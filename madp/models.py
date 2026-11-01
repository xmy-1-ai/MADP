"""Nine explicitly separated architectures for the MADP paper experiments.

All multi-position variants use one shared scalar CTR MLP, evaluated at each
position. Their full predictions never read the current exposure position.
Only PAL uses that position in its forward pass, for examination probability.
"""
import tensorflow as tf

from .config import Config
from .layers import (
    ContextAttention, GradientReversal, GroupTransformer, MLP, TargetAttention,
    TransformerBlock, masked_mean, validate_positions,
)


MULTI_VARIANTS = frozenset({"transformer_multi", "madp_transformer", "madp_din", "dpin"})
ADVERSARIAL_VARIANTS = frozenset({"madp_transformer", "madp_din"})
TRANSFORMER_VARIANTS = frozenset({"transformer_single", "transformer_multi", "madp_transformer", "pal"})


class CTRModel(tf.keras.Model):
    """A model selected by ``Config.variant`` with an explicit input contract.

    Inputs: candidate_id [B], context [B,C], history_ids [B,P,L],
    raw_history_ids [B,P*L], and (PAL only) zero-based position [B]. Padding
    item id is 0. Histories are left-aligned, oldest to newest. An optional
    label field is ignored. Each variant reads only the histories it needs.

    MADP adversarial representation is ``shared_tower([candidate, context,
    mean(nonempty group interests)])``. It participates in every CTR head and
    feeds GRL -> position classifier. The classifier target is the observed
    current display position. Neither display-position embeddings nor that
    observed position enter the reversed-gradient representation. Thus the
    adversarial gradient reaches the history encoder and item/context tower,
    while explicit position embeddings remain available to the CTR heads.
    """

    def __init__(self, config: Config, **kwargs):
        super().__init__(name=kwargs.pop("name", config.variant), **kwargs)
        self.config = config
        self.variant = config.variant
        self.item_embedding = tf.keras.layers.Embedding(
            config.vocab_size, config.embedding_dim, name="item_embedding",
        )
        self.group_encoder = None
        self.interest_dim = config.embedding_dim
        if self.variant in TRANSFORMER_VARIANTS:
            self.group_encoder = GroupTransformer(config, name="group_transformer")
            self.interest_dim = config.model_dim
        elif self.variant in {"din_single", "madp_din"}:
            self.group_encoder = TargetAttention(config.mlp_dims, config.dropout, name="din_attention")
        elif self.variant == "dpin":
            self.group_encoder = ContextAttention(config.mlp_dims, config.dropout, name="context_attention")

        self.click_predictor = MLP(config.mlp_dims, config.dropout, output_dim=1, name="ctr_mlp")
        if self.variant in MULTI_VARIANTS or self.variant == "pal":
            self.position_embedding = tf.keras.layers.Embedding(
                config.num_positions, config.embedding_dim, name="display_position",
            )
        if self.variant in {"transformer_multi", "madp_transformer", "madp_din"}:
            self.shared_tower = MLP((config.model_dim,), config.dropout, name="shared_representation")
        if self.variant in ADVERSARIAL_VARIANTS:
            self.gradient_reversal = GradientReversal(config.grl_strength, name="gradient_reversal")
            self.position_classifier = MLP(
                config.mlp_dims, config.dropout, output_dim=config.num_positions,
                name="position_classifier",
            )
        if self.variant == "pal":
            self.examination_predictor = MLP(
                config.mlp_dims, config.dropout, output_dim=1, name="examination_mlp",
            )
        if self.variant == "dpin":
            self.position_interest_projection = MLP(
                (config.model_dim,), config.dropout, name="nonlinear_position_interest",
            )
            self.position_blocks = [TransformerBlock(
                config.model_dim, config.num_heads, config.ffn_dim, config.dropout,
                name=f"cross_position_block_{i}",
            ) for i in range(config.num_layers)]
            self.base_tower = MLP((config.model_dim,), config.dropout, name="base_item_context")

    def _context(self, batch):
        context = tf.convert_to_tensor(batch["context"], dtype=tf.float32)
        context = tf.ensure_shape(context, [None, self.config.context_dim])
        tf.debugging.assert_all_finite(context, "context must be finite")
        return context

    def _ids(self, values, shape, name, allow_padding=True):
        values = tf.convert_to_tensor(values)
        if not values.dtype.is_integer:
            raise TypeError(f"{name} must have an integer dtype")
        values = tf.ensure_shape(values, shape)
        minimum = 0 if allow_padding else 1
        tf.debugging.assert_greater_equal(values, tf.cast(minimum, values.dtype), message=f"invalid {name}")
        tf.debugging.assert_less(values, tf.cast(self.config.vocab_size, values.dtype), message=f"{name} exceeds vocabulary")
        return values

    def _candidate_and_context(self, batch):
        candidate = self._ids(batch["candidate_id"], [None], "candidate_id", allow_padding=False)
        context = self._context(batch)
        tf.debugging.assert_equal(tf.shape(candidate)[0], tf.shape(context)[0], message="candidate/context batch mismatch")
        return self.item_embedding(candidate), context

    def _group_history(self, batch):
        ids = self._ids(
            batch["history_ids"], [None, self.config.num_positions, self.config.history_length], "history_ids",
        )
        valid = tf.not_equal(ids, 0)
        embeddings = self.item_embedding(ids)
        return embeddings * tf.cast(valid[..., None], embeddings.dtype), valid

    def encode_group_interests(self, batch, training=False):
        """Return interests [B,P,D], before any cross-position interaction.

        Available for every grouped-history variant. The raw sequence baseline
        has no position groups and therefore rejects this operation. This
        method also exposes the within-group isolation for structure tests.
        """
        if self.variant == "sequence_flat":
            raise ValueError("sequence_flat intentionally does not read position-grouped histories")
        history, valid = self._group_history(batch)
        if self.variant in TRANSFORMER_VARIANTS:
            return self.group_encoder(history, valid_mask=valid, training=training)
        if self.variant in {"din_single", "madp_din"}:
            candidate, _ = self._candidate_and_context(batch)
            return self.group_encoder(candidate, history, valid_mask=valid, training=training)
        if self.variant == "dpin":
            context = self._context(batch)
            return self.group_encoder(context, history, valid_mask=valid, training=training)
        return masked_mean(history, valid, axis=2)

    def _all_position_embeddings(self, batch_size):
        positions = self.position_embedding(tf.range(self.config.num_positions))
        return tf.broadcast_to(positions[None, :, :], [batch_size, self.config.num_positions, self.config.embedding_dim])

    def dpin_position_interests(self, batch, training=False):
        """DPIN position tower [B,P,model_dim], independent of candidate_id.

        Context-only attention pools each history group. A nonlinear map of
        [group interest, context, position embedding] precedes Transformer
        attention across positions. Empty groups still have context and a
        position embedding, and remain valid nodes in this interaction layer.
        """
        if self.variant != "dpin":
            raise ValueError("dpin_position_interests requires the dpin variant")
        interests = self.encode_group_interests(batch, training=training)
        context = self._context(batch)
        batch_size = tf.shape(context)[0]
        context = tf.broadcast_to(context[:, None, :], [batch_size, self.config.num_positions, self.config.context_dim])
        positions = self._all_position_embeddings(batch_size)
        x = self.position_interest_projection(tf.concat([interests, context, positions], axis=-1), training=training)
        for block in self.position_blocks:
            x = block(x, training=training)
        return x

    def call(self, batch, training=False):
        candidate, context = self._candidate_and_context(batch)
        base = tf.concat([candidate, context], axis=-1)
        batch_size = tf.shape(candidate)[0]

        if self.variant == "sequence_flat":
            ids = self._ids(batch["raw_history_ids"], [None, self.config.raw_history_length], "raw_history_ids")
            tf.debugging.assert_equal(tf.shape(ids)[0], batch_size, message="history batch mismatch")
            history = self.item_embedding(ids) * tf.cast(tf.not_equal(ids, 0)[..., None], candidate.dtype)
            flattened = tf.reshape(history, [-1, self.config.raw_history_length * self.config.embedding_dim])
            logits = self.click_predictor(tf.concat([base, flattened], axis=-1), training=training)
            return {"logits": logits, "ctr": tf.math.sigmoid(logits)}

        if self.variant == "dpin":
            interests = self.dpin_position_interests(batch, training=training)
            shared = self.base_tower(base, training=training)
            shared = tf.broadcast_to(shared[:, None, :], [batch_size, self.config.num_positions, self.config.model_dim])
            features = tf.concat([shared, interests, self._all_position_embeddings(batch_size)], axis=-1)
            logits = tf.squeeze(self.click_predictor(features, training=training), axis=-1)
            return {"logits": logits, "ctr": tf.math.sigmoid(logits)}

        interests = self.encode_group_interests(batch, training=training)
        tf.debugging.assert_equal(tf.shape(interests)[0], batch_size, message="history batch mismatch")
        if self.variant in {"position_pool", "transformer_single", "din_single", "pal"}:
            flattened = tf.reshape(interests, [-1, self.config.num_positions * self.interest_dim])
            relevance_logits = self.click_predictor(tf.concat([base, flattened], axis=-1), training=training)
            if self.variant != "pal":
                return {"logits": relevance_logits, "ctr": tf.math.sigmoid(relevance_logits)}
            observed_position = validate_positions(batch["position"], self.config.num_positions, batch_size)
            examination_logits = self.examination_predictor(self.position_embedding(observed_position), training=training)
            # Exact log-odds of sigmoid(a)*sigmoid(b), without probability
            # clipping or log(1-p) cancellation near probability one.
            denominator = tf.reduce_logsumexp(tf.stack([
                tf.zeros_like(relevance_logits), relevance_logits, examination_logits,
            ], axis=-1), axis=-1)
            logits = relevance_logits + examination_logits - denominator
            relevance = tf.math.sigmoid(relevance_logits)
            examination = tf.math.sigmoid(examination_logits)
            return {"logits": logits, "ctr": relevance * examination,
                    "relevance": relevance, "examination": examination}

        group_valid = tf.reduce_any(tf.not_equal(batch["history_ids"], 0), axis=-1)
        pooled = masked_mean(interests, group_valid, axis=1)
        shared = self.shared_tower(tf.concat([base, pooled], axis=-1), training=training)
        tiled_shared = tf.broadcast_to(shared[:, None, :], [batch_size, self.config.num_positions, self.config.model_dim])
        head_features = tf.concat([tiled_shared, interests, self._all_position_embeddings(batch_size)], axis=-1)
        logits = tf.squeeze(self.click_predictor(head_features, training=training), axis=-1)
        outputs = {"logits": logits, "ctr": tf.math.sigmoid(logits)}
        if self.variant in ADVERSARIAL_VARIANTS:
            outputs["position_logits"] = self.position_classifier(self.gradient_reversal(shared), training=training)
        return outputs


def build_model(config: Config) -> CTRModel:
    """Construct a lazily built model; call once before loading its weights."""
    if not isinstance(config, Config):
        raise TypeError("config must be a madp.config.Config")
    return CTRModel(config)

