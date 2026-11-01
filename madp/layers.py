"""Small, reusable TensorFlow layers for the MADP experiment family."""
import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    """A pointwise MLP; the same weights apply at every leading index."""

    def __init__(self, widths, dropout=0.0, output_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.network = tf.keras.Sequential(name="network")
        for index, width in enumerate(widths):
            self.network.add(tf.keras.layers.Dense(width, activation="relu", name=f"dense_{index}"))
            if dropout:
                self.network.add(tf.keras.layers.Dropout(dropout, name=f"dropout_{index}"))
        if output_dim is not None:
            self.network.add(tf.keras.layers.Dense(output_dim, name="output"))

    def call(self, inputs, training=False):
        return self.network(inputs, training=training)


def masked_mean(values, mask, axis):
    """Mean over real events; an entirely padded group yields exactly zero."""
    weights = tf.cast(mask, values.dtype)[..., None]
    return tf.math.divide_no_nan(
        tf.reduce_sum(values * weights, axis=axis),
        tf.reduce_sum(weights, axis=axis),
    )


def masked_softmax(logits, mask):
    """Normalize real events only, returning zero weights for an empty group."""
    floor = tf.cast(-1e9, logits.dtype)
    masked = tf.where(mask, logits, floor)
    weights = tf.nn.softmax(masked, axis=-1) * tf.cast(mask, logits.dtype)
    return tf.math.divide_no_nan(weights, tf.reduce_sum(weights, axis=-1, keepdims=True))


class TransformerBlock(tf.keras.layers.Layer):
    """Self attention with safe padding, including wholly empty sequences."""

    def __init__(self, model_dim, num_heads, ffn_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=model_dim // num_heads,
            dropout=dropout, name="self_attention",
        )
        self.attention_dropout = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ffn_dim, activation="relu"),
            tf.keras.layers.Dense(model_dim),
        ], name="ffn")
        self.ffn_dropout = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, valid_mask=None, training=False):
        if valid_mask is None:
            valid_mask = tf.ones(tf.shape(inputs)[:2], dtype=tf.bool)
        # Give empty rows one zero-valued key, then erase padded query outputs.
        # Thus attention never receives a row of entirely masked keys.
        first = tf.equal(tf.range(tf.shape(inputs)[1])[None, :], 0)
        empty = tf.logical_not(tf.reduce_any(valid_mask, axis=-1, keepdims=True))
        safe_keys = tf.logical_or(valid_mask, tf.logical_and(empty, first))
        attended = self.attention(
            inputs, inputs, attention_mask=safe_keys[:, None, :], training=training,
        )
        x = self.norm1(inputs + self.attention_dropout(attended, training=training))
        x = self.norm2(x + self.ffn_dropout(self.ffn(x), training=training))
        return x * tf.cast(valid_mask[..., None], x.dtype)


class GroupTransformer(tf.keras.layers.Layer):
    """One shared Transformer applied independently inside each position group.

    Click-order embeddings describe oldest-to-newest sequence indices. They
    are not display-position embeddings. No attention crosses position groups.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.length = config.history_length
        self.width = config.model_dim
        self.projection = tf.keras.layers.Dense(self.width, name="event_projection")
        self.order_embedding = tf.keras.layers.Embedding(self.length, self.width, name="click_order")
        self.blocks = [TransformerBlock(
            self.width, config.num_heads, config.ffn_dim, config.dropout,
            name=f"block_{i}",
        ) for i in range(config.num_layers)]

    def call(self, embeddings, valid_mask, training=False):
        batch, groups = tf.shape(embeddings)[0], tf.shape(embeddings)[1]
        x = self.projection(embeddings)
        x = x + self.order_embedding(tf.range(self.length))[None, None, :, :]
        x = x * tf.cast(valid_mask[..., None], x.dtype)
        x = tf.reshape(x, [-1, self.length, self.width])
        flat_mask = tf.reshape(valid_mask, [-1, self.length])
        for block in self.blocks:
            x = block(x, valid_mask=flat_mask, training=training)
        pooled = masked_mean(x, flat_mask, axis=1)
        return tf.reshape(pooled, [batch, groups, self.width])


class TargetAttention(tf.keras.layers.Layer):
    """Groupwise target attention with (q, h, q-h, q*h) local activations.

    The query has shape [B,D], histories [B,P,L,D]. Following DIN, local
    activation weights are not softmax-normalized, preserving interest
    intensity. The candidate is the query; empty groups yield zero interests.
    """

    def __init__(self, widths, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.scorer = MLP(widths, dropout, output_dim=1, name="local_activation")

    def call(self, query, history, valid_mask, training=False):
        tf.debugging.assert_equal(tf.shape(query)[0], tf.shape(history)[0], message="query/history batch mismatch")
        q = tf.broadcast_to(query[:, None, None, :], tf.shape(history))
        features = tf.concat([q, history, q - history, q * history], axis=-1)
        scores = tf.squeeze(self.scorer(features, training=training), axis=-1)
        weights = scores * tf.cast(valid_mask, scores.dtype)
        return tf.reduce_sum(history * weights[..., None], axis=2)


class ContextAttention(tf.keras.layers.Layer):
    """DPIN pooling: normalized groupwise attention conditioned on context.

    Attention scores use only concat(history event, current context). The
    candidate item is intentionally absent from this position-interest tower.
    """

    def __init__(self, widths, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.scorer = MLP(widths, dropout, output_dim=1, name="context_activation")

    def call(self, context, history, valid_mask, training=False):
        tf.debugging.assert_equal(tf.shape(context)[0], tf.shape(history)[0], message="context/history batch mismatch")
        target_shape = tf.concat([tf.shape(history)[:3], tf.shape(context)[-1:]], axis=0)
        tiled_context = tf.broadcast_to(context[:, None, None, :], target_shape)
        scores = tf.squeeze(self.scorer(tf.concat([history, tiled_context], axis=-1), training=training), axis=-1)
        weights = masked_softmax(scores, valid_mask)
        return tf.reduce_sum(history * weights[..., None], axis=2)


class GradientReversal(tf.keras.layers.Layer):
    """Identity forward pass; encoder gradients are multiplied by -strength."""

    def __init__(self, strength=1.0, **kwargs):
        super().__init__(**kwargs)
        self.strength = float(strength)

    def call(self, inputs):
        strength = self.strength

        @tf.custom_gradient
        def reverse(x):
            def gradient(upstream):
                return -tf.cast(strength, upstream.dtype) * upstream
            return tf.identity(x), gradient

        return reverse(inputs)


def validate_positions(position, num_positions, batch_size=None):
    """Validate zero-based exposure positions without silently repairing them."""
    position = tf.convert_to_tensor(position)
    if not position.dtype.is_integer:
        raise TypeError("position must have an integer dtype")
    tf.debugging.assert_rank(position, 1, message="position must have shape [B]")
    tf.debugging.assert_greater_equal(position, tf.cast(0, position.dtype), message="position must be >= 0")
    tf.debugging.assert_less(position, tf.cast(num_positions, position.dtype), message="position exceeds num_positions")
    if batch_size is not None:
        tf.debugging.assert_equal(tf.shape(position)[0], batch_size, message="position batch size mismatch")
    return tf.cast(position, tf.int32)

