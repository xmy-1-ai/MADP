"""Observed-position supervision for single- and multi-output CTR models."""
import tensorflow as tf

from .layers import validate_positions
from .models import ADVERSARIAL_VARIANTS


def _observed(values, position, num_positions=None):
    values = tf.convert_to_tensor(values)
    tf.debugging.assert_rank(values, 2, message="prediction values must have shape [B,1] or [B,P]")
    width = values.shape[-1]
    if num_positions is None:
        if width is None or width == 1:
            raise ValueError("num_positions is required for single-output or dynamically shaped predictions")
        num_positions = width
    position = validate_positions(position, num_positions, tf.shape(values)[0])
    actual_width = tf.shape(values)[1]
    tf.debugging.assert_equal(
        tf.logical_or(tf.equal(actual_width, 1), tf.equal(actual_width, num_positions)), True,
        message="prediction width must be 1 or num_positions",
    )
    if width == 1:
        return tf.squeeze(values, axis=-1)
    if width is not None:
        return tf.gather(values, position, axis=1, batch_dims=1)
    return tf.cond(
        tf.equal(actual_width, 1),
        lambda: tf.reshape(values, [-1]),
        lambda: tf.gather(values, position, axis=1, batch_dims=1),
    )


def observed_ctr(outputs, position, num_positions=None):
    """Select observed CTRs [B], validating positions even for scalar models.

    Pass num_positions for single-output models because [B,1] cannot reveal
    the valid exposure range. Multi-output models can infer P from the shape.
    Invalid indices are errors: positions are never clipped or taken modulo P.
    """
    return _observed(outputs["ctr"], position, num_positions)


def compute_losses(outputs, batch, config):
    """Return scalar loss, ctr_loss, adversarial_loss.

    Only the observed display position is supervised. The other P-1 outputs
    are counterfactual predictions, not negative labels. For MADP, minimizing
    positive position cross-entropy trains the classifier; GRL in the model
    reverses its gradient into the shared representation. The representation
    receives coefficient -adversarial_weight * grl_strength. PAL uses ordinary
    binary cross-entropy on the stable logit of its probability product.
    """
    position = validate_positions(batch["position"], config.num_positions)
    logits = _observed(outputs["logits"], position, config.num_positions)
    labels = tf.cast(tf.convert_to_tensor(batch["label"]), logits.dtype)
    labels = tf.ensure_shape(labels, [None])
    tf.debugging.assert_equal(tf.shape(labels), tf.shape(logits), message="label batch size mismatch")
    tf.debugging.assert_all_finite(labels, "labels must be finite")
    tf.debugging.assert_greater_equal(labels, tf.cast(0, labels.dtype), message="labels must lie in [0,1]")
    tf.debugging.assert_less_equal(labels, tf.cast(1, labels.dtype), message="labels must lie in [0,1]")
    ctr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    adversarial_loss = tf.zeros([], dtype=ctr_loss.dtype)
    if config.variant in ADVERSARIAL_VARIANTS:
        position_logits = tf.ensure_shape(outputs["position_logits"], [None, config.num_positions])
        tf.debugging.assert_equal(tf.shape(position_logits)[0], tf.shape(position)[0], message="classifier batch size mismatch")
        adversarial_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=position, logits=position_logits,
        ))
    total = ctr_loss + tf.cast(config.adversarial_weight, ctr_loss.dtype) * adversarial_loss
    return {"loss": total, "ctr_loss": ctr_loss, "adversarial_loss": adversarial_loss}

