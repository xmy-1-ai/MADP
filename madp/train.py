"""Train one variant, selecting its checkpoint only by validation log loss."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
from .config import Config, VARIANTS
from .data import iter_batches, load_npz
from .metrics import binary_metrics


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_tensorflow(seed):
    # Small default pools also make the CPU smoke run practical in constrained
    # environments. Explicit user environment settings take precedence.
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def predict(model, arrays, config):
    from .losses import observed_ctr
    predictions = []
    for batch in iter_batches(arrays, config.batch_size):
        output = model(batch, training=False)
        probability = observed_ctr(output, batch["position"], num_positions=config.num_positions)
        predictions.append(np.asarray(probability).reshape(-1))
    return np.concatenate(predictions)


def train(config, train_path, valid_path, output):
    train_path, valid_path, output = Path(train_path), Path(valid_path), Path(output)
    if train_path.resolve() == valid_path.resolve() or os.path.samefile(train_path, valid_path):
        raise ValueError("Training and validation must be different files/splits")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; choose a new run directory")
    train_data, valid_data = load_npz(train_path, config), load_npz(valid_path, config)
    train_hash, valid_hash = file_sha256(train_path), file_sha256(valid_path)
    if train_hash == valid_hash:
        raise ValueError("Training and validation archives are identical; provide independent splits")
    tf = configure_tensorflow(config.seed)
    from .models import build_model
    from .losses import compute_losses
    model = build_model(config)
    # Build every variable before the first optimizer/checkpoint operation.
    model(next(iter_batches(train_data, config.batch_size)), training=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")
    provenance = {
        "python": platform.python_version(), "tensorflow": tf.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "seed": config.seed, "deterministic_ops": True,
        "selection_metric": "validation_logloss", "command": sys.argv,
        "train": {"path": str(train_path.resolve()), "sha256": train_hash, "count": len(train_data["label"]),
                  "synthetic": (train_path.parent / "SYNTHETIC_DATA.json").exists()},
        "validation": {"path": str(valid_path.resolve()), "sha256": valid_hash, "count": len(valid_data["label"]),
                       "synthetic": (valid_path.parent / "SYNTHETIC_DATA.json").exists()},
    }
    (output / "environment.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    history, best = [], float("inf")
    for epoch in range(config.epochs):
        totals, seen = {}, 0
        for batch in iter_batches(train_data, config.batch_size, shuffle=True, seed=config.seed + epoch):
            with tf.GradientTape() as tape:
                outputs = model(batch, training=True)
                losses = compute_losses(outputs, batch, config)
                loss = losses["loss"]
                tf.debugging.assert_all_finite(loss, "Training loss is not finite")
            gradients = tape.gradient(loss, model.trainable_variables)
            pairs = [(gradient, variable) for gradient, variable in zip(gradients, model.trainable_variables) if gradient is not None]
            if not pairs:
                raise RuntimeError("No trainable gradients were produced")
            for gradient, _ in pairs:
                values = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
                tf.debugging.assert_all_finite(values, "Training gradient is not finite")
            optimizer.apply_gradients(pairs)
            size = len(batch["label"])
            seen += size
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value) * size
        validation = binary_metrics(valid_data["label"], predict(model, valid_data, config))
        improved = validation["logloss"] < best
        if improved:
            best = validation["logloss"]
            model.save_weights(output / "best.weights.h5")
        record = {"epoch": epoch + 1, "train": {name: value / seen for name, value in totals.items()},
                  "validation": validation, "best_checkpoint": improved}
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"epoch={epoch + 1}/{config.epochs} train_loss={record['train']['loss']:.6f} "
              f"validation_logloss={validation['logloss']:.6f} validation_auc={validation['auc']} best={improved}", flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = Config.load(args.config)
    overrides = {name: getattr(args, name) for name in ("variant", "epochs") if getattr(args, name) is not None}
    config = replace(config, **overrides)
    train(config, args.train, args.valid, args.output)


if __name__ == "__main__":
    main()

