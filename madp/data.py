"""Strict, pickle-free input validation and NumPy mini-batches.

IDs are 1-based (zero is padding); positions are 0-based. Histories contain
only events preceding the request and are left-aligned, oldest to newest.
"""
from pathlib import Path
import numpy as np


FIELDS = ("candidate_id", "context", "history_ids", "raw_history_ids", "position", "label")


def validate_arrays(arrays, config):
    """Validate the public NPZ schema and return canonical NumPy arrays."""
    missing, extra = set(FIELDS) - set(arrays), set(arrays) - set(FIELDS)
    if missing or extra:
        raise ValueError(f"Invalid dataset fields; missing={sorted(missing)}, extra={sorted(extra)}")
    values = {key: np.asarray(arrays[key]) for key in FIELDS}
    if values["candidate_id"].ndim != 1:
        raise ValueError("candidate_id must have shape [N]")
    n = len(values["candidate_id"])
    if n == 0:
        raise ValueError("Dataset must contain at least one example")
    shapes = {
        "candidate_id": (n,), "context": (n, config.context_dim),
        "history_ids": (n, config.num_positions, config.history_length),
        "raw_history_ids": (n, config.raw_history_length),
        "position": (n,), "label": (n,),
    }
    for name, value in values.items():
        if value.shape != shapes[name]:
            raise ValueError(f"{name}: expected shape {shapes[name]}, got {value.shape}")
        expected = "iu" if name in ("candidate_id", "history_ids", "raw_history_ids", "position") else "f"
        if value.dtype.kind not in expected:
            raise ValueError(f"{name}: expected {'integer' if expected == 'iu' else 'floating point'} dtype, got {value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    for name in ("candidate_id", "history_ids", "raw_history_ids"):
        minimum = 1 if name == "candidate_id" else 0
        if np.any(values[name] < minimum) or np.any(values[name] >= config.vocab_size):
            raise ValueError(f"{name}: IDs must be in [{minimum}, {config.vocab_size - 1}]")
    if np.any(values["position"] < 0) or np.any(values["position"] >= config.num_positions):
        raise ValueError(f"position must be in [0, {config.num_positions - 1}]")
    if not np.isin(values["label"], [0.0, 1.0]).all():
        raise ValueError("label must contain only 0.0 and 1.0")
    for name in ("history_ids", "raw_history_ids"):
        value = values[name]
        if np.any((value[..., :-1] == 0) & (value[..., 1:] != 0)):
            raise ValueError(f"{name}: nonzero events must be left-aligned, followed only by zero padding")
    # Global order cannot be recovered from position buckets. The raw history
    # must be supplied separately, but it must include the same retained events.
    grouped_sorted = np.sort(values["history_ids"].reshape(n, -1), axis=1)
    if not np.array_equal(grouped_sorted, np.sort(values["raw_history_ids"], axis=1)):
        raise ValueError("raw_history_ids and history_ids must contain the same retained event multiset for each example")
    result = {name: value.astype(np.int32 if name in ("candidate_id", "history_ids", "raw_history_ids", "position") else np.float32)
              for name, value in values.items()}
    if not np.isfinite(result["context"]).all():
        raise ValueError("context overflows float32")
    return result


def load_npz(path, config):
    """Load an NPZ archive without ever permitting Python object unpickling."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    return validate_arrays(arrays, config)


def save_npz(path, arrays, config):
    """Validate first, then write a compressed archive without suffix surprises."""
    values = validate_arrays(arrays, config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **values)


def iter_batches(arrays, batch_size, shuffle=False, seed=42):
    """Yield dictionaries; no tf.data worker/thread pool is required."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    order = np.arange(len(arrays["label"]))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        yield {name: value[indices] for name, value in arrays.items()}

