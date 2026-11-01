"""Generate independently sampled toy splits for pipeline checks, not paper results."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import Config
from .data import save_npz
from .prepare import convert_records


def generate_arrays(config, size, seed):
    if size < 2 * config.num_positions:
        raise ValueError(f"Synthetic split size must be >= {2 * config.num_positions} (two samples per position)")
    rng = np.random.default_rng(seed)
    positions = np.arange(size) % config.num_positions
    rng.shuffle(positions)
    records = []
    for position in positions:
        candidate = int(rng.integers(1, config.vocab_size))
        context = rng.normal(size=config.context_dim)
        history = []
        for group in range(config.num_positions):
            length = int(rng.integers(0, config.history_length + 1))
            history.extend({"item_id": int(rng.integers(1, config.vocab_size)), "position": group}
                           for _ in range(length))
        # A sampled global chronology, not a concatenation of position groups.
        rng.shuffle(history)
        affinity = np.mean([event["item_id"] % 8 == candidate % 8 for event in history]) if history else 0.0
        relevance = 1 / (1 + np.exp(-(0.6 * context[0] + 2 * affinity - 0.2)))
        examination = np.exp(-1.4 * int(position) / (config.num_positions - 1))
        label = int(rng.random() < relevance * examination)
        records.append({"candidate_id": candidate, "context": context.tolist(), "position": int(position),
                        "label": label, "history": history})
    return convert_records(records, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    args = parser.parse_args()
    config = Config.load(args.config)
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output must be a new or empty directory; existing synthetic data will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    split_seeds = np.random.SeedSequence(config.seed).spawn(3)
    metadata = {"synthetic": True, "purpose": "Smoke tests only; never report these numbers as paper results", "seed": config.seed, "splits": {}}
    for split, size, seed in zip(("train", "valid", "test"), (args.train_size, args.val_size, args.test_size), split_seeds):
        arrays = generate_arrays(config, size, seed)
        save_npz(output / f"{split}.npz", arrays, config)
        metadata["splits"][split] = {"count": size, "seed_spawn_key": list(seed.spawn_key)}
        print(f"Synthetic {split}: {size} examples -> {output / f'{split}.npz'}", flush=True)
    config.save(output / "config.json")
    (output / "SYNTHETIC_DATA.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

