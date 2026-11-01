"""Convert chronological, pre-request JSONL histories into the common NPZ schema."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import Config
from .data import save_npz, validate_arrays


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def convert_records(records, config):
    """Keep each position's latest L events and preserve their global chronology.

    Callers must supply histories in oldest-to-newest order, with all events
    occurring before the request. That temporal contract cannot be inferred
    from item IDs; construct chronological, leakage-free splits upstream.
    """
    output = {name: [] for name in ("candidate_id", "context", "history_ids", "raw_history_ids", "position", "label")}
    for row_number, record in enumerate(records, 1):
        try:
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
            candidate_id = _integer(record["candidate_id"], "candidate_id", 1, config.vocab_size - 1)
            position = _integer(record["position"], "position", 0, config.num_positions - 1)
            context = record["context"]
            if not isinstance(context, list) or len(context) != config.context_dim:
                raise ValueError(f"context must be a list of {config.context_dim} numeric values")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in context):
                raise ValueError("context entries must be numeric")
            label = record["label"]
            if isinstance(label, bool) or not isinstance(label, (int, float)) or label not in (0, 1):
                raise ValueError("label must be 0 or 1")
            history = record["history"]
            if not isinstance(history, list):
                raise ValueError("history must be a list")
            groups = [[] for _ in range(config.num_positions)]
            for event_index, event in enumerate(history):
                if not isinstance(event, dict):
                    raise ValueError("history events must be objects")
                item_id = _integer(event["item_id"], "history.item_id", 1, config.vocab_size - 1)
                event_position = _integer(event["position"], "history.position", 0, config.num_positions - 1)
                groups[event_position].append((event_index, item_id))
            grouped = np.zeros((config.num_positions, config.history_length), dtype=np.int32)
            retained = []
            for index, group in enumerate(groups):
                latest = group[-config.history_length:]
                grouped[index, :len(latest)] = [event[1] for event in latest]
                retained.extend(latest)
            retained.sort(key=lambda event: event[0])
            raw = np.zeros(config.raw_history_length, dtype=np.int32)
            raw[:len(retained)] = [event[1] for event in retained]
            output["candidate_id"].append(candidate_id)
            output["position"].append(position)
            output["context"].append(context)
            output["label"].append(label)
            output["history_ids"].append(grouped)
            output["raw_history_ids"].append(raw)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid record {row_number}: {exc}") from exc
    if not output["label"]:
        raise ValueError("JSONL input contains no records")
    arrays = {key: np.asarray(value, dtype=np.float32 if key in ("context", "label") else np.int32)
              for key, value in output.items()}
    return validate_arrays(arrays, config)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line {line_number}; every line must be one record")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc.msg}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="JSONL input, one request per line")
    parser.add_argument("--output", required=True, help="Destination .npz archive")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("input and output must be different files")
    config = Config.load(args.config)
    arrays = convert_records(read_jsonl(args.input), config)
    save_npz(args.output, arrays, config)
    print(f"Prepared {len(arrays['label'])} examples: {args.output}", flush=True)


if __name__ == "__main__":
    main()

