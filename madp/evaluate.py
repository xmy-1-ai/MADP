"""Evaluate one saved best checkpoint with complete per-position reporting."""
import argparse
from pathlib import Path
from .config import Config
from .data import iter_batches, load_npz
from .metrics import evaluate_predictions, write_report
from .train import configure_tensorflow, file_sha256, predict


def evaluate(run, data, output):
    run, data = Path(run), Path(data)
    config = Config.load(run / "config.json")
    arrays = load_npz(data, config)
    configure_tensorflow(config.seed)
    from .models import build_model
    model = build_model(config)
    model(next(iter_batches(arrays, config.batch_size)), training=False)
    model.load_weights(run / "best.weights.h5")
    probabilities = predict(model, arrays, config)
    report = evaluate_predictions(arrays["label"], probabilities, arrays["position"], config.num_positions)
    report["variant"] = config.variant
    report["data"] = {"path": str(data.resolve()), "sha256": file_sha256(data),
                      "synthetic": (data.parent / "SYNTHETIC_DATA.json").exists()}
    report["checkpoint_sha256"] = file_sha256(run / "best.weights.h5")
    write_report(report, output)
    print(f"{config.variant}: count={report['overall']['count']} AUC={report['overall']['auc']} "
          f"logloss={report['overall']['logloss']:.6f} -> {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluate(args.run, args.data, args.output)


if __name__ == "__main__":
    main()

