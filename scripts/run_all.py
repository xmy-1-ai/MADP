"""Run all nine variants on identical splits and aggregate their AUC tables."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

# Support the documented `python scripts/run_all.py` entry point before an
# editable installation, without any machine-specific path dependency.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from madp.config import Config, VARIANTS
from madp.metrics import format_metric


def aggregate(reports, output):
    """Write rows for every position and overall, even when AUC is undefined."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = list(reports)
    reference = reports[names[0]]
    rows = []
    for position_row in reference["positions"]:
        position = position_row["position"]
        row = {"position": position, "sample_count": position_row["count"]}
        row.update({name: reports[name]["positions"][position]["auc"] for name in names})
        rows.append(row)
    rows.append({"position": "overall", "sample_count": reference["overall"]["count"],
                 **{name: reports[name]["overall"]["auc"] for name in names}})
    fields = ["position", "sample_count", *names]
    with (output / "position_auc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: "NA" if value is None else value for key, value in row.items()} for row in rows)
    lines = ["# Position-wise AUC comparison", "",
             "All variants share the same train/validation/test splits, base configuration, and seed. Checkpoints are selected by validation logloss only.", "",
             "Positions are zero-based. NA means AUC is undefined because the position subset is empty or contains only one class.", ""]
    if reference.get("data", {}).get("synthetic"):
        lines.extend(["**SYNTHETIC SMOKE DATA — these values are not paper results.**", ""])
    lines.extend(["| Position | Sample count | " + " | ".join(names) + " |", "| " + " | ".join(["---:"] * len(fields)) + " |"])
    for row in rows:
        lines.append("| " + str(row["position"]) + " | " + str(row["sample_count"]) + " | " + " | ".join(format_metric(row[name]) for name in names) + " |")
    lines.extend(["", "## Overall metrics", "", "| Variant | Count | AUC | Logloss | CTR | Position AUC | Position AUC coverage |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    summary = []
    for name, report in reports.items():
        summary.append({"variant": name, **report["overall"],
                        "sample_weighted_position_auc": report["sample_weighted_position_auc"],
                        "position_auc_sample_coverage": report["position_auc_sample_coverage"]})
        lines.append("| " + name + " | " + " | ".join(format_metric(value) for value in (
            report["overall"]["count"], report["overall"]["auc"], report["overall"]["logloss"], report["overall"]["ctr"],
            report["sample_weighted_position_auc"], report["position_auc_sample_coverage"])) + " |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "overall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows({key: "NA" if value is None else value for key, value in row.items()} for row in summary)
    (output / "comparison.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="runs/comparison")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = Config.load(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    splits = [Path(value).resolve() for value in (args.train, args.valid, args.test)]
    if len(set(splits)) != 3:
        parser.error("train, valid, and test must be three different files/splits")
    # Distinct copied files can still leak the same split.
    from madp.train import file_sha256
    if len({file_sha256(path) for path in splits}) != 3:
        parser.error("train, valid, and test must not be identical archives")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    base_config = output / "base_config.json"
    config.save(base_config)
    reports = {}
    root = Path(__file__).resolve().parents[1]
    for variant in VARIANTS:
        run = output / variant
        print(f"=== {variant} ===", flush=True)
        subprocess.run([sys.executable, "-m", "madp.train", "--config", str(base_config),
                        "--train", str(splits[0]), "--valid", str(splits[1]), "--output", str(run), "--variant", variant],
                       cwd=root, check=True)
        subprocess.run([sys.executable, "-m", "madp.evaluate", "--run", str(run), "--data", str(splits[2]),
                        "--output", str(run / "test")], cwd=root, check=True)
        reports[variant] = json.loads((run / "test" / "metrics.json").read_text(encoding="utf-8"))
    aggregate(reports, output)
    print(f"Comparison written to {output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()

