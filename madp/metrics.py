"""Dependency-light CTR metrics with explicit undefined-AUC handling."""
import csv
import json
from pathlib import Path
import numpy as np


def _validated(labels, probabilities):
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be equally sized 1-D arrays")
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be finite binary values")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return labels, probabilities


def binary_auc(labels, probabilities):
    """Mann–Whitney AUC using average ranks for tied prediction scores."""
    labels, probabilities = _validated(labels, probabilities)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probabilities, kind="stable")
    scores = probabilities[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    boundaries = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1, len(scores)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[start:end] = (start + 1 + end) / 2
    rank_sum = ranks[labels[order] == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def binary_metrics(labels, probabilities):
    labels, probabilities = _validated(labels, probabilities)
    if len(labels) == 0:
        return {"count": 0, "clicks": 0, "ctr": None, "predicted_ctr": None, "auc": None, "logloss": None}
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "count": len(labels), "clicks": int(labels.sum()), "ctr": float(labels.mean()),
        "predicted_ctr": float(probabilities.mean()), "auc": binary_auc(labels, probabilities),
        "logloss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
    }


def evaluate_predictions(labels, probabilities, positions, num_positions):
    labels, probabilities = _validated(labels, probabilities)
    positions = np.asarray(positions)
    if positions.shape != labels.shape or positions.dtype.kind not in "iu":
        raise ValueError("positions must be a 1-D integer array aligned to labels")
    if np.any(positions < 0) or np.any(positions >= num_positions):
        raise ValueError("positions contain out-of-range values")
    rows = []
    for position in range(num_positions):
        mask = positions == position
        rows.append({"position": position, **binary_metrics(labels[mask], probabilities[mask])})
    eligible = [row for row in rows if row["auc"] is not None]
    covered_count = sum(row["count"] for row in eligible)
    return {
        "overall": binary_metrics(labels, probabilities), "positions": rows,
        "sample_weighted_position_auc": sum(row["count"] * row["auc"] for row in eligible) / covered_count if covered_count else None,
        "position_auc_sample_coverage": covered_count / len(labels) if len(labels) else None,
        "position_auc_defined_positions": len(eligible),
    }


def format_metric(value):
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{value:.6f}"


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["position", "count", "clicks", "ctr", "predicted_ctr", "auc", "logloss"]
    with (output / "positions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: "NA" if row[key] is None else row[key] for key in columns} for row in report["positions"])
    overall = report["overall"]
    lines = ["# Evaluation", ""]
    if report.get("data", {}).get("synthetic"):
        lines.extend(["**SYNTHETIC SMOKE DATA — these values are not paper results.**", ""])
    lines.extend(["Position indices are zero-based. NA means an empty subset or undefined single-class AUC.", "",
             f"Overall: count={overall['count']}, AUC={format_metric(overall['auc'])}, logloss={format_metric(overall['logloss'])}, CTR={format_metric(overall['ctr'])}.", "",
             f"Sample-weighted position AUC: {format_metric(report['sample_weighted_position_auc'])}; sample coverage: {format_metric(report['position_auc_sample_coverage'])}.", "",
             "| Position | Count | Clicks | CTR | Predicted CTR | AUC | Logloss |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["positions"]:
        lines.append("| " + " | ".join(format_metric(row[key]) for key in columns) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

