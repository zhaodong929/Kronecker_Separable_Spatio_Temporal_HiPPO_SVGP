#!/usr/bin/env python3
"""Audit one five-seed COVID HiPPO candidate against the retained baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

try:
    from scripts.run_epidemiology_pilot import predictive_metrics
except ModuleNotFoundError:
    from run_epidemiology_pilot import predictive_metrics


METRICS = ("rmse", "nll", "coverage50", "coverage80", "coverage90", "coverage95", "mean_predictive_std", "mean_interval_width90")


def load_run(root: Path, seed: int, *, require_kernel_metadata: bool) -> dict[str, object]:
    online_dir = root / f"seed{seed}" / "cumulative_hippo" / "online"
    calibration_path = root / f"seed{seed}" / "cumulative_hippo" / "calibration" / "result.json"
    result_path = online_dir / "result.json"
    prediction_path = online_dir / "predictions.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if int(result["split_seed"]) != seed:
        raise ValueError(f"Seed mismatch in {result_path}")
    with np.load(prediction_path) as arrays:
        y_true = np.asarray(arrays["y_true"], dtype=np.float64)
        pred_mean = np.asarray(arrays["pred_mean"], dtype=np.float64)
        pred_var = np.asarray(arrays["pred_var"], dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (y_true, pred_mean, pred_var)):
        raise FloatingPointError(f"Non-finite prediction artifact: {prediction_path}")
    if np.any(pred_var <= 0.0):
        raise FloatingPointError(f"Non-positive predictive variance: {prediction_path}")
    recomputed = predictive_metrics(y_true, pred_mean, pred_var)
    for metric in METRICS:
        if not np.isclose(recomputed[metric], result["overall_current_block"][metric], rtol=1e-9, atol=1e-10):
            raise ValueError(f"Metric mismatch for seed {seed}/{metric}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if require_kernel_metadata:
        if calibration.get("temporal_kernel") != result.get("temporal_kernel"):
            raise ValueError(f"Calibration/online temporal kernel mismatch for seed {seed}")
    return {
        "metrics": recomputed,
        "result_path": str(result_path.resolve()),
        "prediction_path": str(prediction_path.resolve()),
        "minimum_variance": float(pred_var.min()),
        "maximum_variance": float(pred_var.max()),
        "temporal_kernel": result.get("temporal_kernel"),
    }


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260810)
    draws = values[rng.integers(0, len(values), size=(100000, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    candidate_rows = []
    paired_rows = []
    audits = []
    for seed in args.seeds:
        candidate = load_run(args.candidate_root, seed, require_kernel_metadata=True)
        baseline = load_run(args.baseline_root, seed, require_kernel_metadata=False)
        candidate_rows.append({"seed": seed, **candidate["metrics"]})
        paired = {"seed": seed}
        for metric in METRICS:
            paired[f"{metric}_candidate_minus_baseline"] = (
                candidate["metrics"][metric] - baseline["metrics"][metric]
            )
        paired_rows.append(paired)
        audits.append(
            {
                "seed": seed,
                "candidate": {key: candidate[key] for key in candidate if key != "metrics"},
                "baseline": {key: baseline[key] for key in baseline if key != "metrics"},
            }
        )

    aggregate = {"candidate": args.candidate_name, "n_seeds": len(candidate_rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in candidate_rows]
        aggregate[f"{metric}_mean"] = mean(values)
        aggregate[f"{metric}_sd"] = stdev(values)
    paired_summary = []
    for metric in METRICS:
        values = np.asarray([row[f"{metric}_candidate_minus_baseline"] for row in paired_rows])
        low, high = bootstrap_ci(values)
        paired_summary.append(
            {
                "metric": metric,
                "mean_difference": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
            }
        )

    args.candidate_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.candidate_root / "metrics_per_seed.csv", candidate_rows)
    write_csv(args.candidate_root / "metrics_aggregate.csv", [aggregate])
    write_csv(args.candidate_root / "paired_vs_retained_baseline.csv", paired_rows)
    write_csv(args.candidate_root / "paired_summary_vs_retained_baseline.csv", paired_summary)
    (args.candidate_root / "artifact_audit.json").write_text(
        json.dumps({"status": "complete", "candidate": args.candidate_name, "audits": audits}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# COVID Q=2 spectral-mixture confirmation",
        "",
        "This is a five-spatial-split feasibility-pilot comparison. Each metric was independently recomputed from saved predictions; candidate calibration and online runs use identical fixed spectral-mixture metadata.",
        "",
        "| Candidate | RMSE | NLL | Coverage90 |",
        "|---|---:|---:|---:|",
        f"| {args.candidate_name} | {aggregate['rmse_mean']:.4f} +/- {aggregate['rmse_sd']:.4f} | {aggregate['nll_mean']:.4f} +/- {aggregate['nll_sd']:.4f} | {aggregate['coverage90_mean']:.4f} +/- {aggregate['coverage90_sd']:.4f} |",
        "",
        "| Metric | Candidate minus retained causal Matérn baseline | Bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for row in paired_summary:
        lines.append(
            f"| {row['metric']} | {row['mean_difference']:+.4f} | "
            f"[{row['bootstrap_95ci_low']:+.4f}, {row['bootstrap_95ci_high']:+.4f}] |"
        )
    (args.candidate_root / "confirmation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "paired_summary": paired_summary}, indent=2))


if __name__ == "__main__":
    main()
