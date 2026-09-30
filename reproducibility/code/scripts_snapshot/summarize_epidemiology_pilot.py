#!/usr/bin/env python3
"""Audit and aggregate completed epidemiology pilot seeds."""

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


METRICS = (
    "rmse",
    "nll",
    "coverage50",
    "coverage80",
    "coverage90",
    "coverage95",
    "mean_predictive_std",
    "mean_interval_width90",
)
ROUTEB_OUTPUTS = {
    "routeb_global_inducing": "global_inducing",
    "routeb_cumulative_hippo": "cumulative_hippo",
}
ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260809)
    draws = rng.choice(array, size=(10000, array.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def _validate_routeb_artifact(seed_dir: Path, method: str, reported: dict) -> dict:
    label = ROUTEB_OUTPUTS[method]
    result_path = seed_dir / label / "online" / "result.json"
    prediction_path = seed_dir / label / "online" / "predictions.npz"
    if not result_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"Missing Route B artifacts for {seed_dir.name}/{method}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(prediction_path) as arrays:
        y_true = np.asarray(arrays["y_true"], dtype=np.float64)
        pred_mean = np.asarray(arrays["pred_mean"], dtype=np.float64)
        pred_var = np.asarray(arrays["pred_var"], dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (y_true, pred_mean, pred_var)):
        raise FloatingPointError(f"Non-finite predictions in {prediction_path}")
    if np.any(pred_var <= 0.0):
        raise FloatingPointError(f"Non-positive predictive variance in {prediction_path}")
    recomputed = predictive_metrics(y_true, pred_mean, pred_var)
    for metric in METRICS:
        if not np.isclose(recomputed[metric], float(reported[metric]), rtol=1e-9, atol=1e-10):
            raise ValueError(
                f"Metric mismatch for {seed_dir.name}/{method}/{metric}: "
                f"reported={reported[metric]}, recomputed={recomputed[metric]}"
            )
    return {
        "result_path": str(result_path.resolve()),
        "prediction_path": str(prediction_path.resolve()),
        "split_seed": int(result["split_seed"]),
        "num_predictions": int(y_true.size),
        "minimum_variance": float(pred_var.min()),
        "maximum_variance": float(pred_var.max()),
    }


def aggregate(input_root: Path, dataset: str, expected_seeds: list[int]) -> dict:
    per_seed: list[dict] = []
    artifact_rows: list[dict] = []
    for seed in expected_seeds:
        seed_dir = input_root / f"seed{seed}"
        protocol_json = (
            ROOT / "data" / "epidemiology" / "protocol" / dataset / f"seed{seed}" / "protocol.json"
        )
        summary_path = seed_dir / "pilot_summary.json"
        if not summary_path.exists() or not protocol_json.exists():
            raise FileNotFoundError(f"Missing completed seed {seed}: {seed_dir}")
        protocol = json.loads(protocol_json.read_text(encoding="utf-8"))
        if int(protocol["split_seed"]) != seed:
            raise ValueError(f"Protocol seed mismatch for {protocol_json}")
        rows = json.loads(summary_path.read_text(encoding="utf-8"))["rows"]
        methods = {row["method"]: row for row in rows}
        if set(ROUTEB_OUTPUTS) - set(methods):
            raise ValueError(f"Missing Route B methods in {summary_path}")
        for method, row in methods.items():
            per_seed.append({"dataset": dataset, "seed": seed, **row})
            if method in ROUTEB_OUTPUTS:
                audit = _validate_routeb_artifact(seed_dir, method, row)
                if audit["split_seed"] != seed:
                    raise ValueError(
                        f"Result seed mismatch for {seed_dir.name}/{method}: "
                        f"expected {seed}, found {audit['split_seed']}"
                    )
                artifact_rows.append({"seed": seed, "method": method, **audit})

    methods = sorted({row["method"] for row in per_seed})
    summary_rows: list[dict] = []
    for method in methods:
        selected = [row for row in per_seed if row["method"] == method]
        row: dict[str, object] = {"dataset": dataset, "method": method, "n_seeds": len(selected)}
        for metric in METRICS:
            values = [float(item[metric]) for item in selected]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(row)

    paired_rows: list[dict] = []
    for seed in expected_seeds:
        by_method = {row["method"]: row for row in per_seed if row["seed"] == seed}
        row = {"dataset": dataset, "seed": seed}
        for metric in METRICS:
            row[f"{metric}_hippo_minus_global"] = (
                float(by_method["routeb_cumulative_hippo"][metric])
                - float(by_method["routeb_global_inducing"][metric])
            )
        paired_rows.append(row)

    paired_summary: list[dict] = []
    for metric in METRICS:
        key = f"{metric}_hippo_minus_global"
        values = [float(row[key]) for row in paired_rows]
        ci_low, ci_high = _bootstrap_mean_ci(values)
        paired_summary.append(
            {
                "dataset": dataset,
                "contrast": "routeb_cumulative_hippo - routeb_global_inducing",
                "metric": metric,
                "n_seeds": len(values),
                "mean_difference": mean(values),
                "sample_sd": stdev(values) if len(values) > 1 else 0.0,
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
            }
        )

    input_root.mkdir(parents=True, exist_ok=True)
    _write_csv(input_root / "metrics_per_seed.csv", per_seed)
    _write_csv(input_root / "metrics_aggregate.csv", summary_rows)
    _write_csv(input_root / "paired_routeb_differences.csv", paired_rows)
    _write_csv(input_root / "paired_routeb_summary.csv", paired_summary)
    audit_payload = {
        "dataset": dataset,
        "expected_seeds": expected_seeds,
        "status": "complete",
        "prediction_artifacts": artifact_rows,
    }
    (input_root / "artifact_audit.json").write_text(
        json.dumps(audit_payload, indent=2), encoding="utf-8"
    )

    lines = [
        f"# {dataset.upper()} local-GPU pilot summary",
        "",
        f"Spatial split seeds: {', '.join(map(str, expected_seeds))}. Values are mean +/- sample SD.",
        "",
        "| Method | RMSE | NLL | Coverage90 | Mean predictive std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} "
            f"| {row['nll_mean']:.4f} +/- {row['nll_sd']:.4f} "
            f"| {row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} "
            f"| {row['mean_predictive_std_mean']:.4f} +/- {row['mean_predictive_std_sd']:.4f} |"
        )
    paired_by_metric = {row["metric"]: row for row in paired_summary}
    lines.extend(
        [
            "",
            "## Paired Route B contrast",
            "",
            "Differences are cumulative HiPPO minus global inducing. Negative RMSE/NLL favors HiPPO.",
            "",
            "| Metric | Mean difference | Bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for metric in ("rmse", "nll", "coverage90"):
        row = paired_by_metric[metric]
        lines.append(
            f"| {metric} | {row['mean_difference']:.6f} "
            f"| [{row['bootstrap_95ci_low']:.6f}, {row['bootstrap_95ci_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This retrospective fixed-snapshot experiment contains only 52 locations and 39 strict-online weeks after calibration.",
            "It is an implementation-feasibility pilot, not evidence for a long-stream HiPPO advantage or real-time vintage performance.",
            "All Route B prediction archives were independently recomputed and checked for finite, positive variances.",
        ]
    )
    (input_root / "aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "complete",
        "dataset": dataset,
        "num_seeds": len(expected_seeds),
        "metrics": summary_rows,
        "paired": paired_summary,
        "output": str(input_root.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["covid", "dengue"], required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    result = aggregate(args.input_root, args.dataset, args.expected_seeds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
