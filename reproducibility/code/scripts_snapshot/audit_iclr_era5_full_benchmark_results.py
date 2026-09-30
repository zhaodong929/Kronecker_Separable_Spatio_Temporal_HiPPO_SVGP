#!/usr/bin/env python3
"""Audit ERA5 benchmark artifacts before aggregation and reporting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPECTED_ONLINE_METHODS = (
    "routeb_analytic_hippo_rff",
    "routeb_inducing_points",
    "bui_osgpr_prior_m128",
    "maddox_streaming_sgpr_m128",
    "official_ohsvgp_m32_rff128",
)


def first_metric(result: dict, metric: str):
    for container in ("overall_current_block", "final"):
        value = result.get(container, {})
        if isinstance(value, dict) and value.get(metric) is not None:
            return value[metric]
    return result.get(metric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("task1_2", "task1_10"), default="task1_10")
    parser.add_argument("--expected-blocks", type=int, default=171)
    parser.add_argument("--expected-times", type=int, default=1674)
    parser.add_argument("--expected-space", type=int, default=200)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit_run(run_dir: Path, expected_blocks: int, expected_shape: tuple[int, int]) -> dict:
    row: dict[str, object] = {
        "run_dir": str(run_dir),
        "status": "complete",
        "issues": [],
    }
    issues: list[str] = row["issues"]  # type: ignore[assignment]

    result_path = run_dir / "result.json"
    blocks_path = run_dir / "blocks.csv"
    predictions_path = run_dir / "predictions.npz"
    for path in (result_path, blocks_path, predictions_path):
        if not path.is_file():
            issues.append(f"missing {path.name}")

    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            row["rmse"] = first_metric(result, "rmse")
            row["nll"] = first_metric(result, "nll")
            row["coverage90"] = first_metric(result, "coverage90")
            row["reported_blocks"] = result.get("num_blocks")
            if result.get("num_blocks") != expected_blocks:
                issues.append(
                    f"result.json num_blocks={result.get('num_blocks')} != {expected_blocks}"
                )
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            issues.append(f"invalid result.json: {exc}")

    if blocks_path.is_file():
        try:
            with blocks_path.open(newline="", encoding="utf-8") as handle:
                block_rows = list(csv.DictReader(handle))
            row["block_rows"] = len(block_rows)
            if len(block_rows) != expected_blocks:
                issues.append(f"blocks.csv rows={len(block_rows)} != {expected_blocks}")
        except (OSError, csv.Error) as exc:
            issues.append(f"invalid blocks.csv: {exc}")

    if predictions_path.is_file():
        try:
            with np.load(predictions_path) as predictions:
                shapes = {key: list(predictions[key].shape) for key in predictions.files}
                row["prediction_shapes"] = shapes
                expected_flat_shapes = {
                    (expected_shape[0] * expected_shape[1],),
                    (expected_shape[0] * expected_shape[1], 1),
                }
                for key in ("y_true", "pred_mean", "pred_var"):
                    if key not in predictions:
                        issues.append(f"predictions.npz missing {key}")
                        continue
                    values = predictions[key]
                    if values.shape != expected_shape and values.shape not in expected_flat_shapes:
                        issues.append(
                            f"{key} shape={values.shape} != {expected_shape}"
                        )
                    if not np.all(np.isfinite(values)):
                        issues.append(f"{key} contains non-finite values")
                if "pred_var" in predictions and np.any(predictions["pred_var"] <= 0):
                    issues.append("pred_var contains non-positive values")
                if all(key in predictions for key in ("y_true", "pred_mean", "pred_var")):
                    y_true = np.asarray(predictions["y_true"], dtype=float).reshape(-1)
                    pred_mean = np.asarray(predictions["pred_mean"], dtype=float).reshape(-1)
                    pred_var = np.maximum(
                        np.asarray(predictions["pred_var"], dtype=float).reshape(-1), 1e-10
                    )
                    error = y_true - pred_mean
                    half = 1.6448536269514722 * np.sqrt(pred_var)
                    recomputed = {
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "nll": float(
                            np.mean(
                                0.5
                                * (
                                    np.log(2.0 * np.pi * pred_var)
                                    + error**2 / pred_var
                                )
                            )
                        ),
                        "coverage90": float(
                            np.mean(
                                (y_true >= pred_mean - half)
                                & (y_true <= pred_mean + half)
                            )
                        ),
                    }
                    row["recomputed_metrics"] = recomputed
                    for metric, recomputed_value in recomputed.items():
                        reported = row.get(metric)
                        if reported is None:
                            issues.append(f"result.json missing aggregate {metric}")
                            continue
                        difference = abs(float(reported) - recomputed_value)
                        row[f"{metric}_absolute_difference"] = difference
                        if difference > 1e-8:
                            issues.append(
                                f"reported {metric} differs from predictions by {difference:.3e}"
                            )
        except (OSError, ValueError) as exc:
            issues.append(f"invalid predictions.npz: {exc}")

    if issues:
        row["status"] = "incomplete"
    return row


def main() -> None:
    args = parse_args()
    online_root = args.benchmark_root / "runs" / args.scope / "online"
    rows = []
    for method in EXPECTED_ONLINE_METHODS:
        for seed in range(3):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    **audit_run(
                        online_root / method / f"seed{seed}",
                        args.expected_blocks,
                        (args.expected_times, args.expected_space),
                    ),
                }
            )

    payload = {
        "scope": args.scope,
        "expected_blocks": args.expected_blocks,
        "expected_prediction_shape": [args.expected_times, args.expected_space],
        "complete_runs": sum(row["status"] == "complete" for row in rows),
        "expected_runs": len(rows),
        "runs": rows,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
