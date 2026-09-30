#!/usr/bin/env python3
"""Summarize the Route-B efficiency training matrix without imputing missing data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import numpy as np


METRICS = (
    "rmse",
    "nll",
    "coverage90",
    "best_iteration",
    "iterations_completed",
    "best_validation_nll",
    "time_to_best_validation_seconds",
    "training_seconds",
    "process_total_seconds",
    "mean_steady_state_iteration_seconds",
    "peak_cuda_allocated_mib",
    "peak_cuda_reserved_mib",
    "conditional_residual_mean",
    "conditional_residual_raw_mean",
)


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def identify_run(path: Path, root: Path) -> tuple[str, str, str, int]:
    relative = path.relative_to(root)
    if len(relative.parts) < 5 or relative.name != "result.json":
        raise ValueError(f"Unexpected result path: {relative}")
    scope, objective, version, seed_dir = relative.parts[-5:-1]
    if not seed_dir.startswith("seed"):
        raise ValueError(f"Expected seed directory in {relative}")
    return scope, objective, version, int(seed_dir.removeprefix("seed"))


def result_row(path: Path, root: Path) -> dict[str, Any]:
    scope, objective, version, seed = identify_run(path, root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    theta = payload.get("learned_theta") or {}
    ell_s = theta.get("ell_s") if isinstance(theta, dict) else None
    ell_s_1 = ell_s[0] if isinstance(ell_s, list) and len(ell_s) >= 1 else None
    ell_s_2 = ell_s[1] if isinstance(ell_s, list) and len(ell_s) >= 2 else None
    noise_std = finite_number(theta.get("noise_std")) if isinstance(theta, dict) else None
    row = {
        "scope": scope,
        "objective": objective,
        "version": version,
        "seed": seed,
        "result_path": str(path),
        "rmse": finite_number(nested(payload, "final", "rmse")),
        "nll": finite_number(nested(payload, "final", "nll")),
        "coverage90": finite_number(nested(payload, "final", "coverage90")),
        "best_iteration": finite_number(payload.get("best_iteration")),
        "iterations_completed": finite_number(payload.get("iterations_completed")),
        "best_validation_nll": finite_number(payload.get("best_validation_nll")),
        "time_to_best_validation_seconds": finite_number(
            payload.get("time_to_best_validation_seconds")
        ),
        "training_seconds": finite_number(nested(payload, "timing", "training_seconds")),
        "process_total_seconds": finite_number(
            nested(payload, "timing", "process_total_seconds")
        ),
        "mean_steady_state_iteration_seconds": finite_number(
            nested(payload, "timing", "mean_steady_state_iteration_seconds")
        ),
        "posterior_setup_seconds": finite_number(
            nested(payload, "timing", "posterior_setup_seconds")
        ),
        "prediction_seconds": finite_number(
            nested(payload, "timing", "prediction_seconds")
        ),
        "conditional_residual_mean": finite_number(
            nested(payload, "final", "diagnostic_avg_nu_star")
        ),
        "conditional_residual_raw_mean": finite_number(
            nested(payload, "final", "diagnostic_avg_nu_star_raw")
        ),
        "peak_cuda_allocated_mib": finite_number(
            nested(payload, "resources", "peak_cuda_allocated_mib")
        ),
        "peak_cuda_reserved_mib": finite_number(
            nested(payload, "resources", "peak_cuda_reserved_mib")
        ),
        "ell_t": finite_number(theta.get("ell_t")) if isinstance(theta, dict) else None,
        "ell_s_1": finite_number(ell_s_1),
        "ell_s_2": finite_number(ell_s_2),
        "kernel_variance": finite_number(theta.get("kernel_variance"))
        if isinstance(theta, dict)
        else None,
        "noise_std": noise_std,
        "noise_variance": None if noise_std is None else float(noise_std) ** 2,
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["scope"]), str(row["objective"]), str(row["version"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (scope, objective, version), group in sorted(group_rows(rows).items()):
        aggregate: dict[str, Any] = {
            "scope": scope,
            "objective": objective,
            "version": version,
            "n_runs": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda r: r["seed"])),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group if finite_number(row.get(metric)) is not None]
            aggregate[f"{metric}_n"] = len(values)
            aggregate[f"{metric}_mean"] = statistics.fmean(values) if values else None
            aggregate[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        output.append(aggregate)
    return output


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["scope"], row["objective"], row["seed"], row["version"]): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    triples = sorted({(row["scope"], row["objective"], row["seed"]) for row in rows})
    for scope, objective, seed in triples:
        original = indexed.get((scope, objective, seed, "original"))
        optimized = indexed.get((scope, objective, seed, "optimized"))
        row: dict[str, Any] = {
            "scope": scope,
            "objective": objective,
            "seed": seed,
            "pair_complete": original is not None and optimized is not None,
        }
        for metric in METRICS:
            left = None if original is None else finite_number(original.get(metric))
            right = None if optimized is None else finite_number(optimized.get(metric))
            row[f"original_{metric}"] = left
            row[f"optimized_{metric}"] = right
            row[f"delta_optimized_minus_original_{metric}"] = (
                None if left is None or right is None else float(right) - float(left)
            )
        if original is not None and optimized is not None:
            original_predictions = Path(str(original["result_path"])).parent / "predictions.npz"
            optimized_predictions = Path(str(optimized["result_path"])).parent / "predictions.npz"
            if original_predictions.is_file() and optimized_predictions.is_file():
                with np.load(original_predictions) as left_payload, np.load(optimized_predictions) as right_payload:
                    for key, label in (("pred_mean", "predictive_mean"), ("pred_var", "predictive_variance")):
                        left_array = np.asarray(left_payload[key], dtype=np.float64)
                        right_array = np.asarray(right_payload[key], dtype=np.float64)
                        denominator = max(1.0, float(np.linalg.norm(left_array.reshape(-1))))
                        row[f"{label}_relative_error"] = float(
                            np.linalg.norm((right_array - left_array).reshape(-1)) / denominator
                        )
                        row[f"original_{label}_finite"] = bool(np.isfinite(left_array).all())
                        row[f"optimized_{label}_finite"] = bool(np.isfinite(right_array).all())
                    row["original_min_predictive_variance"] = float(left_payload["pred_var"].min())
                    row["optimized_min_predictive_variance"] = float(right_payload["pred_var"].min())
                    row["original_nonpositive_variance_count"] = int(
                        np.count_nonzero(left_payload["pred_var"] <= 0.0)
                    )
                    row["optimized_nonpositive_variance_count"] = int(
                        np.count_nonzero(right_payload["pred_var"] <= 0.0)
                    )
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.root.glob("*/*/*/seed*/result.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            rows.append(result_row(path, args.root))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"WARNING: skipped {path}: {exc}", file=sys.stderr)
    if not rows:
        print(f"WARNING: no valid result.json files under {args.root}", file=sys.stderr)

    write_csv(args.output / "training_metrics_per_run.csv", rows)
    write_csv(args.output / "training_metrics_aggregate.csv", aggregate_rows(rows))
    write_csv(args.output / "paired_parity.csv", paired_rows(rows))
    theta_fields = (
        "scope", "objective", "version", "seed", "ell_t", "ell_s_1", "ell_s_2",
        "kernel_variance", "noise_std", "noise_variance",
    )
    write_csv(
        args.output / "learned_hyperparameters.csv",
        [{field: row.get(field) for field in theta_fields} for row in rows],
    )
    print(json.dumps({"valid_runs": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
