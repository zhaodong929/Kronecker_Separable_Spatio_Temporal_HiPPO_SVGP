#!/usr/bin/env python3
"""Re-aggregate a saved Task-2 online stream into causal reporting segments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


Z_BY_COVERAGE = {
    0.50: 0.6744897501960817,
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
}


def segment_slices(num_times: int, num_segments: int) -> list[tuple[int, int]]:
    if num_times <= 0 or num_segments <= 0 or num_segments > num_times:
        raise ValueError("num_segments must be in [1, num_times]")
    base, remainder = divmod(num_times, num_segments)
    lengths = [base + (index < remainder) for index in range(num_segments)]
    starts = np.cumsum([0, *lengths[:-1]], dtype=int)
    return [(int(start), int(start + length)) for start, length in zip(starts, lengths)]


def metrics(y_true: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    mu = np.asarray(mean, dtype=float).reshape(-1)
    var = np.asarray(variance, dtype=float).reshape(-1)
    if y.shape != mu.shape or y.shape != var.shape:
        raise ValueError("y_true, pred_mean and pred_var must have identical shapes")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(var))):
        raise FloatingPointError("Prediction arrays contain non-finite values")
    if np.any(var <= 0.0):
        raise FloatingPointError("Prediction variances must be strictly positive")
    std = np.sqrt(var)
    result = {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "nll": float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var))),
        "mean_predictive_std": float(np.mean(std)),
        "mean_interval_width90": float(2.0 * Z_BY_COVERAGE[0.90] * np.mean(std)),
    }
    for coverage, z in Z_BY_COVERAGE.items():
        result[f"coverage{int(coverage * 100)}"] = float(
            np.mean((y >= mu - z * std) & (y <= mu + z * std))
        )
    return result


def summarize(predictions: Path, output_dir: Path, num_segments: int) -> dict:
    with np.load(predictions) as payload:
        required = {"y_true", "pred_mean", "pred_var"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise KeyError(f"Missing prediction arrays: {missing}")
        y_true = np.asarray(payload["y_true"])
        pred_mean = np.asarray(payload["pred_mean"])
        pred_var = np.asarray(payload["pred_var"])
        times = np.asarray(payload["times"]) if "times" in payload.files else np.arange(y_true.shape[0])

    if y_true.ndim != 2 or pred_mean.shape != y_true.shape or pred_var.shape != y_true.shape:
        raise ValueError("Prediction arrays must have shape [time, heldout_location]")
    if times.shape != (y_true.shape[0],):
        raise ValueError("times must have one value per prediction time")

    slices = segment_slices(y_true.shape[0], num_segments)
    rows = []
    for segment_id, (start, stop) in enumerate(slices, start=1):
        row = {
            "segment_id": segment_id,
            "time_start_index": start,
            "time_stop_index": stop,
            "hours": stop - start,
            "physical_time_start": str(times[start]),
            "physical_time_stop_exclusive": str(times[stop - 1]),
            **metrics(y_true[start:stop], pred_mean[start:stop], pred_var[start:stop]),
        }
        rows.append(row)

    if sum(int(row["hours"]) for row in rows) != y_true.shape[0]:
        raise AssertionError("Segment durations do not cover the prediction stream exactly")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "task2_online_segments.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "predictions": str(predictions),
        "num_times": int(y_true.shape[0]),
        "num_heldout_locations": int(y_true.shape[1]),
        "num_segments": int(num_segments),
        "segment_hours": [int(row["hours"]) for row in rows],
        "segments": rows,
    }
    (output_dir / "task2_online_segments.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-segments", type=int, default=10)
    args = parser.parse_args()
    summary = summarize(args.predictions, args.output_dir, args.num_segments)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
