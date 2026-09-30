#!/usr/bin/env python3
"""Phase-1 COVID diagnostics for the Route B improvement track."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from scripts.run_epidemiology_pilot import predictive_metrics
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_epidemiology_pilot import predictive_metrics


def constant_metrics(y_true: np.ndarray, mean: np.ndarray, variance: float) -> dict[str, float]:
    return predictive_metrics(y_true, mean, np.full_like(mean, max(float(variance), 1e-10)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--routeb-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-aware", action="store_true")
    args = parser.parse_args()

    with np.load(args.protocol_npz) as arrays:
        calibration_y = np.asarray(arrays["calibration_y"], dtype=np.float64)
        stream_y = np.asarray(arrays["stream_y"], dtype=np.float64)
        train = np.asarray(arrays["train_indices"], dtype=int)
        test = np.asarray(arrays["test_indices"], dtype=int)
        task1_stream_mean = np.asarray(arrays["task1_stream_mean"], dtype=np.float64)
        task1_calibration_mean = np.asarray(arrays["task1_calibration_mean"], dtype=np.float64)
    with np.load(args.routeb_predictions) as predictions:
        y_true = np.asarray(predictions["y_true"], dtype=np.float64)
        routeb_mean = np.asarray(predictions["pred_mean"], dtype=np.float64)
        routeb_var = np.asarray(predictions["pred_var"], dtype=np.float64)
    if not np.allclose(y_true, stream_y[:, test]):
        raise ValueError("Route B predictions do not match the protocol held-out labels")

    residual = calibration_y[:, train] - task1_calibration_mean[:, train]
    variance = max(float(np.mean(residual**2)), 1e-10)
    visible_mean = np.mean(stream_y[:, train], axis=1, keepdims=True)
    visible_mean = np.broadcast_to(visible_mean, y_true.shape)
    previous_visible = np.mean(calibration_y[-1, train])
    visible_persistence = np.empty_like(y_true)
    visible_persistence[0] = previous_visible
    visible_persistence[1:] = np.mean(stream_y[:-1, train], axis=1)[:, None]
    heldout_persistence = np.empty_like(y_true)
    heldout_persistence[0] = calibration_y[-1, test]
    heldout_persistence[1:] = stream_y[:-1, test]

    methods = {
        "routeb_cumulative_hippo_q2": (routeb_mean, routeb_var),
        "task1_xlag_ridge_mean": (task1_stream_mean[:, test], np.full_like(y_true, variance)),
        "visible_current_mean": (visible_mean, np.full_like(y_true, variance)),
        "visible_mean_persistence": (visible_persistence, np.full_like(y_true, variance)),
        (
            "persistence"
            if args.history_aware
            else "heldout_persistence_oracle"
        ): (heldout_persistence, np.full_like(y_true, variance)),
    }
    rows: list[dict[str, object]] = []
    for name, (mean, var) in methods.items():
        rows.append({"method": name, **predictive_metrics(y_true, mean, var)})

    per_state: list[dict[str, object]] = []
    for column, location in enumerate(test):
        metrics = predictive_metrics(y_true[:, column], routeb_mean[:, column], routeb_var[:, column])
        per_state.append({"test_column": column, "location_index": int(location), **metrics})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "routeb_per_state.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_state[0].keys())
        writer.writeheader()
        writer.writerows(per_state)
    metadata = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    report = [
        "# COVID Route B Phase-1 diagnostic",
        "",
        "This seed-0 report diagnoses the cold-start spatial protocol; it does not change Route B.",
        (
            "The held-out persistence row is a legal delayed-history baseline: it uses only the previous week's labels."
            if args.history_aware
            else "The held-out persistence row is an oracle: it reads held-out labels from the previous week and is therefore not legal under the permanent cold-start protocol."
        ),
        "",
        "| Method | RMSE | NLL | Coverage90 | Mean std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['method']} | {row['rmse']:.4f} | {row['nll']:.4f} | "
            f"{row['coverage90']:.4f} | {row['mean_predictive_std']:.4f} |"
        )
    report.extend(
        [
            "",
            f"Protocol split seed: {metadata['split_seed']}; train states: {len(train)}; held-out states: {len(test)}; stream weeks: {len(stream_y)}.",
            f"The Route B prediction archive was checked against the protocol labels: {len(per_state)} held-out state trajectories matched.",
        ]
    )
    (args.output_dir / "diagnostic_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
