#!/usr/bin/env python3
"""Plot causal COVID predictions and diagnostics for selected held-out states."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.run_epidemiology_pilot import predictive_metrics
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_epidemiology_pilot import predictive_metrics


DEFAULT_STATES = ("North Dakota", "Hawaii", "Oklahoma", "Louisiana", "California")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--predictions-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    args = parser.parse_args()

    metadata = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    names = list(metadata["location_names"])
    with np.load(args.protocol_npz) as protocol:
        stream_y = np.asarray(protocol["stream_y"], dtype=np.float64)
        test_indices = np.asarray(protocol["test_indices"], dtype=int)
    with np.load(args.predictions_npz) as predictions:
        y_true = np.asarray(predictions["y_true"], dtype=np.float64)
        pred_mean = np.asarray(predictions["pred_mean"], dtype=np.float64)
        pred_var = np.asarray(predictions["pred_var"], dtype=np.float64)
        saved_test_indices = np.asarray(predictions["test_indices"], dtype=int)
    if not np.array_equal(saved_test_indices, test_indices):
        raise ValueError("Prediction and protocol test indices differ")
    expected = stream_y[:, test_indices]
    if not np.allclose(y_true, expected, rtol=0.0, atol=1e-12):
        raise ValueError("Prediction archive does not match protocol labels")
    if not np.isfinite(pred_mean).all() or not np.isfinite(pred_var).all() or (pred_var <= 0.0).any():
        raise ValueError("Predictions must be finite with strictly positive variance")

    selected = []
    for state in args.states:
        if state not in names:
            raise ValueError(f"Unknown protocol state: {state}")
        location_index = names.index(state)
        matches = np.flatnonzero(test_indices == location_index)
        if matches.size != 1:
            raise ValueError(f"{state} is not a unique held-out location")
        selected.append((state, int(location_index), int(matches[0])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    x = np.arange(y_true.shape[0])
    for state, location_index, column in selected:
        truth = y_true[:, column]
        mean = pred_mean[:, column]
        std = np.sqrt(pred_var[:, column])
        metrics = predictive_metrics(truth, mean, pred_var[:, column])
        true_peak = int(np.argmax(truth))
        predicted_peak = int(np.argmax(mean))
        rows.append(
            {
                "state": state,
                "location_index": location_index,
                "rmse": metrics["rmse"],
                "nll": metrics["nll"],
                "coverage90": metrics["coverage90"],
                "mean_bias": float(np.mean(mean - truth)),
                "true_peak_week": true_peak,
                "predicted_peak_week": predicted_peak,
                "peak_lag_weeks": predicted_peak - true_peak,
                "mean_predictive_std": float(np.mean(std)),
            }
        )
        fig, axis = plt.subplots(figsize=(8.5, 4.5))
        axis.plot(x, truth, color="#222222", linewidth=1.8, label="Observed")
        axis.plot(x, mean, color="#1769aa", linewidth=1.8, label="Route B mean")
        axis.fill_between(
            x,
            mean - 1.645 * std,
            mean + 1.645 * std,
            color="#4c9bd3",
            alpha=0.25,
            label="90% predictive interval",
        )
        axis.set_xlabel("Online week")
        axis.set_ylabel("log1p per-capita admissions")
        axis.set_title(f"COVID Route B prediction: {state}")
        axis.legend(frameon=False, ncol=3, loc="best")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{state.lower().replace(' ', '_')}.png", dpi=180)
        plt.close(fig)

    with (args.output_dir / "selected_state_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": str(args.protocol_json.resolve()),
                "predictions": str(args.predictions_npz.resolve()),
                "states": [row["state"] for row in rows],
                "interval": "mean +/- 1.645 * predictive standard deviation",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "states": args.states}, indent=2))


if __name__ == "__main__":
    main()
