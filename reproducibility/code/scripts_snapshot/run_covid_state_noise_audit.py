#!/usr/bin/env python3
"""Audit causal state-aware predictive noise without changing the posterior mean."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--predictions-npz", type=Path, required=True)
    parser.add_argument("--theta-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.protocol_npz) as protocol:
        calibration_y = np.asarray(protocol["calibration_y"], dtype=np.float64)
        calibration_mean = np.asarray(protocol["task1_calibration_mean"], dtype=np.float64)
        stream_y = np.asarray(protocol["stream_y"], dtype=np.float64)
        test_indices = np.asarray(protocol["test_indices"], dtype=int)
        train = np.asarray(protocol["train_indices"], dtype=int)
    with np.load(args.predictions_npz) as predictions:
        y_true = np.asarray(predictions["y_true"], dtype=np.float64)
        pred_mean = np.asarray(predictions["pred_mean"], dtype=np.float64)
        pred_var = np.asarray(predictions["pred_var"], dtype=np.float64)
        saved_test = np.asarray(predictions["test_indices"], dtype=int)
    if not np.array_equal(saved_test, test_indices):
        raise ValueError("Prediction and protocol test indices differ")
    if not np.allclose(y_true, stream_y[:, test_indices], rtol=0.0, atol=1e-12):
        raise ValueError("Prediction archive does not match protocol labels")
    if not np.isfinite(pred_var).all() or (pred_var <= 0.0).any():
        raise ValueError("Predictive variance must be finite and positive")

    theta = json.loads(args.theta_json.read_text(encoding="utf-8"))["learned_theta"]
    global_noise = float(theta["noise_std"]) ** 2
    residual = calibration_y - calibration_mean
    global_residual = float(np.mean(residual[:, train] ** 2))
    state_residual = np.mean(residual**2, axis=0)
    state_residual = np.maximum(state_residual, 1e-6)

    modes = {
        "global_reference": np.full(test_indices.size, global_noise),
        "state_raw": state_residual[test_indices],
        "state_shrink_50": 0.5 * state_residual[test_indices] + 0.5 * global_residual,
        "state_floor_50": np.maximum(state_residual[test_indices], 0.5 * global_noise),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode, noise_by_state in modes.items():
        # The current predictive variance already contains global noise. Replace only that component.
        adjusted = np.maximum(pred_var - global_noise + noise_by_state[None, :], 1e-10)
        metrics = predictive_metrics(y_true, pred_mean, adjusted)
        rows.append({"mode": mode, **metrics, "mean_state_noise_variance": float(np.mean(noise_by_state))})
        np.savez_compressed(
            args.output_dir / f"predictions_{mode}.npz",
            y_true=y_true,
            pred_mean=pred_mean,
            pred_var=adjusted,
            test_indices=test_indices,
        )

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "state_noise.json").write_text(
        json.dumps(
            {
                "global_observation_noise_variance": global_noise,
                "global_train_residual_variance": global_residual,
                "state_residual_variance": state_residual.tolist(),
                "scope": "causal Task-1 residual calibration-only; posterior and mean unchanged",
                "not_claimed": "exact heteroscedastic structured-joint posterior",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
