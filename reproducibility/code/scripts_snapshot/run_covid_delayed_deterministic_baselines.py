#!/usr/bin/env python3
"""Run causal persistence and Task-1 lag-ridge COVID baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

try:
    from scripts.run_epidemiology_pilot import predictive_metrics
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_epidemiology_pilot import predictive_metrics


def save_method(
    output_root: Path,
    method: str,
    *,
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    variance: float,
    test_indices: np.ndarray,
    metadata: dict[str, object],
) -> dict[str, object]:
    started = time.perf_counter()
    pred_var = np.full_like(pred_mean, max(float(variance), 1e-10), dtype=np.float64)
    metrics = predictive_metrics(y_true, pred_mean, pred_var)
    target = output_root / method
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / "predictions.npz",
        y_true=y_true,
        pred_mean=pred_mean,
        pred_var=pred_var,
        test_indices=test_indices,
    )
    payload = {
        "implementation": method,
        "protocol": "delayed-history strict online; no current hidden label is used",
        "split_seed": int(metadata["split_seed"]),
        "overall_current_block": metrics,
        "timing": {"process_total_seconds": time.perf_counter() - started},
        "resources": {"persistent_state_bytes": 0, "history_replay_buffer_bytes": 0},
    }
    (target / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"method": method, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    if int(metadata["split_seed"]) != args.seed:
        raise ValueError("Protocol split seed does not match --seed")
    with np.load(args.protocol_npz) as arrays:
        calibration_y = np.asarray(arrays["calibration_y"], dtype=np.float64)
        stream_y = np.asarray(arrays["stream_y"], dtype=np.float64)
        calibration_mean = np.asarray(arrays["task1_calibration_mean"], dtype=np.float64)
        stream_mean = np.asarray(arrays["task1_stream_mean"], dtype=np.float64)
        train_indices = np.asarray(arrays["train_indices"], dtype=int)
        test_indices = np.asarray(arrays["test_indices"], dtype=int)

    y_true = stream_y[:, test_indices]
    ridge_residual = calibration_y[:, train_indices] - calibration_mean[:, train_indices]
    ridge_variance = float(np.mean(ridge_residual**2))
    rows = [
        save_method(
            args.output_root,
            "lag_ridge",
            y_true=y_true,
            pred_mean=stream_mean[:, test_indices],
            variance=ridge_variance,
            test_indices=test_indices,
            metadata=metadata,
        )
    ]

    persistence_mean = np.vstack([calibration_y[-1, test_indices], stream_y[:-1, test_indices]])
    train_deltas = calibration_y[1:, train_indices] - calibration_y[:-1, train_indices]
    rows.append(
        save_method(
            args.output_root,
            "persistence",
            y_true=y_true,
            pred_mean=persistence_mean,
            variance=float(np.mean(train_deltas**2)),
            test_indices=test_indices,
            metadata=metadata,
        )
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "complete", "seed": args.seed, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
