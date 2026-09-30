#!/usr/bin/env python3
"""Add causal per-state history features to an existing COVID protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def lagged(values: np.ndarray, lag: int) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    output[lag:] = values[:-lag]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.source_npz) as source:
        payload = {name: source[name] for name in source.files}
    calibration_y = np.asarray(payload["calibration_y"], dtype=np.float64)
    stream_y = np.asarray(payload["stream_y"], dtype=np.float64)
    full_y = np.concatenate([calibration_y, stream_y], axis=0)
    calibration_steps, locations = calibration_y.shape
    train = np.asarray(payload["train_indices"], dtype=int)
    coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
    time = np.arange(full_y.shape[0], dtype=np.float64)
    phase = 2.0 * np.pi * time / 52.1775
    lag1, lag2, lag3, lag4 = (lagged(full_y, lag) for lag in (1, 2, 3, 4))
    rolling4 = (lag1 + lag2 + lag3 + lag4) / 4.0
    growth = lag1 - lag2
    visible_mean = np.mean(full_y[:, train], axis=1)
    visible_lag1 = lagged(visible_mean[:, None], 1)[:, 0]
    dynamic = np.stack(
        [
            np.broadcast_to(time[:, None] / max(time[-1], 1.0), full_y.shape),
            np.broadcast_to(np.sin(phase)[:, None], full_y.shape),
            np.broadcast_to(np.cos(phase)[:, None], full_y.shape),
            lag1,
            lag2,
            lag3,
            lag4,
            rolling4,
            growth,
            np.broadcast_to(visible_lag1[:, None], full_y.shape),
            np.broadcast_to(coordinates[None, :, 0], full_y.shape),
            np.broadcast_to(coordinates[None, :, 1], full_y.shape),
        ],
        axis=-1,
    )
    reference = dynamic[:calibration_steps, train].reshape(-1, dynamic.shape[-1])
    mean = reference.mean(axis=0)
    scale = np.maximum(reference.std(axis=0), 1e-12)
    dynamic = (dynamic - mean) / scale
    phi = np.concatenate([np.ones((*full_y.shape, 1)), dynamic], axis=-1)
    design = phi[:calibration_steps, train].reshape(-1, phi.shape[-1])
    target = calibration_y[:, train].reshape(-1)
    beta = np.linalg.solve(design.T @ design + 1e-3 * np.eye(phi.shape[-1]), design.T @ target)
    payload["calibration_phi"] = phi[:calibration_steps].astype(np.float32)
    payload["stream_phi"] = phi[calibration_steps:].astype(np.float32)
    payload["task1_ridge_beta"] = beta
    payload["task1_calibration_mean"] = np.einsum("tsp,p->ts", phi[:calibration_steps], beta).astype(np.float32)
    payload["task1_stream_mean"] = np.einsum("tsp,p->ts", phi[calibration_steps:], beta).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    metadata = json.loads(args.source_json.read_text(encoding="utf-8"))
    metadata["npz"] = str(args.output.resolve())
    metadata["feature_source"] = "history-aware causal state lags"
    metadata["xlag"] = {
        "mode": "history-aware state and visible aggregate lags",
        "features": int(phi.shape[-1]),
        "columns": [
            "intercept", "time_trend", "season_sin", "season_cos", "state_lag1", "state_lag2",
            "state_lag3", "state_lag4", "state_rolling4", "state_growth1", "visible_mean_lag1",
            "latitude", "longitude",
        ],
        "test_target_lags_used": True,
        "test_target_lag_information": "Past held-out labels only; current and future held-out labels are excluded.",
    }
    metadata["protocol_mode"] = "history-aware current-week spatial holdout"
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
