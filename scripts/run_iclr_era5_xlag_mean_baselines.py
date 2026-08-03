#!/usr/bin/env python3
"""Batch and strict-online X-lag mean-only baselines for the ERA5 benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import (  # noqa: E402
    augment_dataset_phi,
    selected_locations_from_dataset,
)
from scripts.run_iclr_era5_routeb_strict_online import TaskPhiCache  # noqa: E402
from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5  # noqa: E402


def metrics(y_true: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    var = np.maximum(np.asarray(variance, dtype=np.float64).reshape(-1), 1e-10)
    half = 1.6448536269514722 * np.sqrt(var)
    return {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "nll": float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var))),
        "coverage90": float(np.mean((y >= mu - half) & (y <= mu + half))),
        "mean_predictive_std": float(np.mean(np.sqrt(var))),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def residual_variance(
    precision: np.ndarray,
    rhs: np.ndarray,
    target_square_sum: float,
    beta: np.ndarray,
    ridge: float,
    count: int,
) -> float:
    gram = precision - ridge * np.eye(precision.shape[0])
    sse = target_square_sum - 2.0 * beta @ rhs + beta @ gram @ beta
    return max(float(sse) / max(int(count), 1), 1e-10)


def block_slices(arrays: np.lib.npyio.NpzFile) -> list[slice]:
    return [
        slice(int(start), int(stop))
        for start, stop in zip(arrays["block_start"], arrays["block_stop"])
    ]


def batch_or_fixed(
    *,
    arrays: np.lib.npyio.NpzFile,
    mode: str,
) -> tuple[dict, list[dict[str, object]], dict[str, np.ndarray], int]:
    y = np.asarray(arrays["stream_y"], dtype=np.float64)
    train = np.asarray(arrays["train_indices"], dtype=int)
    test = np.asarray(arrays["test_indices"], dtype=int)
    if mode == "batch_fixed":
        mean = np.asarray(arrays["batch_stream_mean"], dtype=np.float64)
        calibration = y[:, train] - mean[:, train]
        parameter_source = "all Task-2(+) training locations"
    else:
        mean = np.asarray(arrays["task1_stream_mean"], dtype=np.float64)
        calibration_y = np.asarray(arrays["calibration_y"], dtype=np.float64)
        calibration_mean = np.asarray(arrays["task1_calibration_mean"], dtype=np.float64)
        calibration = calibration_y[:, train] - calibration_mean[:, train]
        parameter_source = "Task 1 only; frozen through Task 2(+)"
    noise_variance = max(float(np.mean(calibration**2)), 1e-10)
    pred_mean = mean[:, test]
    pred_var = np.full_like(pred_mean, noise_variance)
    rows = []
    for block_id, block in enumerate(block_slices(arrays)):
        row = {
            "block_id": block_id,
            "block_start": block.start,
            "block_stop": block.stop,
            "hours": block.stop - block.start,
            "update_seconds": 0.0,
            "prediction_seconds": 0.0,
            **metrics(y[block][:, test], pred_mean[block], pred_var[block]),
        }
        rows.append(row)
    result = {
        "implementation": "X-lag ridge mean only",
        "protocol": mode,
        "parameter_source": parameter_source,
        "noise_variance": noise_variance,
        "num_blocks": len(rows),
        "overall_current_block": metrics(y[:, test], pred_mean, pred_var),
        "final_block": rows[-1],
    }
    predictions = {
        "y_true": y[:, test],
        "pred_mean": pred_mean,
        "pred_var": pred_var,
        "test_indices": test,
        "times": np.asarray(arrays["stream_times"], dtype=np.float64),
    }
    state_bytes = int(arrays["batch_ridge_beta"].nbytes + 8) if mode == "batch_fixed" else int(
        arrays["task1_ridge_beta"].nbytes + 8
    )
    return result, rows, predictions, state_bytes


def recursive_rls(
    *,
    arrays: np.lib.npyio.NpzFile,
    protocol_json: Path,
    ridge: float,
    xlag_length: int,
) -> tuple[dict, list[dict[str, object]], dict[str, np.ndarray], int, float]:
    metadata = json.loads(protocol_json.read_text(encoding="utf-8"))
    data_root = Path(metadata["root"])
    y = np.asarray(arrays["stream_y"], dtype=np.float64)
    calibration_y = np.asarray(arrays["calibration_y"], dtype=np.float64)
    train = np.asarray(arrays["train_indices"], dtype=int)
    test = np.asarray(arrays["test_indices"], dtype=int)

    loading_started = time.perf_counter()
    calibration_raw = load_hipposvgp_era5(
        data_root, tasks=("task_1",), variable_index=0, split="all"
    )
    selected_locations = selected_locations_from_dataset(calibration_raw)
    calibration = augment_dataset_phi(
        calibration_raw, phi_mode="medium_era5_xlag", xlag_length=xlag_length
    )
    phi_calibration = np.asarray(calibration.Phi, dtype=np.float64).reshape(
        calibration_y.shape[0], calibration_y.shape[1], -1
    )
    initial_loading_seconds = time.perf_counter() - loading_started

    x_initial = phi_calibration[:, train].reshape(-1, phi_calibration.shape[-1])
    y_initial = calibration_y[:, train].reshape(-1)
    precision = x_initial.T @ x_initial + ridge * np.eye(x_initial.shape[1])
    rhs = x_initial.T @ y_initial
    target_square_sum = float(y_initial @ y_initial)
    count = int(y_initial.size)
    beta = np.linalg.solve(precision, rhs)
    np.testing.assert_allclose(beta, arrays["task1_ridge_beta"], atol=2e-8, rtol=2e-8)

    cache = TaskPhiCache(data_root, y, xlag_length)
    blocks = block_slices(arrays)
    prediction_mean = np.empty((y.shape[0], test.size), dtype=np.float64)
    prediction_var = np.empty_like(prediction_mean)
    rows = []
    update_total = 0.0
    prediction_total = 0.0
    for block_id, block in enumerate(blocks):
        phi_block, task_index = cache.block(block)
        x_train = np.asarray(phi_block[:, train], dtype=np.float64).reshape(
            -1, phi_block.shape[-1]
        )
        y_train = y[block][:, train].reshape(-1)
        update_started = time.perf_counter()
        precision += x_train.T @ x_train
        rhs += x_train.T @ y_train
        target_square_sum += float(y_train @ y_train)
        count += int(y_train.size)
        beta = np.linalg.solve(precision, rhs)
        noise_variance = residual_variance(
            precision, rhs, target_square_sum, beta, ridge, count
        )
        update_seconds = time.perf_counter() - update_started

        prediction_started = time.perf_counter()
        x_test = np.asarray(phi_block[:, test], dtype=np.float64)
        mean = np.einsum("tsd,d->ts", x_test, beta)
        variance = np.full_like(mean, noise_variance)
        prediction_seconds = time.perf_counter() - prediction_started
        prediction_mean[block] = mean
        prediction_var[block] = variance
        block_metrics = metrics(y[block][:, test], mean, variance)
        rows.append(
            {
                "block_id": block_id,
                "block_start": block.start,
                "block_stop": block.stop,
                "hours": block.stop - block.start,
                "task_index": task_index,
                "update_seconds": update_seconds,
                "prediction_seconds": prediction_seconds,
                "noise_variance": noise_variance,
                **block_metrics,
            }
        )
        update_total += update_seconds
        prediction_total += prediction_seconds

    state_bytes = int(
        precision.nbytes + rhs.nbytes + beta.nbytes + 3 * np.dtype(np.float64).itemsize
    )
    result = {
        "implementation": "strict-online recursive X-lag ridge/RLS mean only",
        "protocol": "Task-1 sufficient-statistic calibration; new-block-only RLS updates; no replay",
        "ridge": ridge,
        "num_blocks": len(rows),
        "overall_current_block": metrics(y[:, test], prediction_mean, prediction_var),
        "final_block": rows[-1],
        "timing": {
            "stream_update_seconds": update_total,
            "stream_prediction_seconds": prediction_total,
            "xlag_feature_loading_seconds": initial_loading_seconds + cache.loading_seconds,
            "mean_block_update_seconds": update_total / len(rows),
            "mean_block_prediction_seconds": prediction_total / len(rows),
        },
    }
    predictions = {
        "y_true": y[:, test],
        "pred_mean": prediction_mean,
        "pred_var": prediction_var,
        "test_indices": test,
        "times": np.asarray(arrays["stream_times"], dtype=np.float64),
    }
    return result, rows, predictions, state_bytes, initial_loading_seconds + cache.loading_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("batch_fixed", "task1_fixed", "recursive_rls"), required=True
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    arrays = np.load(args.protocol_npz)
    if args.mode == "recursive_rls":
        result, rows, predictions, state_bytes, loading_seconds = recursive_rls(
            arrays=arrays,
            protocol_json=args.protocol_json,
            ridge=args.ridge,
            xlag_length=args.xlag_length,
        )
    else:
        result, rows, predictions, state_bytes = batch_or_fixed(arrays=arrays, mode=args.mode)
        loading_seconds = 0.0
    result.update(
        {
            "split_seed": args.seed,
            "num_stream_times": int(arrays["stream_y"].shape[0]),
            "num_train_space": int(arrays["train_indices"].size),
            "num_test_space": int(arrays["test_indices"].size),
            "timing": {
                **result.get("timing", {}),
                "process_total_seconds": time.perf_counter() - started,
            },
            "resources": {
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "persistent_state_bytes": state_bytes,
                "persistent_state_mib": state_bytes / 1024.0**2,
                "history_replay_buffer_bytes": 0,
                "device": "CPU",
                "dtype": "float64",
            },
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "blocks.csv")
    np.savez_compressed(args.output_dir / "predictions.npz", **predictions)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
