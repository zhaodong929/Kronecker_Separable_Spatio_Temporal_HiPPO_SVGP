#!/usr/bin/env python3
"""Strict-streaming ERA5 wrapper for Bui et al.'s official OSGPR_VFE."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CODE = ROOT / "baselines/external/thangbui_streaming_sparse_gp/code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL_CODE))

import gpflow
import numpy as np
import tensorflow as tf

from osgpr import OSGPR_VFE
from stvgp_kronecker.benchmark_runtime import (  # noqa: E402
    configure_tensorflow,
    host_snapshot,
    tensorflow_memory,
)


NP_DTYPE = np.float64


def flatten_inputs(times, coordinates, spatial_indices, block):
    selected_times = np.asarray(times[block], dtype=NP_DTYPE)
    selected_space = np.asarray(coordinates[spatial_indices], dtype=NP_DTYPE)
    return np.column_stack(
        [
            np.repeat(selected_times, selected_space.shape[0]),
            np.tile(selected_space[:, 0], selected_times.shape[0]),
            np.tile(selected_space[:, 1], selected_times.shape[0]),
        ]
    )


def flatten_targets(values, spatial_indices, block):
    return np.asarray(values[block][:, spatial_indices], dtype=NP_DTYPE).reshape(-1, 1)


def product_inducing(times, spatial_inducing, mt):
    temporal = np.linspace(float(np.min(times)), float(np.max(times)), int(mt))
    return np.asarray(
        [[time_value, coord[0], coord[1]] for time_value in temporal for coord in spatial_inducing],
        dtype=NP_DTYPE,
    )


def make_kernel(theta):
    temporal = gpflow.kernels.Matern32(
        variance=float(theta["kernel_variance"]),
        lengthscales=float(theta["ell_t"]),
        active_dims=[0],
    )
    latitude = gpflow.kernels.Matern32(
        variance=1.0,
        lengthscales=float(theta["ell_s"][0]),
        active_dims=[1],
    )
    longitude = gpflow.kernels.Matern32(
        variance=1.0,
        lengthscales=float(theta["ell_s"][1]),
        active_dims=[2],
    )
    kernel = temporal * latitude * longitude
    gpflow.set_trainable(kernel, False)
    return kernel


def metric_row(y_true, mean, variance):
    y = np.asarray(y_true).reshape(-1)
    mu = np.asarray(mean).reshape(-1)
    var = np.maximum(np.asarray(variance).reshape(-1), 1e-10)
    half = 1.6448536269514722 * np.sqrt(var)
    return {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "nll": float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var))),
        "coverage90": float(np.mean((y >= mu - half) & (y <= mu + half))),
        "mean_predictive_std": float(np.mean(np.sqrt(var))),
    }


def posterior_at_z(model, z):
    mean, covariance = model.predict_f(z, full_cov=True)
    covariance = np.asarray(covariance)
    if covariance.ndim == 3:
        covariance = covariance[0]
    return np.asarray(mean), covariance


def predict_current(model, x, noise_variance, chunk_size):
    means = []
    variances = []
    for start in range(0, x.shape[0], chunk_size):
        mean, variance = model.predict_f(x[start : start + chunk_size], full_cov=False)
        means.append(np.asarray(mean))
        variances.append(np.asarray(variance).reshape(-1, 1) + noise_variance)
    return np.concatenate(means), np.concatenate(variances)


def blocks_from_arrays(starts, stops):
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))


def calibration_before_stream(calibration_times, stream_times):
    calibration_times = np.asarray(calibration_times, dtype=NP_DTYPE)
    stream_times = np.asarray(stream_times, dtype=NP_DTYPE)
    if calibration_times.size < 2:
        step = 1.0
    else:
        step = float(np.median(np.diff(calibration_times)))
    return calibration_times - (calibration_times[-1] - stream_times[0] + step)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--theta-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blockwise-output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--mt", type=int, default=2)
    parser.add_argument("--ms", type=int, default=64)
    parser.add_argument("--prediction-chunk-size", type=int, default=4096)
    parser.add_argument("--max-calibration-blocks", type=int, default=0)
    parser.add_argument("--max-stream-blocks", type=int, default=0)
    parser.add_argument(
        "--task1-posterior-warm-start",
        action="store_true",
        help=(
            "Transfer the Task-1 posterior into the stream. By default Task 1 is "
            "used only to calibrate frozen hyperparameters and Task 2 starts from "
            "the GP prior, matching the strict-online benchmark protocol."
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()

    global NP_DTYPE
    runtime = configure_tensorflow(tf, device=args.device, dtype=args.dtype)
    NP_DTYPE = np.float32 if args.dtype == "float32" else np.float64
    process_started = time.perf_counter()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    gpflow.config.set_default_float(NP_DTYPE)
    if runtime["device"].startswith("cuda"):
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except (RuntimeError, ValueError):
            pass
    arrays = np.load(args.protocol_npz)
    calibration = np.asarray(arrays["calibration_y"], dtype=NP_DTYPE)
    stream = np.asarray(arrays["stream_y"], dtype=NP_DTYPE)
    calibration_offset = np.asarray(arrays["task1_calibration_mean"], dtype=NP_DTYPE)
    stream_offset = np.asarray(arrays["task1_stream_mean"], dtype=NP_DTYPE)
    calibration_residual = calibration - calibration_offset
    stream_residual = stream - stream_offset
    calibration_times = np.asarray(arrays["calibration_times"], dtype=NP_DTYPE)
    stream_times = np.asarray(arrays["stream_times"], dtype=NP_DTYPE)
    calibration_times = calibration_before_stream(calibration_times, stream_times)
    coordinates = np.asarray(arrays["coordinates"], dtype=NP_DTYPE)
    train_indices = np.asarray(arrays["train_indices"], dtype=int)
    test_indices = np.asarray(arrays["test_indices"], dtype=int)
    stream_blocks = blocks_from_arrays(arrays["block_start"], arrays["block_stop"])
    calibration_blocks = tuple(
        slice(start, min(calibration_times.size, start + 10))
        for start in range(0, calibration_times.size, 10)
    )
    if args.max_calibration_blocks > 0:
        calibration_blocks = calibration_blocks[: args.max_calibration_blocks]
    if args.max_stream_blocks > 0:
        stream_blocks = stream_blocks[: args.max_stream_blocks]
    theta_payload = json.loads(args.theta_json.read_text(encoding="utf-8"))
    theta = theta_payload["learned_theta"]
    noise_variance = float(theta["noise_std"]) ** 2
    inducing_key = f"inducing_coords_ms{args.ms}"
    spatial_inducing = np.asarray(arrays[inducing_key], dtype=NP_DTYPE)
    combined_time = np.concatenate([calibration_times, stream_times])
    z = product_inducing(combined_time, spatial_inducing, args.mt)

    old_mean = None
    old_covariance = None
    old_kernel_covariance = None
    old_z = z
    calibration_seconds = 0.0
    if args.task1_posterior_warm_start:
        for block_id, block in enumerate(calibration_blocks):
            x_new = flatten_inputs(calibration_times, coordinates, train_indices, block)
            y_new = flatten_targets(calibration_residual, train_indices, block)
            started = time.perf_counter()
            kernel = make_kernel(theta)
            if old_mean is None:
                model = gpflow.models.SGPR(
                    data=(x_new, y_new),
                    kernel=kernel,
                    inducing_variable=z,
                    noise_variance=noise_variance,
                )
                gpflow.set_trainable(model, False)
            else:
                model = OSGPR_VFE(
                    data=(x_new, y_new),
                    kernel=kernel,
                    mu_old=old_mean,
                    Su_old=old_covariance,
                    Kaa_old=old_kernel_covariance,
                    Z_old=old_z,
                    Z=z,
                )
                model.likelihood.variance.assign(noise_variance)
                gpflow.set_trainable(model, False)
            old_mean, old_covariance = posterior_at_z(model, z)
            old_kernel_covariance = np.asarray(kernel(z))
            old_z = z
            calibration_seconds += time.perf_counter() - started
            print(json.dumps({"phase": "calibration", "block": block_id}), flush=True)

    block_rows = []
    all_true = []
    all_mean = []
    all_variance = []
    total_update_seconds = 0.0
    total_prediction_seconds = 0.0
    for block_id, block in enumerate(stream_blocks):
        x_new = flatten_inputs(stream_times, coordinates, train_indices, block)
        y_new = flatten_targets(stream_residual, train_indices, block)
        update_started = time.perf_counter()
        kernel = make_kernel(theta)
        if old_mean is None:
            model = gpflow.models.SGPR(
                data=(x_new, y_new),
                kernel=kernel,
                inducing_variable=z,
                noise_variance=noise_variance,
            )
            gpflow.set_trainable(model, False)
        else:
            model = OSGPR_VFE(
                data=(x_new, y_new),
                kernel=kernel,
                mu_old=old_mean,
                Su_old=old_covariance,
                Kaa_old=old_kernel_covariance,
                Z_old=old_z,
                Z=z,
            )
            model.likelihood.variance.assign(noise_variance)
            gpflow.set_trainable(model, False)
        new_mean, new_covariance = posterior_at_z(model, z)
        update_seconds = time.perf_counter() - update_started

        x_test = flatten_inputs(stream_times, coordinates, test_indices, block)
        y_test = flatten_targets(stream, test_indices, block)
        offset_test = flatten_targets(stream_offset, test_indices, block)
        prediction_started = time.perf_counter()
        mean, variance = predict_current(
            model, x_test, noise_variance, args.prediction_chunk_size
        )
        mean += offset_test
        prediction_seconds = time.perf_counter() - prediction_started
        block_metrics = metric_row(y_test, mean, variance)
        row = {
            "block_id": block_id,
            "block_start": block.start,
            "block_stop": block.stop,
            "hours": block.stop - block.start,
            "update_seconds": update_seconds,
            "prediction_seconds": prediction_seconds,
            **block_metrics,
        }
        block_rows.append(row)
        all_true.append(y_test)
        all_mean.append(mean)
        all_variance.append(variance)
        total_update_seconds += update_seconds
        total_prediction_seconds += prediction_seconds
        old_mean, old_covariance = new_mean, new_covariance
        old_kernel_covariance = np.asarray(kernel(z))
        old_z = z
        print(json.dumps(row), flush=True)

    y_true = np.concatenate(all_true)
    prediction_mean = np.concatenate(all_mean)
    prediction_variance = np.concatenate(all_variance)
    final_metrics = metric_row(y_true, prediction_mean, prediction_variance)
    persistent_bytes = int(
        z.nbytes
        + old_mean.nbytes
        + old_covariance.nbytes
        + old_kernel_covariance.nbytes
    )
    observations_per_full_block = 10 * train_indices.size
    estimated_flops = float(
        len(stream_blocks)
        * 4.0
        * observations_per_full_block
        * z.shape[0] ** 2
    )
    payload = {
        "implementation": "official Bui OSGPR_VFE (thin ERA5 wrapper)",
        "source_repository": "https://github.com/thangbui/streaming_sparse_gp",
        "source_commit": "d95081b",
        "protocol": (
            "strict online; Task-1 hyperparameter calibration only; Task-2(+) starts "
            "from the GP prior; no history replay"
            if not args.task1_posterior_warm_start
            else "diagnostic online; Task-1 posterior warm-start; no history replay"
        ),
        "initial_posterior": (
            "GP prior at the first streaming block"
            if not args.task1_posterior_warm_start
            else "posterior transferred from Task 1"
        ),
        "target_mode": "Task-1 fixed X-lag residual, evaluated on original y",
        "split_seed": args.seed,
        "num_stream_times": int(stream_times.size),
        "num_blocks": len(stream_blocks),
        "calibration_time_range": [float(calibration_times[0]), float(calibration_times[-1])],
        "stream_time_range": [float(stream_times[0]), float(stream_times[-1])],
        "num_train_space": int(train_indices.size),
        "num_test_space": int(test_indices.size),
        "joint_inducing": int(z.shape[0]),
        "temporal_grid_count": args.mt,
        "spatial_grid_count": args.ms,
        "inducing_locations": "fixed global Cartesian product; never optimized",
        "hyperparameters": "Route-B Task-1 empirical-Bayes theta, frozen for controlled posterior-transfer comparison",
        "final": final_metrics,
        "timing": {
            "task1_calibration_seconds": calibration_seconds,
            "stream_update_seconds": total_update_seconds,
            "stream_prediction_seconds": total_prediction_seconds,
            "mean_block_update_seconds": float(np.mean([row["update_seconds"] for row in block_rows])),
            "mean_block_prediction_seconds": float(np.mean([row["prediction_seconds"] for row in block_rows])),
            "first_block_update_seconds": float(block_rows[0]["update_seconds"]),
            "mean_steady_state_block_update_seconds": float(
                np.mean([row["update_seconds"] for row in block_rows[1:]] or [block_rows[0]["update_seconds"]])
            ),
            "process_total_seconds": time.perf_counter() - process_started,
        },
        "resources": {
            **tensorflow_memory(tf, runtime["device"]),
            "persistent_state_bytes": persistent_bytes,
            "persistent_state_mib": persistent_bytes / 1024.0**2,
            "history_replay_buffer_bytes": 0,
            "estimated_streaming_flops": estimated_flops,
            "flops_scope": "dominant O(B M^2) covariance accumulation; analytic estimate",
            "device": runtime["device"],
            "dtype": runtime["dtype"],
            "physical_gpus": runtime["physical_gpus"],
        },
        "environment": host_snapshot(ROOT),
        "versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "gpflow": gpflow.__version__,
            "numpy": np.__version__,
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(block_rows, args.blockwise_output)
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        evaluated_stop = stream_blocks[-1].stop
        np.savez_compressed(
            args.predictions_output,
            y_true=y_true.reshape(evaluated_stop, test_indices.size),
            pred_mean=prediction_mean.reshape(evaluated_stop, test_indices.size),
            pred_var=prediction_variance.reshape(evaluated_stop, test_indices.size),
            test_indices=test_indices,
            times=stream_times[:evaluated_stop],
        )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
