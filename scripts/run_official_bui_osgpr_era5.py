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
from scripts.era5_ncu_ranges import pop_range, profile_this_index, push_range


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


def make_kernel(theta, *, frozen: bool):
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
    gpflow.set_trainable(kernel, not frozen)
    return kernel


def adapt_model(model, *, steps: int, learning_rate: float) -> int:
    """Run a bounded, causal Adam update for the official GPflow model."""

    if steps <= 0:
        return 0
    optimizer = tf.optimizers.Adam(float(learning_rate))
    completed = 0
    for _ in range(int(steps)):
        with tf.GradientTape() as tape:
            loss = model.training_loss()
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        if not bool(tf.math.is_finite(loss)) or any(
            gradient is None or not bool(tf.reduce_all(tf.math.is_finite(gradient)))
            for gradient in gradients
        ):
            raise FloatingPointError("Non-finite Bui adaptive objective or gradient")
        optimizer.apply_gradients(zip(gradients, variables))
        completed += 1
    return completed


def theta_from_model(model) -> dict[str, object]:
    temporal, latitude, longitude = model.kernel.kernels
    return {
        "ell_t": float(temporal.lengthscales.numpy()),
        "ell_s": [float(latitude.lengthscales.numpy()), float(longitude.lengthscales.numpy())],
        "kernel_variance": float(temporal.variance.numpy()),
        "noise_std": float(np.sqrt(model.likelihood.variance.numpy())),
    }


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
    parser.add_argument("--theta-json", type=Path)
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
    parser.add_argument(
        "--delayed-observations",
        action="store_true",
        help="Absorb each scored hidden block once before the next visible update.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Use Bui-owned Task-1 empirical Bayes and bounded causal online "
            "adaptation of kernel, likelihood, and pseudo-input locations."
        ),
    )
    parser.add_argument("--adaptive-calibration-steps", type=int, default=25)
    parser.add_argument("--adaptive-online-steps", type=int, default=5)
    parser.add_argument("--adaptive-learning-rate", type=float, default=0.01)
    parser.add_argument("--initial-ell-t", type=float, default=0.05)
    parser.add_argument("--initial-ell-s", type=float, nargs=2, default=[0.35, 0.35])
    parser.add_argument("--initial-kernel-variance", type=float, default=1.0)
    parser.add_argument("--initial-noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()

    if not args.adaptive and args.theta_json is None:
        raise ValueError("--theta-json is required for controlled Bui OSGPR")
    if args.adaptive and (args.adaptive_calibration_steps < 0 or args.adaptive_online_steps < 0):
        raise ValueError("Adaptive optimization steps must be non-negative")

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
    theta_payload = None if args.theta_json is None else json.loads(args.theta_json.read_text(encoding="utf-8"))
    theta = (
        {
            "ell_t": float(args.initial_ell_t),
            "ell_s": [float(value) for value in args.initial_ell_s],
            "kernel_variance": float(args.initial_kernel_variance),
            "noise_std": float(args.initial_noise),
        }
        if args.adaptive
        else theta_payload["learned_theta"]
    )
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
    task1_warm_start = bool(args.task1_posterior_warm_start or args.adaptive)
    if task1_warm_start:
        for block_id, block in enumerate(calibration_blocks):
            x_new = flatten_inputs(calibration_times, coordinates, train_indices, block)
            y_new = flatten_targets(calibration_residual, train_indices, block)
            started = time.perf_counter()
            kernel = make_kernel(theta, frozen=not args.adaptive)
            if old_mean is None:
                model = gpflow.models.SGPR(
                    data=(x_new, y_new),
                    kernel=kernel,
                    inducing_variable=z,
                    noise_variance=noise_variance,
                )
                if not args.adaptive:
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
                if not args.adaptive:
                    gpflow.set_trainable(model, False)
            if args.adaptive:
                adapt_model(
                    model,
                    steps=args.adaptive_calibration_steps,
                    learning_rate=args.adaptive_learning_rate,
                )
                z = np.asarray(model.inducing_variable.Z)
                noise_variance = float(model.likelihood.variance.numpy())
                theta = theta_from_model(model)
            old_mean, old_covariance = posterior_at_z(model, z)
            old_kernel_covariance = np.asarray(model.kernel(z))
            old_z = z
            calibration_seconds += time.perf_counter() - started
            print(json.dumps({"phase": "calibration", "block": block_id}), flush=True)

    block_rows = []
    all_true = []
    all_mean = []
    all_variance = []
    total_update_seconds = 0.0
    total_prediction_seconds = 0.0
    delayed_rows = 0
    for block_id, block in enumerate(stream_blocks):
        profile_range = profile_this_index(block_id, len(stream_blocks))
        profile_open = push_range("era5_online_block", profile_range)
        update_started = time.perf_counter()
        updates = [(block, train_indices, "current_visible")]
        if args.delayed_observations and block_id > 0:
            updates.insert(0, (stream_blocks[block_id - 1], test_indices, "delayed_hidden"))
        for observation_block, spatial_indices, update_kind in updates:
            x_new = flatten_inputs(stream_times, coordinates, spatial_indices, observation_block)
            y_new = flatten_targets(stream_residual, spatial_indices, observation_block)
            kernel = make_kernel(theta, frozen=not args.adaptive)
            if old_mean is None:
                model = gpflow.models.SGPR(
                    data=(x_new, y_new),
                    kernel=kernel,
                    inducing_variable=z,
                    noise_variance=noise_variance,
                )
                if not args.adaptive:
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
                if not args.adaptive:
                    gpflow.set_trainable(model, False)
            if args.adaptive:
                adapt_model(
                    model,
                    steps=args.adaptive_online_steps,
                    learning_rate=args.adaptive_learning_rate,
                )
                z = np.asarray(model.inducing_variable.Z)
                noise_variance = float(model.likelihood.variance.numpy())
                theta = theta_from_model(model)
            new_mean, new_covariance = posterior_at_z(model, z)
            old_mean, old_covariance = new_mean, new_covariance
            old_kernel_covariance = np.asarray(model.kernel(z))
            old_z = z
            if update_kind == "delayed_hidden":
                delayed_rows += int(x_new.shape[0])
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
        pop_range(profile_open)
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
        old_kernel_covariance = np.asarray(model.kernel(z))
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
            if not task1_warm_start
            else "diagnostic online; Task-1 posterior warm-start; no history replay"
        ),
        "initial_posterior": (
            "GP prior at the first streaming block"
            if not task1_warm_start
            else "posterior transferred from Task 1"
        ),
        "delayed_observations": bool(args.delayed_observations),
        "delayed_observation_rows": delayed_rows,
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
        "inducing_locations": (
            "adaptive pseudo-input locations optimized causally from Task 1 and each online update"
            if args.adaptive
            else "fixed global Cartesian product; never optimized"
        ),
        "adaptive": bool(args.adaptive),
        "adaptive_optimization": (
            None
            if not args.adaptive
            else {
                "Task1_steps_per_block": int(args.adaptive_calibration_steps),
                "online_steps_per_update": int(args.adaptive_online_steps),
                "learning_rate": float(args.adaptive_learning_rate),
                "updated_parameters": ["kernel", "likelihood", "inducing_locations"],
            }
        ),
        "hyperparameters": (
            "Bui-owned Task-1 empirical Bayes followed by bounded causal online adaptation"
            if args.adaptive
            else "Route-B Task-1 empirical-Bayes theta, frozen for controlled posterior-transfer comparison"
        ),
        "learned_theta": theta_from_model(model),
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
