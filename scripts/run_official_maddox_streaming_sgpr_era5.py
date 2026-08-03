#!/usr/bin/env python3
"""Strict-streaming ERA5 wrapper for Maddox et al.'s official StreamingSGPR."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import platform
import sys
import time

import gpytorch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "baselines/external/wjmaddox_online_gp"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from online_gp.models.streaming_sgpr import StreamingSGPR
from stvgp_kronecker.benchmark_runtime import (  # noqa: E402
    SynchronizedTimer,
    host_snapshot,
    resolve_torch_runtime,
)


def flatten_inputs(times, coordinates, spatial_indices, block):
    selected_times = np.asarray(times[block], dtype=np.float64)
    selected_space = np.asarray(coordinates[spatial_indices], dtype=np.float64)
    return np.column_stack(
        [
            np.repeat(selected_times, selected_space.shape[0]),
            np.tile(selected_space[:, 0], selected_times.shape[0]),
            np.tile(selected_space[:, 1], selected_times.shape[0]),
        ]
    )


def flatten_targets(values, spatial_indices, block):
    return np.asarray(values[block][:, spatial_indices], dtype=np.float64).reshape(-1, 1)


def product_inducing(times, spatial_inducing, mt):
    temporal = np.linspace(float(np.min(times)), float(np.max(times)), int(mt))
    return np.asarray(
        [[time_value, coord[0], coord[1]] for time_value in temporal for coord in spatial_inducing],
        dtype=np.float64,
    )


def make_kernel(theta, *, device, dtype):
    temporal = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=(0,))
    latitude = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=(1,))
    longitude = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=(2,))
    temporal.lengthscale = float(theta["ell_t"])
    latitude.lengthscale = float(theta["ell_s"][0])
    longitude.lengthscale = float(theta["ell_s"][1])
    kernel = gpytorch.kernels.ScaleKernel(temporal * latitude * longitude)
    kernel.outputscale = float(theta["kernel_variance"])
    return kernel.to(device=device, dtype=dtype)


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


def predict(model, x, chunk_size, *, device, dtype, synchronize):
    means = []
    variances = []
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, x.shape[0], chunk_size):
            values = torch.as_tensor(
                x[start : start + chunk_size], dtype=dtype, device=device
            )
            prediction = model.likelihood(model(values))
            synchronize()
            means.append(prediction.mean.detach().cpu().numpy().reshape(-1, 1))
            variances.append(prediction.variance.detach().cpu().numpy().reshape(-1, 1))
    return np.concatenate(means), np.concatenate(variances)


def state_bytes(model):
    strategy = model.variational_strategy
    tensors = [
        strategy.inducing_points,
        strategy.variational_distribution.mean,
        strategy.variational_distribution.covariance_matrix,
    ]
    if model._old_C_matrix is not None:
        tensors.append(
            model._old_C_matrix.evaluate()
            if hasattr(model._old_C_matrix, "evaluate")
            else model._old_C_matrix.to_dense()
        )
    tensors.extend(list(model.covar_module.parameters()))
    tensors.extend(list(model.likelihood.parameters()))
    return int(sum(value.numel() * value.element_size() for value in tensors))


def fixed_inducing_fantasy_model(model, x_new, y_new):
    """Apply the official fantasy equations without perturbing fixed inducing points."""
    z_fixed = model.variational_strategy.inducing_points.clone().detach()
    fantasy_model = StreamingSGPR(
        inducing_points=z_fixed,
        likelihood=deepcopy(model.likelihood),
        covar_module=deepcopy(model.covar_module),
        old_strat=model.variational_strategy,
        old_kernel=model.covar_module,
        old_C_matrix=model.current_C_matrix(x_new),
        learn_inducing_locations=False,
        num_data=model.num_data + x_new.size(0),
        jitter=model._jitter,
    )
    with torch.no_grad():
        fantasy_model.update_variational_distribution(x_new, y_new)
    return fantasy_model


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
    parser.add_argument("--jitter", type=float, default=1e-4)
    parser.add_argument("--prediction-chunk-size", type=int, default=4096)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()

    runtime = resolve_torch_runtime(args.device, args.dtype)
    process_started = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(runtime.dtype)
    if runtime.uses_cuda:
        torch.cuda.manual_seed_all(args.seed)
    arrays = np.load(args.protocol_npz)
    theta = json.loads(args.theta_json.read_text(encoding="utf-8"))["learned_theta"]
    times = np.asarray(arrays["stream_times"], dtype=np.float64)
    coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
    stream_y = np.asarray(arrays["stream_y"], dtype=np.float64)
    offset = np.asarray(arrays["task1_stream_mean"], dtype=np.float64)
    residual = stream_y - offset
    train_indices = np.asarray(arrays["train_indices"], dtype=int)
    test_indices = np.asarray(arrays["test_indices"], dtype=int)
    blocks = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(arrays["block_start"], arrays["block_stop"])
    )
    if args.max_blocks > 0:
        blocks = blocks[: args.max_blocks]
    spatial_inducing = np.asarray(arrays[f"inducing_coords_ms{args.ms}"], dtype=np.float64)
    z = torch.as_tensor(
        product_inducing(times, spatial_inducing, args.mt),
        dtype=runtime.dtype,
        device=runtime.device,
    )
    model = StreamingSGPR(
        z,
        covar_module=make_kernel(
            theta, device=runtime.device, dtype=runtime.dtype
        ),
        learn_inducing_locations=False,
        num_data=0,
        jitter=args.jitter,
    ).to(device=runtime.device, dtype=runtime.dtype)
    model.likelihood.noise = float(theta["noise_std"]) ** 2
    for parameter in model.covar_module.parameters():
        parameter.requires_grad_(False)
    for parameter in model.likelihood.parameters():
        parameter.requires_grad_(False)
    runtime.reset_peak_memory()

    rows = []
    all_true = []
    all_mean = []
    all_variance = []
    mean_grid = np.empty((times.size, test_indices.size), dtype=np.float64)
    variance_grid = np.empty_like(mean_grid)
    total_update = 0.0
    total_prediction = 0.0
    for block_id, block in enumerate(blocks):
        x_train = torch.as_tensor(
            flatten_inputs(times, coordinates, train_indices, block),
            dtype=runtime.dtype,
            device=runtime.device,
        )
        y_train = torch.as_tensor(
            flatten_targets(residual, train_indices, block),
            dtype=runtime.dtype,
            device=runtime.device,
        )
        with SynchronizedTimer(runtime.synchronize) as update_timer:
            with torch.no_grad():
                if block_id == 0:
                    model.update_variational_distribution(x_train, y_train)
                    model.num_data = x_train.shape[0]
                else:
                    model = fixed_inducing_fantasy_model(model, x_train, y_train)
                    model = model.to(device=runtime.device, dtype=runtime.dtype)
                    for parameter in model.covar_module.parameters():
                        parameter.requires_grad_(False)
                    for parameter in model.likelihood.parameters():
                        parameter.requires_grad_(False)
        update_seconds = update_timer.elapsed

        x_test = flatten_inputs(times, coordinates, test_indices, block)
        y_test = flatten_targets(stream_y, test_indices, block)
        offset_test = flatten_targets(offset, test_indices, block)
        with SynchronizedTimer(runtime.synchronize) as prediction_timer:
            mean, variance = predict(
                model,
                x_test,
                args.prediction_chunk_size,
                device=runtime.device,
                dtype=runtime.dtype,
                synchronize=runtime.synchronize,
            )
            mean += offset_test
        prediction_seconds = prediction_timer.elapsed
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise FloatingPointError(
                "Maddox StreamingSGPR produced non-finite predictions at "
                f"block {block_id} ({block.start}:{block.stop}) with "
                f"jitter={args.jitter:g}"
            )
        block_metrics = metric_row(y_test, mean, variance)
        block_length = block.stop - block.start
        mean_grid[block] = mean.reshape(block_length, test_indices.size)
        variance_grid[block] = variance.reshape(block_length, test_indices.size)
        all_true.append(y_test)
        all_mean.append(mean)
        all_variance.append(variance)
        total_update += update_seconds
        total_prediction += prediction_seconds
        row = {
            "block_id": block_id,
            "block_start": block.start,
            "block_stop": block.stop,
            "hours": block_length,
            "update_seconds": update_seconds,
            "prediction_seconds": prediction_seconds,
            **block_metrics,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    overall = metric_row(
        np.concatenate(all_true), np.concatenate(all_mean), np.concatenate(all_variance)
    )
    persistent_bytes = state_bytes(model)
    observations = sum((block.stop - block.start) * train_indices.size for block in blocks)
    estimated_flops = float(4.0 * observations * z.shape[0] ** 2)
    payload = {
        "implementation": "official Maddox online_gp StreamingSGPR",
        "source_repository": "https://github.com/wjmaddox/online_gp",
        "source_commit": "3bff4c3",
        "protocol": "strict online; new-block-only labels; no history replay",
        "target_mode": "Task-1 fixed X-lag residual, evaluated on original y",
        "hyperparameters": "Route-B Task-1 empirical-Bayes theta, frozen",
        "inducing_policy": "fixed global Cartesian coordinates; official fantasy equations without the upstream unconditional perturbation",
        "numerical_jitter": args.jitter,
        "split_seed": args.seed,
        "num_stream_times": int(times.size),
        "num_blocks": len(blocks),
        "num_train_space": int(train_indices.size),
        "num_test_space": int(test_indices.size),
        "joint_inducing": int(z.shape[0]),
        "temporal_grid_count": args.mt,
        "spatial_grid_count": args.ms,
        "overall_current_block": overall,
        "final_block": {
            key: rows[-1][key]
            for key in ("rmse", "nll", "coverage90", "mean_predictive_std")
        },
        "timing": {
            "stream_update_seconds": total_update,
            "stream_prediction_seconds": total_prediction,
            "mean_block_update_seconds": float(np.mean([row["update_seconds"] for row in rows])),
            "mean_block_prediction_seconds": float(np.mean([row["prediction_seconds"] for row in rows])),
            "first_block_update_seconds": float(rows[0]["update_seconds"]),
            "mean_steady_state_block_update_seconds": float(
                np.mean([row["update_seconds"] for row in rows[1:]] or [rows[0]["update_seconds"]])
            ),
            "process_total_seconds": time.perf_counter() - process_started,
        },
        "resources": {
            **runtime.resources(),
            "persistent_state_bytes": persistent_bytes,
            "persistent_state_mib": persistent_bytes / 1024.0**2,
            "history_replay_buffer_bytes": 0,
            "estimated_streaming_flops": estimated_flops,
            "flops_scope": "dominant O(N M^2) posterior conditioning; analytic estimate",
        },
        "environment": host_snapshot(ROOT),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpytorch": gpytorch.__version__,
            "numpy": np.__version__,
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(rows, args.blockwise_output)
    if args.predictions_output is not None:
        stop = blocks[-1].stop
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions_output,
            y_true=stream_y[:stop, test_indices],
            pred_mean=mean_grid[:stop],
            pred_var=variance_grid[:stop],
            test_indices=test_indices,
            times=times[:stop],
        )
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
