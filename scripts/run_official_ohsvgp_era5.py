#!/usr/bin/env python3
"""Strict-streaming ERA5 adapter for the official multidimensional OHSVGP class."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "baselines/external/harrisonzhu508_HIPPOSVGP"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.likelihood import GaussianLikelihood
from hipposvgp.multidim import HIPPOOSVGP, SE_kernel
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


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tensor_bytes(values):
    return int(
        sum(value.numel() * value.element_size() for value in values if torch.is_tensor(value))
    )


def configure_kernel(kernel, likelihood, theta, *, device, dtype):
    with torch.no_grad():
        kernel.log_ls.copy_(
            torch.log(
                torch.tensor(
                    [theta["ell_t"], theta["ell_s"][0], theta["ell_s"][1]],
                    dtype=dtype,
                    device=device,
                )
            )
        )
        kernel.log_sf.copy_(
            torch.log(
                torch.tensor(
                    [theta["kernel_variance"]], dtype=dtype, device=device
                )
            )
        )
        likelihood.log_variance.copy_(
            torch.log(
                torch.tensor(
                    float(theta["noise_std"]) ** 2, dtype=dtype, device=device
                )
            )
        )
    kernel.log_ls.requires_grad_(False)
    kernel.log_sf.requires_grad_(False)
    likelihood.log_variance.requires_grad_(False)


def make_model(
    *,
    kernel,
    likelihood,
    z_interpolate,
    rff_sample_size,
    prev_steps,
    hippo,
    inducing_size,
    old_state,
    device,
    dtype,
):
    kwargs = {}
    if old_state is not None:
        kwargs = {
            "num_inducing_old": inducing_size,
            "mv_old": old_state["mv"],
            "Lv_old": old_state["Lv"],
            "Kaa_old": old_state["Kaa"],
            "Z_old": old_state["Z"],
        }
    model = HIPPOOSVGP(
        kernel=deepcopy(kernel),
        likelihood=deepcopy(likelihood),
        Z_interpolate=z_interpolate,
        rff_sample_size=rff_sample_size,
        prev_discrete_steps=prev_steps,
        hippo=hippo,
        inducing_size=inducing_size,
        device=device,
        flag_update_kernel=False,
        **kwargs,
    ).to(device=device, dtype=dtype)
    if old_state is not None:
        model.mv = nn.Parameter(old_state["mv"].clone())
        model.Lv = nn.Parameter(old_state["Lv"].clone())
    return model


def export_state(model, frequencies):
    with torch.no_grad():
        z_old, _, kuu, _, _ = model.Kuu_se(frequencies)
        kuu = torch.exp(model.kernel.log_sf) * kuu
    return {
        "Z": z_old.detach().clone(),
        "Kaa": kuu.detach().clone(),
        "mv": model.mv.detach().clone(),
        "Lv": model.Lv.detach().clone(),
    }


def predict(
    model, frequencies, x, noise_variance, chunk_size, *, device, dtype, synchronize
):
    means = []
    variances = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            x_chunk = torch.as_tensor(
                x[start : start + chunk_size], dtype=dtype, device=device
            )
            mean, latent_variance = model.pred_f(x_chunk, frequencies, full_cov=False)
            synchronize()
            means.append(mean.detach().cpu().numpy())
            variances.append(latent_variance.detach().cpu().numpy() + noise_variance)
    return np.concatenate(means), np.concatenate(variances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--theta-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blockwise-output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--inducing-size", type=int, default=64)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--microbatch-size", type=int, default=200)
    parser.add_argument("--subsampling-lag", type=int, default=10)
    parser.add_argument("--update-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--prediction-chunk-size", type=int, default=200)
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

    kernel = SE_kernel(3, device=runtime.device).to(
        device=runtime.device, dtype=runtime.dtype
    )
    likelihood = GaussianLikelihood(float(theta["noise_std"]) ** 2).to(
        device=runtime.device, dtype=runtime.dtype
    )
    configure_kernel(
        kernel,
        likelihood,
        theta,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    torch.manual_seed(args.seed)
    frequencies = kernel.sample_from_spectral(args.rff_sample_size).detach()
    hippo_steps = 0
    for block in blocks:
        observations = (block.stop - block.start) * train_indices.size
        for start in range(0, observations, args.microbatch_size):
            count = min(args.microbatch_size, observations - start)
            hippo_steps += int(math.ceil(count / args.subsampling_lag))
    hippo = HiPPO_LegS(
        args.inducing_size, runtime.device, max_length=hippo_steps + 1
    ).to(device=runtime.device, dtype=runtime.dtype)
    runtime.reset_peak_memory()
    old_state = None
    prev_steps = 0
    model = None
    block_rows = []
    all_true = []
    all_mean = []
    all_variance = []
    mean_grid = np.empty((times.size, test_indices.size), dtype=np.float64)
    variance_grid = np.empty_like(mean_grid)
    total_update = 0.0
    total_prediction = 0.0
    estimated_flops = 0.0

    for block_id, block in enumerate(blocks):
        x_train = flatten_inputs(times, coordinates, train_indices, block)
        y_train = flatten_targets(residual, train_indices, block)
        order = np.lexsort((x_train[:, 2], x_train[:, 1], x_train[:, 0]))
        x_train = x_train[order]
        y_train = y_train[order]
        with SynchronizedTimer(runtime.synchronize) as update_timer:
            for start in range(0, x_train.shape[0], args.microbatch_size):
                x_np = x_train[start : start + args.microbatch_size]
                y_np = y_train[start : start + args.microbatch_size]
                x = torch.as_tensor(
                    x_np, dtype=runtime.dtype, device=runtime.device
                )
                target = torch.as_tensor(
                    y_np, dtype=runtime.dtype, device=runtime.device
                )
                z_interpolate = x[:: args.subsampling_lag]
                model = make_model(
                    kernel=kernel,
                    likelihood=likelihood,
                    z_interpolate=z_interpolate,
                    rff_sample_size=args.rff_sample_size,
                    prev_steps=prev_steps,
                    hippo=hippo,
                    inducing_size=args.inducing_size,
                    old_state=old_state,
                    device=runtime.device,
                    dtype=runtime.dtype,
                )
                optimizer = torch.optim.Adam(
                    [model.mv, model.Lv], lr=args.learning_rate
                )
                for step in range(args.update_steps):
                    optimizer.zero_grad(set_to_none=True)
                    elbo, _, _ = model.ELBO(
                        x,
                        target,
                        frequencies,
                        recompute_k=step == 0,
                        cache_k=step == 0,
                    )
                    loss = -elbo
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite OHSVGP loss at block {block_id}, offset {start}"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([model.mv, model.Lv], 20.0)
                    optimizer.step()
                old_state = export_state(model, frequencies)
                prev_steps += z_interpolate.shape[0]
                n = x.shape[0]
                m = args.inducing_size
                estimated_flops += args.update_steps * (
                    n**3 / 3.0 + 4.0 * n * n * m
                )
        update_seconds = update_timer.elapsed

        x_test = flatten_inputs(times, coordinates, test_indices, block)
        y_test = flatten_targets(stream_y, test_indices, block)
        offset_test = flatten_targets(offset, test_indices, block)
        with SynchronizedTimer(runtime.synchronize) as prediction_timer:
            pred_mean, pred_variance = predict(
                model,
                frequencies,
                x_test,
                float(theta["noise_std"]) ** 2,
                args.prediction_chunk_size,
                device=runtime.device,
                dtype=runtime.dtype,
                synchronize=runtime.synchronize,
            )
            pred_mean += offset_test
        prediction_seconds = prediction_timer.elapsed
        block_metrics = metric_row(y_test, pred_mean, pred_variance)
        block_length = block.stop - block.start
        mean_grid[block] = pred_mean.reshape(block_length, test_indices.size)
        variance_grid[block] = pred_variance.reshape(block_length, test_indices.size)
        all_true.append(y_test)
        all_mean.append(pred_mean)
        all_variance.append(pred_variance)
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
        block_rows.append(row)
        print(json.dumps(row), flush=True)

    overall = metric_row(
        np.concatenate(all_true), np.concatenate(all_mean), np.concatenate(all_variance)
    )
    persistent_bytes = tensor_bytes(
        [frequencies, old_state["Z"], old_state["Kaa"], old_state["mv"], old_state["Lv"]]
    )
    payload = {
        "implementation": "official HIPPOOSVGP multidimensional class with thin ERA5 wrapper",
        "source_repository": "https://github.com/harrisonzhu508/HIPPOSVGP",
        "source_commit": "a1bff1b",
        "protocol": "strict online; new-block-only labels; no history replay",
        "target_mode": "Task-1 fixed X-lag residual, evaluated on original y",
        "kernel": "official squared-exponential RFF kernel",
        "hyperparameters": "Route-B Task-1 empirical-Bayes theta mapped to SE and frozen",
        "optimizer": "Adam on official variational mv/Lv only",
        "microbatch_policy": "time-sorted observations; every label used once per update step",
        "split_seed": args.seed,
        "num_stream_times": int(times.size),
        "num_blocks": len(blocks),
        "num_train_space": int(train_indices.size),
        "num_test_space": int(test_indices.size),
        "inducing_size": args.inducing_size,
        "rff_sample_size": args.rff_sample_size,
        "hippo_max_length": hippo_steps + 1,
        "overall_current_block": overall,
        "final_block": {
            key: block_rows[-1][key]
            for key in ("rmse", "nll", "coverage90", "mean_predictive_std")
        },
        "timing": {
            "stream_update_seconds": total_update,
            "stream_prediction_seconds": total_prediction,
            "mean_block_update_seconds": float(np.mean([row["update_seconds"] for row in block_rows])),
            "mean_block_prediction_seconds": float(np.mean([row["prediction_seconds"] for row in block_rows])),
            "first_block_update_seconds": float(block_rows[0]["update_seconds"]),
            "mean_steady_state_block_update_seconds": float(
                np.mean([row["update_seconds"] for row in block_rows[1:]] or [block_rows[0]["update_seconds"]])
            ),
            "process_total_seconds": time.perf_counter() - process_started,
        },
        "resources": {
            **runtime.resources(),
            "persistent_state_bytes": persistent_bytes,
            "persistent_state_mib": persistent_bytes / 1024.0**2,
            "history_replay_buffer_bytes": 0,
            "estimated_streaming_flops": estimated_flops,
            "flops_scope": "dense Kff Cholesky and Kff/Kfu products; analytic lower-order estimate",
        },
        "environment": host_snapshot(ROOT),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(block_rows, args.blockwise_output)
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
