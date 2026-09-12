#!/usr/bin/env python3
"""Microbatched OHSVGP adaptation for the locked PEMS-BAY Protocol N."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "baselines/external/harrisonzhu508_HIPPOSVGP"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from baselines.covid_long_setting_b.archive import PredictionArchive
from baselines.traffic_protocol_n import TrafficProtocolN
from hipposvgp.hippo import transition, variable_unroll_matrix, variable_unroll_matrix_sequential
from hipposvgp.likelihood import GaussianLikelihood
from hipposvgp.multidim import SE_kernel
from scripts.run_epidemiology_pilot import predictive_metrics
from scripts.run_official_ohsvgp_era5 import (
    configure_kernel,
    export_state,
    flatten_inputs,
    flatten_targets,
    make_model,
    predict,
)
from stvgp_kronecker.benchmark_runtime import host_snapshot, resolve_torch_runtime


class LazyHiPPOLegS(nn.Module):
    """Materialise LegS transitions only for the current causal microbatch."""

    def __init__(self, order: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.order = int(order)
        matrix, vector = transition("legs", self.order)
        self.register_buffer("base_matrix", torch.as_tensor(matrix, dtype=dtype, device=device))
        self.register_buffer("base_vector", torch.as_tensor(vector.squeeze(-1), dtype=dtype, device=device))
        self.register_buffer("eye", torch.eye(self.order, dtype=dtype, device=device))

    def forward(self, inputs, prev_discrete_steps=0, ini=None, fast=False):
        length = int(inputs.shape[0])
        times = torch.arange(
            int(prev_discrete_steps) + 1,
            int(prev_discrete_steps) + length + 1,
            dtype=self.base_matrix.dtype,
            device=self.base_matrix.device,
        )
        at = self.base_matrix.unsqueeze(0) / times[:, None, None]
        bt = self.base_vector.unsqueeze(0) / times[:, None]
        matrix = torch.linalg.solve_triangular(
            self.eye.unsqueeze(0) - at / 2.0,
            self.eye.unsqueeze(0) + at / 2.0,
            upper=False,
        )
        vector = torch.linalg.solve_triangular(
            self.eye.unsqueeze(0) - at / 2.0,
            bt.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        forcing = (inputs.unsqueeze(-1).transpose(0, -2) * vector).transpose(0, -2)
        if fast:
            return variable_unroll_matrix(matrix, forcing, s=ini)
        return variable_unroll_matrix_sequential(matrix, forcing, s=ini)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--theta-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inducing-size", type=int, default=64)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--microbatch-size", type=int, default=325)
    parser.add_argument("--subsampling-lag", type=int, default=10)
    parser.add_argument("--task1-block-steps", type=int, default=12)
    parser.add_argument("--max-task1-blocks", type=int, default=0)
    parser.add_argument("--update-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--prediction-chunk-size", type=int, default=325)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()
    if min(args.inducing_size, args.rff_sample_size, args.microbatch_size, args.subsampling_lag) < 1:
        raise ValueError("OHSVGP capacities must be positive")

    runtime = resolve_torch_runtime(args.device, args.dtype)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(runtime.dtype)
    if runtime.uses_cuda:
        torch.cuda.manual_seed_all(args.seed)
    protocol = TrafficProtocolN(args.protocol_npz, args.protocol_json)
    arrays = np.load(args.protocol_npz)
    theta = json.loads(args.theta_json.read_text(encoding="utf-8"))["learned_theta"]
    calibration_times = np.asarray(arrays["calibration_times"], dtype=np.float64)
    stream_times = np.asarray(arrays["stream_times"], dtype=np.float64)
    coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
    calibration_y = np.asarray(arrays["calibration_y"], dtype=np.float64)
    stream_y = np.asarray(arrays["stream_y"], dtype=np.float64)
    calibration_offset = np.asarray(arrays["task1_calibration_mean"], dtype=np.float64)
    stream_offset = np.asarray(arrays["task1_stream_mean"], dtype=np.float64)
    calibration_residual = calibration_y - calibration_offset
    stream_residual = stream_y - stream_offset
    visible = protocol.visible_locations
    hidden = protocol.hidden_locations
    requested = protocol.online_weeks if args.max_blocks <= 0 else min(args.max_blocks, protocol.online_weeks)

    kernel = SE_kernel(3, device=runtime.device).to(device=runtime.device, dtype=runtime.dtype)
    likelihood = GaussianLikelihood(float(theta["noise_std"]) ** 2).to(
        device=runtime.device, dtype=runtime.dtype
    )
    configure_kernel(kernel, likelihood, theta, device=runtime.device, dtype=runtime.dtype)
    torch.manual_seed(args.seed)
    frequencies = kernel.sample_from_spectral(args.rff_sample_size).detach()
    hippo = LazyHiPPOLegS(args.inducing_size, runtime.device, runtime.dtype).to(
        device=runtime.device, dtype=runtime.dtype
    )
    old_state = None
    previous_steps = 0
    model = None

    def assimilate(times: np.ndarray, values: np.ndarray, indices: np.ndarray, block: slice) -> None:
        nonlocal old_state, previous_steps, model
        x_values = flatten_inputs(times, coordinates, indices, block)
        y_values = flatten_targets(values, indices, block)
        order = np.lexsort((x_values[:, 2], x_values[:, 1], x_values[:, 0]))
        x_values, y_values = x_values[order], y_values[order]
        for start in range(0, x_values.shape[0], args.microbatch_size):
            stop = min(x_values.shape[0], start + args.microbatch_size)
            x_batch = torch.as_tensor(x_values[start:stop], dtype=runtime.dtype, device=runtime.device)
            y_batch = torch.as_tensor(y_values[start:stop], dtype=runtime.dtype, device=runtime.device)
            z_interpolate = x_batch[:: args.subsampling_lag]
            model = make_model(
                kernel=kernel,
                likelihood=likelihood,
                z_interpolate=z_interpolate,
                rff_sample_size=args.rff_sample_size,
                prev_steps=previous_steps,
                hippo=hippo,
                inducing_size=args.inducing_size,
                old_state=old_state,
                device=runtime.device,
                dtype=runtime.dtype,
            )
            optimizer = torch.optim.Adam([model.mv, model.Lv], lr=args.learning_rate)
            for update in range(args.update_steps):
                optimizer.zero_grad(set_to_none=True)
                elbo, _, _ = model.ELBO(
                    x_batch,
                    y_batch,
                    frequencies,
                    recompute_k=update == 0,
                    cache_k=update == 0,
                )
                loss = -elbo
                if not torch.isfinite(loss):
                    raise FloatingPointError("OHSVGP update became non-finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([model.mv, model.Lv], 20.0)
                optimizer.step()
            old_state = export_state(model, frequencies)
            previous_steps += int(z_interpolate.shape[0])

    started = time.perf_counter()
    task1_started = time.perf_counter()
    task1_starts = list(range(0, protocol.calibration_weeks, args.task1_block_steps))
    if args.max_task1_blocks > 0:
        task1_starts = task1_starts[: args.max_task1_blocks]
    for start in task1_starts:
        assimilate(
            calibration_times,
            calibration_residual,
            visible,
            slice(start, min(protocol.calibration_weeks, start + args.task1_block_steps)),
        )
    runtime.synchronize()
    task1_seconds = time.perf_counter() - task1_started

    rows: list[dict[str, object]] = []
    archive = PredictionArchive(protocol, method="ohsvgp_traffic_protocol_n", seed=args.seed)
    delayed_rows = 0
    for step in range(requested):
        update_started = time.perf_counter()
        if step > 0:
            assimilate(stream_times, stream_residual, hidden, slice(step - 1, step))
            delayed_rows += hidden.size
        assimilate(stream_times, stream_residual, visible, slice(step, step + 1))
        runtime.synchronize()
        update_seconds = time.perf_counter() - update_started
        assert model is not None
        x_query = flatten_inputs(stream_times, coordinates, hidden, slice(step, step + 1))
        prediction_started = time.perf_counter()
        mean, variance = predict(
            model,
            frequencies,
            x_query,
            float(theta["noise_std"]) ** 2,
            args.prediction_chunk_size,
            device=runtime.device,
            dtype=runtime.dtype,
            synchronize=runtime.synchronize,
        )
        mean = mean.reshape(-1) + stream_offset[step, hidden]
        variance = variance.reshape(-1)
        prediction_seconds = time.perf_counter() - prediction_started
        information = protocol.week(step)
        archive.append(information, mean, variance)
        metric = predictive_metrics(stream_y[step, hidden], mean, variance)
        rows.append(
            {
                "step": step,
                "update_seconds": update_seconds,
                "prediction_seconds": prediction_seconds,
                **metric,
            }
        )
        if step % 100 == 0:
            print(json.dumps(rows[-1]), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = archive.write(
        args.output_dir / "predictions.npz",
        require_complete=requested == protocol.online_weeks,
        extra_metadata={
            "adapter": "official_ohsvgp_core_microbatched_traffic_protocol_n",
            "source_commit": "a1bff1bc8162",
            "inducing_size": args.inducing_size,
            "rff_sample_size": args.rff_sample_size,
        },
    )
    write_csv(rows, args.output_dir / "blocks.csv")
    status = {
        "status": "complete",
        "method": "OHSVGP",
        "source_repository": "https://github.com/harrisonzhu508/HIPPOSVGP",
        "source_commit": "a1bff1bc8162",
        "protocol": "pems_bay_protocol_n",
        "seed": args.seed,
        "online_steps": requested,
        "task1_seconds": task1_seconds,
        "delayed_observation_rows": delayed_rows,
        "audit": audit,
        "timing_seconds": time.perf_counter() - started,
        "resources": runtime.resources(),
        "environment": host_snapshot(ROOT),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
