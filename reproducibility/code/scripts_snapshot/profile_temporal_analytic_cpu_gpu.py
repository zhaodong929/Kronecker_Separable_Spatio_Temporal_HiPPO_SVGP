#!/usr/bin/env python3
"""Microbenchmark spherical-Bessel and analytic HiPPO temporal construction."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.temporal_analytic import (
    AnalyticTemporalBuilder,
    TemporalAnalyticConfig,
    TemporalBlockSpec,
    spherical_bessel_j,
)


def timed(device: torch.device, function, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    values = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append(time.perf_counter() - started)
    return values


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_seconds": float(statistics.median(values)),
        "mean_seconds": float(statistics.mean(values)),
        "min_seconds": float(min(values)),
        "max_seconds": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CPU/GPU temporal microbenchmark")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(0)
    cpu_builder = AnalyticTemporalBuilder(
        TemporalAnalyticConfig(
            inducing_size=args.mt,
            rff_sample_size=args.rff_sample_size,
            variance=1.0,
            lengthscale=0.05,
            kernel_type="matern32",
            dtype=dtype,
            seed=0,
        )
    ).to(device="cpu", dtype=dtype)
    cuda_builder = copy.deepcopy(cpu_builder).to(device="cuda", dtype=dtype)
    old = TemporalBlockSpec(-0.1, 0.45, 100, prev_discrete_steps=0, phase_origin=-0.1)
    current = TemporalBlockSpec(-0.1, 1.0, 186, prev_discrete_steps=0, phase_origin=-0.1)
    query_cpu = torch.linspace(0.95, 1.0, 10, dtype=dtype)
    query_cuda = query_cpu.to("cuda")

    kappa_cpu = (
        0.5
        * cpu_builder.current_frequencies()
        * ((current.end - current.start) / current.num_discrete_steps)
        * current.num_discrete_steps
    )
    kappa_cuda = kappa_cpu.to("cuda")
    with torch.no_grad():
        old_basis_cpu, _ = cpu_builder.compute_temporal_basis(old)
        old_basis_cuda, _ = cuda_builder.compute_temporal_basis(old)
    operations = {
        "spherical_bessel": (
            lambda: spherical_bessel_j(args.mt - 1, kappa_cpu),
            lambda: spherical_bessel_j(args.mt - 1, kappa_cuda),
        ),
        "hippo_temporal_basis": (
            lambda: cpu_builder.compute_temporal_basis(current),
            lambda: cuda_builder.compute_temporal_basis(current),
        ),
        "temporal_covariance_bundle_uncached": (
            lambda: cpu_builder.compute_block_covariances(query_cpu, current, old),
            lambda: cuda_builder.compute_block_covariances(query_cuda, current, old),
        ),
        "temporal_covariance_bundle_cached": (
            lambda: cpu_builder.compute_block_covariances_with_basis(
                query_cpu,
                current,
                old_basis=old_basis_cpu,
            ),
            lambda: cuda_builder.compute_block_covariances_with_basis(
                query_cuda,
                current,
                old_basis=old_basis_cuda,
            ),
        ),
    }
    rows = []
    with torch.no_grad():
        for operation, (cpu_function, cuda_function) in operations.items():
            cpu_values = timed(torch.device("cpu"), cpu_function, args.repeats, args.warmup)
            cuda_values = timed(torch.device("cuda"), cuda_function, args.repeats, args.warmup)
            cpu_summary = summarize(cpu_values)
            cuda_summary = summarize(cuda_values)
            rows.append(
                {
                    "operation": operation,
                    "cpu": cpu_summary,
                    "cuda": cuda_summary,
                    "cuda_speedup_vs_cpu": (
                        cpu_summary["median_seconds"] / cuda_summary["median_seconds"]
                    ),
                }
            )

        cpu_bundle = cpu_builder.compute_block_covariances(query_cpu, current, old)
        cuda_bundle = cuda_builder.compute_block_covariances(query_cuda, current, old)
        for cpu_value, cuda_value in zip(cpu_bundle, cuda_bundle):
            assert cpu_value is not None and cuda_value is not None
            torch.testing.assert_close(
                cuda_value.cpu(), cpu_value, rtol=1e-6, atol=1e-7
            )
        cached_cpu = cpu_builder.compute_block_covariances_with_basis(
            query_cpu,
            current,
            old_basis=old_basis_cpu,
        )[:3]
        for expected, actual in zip(cpu_bundle, cached_cpu):
            assert expected is not None and actual is not None
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    payload = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mt": args.mt,
        "rff_sample_size": args.rff_sample_size,
        "dtype": args.dtype,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
