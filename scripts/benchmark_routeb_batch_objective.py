#!/usr/bin/env python3
"""Benchmark exact Route-B objective optimizations with separate timing/profiling."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils import benchmark

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_iclr_era5_routeb_batch import (
    load_joint_phi,
    load_protocol,
)
from scripts.run_routeb_batch_empirical_bayes import tensor_training_data
from stvgp_kronecker.routeb_empirical_bayes import (
    BatchRouteBEmpiricalBayes,
    JointSufficientStatistics,
    joint_sufficient_statistics,
)


VERSIONS = {
    "E0": (False, False, False),
    "E1": (True, False, False),
    "E2": (True, True, False),
    "E3": (True, True, True),
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = max(1.0, float(torch.linalg.vector_norm(reference).detach().cpu()))
    return float(torch.linalg.vector_norm(actual - reference).detach().cpu()) / denominator


def analytical_uncovered_forward_flops(
    *, num_space: int, num_time: int, ms: int, mt: int, num_features: int
) -> dict[str, float]:
    """Lower-bound supplement for major profiler-uncovered forward operators.

    Conventions: Cholesky n^3/3, symmetric EVD 9n^3, and a two-sided
    Cholesky solve with r right-hand sides 2n^2r. Backward costs and scalar
    transcendental/reduction work remain explicitly excluded.
    """

    cholesky = (
        2.0 * ms**3 / 3.0
        + 2.0 * mt**3 / 3.0
        + num_features**3 / 3.0
    )
    eigendecomposition = 9.0 * (ms**3 + mt**3)
    triangular_solves = (
        2.0 * ms**2 * num_space
        + 2.0 * mt**2 * num_time
        + 2.0 * num_features**2
    )
    return {
        "analytical_cholesky_forward_flops": cholesky,
        "analytical_eigh_forward_flops": eigendecomposition,
        "analytical_triangular_solve_forward_flops": triangular_solves,
        "analytical_supplement_forward_flops": (
            cholesky + eigendecomposition + triangular_solves
        ),
    }


def make_model(
    *, data: Any, args: argparse.Namespace, theta: dict[str, Any] | None
) -> BatchRouteBEmpiricalBayes:
    model = BatchRouteBEmpiricalBayes(
        times=data.times,
        spatial_inducing=data.spatial_inducing,
        mt=args.mt,
        representation="analytic_hippo_rff",
        initial_ell_t=0.05,
        initial_ell_s=(0.35, 0.35),
        initial_kernel_variance=1.0,
        initial_noise_std=0.1,
        rff_sample_size=args.rff_sample_size,
        seed=0,
        objective_type=args.objective,
    ).to(device="cuda", dtype=torch.float64)
    if theta is not None:
        model.set_theta(theta)
    return model


def parameter_gradient(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            parameter.grad.detach().reshape(-1)
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=["task2_short", "tasks2_10_long"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective", choices=["finite_dtc", "vfe"], default="finite_dtc")
    parser.add_argument("--versions", nargs="+", choices=list(VERSIONS), default=list(VERSIONS))
    parser.add_argument("--theta-json", type=Path)
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--beta-prior-variance", type=float, default=1000.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--cross-contraction",
        choices=["auto", "einsum", "spatial_first", "temporal_first"],
        default="einsum",
    )
    parser.add_argument("--feature-block-size", type=int)
    parser.add_argument("--feature-projection-npz", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    data, fit_indices, _ = load_protocol(args.protocol_npz, args.ms, "stream")
    with np.load(args.protocol_npz) as arrays:
        data.phi, _ = load_joint_phi(
            arrays=arrays,
            protocol_json=args.protocol_json,
            data_root=args.data_root,
            xlag_length=10,
            data_part="stream",
        )
    projection_path = args.feature_projection_npz
    if projection_path is not None:
        with np.load(projection_path) as payload:
            basis = np.asarray(payload["basis"], dtype=np.float64)
        data.phi = np.einsum(
            "stp,pr->str", np.asarray(data.phi, dtype=np.float64), basis, optimize=True
        )
    y, phi, coordinates = tensor_training_data(
        data,
        fit_indices,
        device=torch.device("cuda"),
        dtype=torch.float64,
    )
    theta = None
    if args.theta_json is not None:
        theta = json.loads(args.theta_json.read_text(encoding="utf-8"))["learned_theta"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    reference: dict[str, torch.Tensor] | None = None
    template_model = make_model(data=data, args=args, theta=theta)
    template_state = {
        key: value.detach().clone()
        for key, value in template_model.state_dict().items()
    }
    del template_model
    for version in args.versions:
        cache_stats, combine_basis, remove_solve = VERSIONS[version]
        model = make_model(data=data, args=args, theta=theta)
        model.load_state_dict(template_state)
        statistics: JointSufficientStatistics | None = (
            joint_sufficient_statistics(y, phi) if cache_stats else None
        )
        options = {
            "sufficient_statistics": statistics,
            "combine_basis_transforms": combine_basis,
            "remove_redundant_solve": remove_solve,
            "cross_contraction": args.cross_contraction,
            "feature_block_size": args.feature_block_size,
        }

        def forward_backward() -> Any:
            model.zero_grad(set_to_none=True)
            diagnostics = model.objective(
                y_matrix=y,
                phi_tensor=phi,
                spatial_coordinates=coordinates,
                beta_prior_variance=args.beta_prior_variance,
                **options,
            )
            diagnostics.nlml_per_observation.backward()
            return diagnostics

        parity_result = forward_backward()
        assert parity_result.beta_mean is not None
        assert parity_result.u_mean is not None
        assert parity_result.beta_precision is not None
        current = {
            "loss": parity_result.nlml_per_observation.detach(),
            "gradient": parameter_gradient(model),
            "posterior_mean": torch.cat(
                (
                    parity_result.beta_mean.detach(),
                    parity_result.u_mean.detach().reshape(-1),
                )
            ),
            "beta_covariance": torch.linalg.inv(parity_result.beta_precision.detach()),
        }
        if reference is None:
            reference = current
        parity = {
            "objective_relative_error": relative_error(current["loss"], reference["loss"]),
            "gradient_relative_error": relative_error(current["gradient"], reference["gradient"]),
            "posterior_mean_relative_error": relative_error(
                current["posterior_mean"], reference["posterior_mean"]
            ),
            "beta_covariance_relative_error": relative_error(
                current["beta_covariance"], reference["beta_covariance"]
            ),
        }
        passes_exact_parity = not (
            parity["objective_relative_error"] >= 1e-8
            or parity["gradient_relative_error"] >= 1e-7
            or parity["posterior_mean_relative_error"] >= 1e-8
            or parity["beta_covariance_relative_error"] >= 1e-8
        )
        if not passes_exact_parity:
            print(
                json.dumps(
                    {"warning": "exact parity threshold failed", "version": version, **parity}
                ),
                flush=True,
            )

        for _ in range(args.warmup):
            forward_backward()
        torch.cuda.synchronize()
        timer = benchmark.Timer(
            stmt="forward_backward()",
            globals={"forward_backward": forward_backward},
            num_threads=1,
        )
        timing = timer.timeit(args.repeats)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            forward_backward()
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        model.zero_grad(set_to_none=True)
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as profiler:
            forward_backward()
        events = profiler.key_averages(group_by_input_shape=True)
        profiler_flops = int(sum(int(event.flops or 0) for event in events))
        for event in events:
            operator_rows.append(
                {
                    "scope": args.scope,
                    "seed": args.seed,
                    "objective": args.objective,
                    "version": version,
                    "operator": event.key,
                    "input_shapes": str(event.input_shapes),
                    "calls": int(event.count),
                    "profiler_counted_flops": int(event.flops or 0),
                    "self_cpu_time_us": float(event.self_cpu_time_total),
                    "device_time_us": float(getattr(event, "device_time_total", 0.0)),
                    "self_device_memory_bytes": int(
                        getattr(event, "self_device_memory_usage", 0)
                    ),
                }
            )
        supplement = analytical_uncovered_forward_flops(
            num_space=y.shape[0],
            num_time=y.shape[1],
            ms=args.ms,
            mt=args.mt,
            num_features=phi.shape[-1],
        )
        row = {
            "method": "Route B cumulative HiPPO",
            "scope": args.scope,
            "mode": "batch_objective_forward_backward",
            "seed": args.seed,
            "objective": args.objective,
            "version": version,
            "num_time": int(y.shape[1]),
            "num_space": int(y.shape[0]),
            "num_features": int(phi.shape[-1]),
            "mt": args.mt,
            "ms": args.ms,
            "warmup_steps": args.warmup,
            "timed_repeats": args.repeats,
            "steady_runtime_seconds_per_step": float(timing.mean),
            "steady_runtime_iqr_seconds": float(timing.iqr),
            "peak_allocated_bytes": int(peak_allocated),
            "peak_reserved_bytes": int(peak_reserved),
            "profiler_counted_flops_per_step": profiler_flops,
            "profiler_counted_gflops_per_step": profiler_flops / 1e9,
            **supplement,
            "profiler_plus_forward_supplement_flops": (
                profiler_flops + supplement["analytical_supplement_forward_flops"]
            ),
            "counting_method": "PyTorch Profiler with_flops=True plus explicit forward-only analytical lower-bound supplement",
            "excluded_flops": "factorization/solve backward, elementwise kernels, transcendental functions, reductions, optimizer, validation and prediction",
            "cross_contraction": args.cross_contraction,
            "feature_block_size": args.feature_block_size,
            "feature_projection": None if projection_path is None else str(projection_path),
            "passes_exact_parity": passes_exact_parity,
            **parity,
        }
        result_rows.append(row)
        write_csv(args.output_dir / "batch_ablation.csv", result_rows)
        write_csv(args.output_dir / "profiler_operator_breakdown.csv", operator_rows)
        (args.output_dir / "batch_ablation.json").write_text(
            json.dumps(result_rows, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(row), flush=True)
        del model, statistics, parity_result, current
        gc.collect()
        torch.cuda.empty_cache()

    write_csv(args.output_dir / "batch_ablation.csv", result_rows)
    write_csv(args.output_dir / "profiler_operator_breakdown.csv", operator_rows)
    (args.output_dir / "batch_ablation.json").write_text(
        json.dumps(result_rows, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
