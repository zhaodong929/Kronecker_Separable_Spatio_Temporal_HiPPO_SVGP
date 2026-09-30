#!/usr/bin/env python3
"""Audit strict-online Route-B update+prediction FLOPs separately from runtime.

The timing pass executes the unmodified strict-online runner in a subprocess.
The profiling pass monkey-patches only the Torch update/prediction entry points,
so it observes the same causal state recursion without changing model outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.joint_ssgp_kron.torch_backend import (
    TorchJointSSGPKronHiPPOSVGP,
)


COUNTING_METHOD = (
    "PyTorch Profiler with_flops=True formula counts for supported ATen operators "
    "plus a separately labelled forward-only analytical lower-bound supplement"
)
EXCLUDED_FLOPS = (
    "factorization/solve backward (online recursion has no autograd backward), "
    "elementwise/transcendental/reduction kernels, CPU HiPPO/RFF factor construction, "
    "feature loading, metric calculation, serialization, and unsupported ATen operators"
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def factorization_lower_bound_flops(
    *, ms: int, mt: int, num_features: int
) -> dict[str, float]:
    """Major forward-only work not covered reliably by profiler FLOP formulas.

    Conventions are Cholesky n^3/3, symmetric EVD 9n^3 and a two-sided
    Cholesky solve with r RHS 2n^2r. Counts reflect the current Torch backend's
    first successful jitter attempt; retries and all elementwise work are absent.
    """

    p = num_features
    # update: beta-prior inverse, Kt inverse, two generalized eigensystems,
    # beta solve and beta covariance inverse. Prediction repeats Kt inverse and
    # two generalized eigensystems.
    cholesky = (
        3.0 * p**3 / 3.0
        + 3.0 * ms**3 / 3.0
        + 5.0 * mt**3 / 3.0
    )
    eigendecomposition = 18.0 * (ms**3 + mt**3)
    # Principal inverse/solve RHS only. Generalized-eigen triangular transforms
    # are deliberately left excluded rather than assigned an uncertain formula.
    triangular_solve = (
        2.0 * p**3 * 3.0
        + 2.0 * mt**3 * 2.0
        + 2.0 * ms**3
    )
    total = cholesky + eigendecomposition + triangular_solve
    return {
        "analytical_cholesky_forward_flops": cholesky,
        "analytical_eigh_forward_flops": eigendecomposition,
        "analytical_principal_solve_forward_flops": triangular_solve,
        "analytical_supplement_forward_flops": total,
    }


def setup_lower_bound_flops(*, ms: int, num_space: int) -> float:
    """Spatial Gram, Cholesky inverse and projection lower bound."""

    return float(2 * num_space * ms**2 + ms**3 / 3.0 + 2 * ms**3)


def total_flops(
    *, setup_flops: float, block_profiler_flops: list[float], block_supplements: list[float]
) -> float:
    if len(block_profiler_flops) != len(block_supplements):
        raise ValueError("Profiler and supplement block lists must have equal lengths")
    return float(setup_flops + sum(block_profiler_flops) + sum(block_supplements))


def profiler_flops(call: Callable[[], Any]) -> tuple[Any, int, list[dict[str, Any]]]:
    # CPU activity records ATen dispatch and formula FLOPs even when tensors are
    # on CUDA. CUDA activity is intentionally omitted because CUPTI is not
    # available reliably under the local WSL setup.
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        value = call()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    events = prof.key_averages(group_by_input_shape=True)
    rows = [
        {
            "operator": event.key,
            "input_shapes": str(event.input_shapes),
            "calls": int(event.count),
            "profiler_counted_flops": int(event.flops or 0),
            "self_cpu_time_us": float(event.self_cpu_time_total),
        }
        for event in events
    ]
    return value, int(sum(row["profiler_counted_flops"] for row in rows)), rows


def runner_arguments(args: argparse.Namespace, *, output_dir: Path) -> list[str]:
    runner = ROOT / "scripts/run_iclr_era5_routeb_strict_online.py"
    command = [
        str(args.python), str(runner),
        "--protocol-npz", str(args.protocol_npz),
        "--protocol-json", str(args.protocol_json),
        "--data-root", str(args.data_root),
        "--theta-json", str(args.theta_json),
        "--output", str(output_dir / "result.json"),
        "--blockwise-output", str(output_dir / "blocks.csv"),
        "--predictions-output", str(output_dir / "predictions.npz"),
        "--representation", "analytic_hippo_rff",
        "--mt", str(args.mt), "--ms", str(args.ms),
        "--rff-sample-size", str(args.rff_sample_size),
        "--prediction-chunk-size", str(args.prediction_chunk_size),
        "--beta-prior-variance", str(args.beta_prior_variance),
        "--seed", str(args.seed),
        "--solver-backend", "torch", "--device", "cuda", "--dtype", "float64",
        "--temporal-factor-device", args.temporal_factor_device,
        "--include-conditional-residual-variance",
    ]
    if args.max_blocks:
        command.extend(("--max-blocks", str(args.max_blocks)))
    if args.feature_projection_npz is not None:
        command.extend(("--feature-projection-npz", str(args.feature_projection_npz)))
    return command


def run_timing(args: argparse.Namespace) -> dict[str, Any]:
    timing_dir = args.output_dir / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    command = runner_arguments(args, output_dir=timing_dir)
    with (timing_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((timing_dir / "result.json").read_text(encoding="utf-8"))


def run_profile(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import scripts.run_iclr_era5_routeb_strict_online as runner

    profile_dir = args.output_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    block_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    original_update = TorchJointSSGPKronHiPPOSVGP.update_block_structured_joint_ssgp_transfer
    original_predict = TorchJointSSGPKronHiPPOSVGP.predict_with_C

    def wrapped_update(self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        block_id = len(block_rows)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        value, flops, events = profiler_flops(
            lambda: original_update(self, *call_args, **call_kwargs)
        )
        row = {
            "block_id": block_id,
            "update_profiler_counted_flops": flops,
            "prediction_profiler_counted_flops": None,
            "profiler_counted_flops": None,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        }
        block_rows.append(row)
        operator_rows.extend({"block_id": block_id, "phase": "update", **event} for event in events)
        return value

    def wrapped_predict(self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        block_id = len(block_rows) - 1
        if block_id < 0:
            raise RuntimeError("Prediction was called before an online update")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        value, flops, events = profiler_flops(
            lambda: original_predict(self, *call_args, **call_kwargs)
        )
        block_rows[block_id]["prediction_profiler_counted_flops"] = flops
        block_rows[block_id]["profiler_counted_flops"] = (
            block_rows[block_id]["update_profiler_counted_flops"] + flops
        )
        if torch.cuda.is_available():
            block_rows[block_id]["peak_allocated_bytes"] = max(
                block_rows[block_id]["peak_allocated_bytes"], int(torch.cuda.max_memory_allocated())
            )
            block_rows[block_id]["peak_reserved_bytes"] = max(
                block_rows[block_id]["peak_reserved_bytes"], int(torch.cuda.max_memory_reserved())
            )
        operator_rows.extend({"block_id": block_id, "phase": "prediction", **event} for event in events)
        return value

    TorchJointSSGPKronHiPPOSVGP.update_block_structured_joint_ssgp_transfer = wrapped_update
    TorchJointSSGPKronHiPPOSVGP.predict_with_C = wrapped_predict
    command = runner_arguments(args, output_dir=profile_dir)
    old_argv = sys.argv
    try:
        sys.argv = command[1:]
        with (profile_dir / "run.log").open("w", encoding="utf-8") as log:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = log
            try:
                runner.main()
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
    finally:
        sys.argv = old_argv
        TorchJointSSGPKronHiPPOSVGP.update_block_structured_joint_ssgp_transfer = original_update
        TorchJointSSGPKronHiPPOSVGP.predict_with_C = original_predict
    payload = json.loads((profile_dir / "result.json").read_text(encoding="utf-8"))
    return payload, block_rows, operator_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--theta-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("task2_short", "tasks2_10_long"), required=True)
    parser.add_argument("--objective-source", choices=("finite_dtc", "vfe"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument(
        "--num-train-space",
        type=int,
        default=0,
        help="Override setup count; 0 reads the actual train-space count from the runner.",
    )
    parser.add_argument("--num-features", type=int, default=133)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument("--beta-prior-variance", type=float, default=1000.0)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--temporal-factor-device", choices=("auto", "cpu", "solver"), default="auto")
    parser.add_argument("--feature-projection-npz", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()
    if args.skip_timing and args.skip_profile:
        raise ValueError("At least one of timing or profiling must be enabled")
    if args.feature_projection_npz is not None:
        with __import__("numpy").load(args.feature_projection_npz) as projection:
            args.num_features = int(projection["basis"].shape[1])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timing_payload = None if args.skip_timing else run_timing(args)
    profile_payload: dict[str, Any] | None = None
    block_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    if not args.skip_profile:
        profile_payload, block_rows, operator_rows = run_profile(args)
        supplement = factorization_lower_bound_flops(
            ms=args.ms, mt=args.mt, num_features=args.num_features
        )
        for row in block_rows:
            row.update(supplement)
            row["profiler_plus_forward_supplement_flops"] = (
                row["profiler_counted_flops"] + supplement["analytical_supplement_forward_flops"]
            )
            row["counting_method"] = COUNTING_METHOD
            row["excluded_flops"] = EXCLUDED_FLOPS

    result_for_shape = timing_payload or profile_payload
    if result_for_shape is None:
        raise RuntimeError("No online result was produced")
    num_train_space = (
        args.num_train_space
        if args.num_train_space > 0
        else int(result_for_shape["num_train_space"])
    )
    setup_flops = setup_lower_bound_flops(ms=args.ms, num_space=num_train_space)
    profiler_counts = [float(row["profiler_counted_flops"]) for row in block_rows]
    supplements = [float(row["analytical_supplement_forward_flops"]) for row in block_rows]
    summary = {
        "method": "Route B cumulative-changing HiPPO strict online",
        "scope": args.scope,
        "objective_source": args.objective_source,
        "objective_note": (
            "DTC/VFE labels identify Task-1 empirical-Bayes calibration only; "
            "the strict-online posterior recursion is the same finite structured update."
        ),
        "seed": args.seed,
        "num_train_space": num_train_space,
        "num_blocks_profiled": len(block_rows),
        "profiler_counted_flops_total": sum(profiler_counts) if block_rows else None,
        "analytical_setup_lower_bound_flops": setup_flops,
        "analytical_block_supplement_flops_total": sum(supplements) if block_rows else None,
        "profiler_plus_analytical_lower_bound_total_flops": (
            total_flops(
                setup_flops=setup_flops,
                block_profiler_flops=profiler_counts,
                block_supplements=supplements,
            )
            if block_rows else None
        ),
        "counting_method": COUNTING_METHOD,
        "excluded_flops": EXCLUDED_FLOPS,
        "online_optimization_status": {
            "E1_cached_statistics": "already inherent in accumulated online sufficient statistics",
            "E2_grouped_basis_RHS": "already implemented by one grouped Sylvester solve",
            "E3_remove_final_solve": "already implemented as m_u = v_h - W m_beta",
        },
        "timing_result": timing_payload,
        "profile_result": profile_payload,
    }
    write_csv(args.output_dir / "online_efficiency_blocks.csv", block_rows)
    write_csv(args.output_dir / "online_profiler_operator_breakdown.csv", operator_rows)
    (args.output_dir / "online_efficiency.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"timing_result", "profile_result"}}, indent=2))


if __name__ == "__main__":
    main()
