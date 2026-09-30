#!/usr/bin/env python3
"""Microbenchmark Route-B feature/GP cross contractions and torch.compile."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils import benchmark

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_iclr_era5_routeb_batch import load_joint_phi, load_protocol
from scripts.run_routeb_batch_empirical_bayes import tensor_training_data
from stvgp_kronecker.routeb_empirical_bayes import (
    BatchRouteBEmpiricalBayes,
    feature_gp_cross,
)


ORDERS = ("einsum", "spatial_first", "temporal_first")


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


def largest_intermediate(
    *, order: str, ns: int, nt: int, ms: int, mt: int, block: int, dtype_bytes: int
) -> tuple[int, str]:
    output_elements = block * ms * mt
    if order == "spatial_first":
        intermediate = ms * nt * block
        label = f"G[{ms},{nt},{block}]"
    elif order == "temporal_first":
        intermediate = ns * block * mt
        label = f"G[{ns},{block},{mt}]"
    else:
        intermediate = output_elements
        label = "backend-selected einsum path; output lower bound"
    return max(output_elements, intermediate) * dtype_bytes, label


def timed_mean(function: Callable[[], Any], repeats: int) -> float:
    timer = benchmark.Timer(
        stmt="function()", globals={"function": function}, num_threads=1
    )
    return float(timer.timeit(repeats).mean)


def profiler_flops(function: Callable[[], Any]) -> int:
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as profiler:
        function()
    return int(sum(int(event.flops or 0) for event in profiler.key_averages()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--theta-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=["task2_short", "tasks2_10_long"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--blocks", type=int, nargs="+", default=[16, 32, 64, 133])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
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
    _, phi, coordinates = tensor_training_data(
        data, fit_indices, device="cuda", dtype=torch.float64
    )
    theta = json.loads(args.theta_json.read_text(encoding="utf-8"))["learned_theta"]
    model = BatchRouteBEmpiricalBayes(
        times=data.times,
        spatial_inducing=data.spatial_inducing,
        mt=args.mt,
        representation="analytic_hippo_rff",
        initial_ell_t=0.05,
        initial_ell_s=(0.35, 0.35),
        initial_kernel_variance=1.0,
        initial_noise_std=0.1,
        rff_sample_size=256,
        seed=0,
    ).to(device="cuda", dtype=torch.float64)
    model.set_theta(theta)
    with torch.no_grad():
        t_base, c_base, _, _ = model.factor_matrices(coordinates)
    del model
    weight = torch.randn(
        phi.shape[-1], args.ms, args.mt, device="cuda", dtype=torch.float64
    )

    c_reference = c_base.detach().clone().requires_grad_()
    t_reference = t_base.detach().clone().requires_grad_()
    reference_output = feature_gp_cross(c_reference, phi, t_reference)
    torch.sum(reference_output * weight).backward()
    reference_gradient = torch.cat(
        (c_reference.grad.reshape(-1), t_reference.grad.reshape(-1))
    ).detach()
    reference_output = reference_output.detach()

    rows: list[dict[str, Any]] = []
    functions: dict[tuple[str, int], Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = {}
    for order in ORDERS:
        for requested_block in args.blocks:
            block_size = min(requested_block, phi.shape[-1])

            def contraction(
                c: torch.Tensor,
                features: torch.Tensor,
                t: torch.Tensor,
                *,
                selected_order: str = order,
                selected_block: int = block_size,
            ) -> torch.Tensor:
                return feature_gp_cross(
                    c,
                    features,
                    t,
                    order=selected_order,
                    feature_block_size=selected_block,
                )

            functions[(order, block_size)] = contraction
            c = c_base.detach().clone().requires_grad_()
            t = t_base.detach().clone().requires_grad_()

            def forward() -> torch.Tensor:
                with torch.no_grad():
                    return contraction(c, phi, t)

            def forward_backward() -> torch.Tensor:
                c.grad = None
                t.grad = None
                output = contraction(c, phi, t)
                torch.sum(output * weight).backward()
                return output

            output = forward_backward().detach()
            gradient = torch.cat((c.grad.reshape(-1), t.grad.reshape(-1))).detach()
            output_error = relative_error(output, reference_output)
            gradient_error = relative_error(gradient, reference_gradient)
            if output_error >= 1e-8 or gradient_error >= 1e-7:
                raise RuntimeError(
                    f"Cross parity failure for {order}/{block_size}: "
                    f"output={output_error}, gradient={gradient_error}"
                )
            for _ in range(args.warmup):
                forward()
                forward_backward()
            torch.cuda.synchronize()
            forward_seconds = timed_mean(forward, args.repeats)
            forward_backward_seconds = timed_mean(forward_backward, args.repeats)

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.repeats):
                forward_backward()
            torch.cuda.synchronize()
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            counted_flops = profiler_flops(forward_backward)
            largest_bytes, intermediate_label = largest_intermediate(
                order=order,
                ns=phi.shape[0],
                nt=phi.shape[1],
                ms=args.ms,
                mt=args.mt,
                block=block_size,
                dtype_bytes=phi.element_size(),
            )
            row = {
                "scope": args.scope,
                "seed": args.seed,
                "order": order,
                "feature_block_size": block_size,
                "num_space": int(phi.shape[0]),
                "num_time": int(phi.shape[1]),
                "num_features": int(phi.shape[2]),
                "forward_seconds": forward_seconds,
                "forward_backward_seconds": forward_backward_seconds,
                "profiler_counted_flops_forward_backward": counted_flops,
                "profiler_counted_gflops_forward_backward": counted_flops / 1e9,
                "largest_explicit_intermediate_bytes": largest_bytes,
                "largest_intermediate_description": intermediate_label,
                "peak_allocated_bytes": int(peak_allocated),
                "peak_reserved_bytes": int(peak_reserved),
                "output_relative_error": output_error,
                "gradient_relative_error": gradient_error,
                "counting_method": "PyTorch Profiler with_flops=True; unsupported operators excluded",
            }
            rows.append(row)
            write_csv(args.output_dir / "cross_microbenchmark.csv", rows)
            print(json.dumps(row), flush=True)
            del c, t, output, gradient
            gc.collect()

    best = min(rows, key=lambda row: float(row["forward_backward_seconds"]))
    best_key = (str(best["order"]), int(best["feature_block_size"]))
    eager_function = functions[best_key]
    compile_rows: list[dict[str, Any]] = []
    for mode in ("default", "reduce-overhead"):
        torch._dynamo.reset()
        compiled = torch.compile(eager_function, mode=mode)
        c = c_base.detach().clone().requires_grad_()
        t = t_base.detach().clone().requires_grad_()

        def compiled_forward_backward() -> torch.Tensor:
            c.grad = None
            t.grad = None
            output = compiled(c, phi, t)
            torch.sum(output * weight).backward()
            return output

        started = time.perf_counter()
        compiled_output = compiled_forward_backward().detach()
        torch.cuda.synchronize()
        compilation_seconds = time.perf_counter() - started
        compiled_gradient = torch.cat((c.grad.reshape(-1), t.grad.reshape(-1))).detach()
        output_error = relative_error(compiled_output, reference_output)
        gradient_error = relative_error(compiled_gradient, reference_gradient)
        for _ in range(args.warmup):
            compiled_forward_backward()
        torch.cuda.synchronize()
        steady_seconds = timed_mean(compiled_forward_backward, args.repeats)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            compiled_forward_backward()
        torch.cuda.synchronize()
        row = {
            "scope": args.scope,
            "seed": args.seed,
            "compile_mode": mode,
            "selected_order": best_key[0],
            "selected_feature_block_size": best_key[1],
            "compilation_latency_seconds": compilation_seconds,
            "steady_forward_backward_seconds": steady_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "output_relative_error": output_error,
            "gradient_relative_error": gradient_error,
        }
        compile_rows.append(row)
        write_csv(args.output_dir / "compile_microbenchmark.csv", compile_rows)
        print(json.dumps(row), flush=True)
        del compiled, c, t
        gc.collect()
        torch.cuda.empty_cache()

    summary = {"best_eager": best, "compile": compile_rows}
    (args.output_dir / "cross_microbenchmark.json").write_text(
        json.dumps({"rows": rows, **summary}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
