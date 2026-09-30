#!/usr/bin/env python3
"""Run isolated repeated timing audits for the long-stream Route B methods."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_float(path: Path, field: str) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(field + ":"):
            return float(line.split()[1]) / 1024.0
    return float("nan")


def gpu_memory_mib() -> float:
    if shutil.which("nvidia-smi") is None:
        return float("nan")
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=False,
        text=True,
    )
    try:
        return float(completed.stdout.splitlines()[0].strip())
    except (IndexError, ValueError):
        return float("nan")


def gpu_metadata() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"status": "not_available"}
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=False,
        text=True,
    )
    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            devices.append(
                {
                    "name": fields[0],
                    "memory_total_mib": float(fields[1]),
                    "driver_version": fields[2],
                }
            )
        except ValueError:
            continue
    return {"status": "available" if devices else "unavailable", "devices": devices}


def process_rss_mib(pid: int) -> float:
    status = Path(f"/proc/{pid}/status")
    return read_float(status, "VmRSS") if status.is_file() else float("nan")


def run_once(command: list[str], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak_gpu = gpu_memory_mib()
    peak_rss = float("nan")
    with (output / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            peak_gpu = np.nanmax([peak_gpu, gpu_memory_mib()])
            peak_rss = np.nanmax([peak_rss, process_rss_mib(process.pid)])
            time.sleep(0.05)
    if process.returncode != 0:
        raise RuntimeError(f"Efficiency runner failed; inspect {output / 'run.log'}")
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    with (output / "blocks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Runner wrote no online blocks")
    block = rows[-1]
    resources = result.get("resources", {})
    return {
        "process_seconds": time.perf_counter() - started,
        "update_seconds": float(block["update_seconds"]),
        "prediction_seconds": float(block["prediction_seconds"]),
        "peak_nvidia_smi_mib": float(peak_gpu),
        "peak_cpu_rss_mib": float(peak_rss),
        "peak_allocated_mib": float(block.get("peak_cuda_allocated_mib", "nan")),
        "peak_reserved_mib": float(resources.get("peak_cuda_reserved_mib", "nan")),
        "persistent_state_mib": float(resources.get("persistent_state_mib", "nan")),
        "solver_device": str(resources.get("solver_device", result.get("solver_device", "unknown"))),
    }


def command_for(
    *, python: Path, protocol: Path, metadata: Path, theta: Path, output: Path, representation: str
) -> list[str]:
    command = [
        str(python), "scripts/run_iclr_era5_routeb_strict_online.py",
        "--protocol-npz", str(protocol), "--protocol-json", str(metadata),
        "--theta-json", str(theta), "--output", str(output / "result.json"),
        "--blockwise-output", str(output / "blocks.csv"), "--representation", representation,
        "--mt", "32", "--ms", "32", "--seed", "0", "--max-blocks", "5",
        "--delayed-observations", "--task1-posterior-init", "--solver-backend", "torch",
        "--device", "cuda", "--dtype", "float64", "--include-conditional-residual-variance",
    ]
    if representation == "analytic_hippo_rff":
        command.extend(["--rff-sample-size", "64", "--temporal-kernel", "spectral_mixture", "--spectral-mixture-json", "configs/covid_sm_q2.json"])
    return command


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def analytical_flops(num_features: int = 13, mt: int = 32, ms: int = 32) -> dict[str, float]:
    cholesky = 3.0 * num_features**3 / 3.0 + 3.0 * ms**3 / 3.0 + 5.0 * mt**3 / 3.0
    eigh = 18.0 * (ms**3 + mt**3)
    solve = 2.0 * num_features**3 * 3.0 + 2.0 * mt**3 * 2.0 + 2.0 * ms**3
    return {
        "analytical_cholesky_forward_flops": cholesky,
        "analytical_eigh_forward_flops": eigh,
        "analytical_principal_solve_forward_flops": solve,
        "analytical_forward_lower_bound_flops": cholesky + eigh + solve,
        "analytical_scope": "backend-neutral forward lower bound; excludes elementwise kernels, CPU temporal factors and metric/serialization work",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed0"))
    parser.add_argument("--results-root", type=Path, default=Path("results/diagnostics/covid_long_stream_2020_2024_mandatory/seed0"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/covid_long_stream_2020_2024_mandatory/efficiency"))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.warmups != 10 or args.repeats < 30:
        raise ValueError("The formal audit requires exactly 10 warm-ups and at least 30 steady-state repeats")

    protocol_root = (ROOT / args.protocol_root).resolve()
    results_root = (ROOT / args.results_root).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = ROOT / ".venv/bin/python"
    methods = (("routeb_ordinary", "inducing_points"), ("routeb_cumulative", "analytic_hippo_rff"))
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "status": "complete",
        "warmups": args.warmups,
        "steady_state_repeats": args.repeats,
        "timing_scope": "fifth real one-week update plus prediction after four causal in-process updates; the runner synchronizes CUDA around measured phases",
        "compile_jit_scope": "first isolated eager-PyTorch invocation; no torch.compile or JIT is enabled",
        "nsight_compute": {
            "status": "not_instrumented" if shutil.which("ncu") is None else "available_not_run",
            "reason": "Profiler is separated from timing; this local audit does not claim Nsight executed FLOPs.",
        },
        "framework_profiler_count": {"status": "not_instrumented", "reason": "Profiler is excluded from formal timing and has no claim in this audit."},
        "gpu": gpu_metadata(),
        "methods": {},
    }
    for method, representation in methods:
        theta = results_root / method / "calibration/result.json"
        if not theta.is_file():
            raise FileNotFoundError(f"Missing calibrated theta: {theta}")
        method_rows: list[dict[str, Any]] = []
        for phase, count in (("warmup", args.warmups), ("steady", args.repeats)):
            for repeat in range(count):
                run_dir = output / method / phase / f"repeat_{repeat:02d}"
                record = run_once(
                    command_for(
                        python=python,
                        protocol=protocol_root / "protocol.npz",
                        metadata=protocol_root / "protocol.json",
                        theta=theta,
                        output=run_dir,
                        representation=representation,
                    ),
                    run_dir,
                )
                row = {"method": method, "representation": representation, "phase": phase, "repeat": repeat, **record}
                rows.append(row)
                if phase == "steady":
                    method_rows.append(row)
        summary["methods"][method] = {
            "steady_update_seconds_mean": float(np.mean([row["update_seconds"] for row in method_rows])),
            "steady_prediction_seconds_mean": float(np.mean([row["prediction_seconds"] for row in method_rows])),
            "steady_update_prediction_seconds_mean": float(np.mean([row["update_seconds"] + row["prediction_seconds"] for row in method_rows])),
            "steady_process_seconds_mean": float(np.mean([row["process_seconds"] for row in method_rows])),
            "first_invocation_process_seconds": float(rows[-(args.repeats + args.warmups)]["process_seconds"]),
            "peak_allocated_mib": float(np.nanmax([row["peak_allocated_mib"] for row in method_rows])),
            "peak_reserved_mib": float(np.nanmax([row["peak_reserved_mib"] for row in method_rows])),
            "peak_nvidia_smi_mib": float(np.nanmax([row["peak_nvidia_smi_mib"] for row in method_rows])),
            "peak_cpu_rss_mib": float(np.nanmax([row["peak_cpu_rss_mib"] for row in method_rows])),
            "persistent_state_mib": float(np.mean([row["persistent_state_mib"] for row in method_rows])),
            **analytical_flops(),
        }
    write_csv(output / "steady_state_repeats.csv", rows)
    (output / "efficiency_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
