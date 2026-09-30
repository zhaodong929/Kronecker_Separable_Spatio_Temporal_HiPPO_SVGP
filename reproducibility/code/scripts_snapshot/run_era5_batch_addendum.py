#!/usr/bin/env python3
"""Run the independent ERA5 batch addendum without touching old archives.

The addendum compares the official AaltoML ST-SVGP wrapper with the two
Kronecker VFE representations under the existing shared ERA5 batch protocol.
Each job owns a directory containing its command, log, status, result and
prediction archive.  Missing or failed jobs are never replaced by old results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = (
    ROOT
    / "results"
    / "experiments_era5_ohsvgp_heldout_fullspace"
    / "paper_ready"
    / "ICLR Formal experiment"
    / "iclr_era5_full_benchmark"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "experiments_era5_ohsvgp_heldout_fullspace"
    / "paper_ready"
    / "ICLR Formal experiment"
    / "era5_batch_addendum_20260827"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--routeb-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument(
        "--official-python",
        type=Path,
        default=ROOT / ".envs/stvgp_official_py37/bin/python",
    )
    parser.add_argument("--scopes", nargs="+", default=["task1_2", "task1_10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--official-ms", type=int, default=128)
    parser.add_argument("--official-iterations", type=int, default=100)
    parser.add_argument("--routeb-iterations", type=int, default=100)
    parser.add_argument(
        "--target-mode",
        choices=["direct", "shared_xlag_residual", "joint_xlag"],
        default="joint_xlag",
    )
    parser.add_argument(
        "--temporal-kernel",
        choices=["rbf", "matern32", "spectral_mixture"],
        default="matern32",
    )
    parser.add_argument("--spectral-mixture-json", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["official_st_svgp", "kron_stgp_vfe", "kronhippo_stgp_vfe"],
        default=["official_st_svgp", "kron_stgp_vfe", "kronhippo_stgp_vfe"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def shell_join(parts: list[str]) -> str:
    """Format an argv list on Python versions before shlex.join was added."""
    return " ".join(shlex.quote(str(part)) for part in parts)


def export_official_protocol(protocol_npz: Path, output: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/export_iclr_protocol_for_official_stvgp.py"),
        "--protocol-npz",
        str(protocol_npz),
        "--output",
        str(output),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, check=True)


def command_for(
    *,
    method: str,
    scope: str,
    seed: int,
    args: argparse.Namespace,
    output: Path,
    official_data: Path,
    protocol_npz: Path,
    protocol_json: Path,
) -> tuple[list[str], dict[str, str]]:
    prediction = output / "predictions.npz"
    if method == "official_st_svgp":
        command = [
            str(args.official_python),
            str(ROOT / "scripts/run_official_stvgp_cuda_compat.py"),
            "--model",
            "st_svgp",
            "--data-npz",
            str(official_data),
            "--num-spatial-inducing",
            str(args.official_ms),
            "--fixed-spatial-inducing",
            "--iterations",
            str(args.official_iterations),
            "--seed",
            str(seed),
            "--use-xlag-mean",
            "--jit",
            "--parallel",
            "--trajectory-every",
            "10",
            "--predictions-output",
            str(prediction),
            "--trajectory-output",
            str(output / "trajectory.json"),
            "--output",
            str(output / "result.json"),
        ]
        env = {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
        }
        return command, env

    representation = (
        "inducing_points" if method == "kron_stgp_vfe" else "analytic_hippo_rff"
    )
    command = [
        str(args.routeb_python),
        str(ROOT / "scripts/run_iclr_era5_routeb_batch.py"),
        "--protocol-npz",
        str(protocol_npz),
        "--protocol-json",
        str(protocol_json),
        "--output-dir",
        str(output),
        "--data-part",
        "stream",
        "--target-mode",
        args.target_mode,
        "--representation",
        representation,
        "--mt",
        "128",
        "--ms",
        "128",
        "--iterations",
        str(args.routeb_iterations),
        "--learning-rate",
        "0.02",
        "--validation-every",
        "5",
        "--beta-prior-variance",
        "1000",
        "--rff-sample-size",
        "256",
        "--xlag-length",
        "10",
        "--prediction-chunk-size",
        "8192",
        "--split-seed",
        str(seed),
        "--model-seed",
        "0",
        "--training-objective",
        "vfe",
        "--temporal-kernel",
        args.temporal_kernel,
        "--include-conditional-residual-variance",
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--evaluation-backend",
        "torch",
        "--warmup-steps",
        "1",
        "--predictions-output",
        str(prediction),
    ]
    if args.spectral_mixture_json is not None:
        command.extend(["--spectral-mixture-json", str(args.spectral_mixture_json)])
    return command, {}


def run_job(
    *,
    method: str,
    scope: str,
    seed: int,
    args: argparse.Namespace,
    protocol_npz: Path,
    protocol_json: Path,
    official_data: Path,
) -> dict:
    output = args.output_root / "runs" / scope / method / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    command, extra_env = command_for(
        method=method,
        scope=scope,
        seed=seed,
        args=args,
        output=output,
        official_data=official_data,
        protocol_npz=protocol_npz,
        protocol_json=protocol_json,
    )
    command_text = shell_join(command)
    (output / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    status_path = output / "status.json"
    result_path = output / "result.json"
    if result_path.is_file() and result_path.stat().st_size and not args.force:
        status = {
            "status": "complete_existing",
            "method": method,
            "scope": scope,
            "seed": seed,
            "result": str(result_path),
            "command": command_text,
        }
        write_json(status_path, status)
        return status
    if args.dry_run:
        status = {
            "status": "dry_run",
            "method": method,
            "scope": scope,
            "seed": seed,
            "command": command_text,
        }
        write_json(status_path, status)
        return status

    started = time.time()
    status = {
        "status": "running",
        "method": method,
        "scope": scope,
        "seed": seed,
        "command": command_text,
        "started_unix": started,
    }
    write_json(status_path, status)
    env = os.environ.copy()
    env.update(extra_env)
    with (output / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status.update(
        {
            "status": "complete" if completed.returncode == 0 and result_path.is_file() else "failed",
            "returncode": completed.returncode,
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "result_exists": result_path.is_file(),
        }
    )
    write_json(status_path, status)
    return status


def main() -> None:
    args = parse_args()
    if not args.seeds or not args.scopes or not args.methods:
        raise ValueError("At least one scope, seed and method is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol_manifest = args.benchmark_root / "protocol/manifest.json"
    if not protocol_manifest.is_file():
        raise FileNotFoundError(protocol_manifest)

    records = []
    for scope in args.scopes:
        for seed in args.seeds:
            protocol_dir = args.benchmark_root / "protocol" / scope / f"seed{seed}"
            protocol_npz = protocol_dir / "protocol.npz"
            protocol_json = protocol_dir / "protocol.json"
            if not protocol_npz.is_file() or not protocol_json.is_file():
                raise FileNotFoundError(f"Missing protocol for {scope} seed {seed}")
            official_data = args.output_root / "official_protocol" / scope / f"seed{seed}/data.npz"
            if not official_data.is_file() and not args.dry_run:
                export_official_protocol(protocol_npz, official_data)
            for method in args.methods:
                records.append(
                    run_job(
                        method=method,
                        scope=scope,
                        seed=seed,
                        args=args,
                        protocol_npz=protocol_npz,
                        protocol_json=protocol_json,
                        official_data=official_data,
                    )
                )
    write_json(
        args.output_root / "run_manifest.json",
        {
            "schema_version": 1,
            "purpose": "ERA5 short/long batch addendum",
            "benchmark_root": str(args.benchmark_root),
            "output_root": str(args.output_root),
            "scopes": args.scopes,
            "seeds": args.seeds,
            "methods": args.methods,
            "official_ms": args.official_ms,
            "official_iterations": args.official_iterations,
            "routeb_iterations": args.routeb_iterations,
            "records": records,
        },
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
