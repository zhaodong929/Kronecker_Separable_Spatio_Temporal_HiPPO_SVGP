#!/usr/bin/env python3
"""Run the local ERA5 spectral-mixture/VFE formal suite.

The Route-B jobs use the shared effective X-lag protocol and keep each job's
command, log, status, result, and prediction archive in an isolated directory.
The official ST-SVGP job is intentionally the historical CPU Bayes-Newton
entrypoint and is recorded separately from the Route-B kernel comparison.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "autodl_results/rtx4090_gpu_only_published_20260820T145100Z/benchmark"
)
DEFAULT_OUTPUT = (
    ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/"
    "ICLR Formal experiment/era5_sm_vfe_formal_20260827"
)
DEFAULT_DATA_ROOT = ROOT / "data/era5/processed_timeseries_4_task1_10_extension"
METHODS = {
    "kron_stgp_vfe": "inducing_points",
    "kronhippo_stgp_vfe": "analytic_hippo_rff",
}


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def make_official_direct_data(source: Path, output: Path) -> None:
    """Preserve the train/test and X-lag arrays for the direct official path."""
    with np.load(source) as data:
        payload = {
            "times": data["times"],
            "train_coords": data["train_coords"],
            "test_coords": data["test_coords"],
            "y_train": data["y_train"],
            "y_test": data["y_test"],
            "xlag_mean_train": data["xlag_mean_train"],
            "xlag_mean_test": data["xlag_mean_test"],
            "train_indices": data["train_indices"],
            "test_indices": data["test_indices"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def run_job(name: str, command: list[str], output: Path, env_updates: dict[str, str] | None = None, artifact: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json" if artifact is None else artifact
    status_path = output / "status.json"
    command_string = command_text(command)
    (output / "command.txt").write_text(command_string + "\n", encoding="utf-8")
    if result_path.is_file() and result_path.stat().st_size > 0:
        status = {"name": name, "status": "complete_existing", "artifact": str(result_path), "command": command_string}
        write_json(status_path, status)
        return status

    status = {"name": name, "status": "running", "command": command_string, "started_unix": time.time()}
    write_json(status_path, status)
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    started = time.perf_counter()
    with (output / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    status.update({
        "status": "complete" if completed.returncode == 0 and result_path.is_file() else "failed",
        "returncode": completed.returncode,
        "result_exists": result_path.is_file(),
        "artifact": str(result_path),
        "elapsed_seconds": time.perf_counter() - started,
        "finished_unix": time.time(),
    })
    write_json(status_path, status)
    return status


def routeb_common(protocol: Path, protocol_json: Path, output: Path, seed: int, representation: str, *, data_part: str, data_root: Path) -> list[str]:
    initial_ell_t = "0.05" if data_part == "calibration" else "0.5"
    return [
        str(ROOT / ".venv_cuda128/bin/python"),
        str(ROOT / "scripts/run_iclr_era5_routeb_batch.py"),
        "--protocol-npz", str(protocol),
        "--protocol-json", str(protocol_json),
        "--data-root", str(data_root),
        "--output-dir", str(output),
        "--data-part", data_part,
        "--target-mode", "joint_xlag",
        "--representation", representation,
        "--mt", "128", "--ms", "128",
        "--iterations", "100", "--learning-rate", "0.02",
        "--validation-every", "5", "--beta-prior-variance", "1000",
        "--rff-sample-size", "256", "--xlag-length", "10",
        "--prediction-chunk-size", "8192",
        "--split-seed", str(seed), "--model-seed", "0",
        "--training-objective", "vfe",
        "--temporal-kernel", "spectral_mixture",
        "--spectral-mixture-json", str(ROOT / "configs/era5_sm_q3.json"),
        "--initial-ell-t", initial_ell_t,
        "--include-conditional-residual-variance",
        "--device", "cuda", "--dtype", "float64",
        "--evaluation-backend", "torch", "--warmup-steps", "1",
        "--predictions-output", str(output / "predictions.npz"),
    ]


def online_command(protocol: Path, protocol_json: Path, theta: Path, output: Path, seed: int, representation: str, data_root: Path) -> list[str]:
    return [
        str(ROOT / ".venv_cuda128/bin/python"),
        str(ROOT / "scripts/run_iclr_era5_routeb_strict_online.py"),
        "--protocol-npz", str(protocol), "--protocol-json", str(protocol_json),
        "--data-root", str(data_root),
        "--theta-json", str(theta),
        "--output", str(output / "result.json"),
        "--blockwise-output", str(output / "blockwise.csv"),
        "--predictions-output", str(output / "predictions.npz"),
        "--representation", representation,
        "--mt", "128", "--ms", "128", "--rff-sample-size", "256",
        "--temporal-kernel", "spectral_mixture",
        "--spectral-mixture-json", str(ROOT / "configs/era5_sm_q3.json"),
        "--prediction-chunk-size", "8192", "--beta-prior-variance", "1000",
        "--seed", str(seed), "--solver-backend", "torch",
        "--device", "cuda", "--dtype", "float64",
        "--temporal-factor-device", "solver",
        "--include-conditional-residual-variance",
    ]


def official_command(data: Path, output: Path, seed: int) -> list[str]:
    return [
        str(ROOT / ".envs/stvgp_official_py37/bin/python"),
        str(ROOT / "scripts/run_official_stvgp_legacy.py"),
        "--model", "st_svgp", "--data-npz", str(data),
        "--num-spatial-inducing", "30", "--fixed-spatial-inducing",
        "--iterations", "100", "--temporal-lengthscale", "0.05",
        "--spatial-lengthscale", "0.35", "--likelihood-variance", "0.01",
        "--learning-rate", "0.05", "--newton-rate", "1.0",
        "--early-stop-relative-tol", "0.002", "--early-stop-patience", "10",
        "--early-stop-min-iterations", "30", "--seed", str(seed), "--use-xlag-mean",
        "--jit", "--trajectory-every", "10",
        "--predictions-output", str(output / "predictions.npz"),
        "--trajectory-output", str(output / "trajectory.json"),
        "--output", str(output / "result.json"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force-long",
        action="store_true",
        help="Clear only task1_10 batch/online artifacts before rerunning them.",
    )
    parser.add_argument(
        "--skip-long-batch",
        action="store_true",
        help="Run task1_10 online only; do not create or rerun long batch jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    scopes = {"task1_2": "short", "task1_10": "long"}

    config = {
        "suite": "ERA5 spectral mixture VFE formal rerun",
        "benchmark_root": str(args.benchmark_root),
        "data_root": str(args.data_root),
        "output_root": str(output_root),
        "seeds": args.seeds,
        "routeb": {"target_mode": "joint_xlag", "xlag_length": 10, "xlag_features": 133, "objective": "vfe", "temporal_kernel": "spectral_mixture", "mixture_config": str(ROOT / "configs/era5_sm_q3.json"), "Mt": 128, "Ms": 128, "RFF": 256, "conditional_residual_variance": True},
        "numerical_stability": {"batch_initial_temporal_lengthscale": 0.5, "reason": "long-axis SM-VFE Cholesky-gradient stability; kernel family and mixture parameters unchanged"},
        "official_st_svgp": {"entrypoint": "run_official_stvgp_legacy.py", "backend": "CPU", "scope": "task1_2", "Ms": 30, "jit": True, "protocol_path": "direct 800-train/200-test path with fixed X-lag mean", "reason": "the historical official short-batch path is direct; validation-only arrays are omitted to avoid a selection/refit memory overlap on the local 16 GiB host"},
    }
    write_json(output_root / "experiment_config.json", config)

    for seed in args.seeds:
        protocol = args.benchmark_root / "protocol/task1_2" / f"seed{seed}/protocol.npz"
        protocol_json = protocol.with_suffix(".json")
        if not protocol.is_file() or not protocol_json.is_file():
            raise FileNotFoundError(f"Missing task1_2 protocol for seed {seed}")
        for method, representation in METHODS.items():
            calibration = output_root / f"calibration/{method}/seed{seed}"
            command = routeb_common(protocol, protocol_json, calibration, seed, representation, data_part="calibration", data_root=args.data_root)
            if args.force:
                (calibration / "result.json").unlink(missing_ok=True)
            records.append(run_job(f"calibration/{method}/seed{seed}", command, calibration))

    for scope, label in scopes.items():
        for seed in args.seeds:
            protocol = args.benchmark_root / "protocol" / scope / f"seed{seed}/protocol.npz"
            protocol_json = protocol.with_suffix(".json")
            if not protocol.is_file() or not protocol_json.is_file():
                raise FileNotFoundError(f"Missing {scope} protocol for seed {seed}")
            for method, representation in METHODS.items():
                calibration = output_root / f"calibration/{method}/seed{seed}/result.json"
                batch = output_root / f"runs/{scope}/batch/{method}/seed{seed}"
                if not (scope == "task1_10" and args.skip_long_batch):
                    command = routeb_common(protocol, protocol_json, batch, seed, representation, data_part="stream", data_root=args.data_root)
                    if args.force or (args.force_long and scope == "task1_10"):
                        (batch / "result.json").unlink(missing_ok=True)
                    records.append(run_job(f"runs/{scope}/batch/{method}/seed{seed}", command, batch))

                if scope == "task1_2":
                    online = output_root / f"runs/{scope}/online/{method}/seed{seed}"
                    command = online_command(protocol, protocol_json, calibration, online, seed, representation, args.data_root)
                    if args.force or (args.force_long and scope == "task1_10"):
                        (online / "result.json").unlink(missing_ok=True)
                    records.append(run_job(f"runs/{scope}/online/{method}/seed{seed}", command, online))
                else:
                    online = output_root / f"runs/{scope}/online/{method}/seed{seed}"
                    command = online_command(protocol, protocol_json, calibration, online, seed, representation, args.data_root)
                    if args.force or (args.force_long and scope == "task1_10"):
                        (online / "result.json").unlink(missing_ok=True)
                    records.append(run_job(f"runs/{scope}/online/{method}/seed{seed}", command, online))

    if not args.skip_official:
        for seed in args.seeds:
            data = output_root / f"official_protocol/task1_2/seed{seed}/data.npz"
            if not data.is_file():
                export = [str(ROOT / ".venv_cuda128/bin/python"), str(ROOT / "scripts/export_iclr_protocol_for_official_stvgp.py"), "--protocol-npz", str(args.benchmark_root / f"protocol/task1_2/seed{seed}/protocol.npz"), "--output", str(data)]
                export_status = run_job(f"official_protocol/task1_2/seed{seed}", export, data.parent, artifact=data)
                records.append(export_status)
            direct_data = data.with_name("direct_data.npz")
            if not direct_data.is_file():
                make_official_direct_data(data, direct_data)
            official = output_root / f"runs/task1_2/official_st_svgp/seed{seed}"
            command = official_command(direct_data, official, seed)
            if args.force:
                (official / "result.json").unlink(missing_ok=True)
            records.append(run_job(f"runs/task1_2/official_st_svgp/seed{seed}", command, official, {"JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}))

    write_json(output_root / "run_manifest.json", {**config, "records": records})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
