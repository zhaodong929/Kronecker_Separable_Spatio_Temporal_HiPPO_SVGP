#!/usr/bin/env python3
"""Auditable Stage2plus ERA5 baseline matrix.

The matrix reruns only cells that existed historically: short batch, short
online, and long online.  Long batch is intentionally omitted.  Route-B uses
the requested spectral-mixture VFE configuration; wrappers without a real
spectral-mixture implementation retain and report their actual kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = Path(
    "/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/"
    "kronecker+s2vgp/ICLR Formal experiment/iclr_era5_stage2plus/protocol"
)
DEFAULT_OUTPUT = ROOT / "results/era5_stage2plus_sm_baseline_rerun_20260828"
SEEDS = (0, 1, 2, 3, 4)
SCOPES = {"short": "task1_2", "long": "task1_10"}
ECE_LEVELS = np.round(np.arange(0.05, 1.0, 0.1), 2)

# Capacity is matched by meaning.  The structured method has two Kronecker
# factors; dense wrappers cannot afford their 16,384 Cartesian points.  They
# share the 128 spatial locations and use the largest practical dense grid.
CAPACITY_POLICY = {
    "routeb": {
        "mt": 128,
        "ms": 128,
        "joint_inducing": 16384,
        "rff": 256,
        "rationale": "structured Kronecker factors; controlled methods match exactly",
    },
    "dense_svgp": {
        "mt": 8,
        "ms": 128,
        "joint_inducing": 1024,
        "rationale": "same 128 spatial locations; dense MxM algebra makes 16,384 points impractical",
    },
    "maddox_short": {
        "mt": 8,
        "ms": 128,
        "joint_inducing": 1024,
        "rationale": "short-stream stable Maddox configuration",
    },
    "maddox_long": {
        "mt": 1,
        "ms": 128,
        "joint_inducing": 128,
        "rationale": "long-stream numerical-stability reduction; minimum allowed inducing capacity",
    },
    "ohsvgp": {
        "inducing_size": 128,
        "rff": 256,
        "rationale": "temporal memory order and RFF count aligned with Route-B; no separate spatial grid",
    },
    "xlag": {"rationale": "linear sufficient-statistic baseline; no inducing variables"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def shell_command(command: list[str]) -> str:
    return shlex.join([str(item) for item in command])


def gaussian_crps(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
    sigma = np.sqrt(np.maximum(np.asarray(variance, dtype=float), 1e-12))
    z = (np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)) / sigma
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return float(np.mean(sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))))


def gaussian_ece(y: np.ndarray, mean: np.ndarray, variance: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(1_000_000 + seed)
    y_flat = np.asarray(y, dtype=float).reshape(-1)
    mean_flat = np.asarray(mean, dtype=float).reshape(-1)
    std_flat = np.sqrt(np.maximum(np.asarray(variance, dtype=float), 1e-12)).reshape(-1)
    covered = np.zeros(len(ECE_LEVELS), dtype=np.int64)

    # Long ERA5 streams are too large for a full (100, time, space) draw
    # tensor.  Each chunk uses the same deterministic Monte Carlo estimator.
    for start in range(0, y_flat.size, 4096):
        stop = min(start + 4096, y_flat.size)
        draws = mean_flat[None, start:stop] + std_flat[None, start:stop] * rng.standard_normal((100, stop - start))
        for index, level in enumerate(ECE_LEVELS):
            lower = np.quantile(draws, (1.0 - level) / 2.0, axis=0)
            upper = np.quantile(draws, (1.0 + level) / 2.0, axis=0)
            covered[index] += np.count_nonzero(
                (y_flat[start:stop] >= lower) & (y_flat[start:stop] <= upper)
            )
    empirical = covered / y_flat.size
    return float(np.mean(np.abs(empirical - ECE_LEVELS)))


def metrics_from_archive(path: Path, seed: int) -> dict[str, float]:
    with np.load(path) as data:
        required = {"y_true", "pred_mean", "pred_var"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        y = np.asarray(data["y_true"], dtype=float)
        mean = np.asarray(data["pred_mean"], dtype=float)
        var = np.asarray(data["pred_var"], dtype=float)
    if y.shape != mean.shape or y.shape != var.shape:
        raise ValueError(f"{path}: shape mismatch {y.shape}, {mean.shape}, {var.shape}")
    if not np.isfinite(y).all() or not np.isfinite(mean).all() or not np.isfinite(var).all():
        raise FloatingPointError(f"{path}: non-finite archive")
    if np.any(var <= 0.0):
        raise ValueError(f"{path}: predictive variance is not strictly positive")
    z90 = 1.6448536269514722
    return {
        "n_points": int(y.size),
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "crps": gaussian_crps(y, mean, var),
        "gaussian_nlpd": float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y - mean) ** 2 / var))),
        "ece": gaussian_ece(y, mean, var, seed),
        "coverage90": float(np.mean((y >= mean - z90 * np.sqrt(var)) & (y <= mean + z90 * np.sqrt(var)))),
        "mean_predictive_std": float(np.mean(np.sqrt(var))),
    }


def protocol_paths(args: argparse.Namespace, scope: str, seed: int) -> tuple[Path, Path]:
    directory = args.protocol / SCOPES[scope] / f"seed{seed}"
    return directory / "protocol.npz", directory / "protocol.json"


def data_root_for_scope(args: argparse.Namespace, scope: str) -> Path:
    if scope == "long":
        return args.data_root.parent / "processed_timeseries_4_task1_10_extension"
    return args.data_root


def routeb_batch_command(
    args: argparse.Namespace,
    *,
    scope: str,
    method: str,
    seed: int,
    output: Path,
    data_part: str = "stream",
    kernel: str = "spectral_mixture",
) -> list[str]:
    protocol, protocol_json = protocol_paths(args, scope, seed)
    representation = "inducing_points" if method in {"kron_stgp", "kron_stgp_residual"} else "analytic_hippo_rff"
    target_mode = "joint_xlag" if method in {"kron_stgp", "kronhippo_stgp"} else "shared_xlag_residual"
    command = [
        str(args.cuda_python), str(ROOT / "scripts/run_iclr_era5_routeb_batch.py"),
        "--protocol-npz", str(protocol), "--protocol-json", str(protocol_json),
        "--data-root", str(data_root_for_scope(args, scope)), "--output-dir", str(output),
        "--data-part", data_part, "--target-mode", target_mode,
        "--representation", representation,
        "--mt", str(CAPACITY_POLICY["routeb"]["mt"]), "--ms", str(CAPACITY_POLICY["routeb"]["ms"]),
        "--iterations", str(args.routeb_iterations), "--learning-rate", "0.02",
        "--validation-every", "5", "--beta-prior-variance", "1000",
        "--rff-sample-size", str(CAPACITY_POLICY["routeb"]["rff"]),
        "--xlag-length", "10", "--prediction-chunk-size", "8192",
        "--split-seed", str(seed), "--model-seed", "0", "--training-objective", "vfe",
        "--temporal-kernel", kernel, "--include-conditional-residual-variance",
        "--device", args.device, "--dtype", "float64", "--evaluation-backend", "torch",
        "--warmup-steps", "1",
    ]
    if kernel == "spectral_mixture":
        command += ["--spectral-mixture-json", str(args.sm_config)]
    if data_part == "stream":
        command += ["--predictions-output", str(output / "predictions.npz")]
    return command


def routeb_online_command(
    args: argparse.Namespace,
    *,
    scope: str,
    method: str,
    seed: int,
    output: Path,
    theta_json: Path,
) -> list[str]:
    protocol, protocol_json = protocol_paths(args, scope, seed)
    representation = "inducing_points" if method == "kron_stgp" else "analytic_hippo_rff"
    command = [
        str(args.cuda_python), str(ROOT / "scripts/run_iclr_era5_routeb_strict_online.py"),
        "--protocol-npz", str(protocol), "--protocol-json", str(protocol_json),
        "--data-root", str(data_root_for_scope(args, scope)), "--theta-json", str(theta_json),
        "--output", str(output / "result.json"), "--blockwise-output", str(output / "blocks.csv"),
        "--predictions-output", str(output / "predictions.npz"), "--representation", representation,
        "--mt", str(CAPACITY_POLICY["routeb"]["mt"]), "--ms", str(CAPACITY_POLICY["routeb"]["ms"]),
        "--rff-sample-size", str(CAPACITY_POLICY["routeb"]["rff"]),
        "--temporal-kernel", "spectral_mixture", "--spectral-mixture-json", str(args.sm_config),
        "--prediction-chunk-size", "8192", "--beta-prior-variance", "1000", "--seed", str(seed),
        "--solver-backend", "torch", "--device", args.device, "--temporal-factor-device", "solver",
        "--dtype", "float64",
    ]
    if args.smoke:
        command += ["--max-blocks", str(args.smoke_blocks)]
    return command


def xlag_command(args: argparse.Namespace, *, scope: str, mode: str, seed: int, output: Path) -> list[str]:
    protocol, protocol_json = protocol_paths(args, scope, seed)
    return [
        str(args.cuda_python), str(ROOT / "scripts/run_iclr_era5_xlag_mean_baselines.py"),
        "--protocol-npz", str(protocol), "--protocol-json", str(protocol_json),
        "--data-root", str(data_root_for_scope(args, scope)),
        "--output-dir", str(output), "--mode", mode, "--xlag-length", "10", "--seed", str(seed),
    ]


def external_command(
    args: argparse.Namespace,
    *,
    scope: str,
    method: str,
    seed: int,
    output: Path,
    theta_json: Path,
) -> list[str]:
    protocol, _ = protocol_paths(args, scope, seed)
    if method == "gpflow_svgp":
        return [
            str(args.gpflow_python), str(ROOT / "scripts/run_official_gpflow_svgp_era5.py"),
            "--protocol-npz", str(protocol), "--output", str(output / "result.json"),
            "--predictions-output", str(output / "predictions.npz"), "--target-mode", "shared_xlag_residual",
            "--mt", str(CAPACITY_POLICY["dense_svgp"]["mt"]), "--ms", str(CAPACITY_POLICY["dense_svgp"]["ms"]),
            "--iterations", str(args.external_iterations), "--batch-size", "2048", "--learning-rate", "0.01",
            "--natgrad-gamma", "0.1", "--validation-every", "10", "--prediction-chunk-size", "8192",
            "--seed", str(seed), "--device", "cpu", "--dtype", "float64",
        ]
    if method == "maddox_streaming_sgpr":
        maddox_capacity = CAPACITY_POLICY["maddox_long" if scope == "long" else "maddox_short"]
        command = [
            str(args.cuda_python), str(ROOT / "scripts/run_official_maddox_streaming_sgpr_era5.py"),
            "--protocol-npz", str(protocol), "--theta-json", str(theta_json), "--output", str(output / "result.json"),
            "--blockwise-output", str(output / "blocks.csv"), "--predictions-output", str(output / "predictions.npz"),
            "--mt", str(maddox_capacity["mt"]), "--ms", str(maddox_capacity["ms"]),
            "--jitter", "1e-3" if scope == "long" else "1e-4",
            "--resample-ratio", "0.0" if scope == "long" else "0.2",
            "--seed", str(seed), "--device", args.device, "--dtype", "float64",
        ]
    elif method == "bui_osgpr":
        command = [
            str(args.gpflow_python), str(ROOT / "scripts/run_official_bui_osgpr_era5.py"),
            "--protocol-npz", str(protocol), "--theta-json", str(theta_json), "--output", str(output / "result.json"),
            "--blockwise-output", str(output / "blocks.csv"), "--predictions-output", str(output / "predictions.npz"),
            "--mt", str(CAPACITY_POLICY["dense_svgp"]["mt"]), "--ms", str(CAPACITY_POLICY["dense_svgp"]["ms"]),
            "--seed", str(seed), "--device", "cpu", "--dtype", "float64",
        ]
    elif method == "ohsvgp":
        command = [
            str(args.cuda_python), str(ROOT / "scripts/run_official_ohsvgp_era5.py"),
            "--protocol-npz", str(protocol), "--theta-json", str(theta_json), "--output", str(output / "result.json"),
            "--blockwise-output", str(output / "blocks.csv"), "--predictions-output", str(output / "predictions.npz"),
            "--inducing-size", str(CAPACITY_POLICY["ohsvgp"]["inducing_size"]),
            "--rff-sample-size", str(CAPACITY_POLICY["ohsvgp"]["rff"]), "--microbatch-size", "200",
            "--update-steps", "1", "--seed", str(seed), "--device", args.device, "--dtype", args.ohsvgp_dtype,
        ]
    else:
        raise ValueError(method)
    if args.smoke and method != "gpflow_svgp":
        command += [
            "--max-stream-blocks" if method == "bui_osgpr" else "--max-blocks",
            str(args.smoke_blocks),
        ]
    return command


def add_record(
    records: list[dict[str, object]], *, name: str, method: str, scope: str | None,
    setting: str, kernel: str, capacity: dict[str, object], output: Path,
    command: list[str], evaluation: bool, requires: list[str] | None = None,
) -> None:
    records.append({
        "name": name, "method": method, "scope": scope, "setting": setting,
        "kernel": kernel, "capacity": capacity, "seed": int(name.rsplit("seed", 1)[1]),
        "output": str(output), "command": command, "evaluation": evaluation,
        "requires": requires or [], "status": "pending",
    })


def build_records(args: argparse.Namespace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed in SEEDS:
        for calibration_name, method, kernel in (
            ("routeb_kron_stgp_sm", "kron_stgp", "spectral_mixture"),
            ("routeb_kronhippo_stgp_sm", "kronhippo_stgp", "spectral_mixture"),
            ("external_matern32_control", "kron_stgp", "matern32_fallback"),
        ):
            output = args.output / "calibration" / calibration_name / f"seed{seed}"
            add_record(
                records, name=f"calibration/{calibration_name}/seed{seed}", method=calibration_name,
                scope=None, setting="task1_calibration", kernel=kernel, capacity=CAPACITY_POLICY["routeb"],
                output=output, command=routeb_batch_command(
                    args, scope="short", method=method, seed=seed, output=output, data_part="calibration",
                    kernel="matern32" if kernel == "matern32_fallback" else kernel,
                ), evaluation=False,
            )

    # Historical Stage2plus cells: short batch only, no long batch.
    for seed in SEEDS:
        for method in ("kron_stgp", "kronhippo_stgp", "kron_stgp_residual", "kronhippo_stgp_residual"):
            output = args.output / "runs" / "short" / "batch" / method / f"seed{seed}"
            add_record(
                records, name=f"short/batch/{method}/seed{seed}", method=method, scope="short", setting="batch",
                kernel="spectral_mixture", capacity=CAPACITY_POLICY["routeb"], output=output,
                command=routeb_batch_command(args, scope="short", method=method, seed=seed, output=output), evaluation=True,
            )
        output = args.output / "runs" / "short" / "batch" / "gpflow_svgp" / f"seed{seed}"
        add_record(
            records, name=f"short/batch/gpflow_svgp/seed{seed}", method="gpflow_svgp", scope="short", setting="batch",
            kernel="matern32_fallback", capacity=CAPACITY_POLICY["dense_svgp"], output=output,
            command=external_command(
                args, scope="short", method="gpflow_svgp", seed=seed, output=output,
                theta_json=args.output / "calibration" / "external_matern32_control" / f"seed{seed}" / "result.json",
            ), evaluation=True,
        )
        output = args.output / "runs" / "short" / "batch" / "frozen_xlag" / f"seed{seed}"
        add_record(
            records, name=f"short/batch/frozen_xlag/seed{seed}", method="frozen_xlag", scope="short", setting="batch",
            kernel="not_applicable", capacity=CAPACITY_POLICY["xlag"], output=output,
            command=xlag_command(args, scope="short", mode="batch_fixed", seed=seed, output=output), evaluation=True,
        )

    # Historical online cells: both short and long.
    for scope in SCOPES:
        for seed in SEEDS:
            for method, calibration_name in (("kron_stgp", "routeb_kron_stgp_sm"), ("kronhippo_stgp", "routeb_kronhippo_stgp_sm")):
                output = args.output / "runs" / scope / "online" / method / f"seed{seed}"
                add_record(
                    records, name=f"{scope}/online/{method}/seed{seed}", method=method, scope=scope, setting="online",
                    kernel="spectral_mixture", capacity=CAPACITY_POLICY["routeb"], output=output,
                    command=routeb_online_command(
                        args, scope=scope, method=method, seed=seed, output=output,
                        theta_json=args.output / "calibration" / calibration_name / f"seed{seed}" / "result.json",
                    ), evaluation=True, requires=[calibration_name],
                )
            theta = args.output / "calibration" / "external_matern32_control" / f"seed{seed}" / "result.json"
            for method, kernel, capacity in (
                ("maddox_streaming_sgpr", "matern32_fallback", CAPACITY_POLICY["dense_svgp"]),
                ("bui_osgpr", "matern32_fallback", CAPACITY_POLICY["dense_svgp"]),
                ("ohsvgp", "squared_exponential_official", CAPACITY_POLICY["ohsvgp"]),
            ):
                if method == "maddox_streaming_sgpr":
                    capacity = CAPACITY_POLICY["maddox_long" if scope == "long" else "maddox_short"]
                output = args.output / "runs" / scope / "online" / method / f"seed{seed}"
                add_record(
                    records, name=f"{scope}/online/{method}/seed{seed}", method=method, scope=scope, setting="online",
                    kernel=kernel, capacity=capacity, output=output,
                    command=external_command(args, scope=scope, method=method, seed=seed, output=output, theta_json=theta),
                    evaluation=True, requires=["external_matern32_control"],
                )
            for method, mode in (("recursive_xlag", "recursive_rls"), ("frozen_xlag", "task1_fixed")):
                output = args.output / "runs" / scope / "online" / method / f"seed{seed}"
                add_record(
                    records, name=f"{scope}/online/{method}/seed{seed}", method=method, scope=scope, setting="online",
                    kernel="not_applicable", capacity=CAPACITY_POLICY["xlag"], output=output,
                    command=xlag_command(args, scope=scope, mode=mode, seed=seed, output=output), evaluation=True,
                )
    return records


def record_complete(record: dict[str, object]) -> bool:
    output = Path(str(record["output"]))
    return (output / "result.json").is_file() and (
        not record["evaluation"] or (output / "predictions.npz").is_file()
    )


def run_job(args: argparse.Namespace, record: dict[str, object]) -> None:
    output = Path(str(record["output"]))
    output.mkdir(parents=True, exist_ok=True)
    command = [str(item) for item in record["command"]]
    (output / "command.txt").write_text(shell_command(command) + "\n", encoding="utf-8")
    if record_complete(record) and not (
        args.force or (args.force_evaluation and record["evaluation"])
    ):
        record["status"] = "complete_existing"
        if record["evaluation"]:
            record["metrics"] = metrics_from_archive(output / "predictions.npz", int(record["seed"]))
        write_json(output / "status.json", record)
        return
    record.update(status="running", started_unix=time.time())
    write_json(output / "status.json", record)
    started = time.perf_counter()
    with (output / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    complete = process.returncode == 0 and record_complete(record)
    record.update(status="complete" if complete else "failed", returncode=process.returncode, elapsed_seconds=time.perf_counter() - started)
    if complete and record["evaluation"]:
        record["metrics"] = metrics_from_archive(output / "predictions.npz", int(record["seed"]))
    write_json(output / "status.json", record)
    if not complete:
        raise RuntimeError(f"job failed: {record['name']} returncode={process.returncode}; see {output / 'run.log'}")


def aggregate(records: list[dict[str, object]], output: Path) -> None:
    rows: list[dict[str, object]] = []
    for record in records:
        if (
            not record["evaluation"]
            or record.get("status") not in {"complete", "complete_existing"}
            or "metrics" not in record
        ):
            continue
        rows.append({
            "scope": record["scope"], "setting": record["setting"], "method": record["method"],
            "kernel": record["kernel"], "capacity": json.dumps(record["capacity"], sort_keys=True),
            "seed": record["seed"], **record["metrics"],
        })
    fields = ["scope", "setting", "method", "kernel", "capacity", "seed", "n_points", "rmse", "crps", "gaussian_nlpd", "ece", "coverage90", "mean_predictive_std"]
    with (output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    groups = {(row["scope"], row["setting"], row["method"], row["kernel"], row["capacity"]) for row in rows}
    summary: list[dict[str, object]] = []
    for key in sorted(groups):
        group = [row for row in rows if tuple(row[field] for field in ("scope", "setting", "method", "kernel", "capacity")) == key]
        result: dict[str, object] = dict(zip(("scope", "setting", "method", "kernel", "capacity"), key))
        result["completed_seeds"] = len(group)
        for metric in ("rmse", "crps", "gaussian_nlpd", "ece", "coverage90"):
            values = np.asarray([float(row[metric]) for row in group])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)
    summary_fields = ["scope", "setting", "method", "kernel", "capacity", "completed_seeds"] + [item for metric in ("rmse", "crps", "gaussian_nlpd", "ece", "coverage90") for item in (f"{metric}_mean", f"{metric}_sd")]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields); writer.writeheader(); writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/era5/processed_timeseries_4")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cuda-python", type=Path, default=ROOT / ".venv_cuda128/bin/python")
    parser.add_argument("--gpflow-python", type=Path, default=ROOT / ".venv_osgpr/bin/python")
    parser.add_argument("--sm-config", type=Path, default=ROOT / "configs/era5_sm_q3.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ohsvgp-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--routeb-iterations", type=int, default=100)
    parser.add_argument("--external-iterations", type=int, default=100)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-blocks", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for attribute in ("protocol", "data_root", "output", "sm_config"):
        setattr(args, attribute, Path(getattr(args, attribute)).resolve())
    # Keep virtual-environment launchers intact.  Resolving the symlink behind
    # ``bin/python`` removes the environment's site-packages from sys.path.
    args.cuda_python = Path(args.cuda_python).absolute()
    args.gpflow_python = Path(args.gpflow_python).absolute()
    if args.smoke:
        args.routeb_iterations = min(args.routeb_iterations, 10)
        args.external_iterations = min(args.external_iterations, 10)
    args.output.mkdir(parents=True, exist_ok=True)
    all_records = build_records(args)
    records = all_records if not args.methods else [
        record for record in all_records
        if record["evaluation"] and record["method"] in set(args.methods)
        or (not record["evaluation"] and record["method"] in {
            dependency for candidate in all_records if candidate["evaluation"] and candidate["method"] in set(args.methods) for dependency in candidate["requires"]
        })
    ]
    if args.smoke:
        records = [record for record in records if int(record["seed"]) == 0]
    for scope_name in SCOPES.values():
        for seed in SEEDS:
            path = args.protocol / scope_name / f"seed{seed}" / "protocol.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
    manifest = {
        "schema_version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_root": str(args.protocol),
        "protocol_sha256": {f"{scope}/seed{seed}": sha256(args.protocol / protocol_scope / f"seed{seed}" / "protocol.npz") for scope, protocol_scope in SCOPES.items() for seed in SEEDS},
        "seeds": list(SEEDS),
        "requested_cells": {"short_batch": True, "short_online": True, "long_online": True, "long_batch": False},
        "metrics": ["rmse", "crps", "gaussian_nlpd", "ece", "coverage90"],
        "kernel_policy": {"requested": "spectral_mixture", "fallback": "actual wrapper kernel is recorded; no external method is mislabeled as SM"},
        "capacity_policy": CAPACITY_POLICY,
        "records": records,
    }
    write_json(args.output / "run_manifest.json", manifest)
    if not args.dry_run:
        failure: str | None = None
        try:
            for record in records:
                run_job(args, record)
                manifest["records"] = records
                write_json(args.output / "run_manifest.json", manifest)
        except Exception as error:
            failure = repr(error)
        aggregate(records, args.output)
        status = {"status": "failed" if failure else "complete", "failure": failure, "records": len(records), "completed": sum(record.get("status") in {"complete", "complete_existing"} for record in records)}
        write_json(args.output / "status.json", status)
        if failure:
            raise RuntimeError(failure)


if __name__ == "__main__":
    main()
