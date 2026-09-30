#!/usr/bin/env python3
"""Run the Stage2plus ERA5 baseline rerun and build a metric audit.

The rerun is deliberately isolated from historical archives.  Route-B
controls use the requested spectral-mixture VFE configuration.  External
wrappers retain their upstream kernel when they cannot express spectral
mixtures; this is recorded explicitly in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
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
    std = np.sqrt(np.maximum(np.asarray(variance, dtype=float), 1e-12))
    draws = np.asarray(mean, dtype=float)[None, ...] + std[None, ...] * rng.standard_normal((100, *mean.shape))
    lower = np.quantile(draws, (1.0 - ECE_LEVELS)[:, None, None] / 2.0, axis=0)
    upper = np.quantile(draws, (1.0 + ECE_LEVELS)[:, None, None] / 2.0, axis=0)
    empirical = np.mean((y[None, ...] >= lower) & (y[None, ...] <= upper), axis=(1, 2))
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


def routeb_command(args: argparse.Namespace, scope: str, method: str, seed: int, output: Path) -> list[str]:
    representation = "inducing_points" if method in {"kron_stgp", "kron_stgp_residual"} else "analytic_hippo_rff"
    target = "joint_xlag" if method in {"kron_stgp", "kronhippo_stgp"} else "shared_xlag_residual"
    return [
        str(args.python), str(ROOT / "scripts/run_iclr_era5_routeb_batch.py"),
        "--protocol-npz", str(args.protocol / scope / f"seed{seed}/protocol.npz"),
        "--protocol-json", str(args.protocol / scope / f"seed{seed}/protocol.json"),
        "--data-root", str(args.data_root), "--output-dir", str(output), "--data-part", "stream",
        "--target-mode", target, "--representation", representation, "--mt", "128", "--ms", "128",
        "--iterations", str(args.routeb_iterations), "--learning-rate", "0.02", "--validation-every", "5",
        "--beta-prior-variance", "1000", "--rff-sample-size", "256", "--xlag-length", "10",
        "--prediction-chunk-size", "8192", "--split-seed", str(seed), "--model-seed", "0",
        "--training-objective", "vfe", "--temporal-kernel", "spectral_mixture",
        "--spectral-mixture-json", str(args.sm_config), "--include-conditional-residual-variance",
        "--device", args.device, "--dtype", "float64", "--evaluation-backend", "torch",
        "--warmup-steps", "1", "--predictions-output", str(output / "predictions.npz"),
    ]


def xlag_command(args: argparse.Namespace, scope: str, mode: str, seed: int, output: Path) -> list[str]:
    return [
        str(args.python), str(ROOT / "scripts/run_iclr_era5_xlag_mean_baselines.py"),
        "--protocol-npz", str(args.protocol / scope / f"seed{seed}/protocol.npz"),
        "--protocol-json", str(args.protocol / scope / f"seed{seed}/protocol.json"),
        "--output-dir", str(output), "--mode", mode, "--seed", str(seed),
    ]


def external_command(args: argparse.Namespace, scope: str, method: str, seed: int, output: Path) -> list[str]:
    protocol = args.protocol / scope / f"seed{seed}/protocol.npz"
    base = [str(args.python), str(ROOT / f"scripts/run_official_{method}_era5.py"), "--protocol-npz", str(protocol)]
    if method == "gpflow_svgp":
        return base + ["--output", str(output / "result.json"), "--predictions-output", str(output / "predictions.npz"),
                       "--target-mode", "shared_xlag_residual", "--mt", "8", "--ms", "64", "--iterations", str(args.external_iterations),
                       "--batch-size", "2048", "--learning-rate", "0.01", "--natgrad-gamma", "0.1", "--validation-every", "10",
                       "--prediction-chunk-size", "8192", "--seed", str(seed), "--device", args.device, "--dtype", "float64"]
    raise ValueError(method)


def run_job(args: argparse.Namespace, record: dict[str, object]) -> None:
    output = Path(str(record["output"]))
    output.mkdir(parents=True, exist_ok=True)
    command = [str(item) for item in record["command"]]
    (output / "command.txt").write_text(shell_command(command) + "\n", encoding="utf-8")
    status_path = output / "status.json"
    result_path = output / "result.json"
    prediction_path = output / "predictions.npz"
    if result_path.is_file() and prediction_path.is_file() and not args.force:
        record.update(status="complete_existing")
        write_json(status_path, record)
        return
    record.update(status="running", started_unix=time.time())
    write_json(status_path, record)
    started = time.perf_counter()
    with (output / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    record.update(status="complete" if proc.returncode == 0 and result_path.is_file() and prediction_path.is_file() else "failed",
                  returncode=proc.returncode, elapsed_seconds=time.perf_counter() - started)
    if record["status"] == "complete":
        record["metrics"] = metrics_from_archive(prediction_path, int(record["seed"]))
    write_json(status_path, record)
    if record["status"] != "complete":
        raise RuntimeError(f"job failed: {record['name']} returncode={proc.returncode}; see {output / 'run.log'}")


def build_records(args: argparse.Namespace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    def add(name: str, method: str, scope: str, setting: str, kernel: str, command: list[str]) -> None:
        records.append({"name": name, "method": method, "scope": scope, "setting": setting, "kernel": kernel,
                        "seed": int(name.rsplit("seed", 1)[1]), "output": str(args.output / "runs" / scope / method / name.rsplit("/", 1)[-1]),
                        "command": command, "status": "pending"})
    for short_name, scope in SCOPES.items():
        for seed in SEEDS:
            add(f"{short_name}/kron_stgp/seed{seed}", "kron_stgp", scope, "batch", "spectral_mixture",
                routeb_command(args, scope, "kron_stgp", seed, args.output / "runs" / scope / "kron_stgp" / f"seed{seed}"))
            add(f"{short_name}/kronhippo_stgp/seed{seed}", "kronhippo_stgp", scope, "batch", "spectral_mixture",
                routeb_command(args, scope, "kronhippo_stgp", seed, args.output / "runs" / scope / "kronhippo_stgp" / f"seed{seed}"))
            add(f"{short_name}/frozen_xlag/seed{seed}", "frozen_xlag", scope, "batch", "none",
                xlag_command(args, scope, "batch_fixed", seed, args.output / "runs" / scope / "frozen_xlag" / f"seed{seed}"))
        if short_name == "short":
            for method in ("kron_stgp_residual", "kronhippo_stgp_residual"):
                for seed in SEEDS:
                    add(f"{short_name}/{method}/seed{seed}", method, scope, "batch", "spectral_mixture",
                        routeb_command(args, scope, method, seed, args.output / "runs" / scope / method / f"seed{seed}"))
            for seed in SEEDS:
                add(f"{short_name}/gpflow_svgp/seed{seed}", "gpflow_svgp", scope, "batch", "matern32_fallback",
                    external_command(args, scope, "gpflow_svgp", seed, args.output / "runs" / scope / "gpflow_svgp" / f"seed{seed}"))
    return records


def aggregate(records: list[dict[str, object]], output: Path) -> None:
    rows = []
    for record in records:
        if record.get("status") not in {"complete", "complete_existing"} or "metrics" not in record:
            continue
        rows.append({"scope": record["scope"], "setting": record["setting"], "method": record["method"],
                     "kernel": record["kernel"], "seed": record["seed"], **record["metrics"]})
    with (output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["scope", "setting", "method", "kernel", "seed", "n_points", "rmse", "crps", "gaussian_nlpd", "ece", "coverage90", "mean_predictive_std"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = []
    for scope, method, kernel in sorted({(r["scope"], r["method"], r["kernel"]) for r in rows}):
        selected = [r for r in rows if r["scope"] == scope and r["method"] == method and r["kernel"] == kernel]
        if len(selected) != len(SEEDS):
            continue
        row = {"scope": scope, "method": method, "kernel": kernel, "completed_seeds": len(selected)}
        for metric in ("rmse", "crps", "gaussian_nlpd", "ece", "coverage90"):
            values = np.asarray([float(r[metric]) for r in selected])
            row[f"{metric}_mean"] = float(values.mean()); row[f"{metric}_sd"] = float(values.std(ddof=1))
        summary.append(row)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["scope", "method", "kernel", "completed_seeds"] + [x for m in ("rmse", "crps", "gaussian_nlpd", "ece", "coverage90") for x in (f"{m}_mean", f"{m}_sd")]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/era5/processed_timeseries_4")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv_cuda128/bin/python")
    parser.add_argument("--sm-config", type=Path, default=ROOT / "configs/era5_sm_q3.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--routeb-iterations", type=int, default=100)
    parser.add_argument("--external-iterations", type=int, default=100)
    parser.add_argument("--methods", nargs="*", default=None, help="Optional method subset for smoke/continuation")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve(); args.protocol = args.protocol.resolve(); args.data_root = args.data_root.resolve(); args.python = args.python.resolve(); args.sm_config = args.sm_config.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    records = build_records(args)
    if args.methods:
        records = [record for record in records if record["method"] in set(args.methods)]
    manifest = {"schema_version": 1, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "protocol_root": str(args.protocol), "protocol_sha256": {}, "seeds": list(SEEDS),
                "scopes": SCOPES, "long_batch": "skipped_by_request", "metrics": ["rmse", "crps", "gaussian_nlpd", "ece", "coverage90"],
                "kernel_policy": {"requested": "spectral_mixture", "fallback": "matern32_only_when_wrapper_has_no_SM_support"}, "records": records}
    for scope in SCOPES.values():
        for seed in SEEDS:
            p = args.protocol / scope / f"seed{seed}/protocol.npz"
            if not p.is_file(): raise FileNotFoundError(p)
            manifest["protocol_sha256"][str(p)] = sha256(p)
    write_json(args.output / "run_manifest.json", manifest)
    if not args.dry_run:
        for record in records:
            run_job(args, record)
            manifest["records"] = records
            write_json(args.output / "run_manifest.json", manifest)
    aggregate(records, args.output)
    write_json(args.output / "status.json", {"status": "complete", "records": len(records), "completed": sum(r.get("status") in {"complete", "complete_existing"} for r in records)})


if __name__ == "__main__":
    main()
