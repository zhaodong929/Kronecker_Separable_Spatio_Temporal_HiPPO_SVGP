#!/usr/bin/env python3
"""Run the frozen AISTATS 2027 zero-cross and identity-reuse controls."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A_SOURCE_ROOT = Path(
    "/home/zd929/projects/stvgp_kronecker_aistats_controls_0398"
)
TRANSFER_RUNNER = ROOT / "scripts/run_iclr_era5_transfer_mechanism.py"
DEFAULT_PROTOCOL_ROOT = Path(
    "/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/"
    "kronecker+s2vgp/ICLR Formal experiment/iclr_era5_stage2plus/protocol"
)
DEFAULT_A_THETA_ROOT = (
    ROOT
    / "results/diagnostics/routeb_task1_10_vfe_budget_comparison/"
    "calibration/converged/vfe"
)
DEFAULT_B_THETA_ROOT = (
    ROOT
    / "results/era5_stage2plus_sm_baseline_rerun_20260828_formal/"
    "calibration/routeb_kronhippo_stgp_sm"
)
DEFAULT_SM_CONFIG = ROOT / "configs/era5_sm_q3.json"
METRICS = ("rmse", "crps", "nll", "ece", "coverage90")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def input_record(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    record: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        record[name] = {"path": str(resolved), "sha256": sha256(resolved)}
    return record


def run_job(
    *,
    output_dir: Path,
    command: list[str],
    inputs: dict[str, Path],
    experiment: str,
    variant: str,
    seed: int,
    mt: int,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Immutable run directory already exists; refusing to reuse it: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    git_status = git_value("status", "--short")
    provenance = {
        "status": "running",
        "experiment": experiment,
        "variant": variant,
        "seed": seed,
        "mt": mt,
        "started_utc": utc_now(),
        "command": command,
        "cwd": str(ROOT),
        "python": sys.executable,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_sha256": hashlib.sha256(git_status.encode("utf-8")).hexdigest(),
        "git_status_entry_count": len(git_status.splitlines()),
        "inputs": input_record(inputs),
        "driver_sha256": sha256(Path(__file__)),
    }
    write_json(provenance, output_dir / "provenance.json")
    write_json({"command": command}, output_dir / "command.json")
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result_path = output_dir / "result.json"
    result_status = None
    if result_path.is_file():
        result_status = json.loads(result_path.read_text(encoding="utf-8")).get("status")
    provenance.update(
        {
            "ended_utc": utc_now(),
            "returncode": completed.returncode,
            "result_status": result_status,
            "status": (
                "complete"
                if completed.returncode == 0 and result_status in {None, "complete"}
                else "failed"
            ),
        }
    )
    write_json(provenance, output_dir / "provenance.json")
    if provenance["status"] != "complete":
        raise RuntimeError(
            f"Experiment failed without retry: {output_dir} (return code {completed.returncode})"
        )


def experiment_a(args: argparse.Namespace, *, resume: bool = False) -> None:
    stage_root = args.output_root / (
        "experiment_a_benchmark_zero_cross_exact_archive_continuation"
        if resume
        else "experiment_a_benchmark_zero_cross_exact_archive"
    )
    if stage_root.exists():
        raise FileExistsError(f"Stage directory already exists: {stage_root}")
    schedule = (
        (("zero_cross", True, range(2, 5)),)
        if resume
        else (
            ("full_joint_changing", False, range(5)),
            ("zero_cross", True, range(5)),
        )
    )
    for variant, zero_cross, seeds in schedule:
        for seed in seeds:
            output_dir = stage_root / variant / f"seed{seed}"
            strict_runner = args.a_source_root / "scripts/run_iclr_era5_routeb_strict_online.py"
            strict_backend = (
                args.a_source_root
                / "stvgp_kronecker/joint_ssgp_kron/torch_backend.py"
            )
            protocol = args.protocol_root / "task1_10" / f"seed{seed}"
            theta = args.a_theta_root / f"seed{seed}" / "result.json"
            command = [
                sys.executable,
                str(strict_runner),
                "--protocol-npz",
                str(protocol / "protocol.npz"),
                "--protocol-json",
                str(protocol / "protocol.json"),
                "--data-root",
                str(ROOT / "data/era5/processed_timeseries_4_task1_10_extension"),
                "--theta-json",
                str(theta),
                "--output",
                str(output_dir / "result.json"),
                "--blockwise-output",
                str(output_dir / "blocks.csv"),
                "--predictions-output",
                str(output_dir / "predictions.npz"),
                "--representation",
                "analytic_hippo_rff",
                "--mt",
                "128",
                "--ms",
                "128",
                "--rff-sample-size",
                "256",
                "--prediction-chunk-size",
                "8192",
                "--include-conditional-residual-variance",
                "--beta-prior-variance",
                "1000.0",
                "--seed",
                str(seed),
                "--solver-backend",
                "torch",
                "--device",
                args.device,
                "--dtype",
                "float64",
                "--temporal-factor-device",
                "cpu",
            ]
            if zero_cross:
                command.append("--zero-cross")
            run_job(
                output_dir=output_dir,
                command=command,
                inputs={
                    "runner": strict_runner,
                    "backend": strict_backend,
                    "protocol_npz": protocol / "protocol.npz",
                    "protocol_json": protocol / "protocol.json",
                    "theta_checkpoint": theta,
                },
                experiment="A_benchmark_operating_point_zero_cross",
                variant=variant,
                seed=seed,
                mt=128,
            )
    write_json({"status": "complete", "ended_utc": utc_now()}, stage_root / "stage.json")


def experiment_b(args: argparse.Namespace, capacities: tuple[int, ...]) -> None:
    stage_root = args.output_root / "experiment_b_identity_reuse"
    for mt in capacities:
        capacity_root = stage_root / f"mt{mt}"
        if capacity_root.exists():
            raise FileExistsError(f"Capacity directory already exists: {capacity_root}")
        for variant, mode in (
            ("conditional_transport", "structured_changing"),
            ("identity_reuse", "identity_reuse_changing"),
        ):
            for seed in range(5):
                output_dir = capacity_root / variant / f"seed{seed}"
                protocol = args.protocol_root / "task1_2" / f"seed{seed}"
                theta = args.b_theta_root / f"seed{seed}" / "result.json"
                command = [
                    sys.executable,
                    str(TRANSFER_RUNNER),
                    "--protocol-npz",
                    str(protocol / "protocol.npz"),
                    "--protocol-json",
                    str(protocol / "protocol.json"),
                    "--data-root",
                    str(ROOT / "data/era5/processed_timeseries_4"),
                    "--theta-json",
                    str(theta),
                    "--spectral-mixture-json",
                    str(args.sm_config),
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    mode,
                    "--seed",
                    str(seed),
                    "--ms",
                    "128",
                    "--mt",
                    str(mt),
                    "--rff-sample-size",
                    "256",
                    "--prediction-chunk-size",
                    "8192",
                    "--beta-prior-variance",
                    "1000.0",
                    "--device",
                    args.device,
                    "--batch-reference",
                ]
                run_job(
                    output_dir=output_dir,
                    command=command,
                    inputs={
                        "runner": TRANSFER_RUNNER,
                        "protocol_npz": protocol / "protocol.npz",
                        "protocol_json": protocol / "protocol.json",
                        "theta_checkpoint": theta,
                        "spectral_mixture_config": args.sm_config,
                    },
                    experiment="B_identity_reuse_coordinate_control",
                    variant=variant,
                    seed=seed,
                    mt=mt,
                )
        write_json(
            {"status": "complete", "mt": mt, "ended_utc": utc_now()},
            capacity_root / "stage.json",
        )


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def gaussian_prediction_metrics(path: Path) -> dict[str, float]:
    with np.load(path) as payload:
        y = np.asarray(payload["y_true"], dtype=float).reshape(-1)
        mean = np.asarray(payload["pred_mean"], dtype=float).reshape(-1)
        variance = np.maximum(
            np.asarray(payload["pred_var"], dtype=float).reshape(-1), 1e-10
        )
    std = np.sqrt(variance)
    z = (y - mean) / std
    cdf = 0.5 * (
        1.0 + np.vectorize(math.erf, otypes=[np.float64])(z / math.sqrt(2.0))
    )
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    crps = std * (
        z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi)
    )
    levels = tuple(np.arange(0.05, 1.0, 0.1))
    normal = NormalDist()
    ece = np.mean(
        [
            abs(
                float(
                    np.mean(
                        np.abs(y - mean)
                        <= normal.inv_cdf((1.0 + level) / 2.0) * std
                    )
                )
                - level
            )
            for level in levels
        ]
    )
    return {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "crps": float(np.mean(np.maximum(crps, 0.0))),
        "nll": float(
            np.mean(
                0.5
                * (
                    np.log(2.0 * np.pi * variance)
                    + (y - mean) ** 2 / variance
                )
            )
        ),
        "ece": float(ece),
        "coverage90": float(
            np.mean(np.abs(y - mean) <= 1.6448536269514722 * std)
        ),
    }


def aggregate_a(output_root: Path, aggregate_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    root = output_root / "experiment_a_benchmark_zero_cross_exact_archive"
    continuation = (
        output_root
        / "experiment_a_benchmark_zero_cross_exact_archive_continuation"
    )
    for variant in ("full_joint_changing", "zero_cross"):
        values = {metric: [] for metric in METRICS}
        for seed in range(5):
            run_root = (
                continuation
                if variant == "zero_cross" and seed >= 2
                else root
            )
            run_dir = run_root / variant / f"seed{seed}"
            metric_payload = gaussian_prediction_metrics(run_dir / "predictions.npz")
            row = {"variant": variant, "seed": seed}
            for metric in METRICS:
                row[metric] = float(metric_payload[metric])
                values[metric].append(row[metric])
            rows.append(row)
        summary: dict[str, Any] = {"variant": variant, "splits": 5}
        for metric in METRICS:
            summary[f"{metric}_mean"], summary[f"{metric}_sample_sd"] = mean_sd(
                values[metric]
            )
        summaries.append(summary)
    write_csv(rows, aggregate_root / "experiment_a_per_split.csv")
    write_csv(summaries, aggregate_root / "experiment_a_summary.csv")
    return {"per_split": rows, "summary": summaries}


def aggregate_b(output_root: Path, aggregate_root: Path) -> dict[str, Any]:
    root = output_root / "experiment_b_identity_reuse"
    capacity_dirs = sorted(
        (path for path in root.glob("mt*") if path.is_dir()),
        key=lambda path: int(path.name[2:]),
    )
    per_split: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    blockwise: list[dict[str, Any]] = []
    for capacity_dir in capacity_dirs:
        mt = int(capacity_dir.name[2:])
        for variant in ("conditional_transport", "identity_reuse"):
            values = {metric: [] for metric in METRICS}
            kl_values: list[float] = []
            for seed in range(5):
                run_dir = capacity_dir / variant / f"seed{seed}"
                payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
                row = {"mt": mt, "variant": variant, "seed": seed}
                for metric in METRICS:
                    row[metric] = float(payload["aggregate"][metric])
                    values[metric].append(row[metric])
                with (run_dir / "blocks.csv").open(newline="", encoding="utf-8") as handle:
                    blocks = list(csv.DictReader(handle))
                steady_kl: list[float] = []
                for block in blocks:
                    block_row = {
                        "mt": mt,
                        "variant": variant,
                        "seed": seed,
                        "block_id": int(block["block_id"]),
                        "reference_predictive_gaussian_kl": float(
                            block["reference_predictive_gaussian_kl"]
                        ),
                    }
                    blockwise.append(block_row)
                    if block_row["block_id"] > 0:
                        steady_kl.append(block_row["reference_predictive_gaussian_kl"])
                row["predictive_gaussian_kl_to_reference"] = float(np.mean(steady_kl))
                kl_values.append(row["predictive_gaussian_kl_to_reference"])
                per_split.append(row)
            summary: dict[str, Any] = {"mt": mt, "variant": variant, "splits": 5}
            for metric in METRICS:
                summary[f"{metric}_mean"], summary[f"{metric}_sample_sd"] = mean_sd(
                    values[metric]
                )
            (
                summary["predictive_gaussian_kl_to_reference_mean"],
                summary["predictive_gaussian_kl_to_reference_sample_sd"],
            ) = mean_sd(kl_values)
            summaries.append(summary)
    write_csv(per_split, aggregate_root / "experiment_b_per_split.csv")
    write_csv(summaries, aggregate_root / "experiment_b_summary.csv")
    write_csv(blockwise, aggregate_root / "experiment_b_blockwise_kl.csv")
    return {
        "aggregation": (
            "KL(online||recomputed_reference), averaged over query values within "
            "each block by the runner, then over non-warm-up blocks within seed, "
            "then across the five seeds"
        ),
        "per_split": per_split,
        "summary": summaries,
    }


def aggregate(args: argparse.Namespace) -> None:
    aggregate_root = args.output_root / "aggregate"
    if aggregate_root.exists():
        raise FileExistsError(f"Aggregate directory already exists: {aggregate_root}")
    aggregate_root.mkdir(parents=True)
    payload = {
        "status": "complete",
        "generated_utc": utc_now(),
        "experiment_a": aggregate_a(args.output_root, aggregate_root),
        "experiment_b": aggregate_b(args.output_root, aggregate_root),
    }
    write_json(payload, aggregate_root / "aggregate.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "experiment-a",
            "experiment-a-resume",
            "experiment-b128",
            "experiment-b-sweep",
            "aggregate",
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--a-theta-root", type=Path, default=DEFAULT_A_THETA_ROOT)
    parser.add_argument("--a-source-root", type=Path, default=DEFAULT_A_SOURCE_ROOT)
    parser.add_argument("--b-theta-root", type=Path, default=DEFAULT_B_THETA_ROOT)
    parser.add_argument("--sm-config", type=Path, default=DEFAULT_SM_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "experiment-a":
        experiment_a(args)
    elif args.stage == "experiment-a-resume":
        experiment_a(args, resume=True)
    elif args.stage == "experiment-b128":
        experiment_b(args, (128,))
    elif args.stage == "experiment-b-sweep":
        experiment_b(args, (16, 32, 64))
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
