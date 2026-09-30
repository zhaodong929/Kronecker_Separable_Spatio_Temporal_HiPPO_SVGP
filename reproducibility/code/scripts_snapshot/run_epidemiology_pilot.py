#!/usr/bin/env python3
"""Run small local-GPU Route B pilots on an audited epidemiology protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def predictive_metrics(y_true, mean, variance) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    var = np.maximum(np.asarray(variance, dtype=np.float64).reshape(-1), 1e-10)
    std = np.sqrt(var)
    result = {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "nll": float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var))),
        "mean_predictive_std": float(std.mean()),
    }
    for level, z in ((50, 0.6744897501960817), (80, 1.2815515655446004), (90, 1.6448536269514722), (95, 1.959963984540054)):
        result[f"coverage{level}"] = float(np.mean(np.abs(y - mu) <= z * std))
    result["mean_interval_width90"] = float(np.mean(2.0 * 1.6448536269514722 * std))
    return result


def baseline_rows(protocol: Path) -> list[dict]:
    with np.load(protocol) as arrays:
        train = np.asarray(arrays["train_indices"], dtype=int)
        test = np.asarray(arrays["test_indices"], dtype=int)
        calibration_y = np.asarray(arrays["calibration_y"], dtype=np.float64)
        stream_y = np.asarray(arrays["stream_y"], dtype=np.float64)
        calibration_mean = np.asarray(arrays["task1_calibration_mean"], dtype=np.float64)
        stream_mean = np.asarray(arrays["task1_stream_mean"], dtype=np.float64)
    residual = calibration_y[:, train] - calibration_mean[:, train]
    variance = max(float(np.mean(residual**2)), 1e-6)
    rows = [
        {
            "method": "shared_causal_ridge_mean",
            **predictive_metrics(
                stream_y[:, test], stream_mean[:, test], np.full_like(stream_y[:, test], variance)
            ),
        }
    ]
    visible_mean = np.mean(stream_y[:, train], axis=1, keepdims=True)
    spatial_mean = np.broadcast_to(visible_mean, (stream_y.shape[0], test.size))
    mean_residual = stream_y[:, train] - visible_mean
    rows.append(
        {
            "method": "current_block_visible_spatial_mean",
            **predictive_metrics(
                stream_y[:, test], spatial_mean, np.full_like(spatial_mean, max(float(np.mean(mean_residual**2)), 1e-6))
            ),
        }
    )
    return rows


def run(command: list[str], *, dry_run: bool) -> None:
    print(json.dumps({"command": command}), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mt", type=int, default=16)
    parser.add_argument("--ms", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--rff-sample-size", type=int, default=64)
    parser.add_argument(
        "--representations",
        choices=["global_inducing", "cumulative_hippo"],
        nargs="+",
        default=["global_inducing", "cumulative_hippo"],
    )
    parser.add_argument(
        "--temporal-kernel",
        choices=["matern32", "spectral_mixture"],
        default="matern32",
    )
    parser.add_argument("--spectral-mixture-json", type=Path)
    parser.add_argument("--validation-every", type=int)
    parser.add_argument("--early-stopping-patience-validations", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol_metadata = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    split_seed = int(protocol_metadata["split_seed"])

    if args.temporal_kernel == "spectral_mixture" and "cumulative_hippo" not in args.representations:
        raise ValueError("The spectral-mixture screen requires cumulative_hippo")

    baseline = baseline_rows(args.protocol_npz)
    python = sys.executable
    representation_by_label = {
        "global_inducing": "inducing_points",
        "cumulative_hippo": "analytic_hippo_rff",
    }
    representations = tuple(
        (representation_by_label[label], label) for label in args.representations
    )
    for representation, label in representations:
        calibration_dir = args.output_root / label / "calibration"
        calibration_command = [
            python,
            "scripts/run_iclr_era5_routeb_batch.py",
            "--protocol-npz",
            str(args.protocol_npz),
            "--protocol-json",
            str(args.protocol_json),
            "--output-dir",
            str(calibration_dir),
            "--data-part",
            "calibration",
            "--target-mode",
            "joint_xlag",
            "--representation",
            representation,
            "--mt",
            str(args.mt),
            "--ms",
            str(args.ms),
            "--iterations",
            str(args.iterations),
            "--learning-rate",
            str(args.learning_rate),
            "--validation-every",
            str(args.validation_every or max(1, args.iterations // 5)),
            "--early-stopping-patience-validations",
            str(args.early_stopping_patience_validations),
            "--early-stopping-min-delta",
            str(args.early_stopping_min_delta),
            "--split-seed",
            str(split_seed),
            "--device",
            args.device,
            "--dtype",
            "float64",
            "--evaluation-backend",
            "torch",
            "--objective-optimization-version",
            "E3",
            "--include-conditional-residual-variance",
        ]
        if representation == "analytic_hippo_rff":
            calibration_command.extend(["--rff-sample-size", str(args.rff_sample_size)])
            calibration_command.extend(["--temporal-kernel", args.temporal_kernel])
            if args.spectral_mixture_json is not None:
                calibration_command.extend(["--spectral-mixture-json", str(args.spectral_mixture_json)])
        run(calibration_command, dry_run=args.dry_run)

        result_dir = args.output_root / label / "online"
        online_command = [
            python,
            "scripts/run_iclr_era5_routeb_strict_online.py",
            "--protocol-npz",
            str(args.protocol_npz),
            "--protocol-json",
            str(args.protocol_json),
            "--theta-json",
            str(calibration_dir / "result.json"),
            "--output",
            str(result_dir / "result.json"),
            "--blockwise-output",
            str(result_dir / "blocks.csv"),
            "--predictions-output",
            str(result_dir / "predictions.npz"),
            "--representation",
            representation,
            "--mt",
            str(args.mt),
            "--ms",
            str(args.ms),
            "--seed",
            str(split_seed),
            "--solver-backend",
            "torch",
            "--device",
            args.device,
            "--dtype",
            "float64",
            "--include-conditional-residual-variance",
        ]
        if args.max_blocks > 0:
            online_command.extend(["--max-blocks", str(args.max_blocks)])
        if representation == "analytic_hippo_rff":
            online_command.extend(["--rff-sample-size", str(args.rff_sample_size)])
            online_command.extend(["--temporal-kernel", args.temporal_kernel])
            if args.spectral_mixture_json is not None:
                online_command.extend(["--spectral-mixture-json", str(args.spectral_mixture_json)])
        run(online_command, dry_run=args.dry_run)

    if args.dry_run:
        return
    rows = list(baseline)
    for _, label in representations:
        result = json.loads(
            (args.output_root / label / "online" / "result.json").read_text(encoding="utf-8")
        )
        rows.append({"method": f"routeb_{label}", **result["overall_current_block"]})
    fields = sorted({key for row in rows for key in row})
    with (args.output_root / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "pilot_summary.json").write_text(
        json.dumps({"protocol": str(args.protocol_npz), "rows": rows}, indent=2), encoding="utf-8"
    )
    report = [
        "# Epidemiology local-GPU pilot",
        "",
        f"This is spatial split seed {split_seed} on a retrospective data snapshot, not a final benchmark.",
        "",
        "| Method | RMSE | NLL | Coverage50 | Coverage80 | Coverage90 | Coverage95 | Mean std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['method']} | {row['rmse']:.4f} | {row['nll']:.4f} "
            f"| {row.get('coverage50', float('nan')):.4f} | {row.get('coverage80', float('nan')):.4f} "
            f"| {row.get('coverage90', float('nan')):.4f} | {row.get('coverage95', float('nan')):.4f} "
            f"| {row['mean_predictive_std']:.4f} |"
        )
    report.extend(
        [
            "",
            "COVID has only 52 locations and 39 post-calibration weeks in this snapshot.",
            "It validates implementation feasibility but is too short to establish a long-memory HiPPO advantage.",
        ]
    )
    (args.output_root / "pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "output": str(args.output_root.resolve())}, indent=2))


if __name__ == "__main__":
    main()
