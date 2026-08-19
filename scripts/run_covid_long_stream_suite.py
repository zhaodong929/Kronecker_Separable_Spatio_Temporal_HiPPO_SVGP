#!/usr/bin/env python3
"""Run the paired long-history COVID baseline suite without overwriting pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "deterministic",
    "ohsvgp_rbf",
    "routeb_ordinary",
    "routeb_cumulative",
    "bui_controlled",
    "bui_adaptive",
)


def complete(path: Path, expected_weeks: int) -> bool:
    if not (path / "result.json").is_file() and not (path / "predictions.npz").is_file():
        return False
    if not (path / "result.json").is_file() or not (path / "predictions.npz").is_file():
        return False
    try:
        with np.load(path / "predictions.npz") as archive:
            return int(np.asarray(archive["y_true"]).shape[0]) == int(expected_weeks)
    except (KeyError, OSError, ValueError):
        return False


def run_command(command: list[str], *, output: Path, dry_run: bool) -> dict[str, object]:
    if dry_run:
        return {"status": "dry_run", "command": command}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode == 0:
        return {"status": "complete", "command": command}
    status = {"status": "failed", "returncode": process.returncode, "command": command}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def routeb_commands(
    *,
    python: str,
    protocol: Path,
    protocol_json: Path,
    calibration: Path,
    online: Path,
    seed: int,
    representation: str,
    device: str,
    max_blocks: int,
    delay_weeks: int,
    forecast_without_current_visible_observations: bool,
) -> tuple[list[str], list[str]]:
    calibration_command = [
        python,
        "scripts/run_iclr_era5_routeb_batch.py",
        "--protocol-npz", str(protocol), "--protocol-json", str(protocol_json),
        "--output-dir", str(calibration), "--data-part", "calibration",
        "--target-mode", "joint_xlag", "--representation", representation,
        "--mt", "32", "--ms", "32", "--iterations", "500", "--learning-rate", "0.02",
        "--validation-every", "10", "--early-stopping-patience-validations", "10",
        "--split-seed", str(seed), "--device", device, "--dtype", "float64",
        "--evaluation-backend", "torch", "--objective-optimization-version", "E3",
        "--include-conditional-residual-variance",
    ]
    online_command = [
        python,
        "scripts/run_iclr_era5_routeb_strict_online.py",
        "--protocol-npz", str(protocol), "--protocol-json", str(protocol_json),
        "--theta-json", str(calibration / "result.json"),
        "--output", str(online / "result.json"),
        "--blockwise-output", str(online / "blocks.csv"),
        "--predictions-output", str(online / "predictions.npz"),
        "--representation", representation, "--mt", "32", "--ms", "32",
        "--seed", str(seed), "--task1-posterior-init",
        "--solver-backend", "torch", "--device", device, "--dtype", "float64",
        "--include-conditional-residual-variance",
    ]
    if representation == "analytic_hippo_rff":
        calibration_command.extend(
            ["--rff-sample-size", "64", "--temporal-kernel", "spectral_mixture", "--spectral-mixture-json", "configs/covid_sm_q2.json"]
        )
        online_command.extend(
            ["--temporal-kernel", "spectral_mixture", "--spectral-mixture-json", "configs/covid_sm_q2.json"]
        )
    if max_blocks > 0:
        online_command.extend(["--max-blocks", str(max_blocks)])
    if delay_weeks == 1:
        online_command.append("--delayed-observations")
    else:
        online_command.extend(["--delayed-observation-blocks", str(delay_weeks)])
    if forecast_without_current_visible_observations:
        online_command.append("--forecast-without-current-visible-observations")
    return calibration_command, online_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"))
    parser.add_argument("--output-root", type=Path, default=Path("results/diagnostics/covid_long_stream_2020_2024_mandatory"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bui-python", type=Path, default=Path(".venv_osgpr/bin/python"))
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument(
        "--reuse-calibration-root",
        type=Path,
        help="Optional root containing existing per-seed Route B Task-1 calibration artifacts.",
    )
    parser.add_argument(
        "--forecast-without-current-visible-observations",
        action="store_true",
        help="Run the Route B one-step forecasting ablation without current visible labels.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    python = str(ROOT / ".venv/bin/python")
    # Do not resolve a virtualenv's interpreter symlink: resolving it discards
    # the virtualenv prefix and imports packages from the system Python.
    bui_python = str(ROOT / args.bui_python)
    protocol_root = (ROOT / args.protocol_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    calibration_root = (
        None
        if args.reuse_calibration_root is None
        else (ROOT / args.reuse_calibration_root).resolve()
    )
    if args.forecast_without_current_visible_observations:
        unsupported = set(args.methods) - {"routeb_ordinary", "routeb_cumulative"}
        if unsupported:
            raise ValueError(
                "The one-step forecasting ablation is defined only for Route B; "
                f"unsupported methods: {sorted(unsupported)}"
            )
    manifest: list[dict[str, object]] = []
    for seed in args.seeds:
        protocol = protocol_root / f"seed{seed}" / "protocol.npz"
        protocol_json = protocol.with_suffix(".json")
        if not protocol.is_file() or not protocol_json.is_file():
            raise FileNotFoundError(f"Missing long-stream protocol for seed {seed}: {protocol}")
        seed_root = output_root / f"seed{seed}"
        protocol_metadata = json.loads(protocol_json.read_text(encoding="utf-8"))
        delay_weeks = int(protocol_metadata["xlag"]["delay_weeks"])
        expected_weeks = int(protocol_metadata["num_stream_times"])
        cumulative_calibration = (
            calibration_root / f"seed{seed}" / "routeb_cumulative" / "calibration"
            if calibration_root is not None
            else seed_root / "routeb_cumulative" / "calibration"
        )

        if delay_weeks > 1 and any(
            method in args.methods
            for method in ("deterministic", "ohsvgp_rbf", "bui_controlled", "bui_adaptive")
        ):
            raise ValueError(
                "Only Route B supports delayed-observation stress delays above one week; "
                "persistence and the pinned OHSVGP/Bui wrappers would use unavailable labels."
            )

        if "deterministic" in args.methods:
            root = seed_root / "deterministic"
            persistence = root / "persistence"
            if complete(persistence, expected_weeks):
                manifest.append({"seed": seed, "method": "deterministic", "status": "skipped_complete"})
            else:
                command = [
                    python, "scripts/run_covid_delayed_deterministic_baselines.py",
                    "--protocol-npz", str(protocol), "--protocol-json", str(protocol_json),
                    "--output-root", str(root), "--seed", str(seed),
                ]
                manifest.append({"seed": seed, "method": "deterministic", **run_command(command, output=root, dry_run=args.dry_run)})

        if "ohsvgp_rbf" in args.methods:
            root = seed_root / "ohsvgp_rbf"
            if complete(root, expected_weeks):
                manifest.append({"seed": seed, "method": "ohsvgp_rbf", "status": "skipped_complete"})
            else:
                command = [
                    python, "scripts/run_covid_ohsvgp_own_theta.py",
                    "--protocol-npz", str(protocol), "--protocol-json", str(protocol_json),
                    "--output-dir", str(root), "--kernel", "rbf", "--inducing-size", "32",
                    "--rff-sample-size", "64", "--calibration-iterations", "500",
                    "--validation-every", "10", "--early-stopping-patience-validations", "10",
                    "--learning-rate", "0.001", "--update-steps", "1", "--delayed-observations",
                    "--seed", str(seed), "--device", args.device, "--dtype", "float64",
                ]
                manifest.append({"seed": seed, "method": "ohsvgp_rbf", **run_command(command, output=root, dry_run=args.dry_run)})

        for method, representation in (
            ("routeb_ordinary", "inducing_points"),
            ("routeb_cumulative", "analytic_hippo_rff"),
        ):
            if method not in args.methods:
                continue
            calibration = (
                calibration_root / f"seed{seed}" / method / "calibration"
                if calibration_root is not None
                else seed_root / method / "calibration"
            )
            online = seed_root / method / "online"
            if complete(online, expected_weeks):
                manifest.append({"seed": seed, "method": method, "status": "skipped_complete"})
                continue
            calibration_command, online_command = routeb_commands(
                python=python, protocol=protocol, protocol_json=protocol_json,
                calibration=calibration, online=online, seed=seed, representation=representation,
                device=args.device, max_blocks=args.max_blocks, delay_weeks=delay_weeks,
                forecast_without_current_visible_observations=(
                    args.forecast_without_current_visible_observations
                ),
            )
            if not (calibration / "result.json").is_file() and calibration_root is not None:
                raise FileNotFoundError(
                    f"Missing reused Task-1 calibration for seed {seed}, {method}: {calibration}"
                )
            if not (calibration / "result.json").is_file():
                record = run_command(calibration_command, output=calibration, dry_run=args.dry_run)
                manifest.append({"seed": seed, "method": f"{method}_calibration", **record})
                if record["status"] == "failed":
                    continue
            manifest.append({"seed": seed, "method": method, **run_command(online_command, output=online, dry_run=args.dry_run)})

        if "bui_controlled" in args.methods:
            root = seed_root / "bui_osgpr_controlled"
            if complete(root, expected_weeks):
                manifest.append({"seed": seed, "method": "bui_controlled", "status": "skipped_complete"})
            elif not (cumulative_calibration / "result.json").is_file() and not args.dry_run:
                status = {"status": "blocked", "reason": "controlled Bui requires Route B cumulative Task-1 theta"}
                root.mkdir(parents=True, exist_ok=True)
                (root / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                manifest.append({"seed": seed, "method": "bui_controlled", **status})
            else:
                command = [
                    bui_python, "scripts/run_official_bui_osgpr_era5.py",
                    "--protocol-npz", str(protocol), "--theta-json", str(cumulative_calibration / "result.json"),
                    "--output", str(root / "result.json"), "--blockwise-output", str(root / "blocks.csv"),
                    "--predictions-output", str(root / "predictions.npz"), "--mt", "32", "--ms", "32",
                    "--task1-posterior-warm-start", "--delayed-observations", "--seed", str(seed),
                    "--device", "cpu", "--dtype", "float64",
                ]
                if args.max_blocks > 0:
                    command.extend(["--max-stream-blocks", str(args.max_blocks)])
                manifest.append({"seed": seed, "method": "bui_controlled", **run_command(command, output=root, dry_run=args.dry_run)})

        if "bui_adaptive" in args.methods:
            root = seed_root / "bui_osgpr_adaptive"
            if complete(root, expected_weeks):
                manifest.append({"seed": seed, "method": "bui_adaptive", "status": "skipped_complete"})
            else:
                command = [
                    bui_python, "scripts/run_official_bui_osgpr_era5.py",
                    "--protocol-npz", str(protocol), "--output", str(root / "result.json"),
                    "--blockwise-output", str(root / "blocks.csv"), "--predictions-output", str(root / "predictions.npz"),
                    "--mt", "32", "--ms", "32", "--adaptive", "--adaptive-calibration-steps", "25",
                    "--adaptive-online-steps", "5", "--adaptive-learning-rate", "0.01",
                    "--delayed-observations", "--seed", str(seed), "--device", "cpu", "--dtype", "float64",
                ]
                if args.max_blocks > 0:
                    command.extend(["--max-stream-blocks", str(args.max_blocks)])
                manifest.append({"seed": seed, "method": "bui_adaptive", **run_command(command, output=root, dry_run=args.dry_run)})

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "records": len(manifest), "manifest": str(output_root / "suite_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
