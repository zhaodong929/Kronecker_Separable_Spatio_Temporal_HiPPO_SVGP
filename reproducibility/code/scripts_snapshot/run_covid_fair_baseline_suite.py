#!/usr/bin/env python3
"""Run the same-protocol COVID baseline suite for independent split seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def completed(path: Path) -> bool:
    return (path / "result.json").is_file() and (path / "predictions.npz").is_file()


def run(command: list[str], cwd: Path) -> dict[str, object]:
    subprocess.run(command, cwd=cwd, check=True)
    return {"command": command, "status": "complete"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=Path("data/epidemiology/protocol/covid_history_aware"))
    parser.add_argument("--output-root", type=Path, default=Path("results/diagnostics/covid_delayed_baseline_comparison"))
    parser.add_argument("--routeb-root", type=Path, default=Path("results/diagnostics/covid_routeb_upgrade/phase9_confirmation_seeds5_9"))
    parser.add_argument("--bui-python", type=Path, default=Path(".venv_osgpr/bin/python"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-iterations", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Preserve venv entrypoints; resolving their symlinks drops the venv site-packages.
    python = str(ROOT / ".venv/bin/python")
    bui_python = str(ROOT / args.bui_python)
    manifest: list[dict[str, object]] = []
    for seed in args.seeds:
        protocol = (ROOT / args.protocol_root / f"seed{seed}/protocol.npz").resolve()
        protocol_json = protocol.with_suffix(".json")
        seed_root = (ROOT / args.output_root / f"seed{seed}").resolve()
        theta = (ROOT / args.routeb_root / f"seed{seed}/calibration/result.json").resolve()
        deterministic_root = seed_root / "deterministic"
        deterministic_command = [
            python,
            "scripts/run_covid_delayed_deterministic_baselines.py",
            "--protocol-npz", str(protocol),
            "--protocol-json", str(protocol_json),
            "--output-root", str(deterministic_root),
            "--seed", str(seed),
        ]
        if not args.dry_run and not completed(deterministic_root / "persistence"):
            manifest.append({"seed": seed, "method": "deterministic", **run(deterministic_command, ROOT)})
        else:
            manifest.append({"seed": seed, "method": "deterministic", "command": deterministic_command, "status": "skipped"})

        ohsvgp_root = seed_root / "ohsvgp_rbf"
        ohsvgp_command = [
            python,
            "scripts/run_covid_ohsvgp_own_theta.py",
            "--protocol-npz", str(protocol),
            "--protocol-json", str(protocol_json),
            "--output-dir", str(ohsvgp_root),
            "--kernel", "rbf",
            "--inducing-size", "32",
            "--rff-sample-size", "64",
            "--calibration-iterations", str(args.calibration_iterations),
            "--validation-every", "10",
            "--early-stopping-patience-validations", "10",
            "--learning-rate", "0.001",
            "--update-steps", "1",
            "--delayed-observations",
            "--seed", str(seed),
            "--device", args.device,
            "--dtype", "float64",
        ]
        if not args.dry_run and not completed(ohsvgp_root):
            manifest.append({"seed": seed, "method": "ohsvgp_rbf", **run(ohsvgp_command, ROOT)})
        else:
            manifest.append({"seed": seed, "method": "ohsvgp_rbf", "command": ohsvgp_command, "status": "skipped"})

        ordinary_calibration = seed_root / "routeb_ordinary" / "calibration"
        ordinary_online = seed_root / "routeb_ordinary" / "online"
        ordinary_calibration_command = [
            python,
            "scripts/run_iclr_era5_routeb_batch.py",
            "--protocol-npz", str(protocol),
            "--protocol-json", str(protocol_json),
            "--output-dir", str(ordinary_calibration),
            "--data-part", "calibration",
            "--target-mode", "joint_xlag",
            "--representation", "inducing_points",
            "--mt", "32", "--ms", "32",
            "--iterations", str(args.calibration_iterations),
            "--learning-rate", "0.02",
            "--validation-every", "10",
            "--early-stopping-patience-validations", "10",
            "--split-seed", str(seed),
            "--device", args.device,
            "--dtype", "float64",
            "--evaluation-backend", "torch",
            "--objective-optimization-version", "E3",
            "--include-conditional-residual-variance",
        ]
        ordinary_online_command = [
            python,
            "scripts/run_iclr_era5_routeb_strict_online.py",
            "--protocol-npz", str(protocol),
            "--protocol-json", str(protocol_json),
            "--theta-json", str(ordinary_calibration / "result.json"),
            "--output", str(ordinary_online / "result.json"),
            "--blockwise-output", str(ordinary_online / "blocks.csv"),
            "--predictions-output", str(ordinary_online / "predictions.npz"),
            "--representation", "inducing_points",
            "--mt", "32", "--ms", "32",
            "--seed", str(seed),
            "--delayed-observations", "--task1-posterior-init",
            "--solver-backend", "torch", "--device", args.device, "--dtype", "float64",
            "--include-conditional-residual-variance",
        ]
        if not args.dry_run and not completed(ordinary_online):
            if not (ordinary_calibration / "result.json").is_file():
                manifest.append({"seed": seed, "method": "routeb_ordinary_calibration", **run(ordinary_calibration_command, ROOT)})
            if not ordinary_online.parent.exists():
                ordinary_online.mkdir(parents=True)
            manifest.append({"seed": seed, "method": "routeb_ordinary_online", **run(ordinary_online_command, ROOT)})
        else:
            manifest.append({"seed": seed, "method": "routeb_ordinary", "command": ordinary_online_command, "status": "skipped"})

        bui_root = seed_root / "bui_osgpr_controlled"
        bui_command = [
            bui_python,
            "scripts/run_official_bui_osgpr_era5.py",
            "--protocol-npz", str(protocol),
            "--theta-json", str(theta),
            "--output", str(bui_root / "result.json"),
            "--blockwise-output", str(bui_root / "blocks.csv"),
            "--predictions-output", str(bui_root / "predictions.npz"),
            "--mt", "32", "--ms", "32",
            "--task1-posterior-warm-start", "--delayed-observations",
            "--seed", str(seed), "--device", "cpu", "--dtype", "float64",
        ]
        if not args.dry_run and not completed(bui_root):
            manifest.append({"seed": seed, "method": "bui_osgpr_controlled", **run(bui_command, ROOT)})
        else:
            manifest.append({"seed": seed, "method": "bui_osgpr_controlled", "command": bui_command, "status": "skipped"})

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "jobs": len(manifest), "manifest": str((args.output_root / "suite_manifest.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
