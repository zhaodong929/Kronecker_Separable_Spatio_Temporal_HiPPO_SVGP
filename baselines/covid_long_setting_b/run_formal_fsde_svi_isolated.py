#!/usr/bin/env python3
"""Run the frozen official-core FSDE-SVI Setting B protocol on formal seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("baselines/covid_long_setting_b/results/formal_selected_fsde_svi_isolated"),
    )
    parser.add_argument(
        "--fsde-python",
        type=Path,
        default=Path("baselines/.venvs/factorial_sde_fsde39/bin/python"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--temporal-inducing", type=int, default=4)
    parser.add_argument("--latent-rank", type=int, default=2)
    parser.add_argument("--task1-iterations", type=int, default=50)
    parser.add_argument("--online-inference-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def archive_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as values:
            return all(
                np.asarray(values[key]).shape == (143, 10)
                and np.isfinite(np.asarray(values[key])).all()
                for key in ("y_true", "pred_mean", "pred_var")
            ) and (np.asarray(values["pred_var"]) >= 0.0).all()
    except (KeyError, OSError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    protocol_root, output_root, interpreter = map(
        absolute, (args.protocol_root, args.output_root, args.fsde_python)
    )
    worker = ROOT / "baselines/covid_long_setting_b/run_factorial_fsde_svi_isolated_stream.py"
    manifest = []
    for seed in args.seeds:
        protocol = protocol_root / f"seed{seed}" / "protocol.npz"
        output = output_root / f"seed{seed}" / "fsde_svi"
        if not protocol.is_file():
            raise FileNotFoundError(f"Missing formal protocol: {protocol}")
        command = [
            str(interpreter), str(worker), "--protocol-npz", str(protocol),
            "--output-dir", str(output), "--seed", str(seed),
            "--temporal-inducing", str(args.temporal_inducing),
            "--latent-rank", str(args.latent_rank),
            "--task1-iterations", str(args.task1_iterations),
            "--online-inference-steps", str(args.online_inference_steps),
            "--batch-size", str(args.batch_size),
        ]
        if archive_complete(output / "predictions.npz"):
            manifest.append({"seed": seed, "status": "reused_complete", "command": command})
            continue
        if args.dry_run:
            manifest.append({"seed": seed, "status": "dry_run", "command": command})
            continue
        output.mkdir(parents=True, exist_ok=True)
        with (output / "run.log").open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        manifest.append(
            {
                "seed": seed,
                "status": "complete" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "command": command,
            }
        )
        if completed.returncode:
            break
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "dry_run" if args.dry_run else "complete",
        "capacity": {"temporal_inducing": args.temporal_inducing, "latent_rank": args.latent_rank},
        "online_inference_steps": args.online_inference_steps,
        "seeds": args.seeds,
        "manifest": manifest,
    }
    (output_root / "formal_run_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
