#!/usr/bin/env python3
"""Execute the unmodified official OHSVGP 1D and multidimensional gates.

The Setting B adapter is not eligible until both commands pass in a dedicated
environment.  The default is non-executing and records the exact pinned
commands required on the 4090 node; ``--execute`` writes independent logs and
does not modify the upstream source tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = ROOT / "baselines/external/harrisonzhu508_HIPPOSVGP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baselines/covid_long_setting_b/reproduction/convergence_repair_v1/ohsvgp"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--inducing-size", type=int, default=16)
    parser.add_argument("--rff-sample-size", type=int, default=128)
    parser.add_argument("--multidim-iterations", type=int, default=100)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=OFFICIAL_ROOT, text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def main() -> None:
    args = parse_args()
    output = absolute(args.output_dir)
    python = absolute(args.python)
    if not OFFICIAL_ROOT.is_dir():
        raise FileNotFoundError(f"Pinned OHSVGP source is missing: {OFFICIAL_ROOT}")
    if not python.is_file():
        raise FileNotFoundError(f"Requested OHSVGP environment is missing: {python}")
    output.mkdir(parents=True, exist_ok=True)
    commands = {
        "official_1d_online": [
            str(python),
            "scripts/onedim/solar/test_hsgp_online_fixedkernel.py",
            "--data", "solar",
            "--inducing_size", str(args.inducing_size),
            "--rff_sample_size", str(args.rff_sample_size),
            "--rff_sample_size_qb", str(args.rff_sample_size),
            "--rff_sample_size_pred", str(args.rff_sample_size),
        ],
        "official_multidimensional": [
            str(python),
            "scripts/multidim/main_hsgp_online_toy.py",
            "--data", "moon",
            "--inducing_size", str(args.inducing_size),
            "--rff_sample_size", str(args.rff_sample_size),
            "--num_iters", str(args.multidim_iterations),
        ],
    }
    checks: dict[str, Any] = {}
    for name, command in commands.items():
        check: dict[str, Any] = {"command": command, "cwd": str(OFFICIAL_ROOT)}
        if args.execute:
            log = output / f"{name}.log"
            with log.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(command, cwd=OFFICIAL_ROOT, stdout=handle, stderr=subprocess.STDOUT)
            check.update({"status": "passed" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "log": str(log)})
        else:
            check["status"] = "planned_not_executed"
        checks[name] = check
    overall = "passed" if args.execute and all(item["status"] == "passed" for item in checks.values()) else "pending"
    payload = {
        "status": overall,
        "purpose": "Unmodified official OHSVGP gates before multidimensional COVID Setting B adaptation",
        "source_repository": "https://github.com/harrisonzhu508/HIPPOSVGP",
        "source_commit": source_commit(),
        "environment_python": str(python),
        "checks": checks,
        "adapter_boundary": "The subsequent CDC Setting B RFF choices are adapter-specific and must not be described as the universal official OHSVGP configuration.",
    }
    (output / "gate_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": overall, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
