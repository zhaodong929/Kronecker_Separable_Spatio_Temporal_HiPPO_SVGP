#!/usr/bin/env python3
"""Capture immutable package/environment evidence for the cloud formal run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def parse_assignment(value: str) -> tuple[str, Path]:
    name, separator, executable = value.partition("=")
    if not separator or not name or not executable:
        raise argparse.ArgumentTypeError("environment must have the form label=/absolute/path/to/python")
    return name, Path(executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        action="append",
        type=parse_assignment,
        required=True,
        help="Repeat for each isolated baseline environment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("baselines/covid_long_setting_b/results/convergence_repair_v1/cloud_environment_locks.json"),
    )
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run(executable: Path, *arguments: str) -> str:
    completed = subprocess.run([str(executable), *arguments], text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def main() -> None:
    args = parse_args()
    output = absolute(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace environment lock: {output}")
    environments = {}
    for name, executable in args.environment:
        executable = absolute(executable)
        if not executable.is_file():
            raise FileNotFoundError(f"Missing {name} Python executable: {executable}")
        freeze = run(executable, "-m", "pip", "freeze")
        environments[name] = {
            "python": str(executable),
            "python_version": run(executable, "--version"),
            "pip_freeze": freeze.splitlines(),
            "pip_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "complete", "environments": environments}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "environments": sorted(environments), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
