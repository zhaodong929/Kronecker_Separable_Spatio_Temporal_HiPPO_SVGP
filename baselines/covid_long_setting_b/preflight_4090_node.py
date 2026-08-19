#!/usr/bin/env python3
"""Record and enforce the hardware gate for repaired COVID baseline runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("baselines/covid_long_setting_b/results/convergence_repair_v1/cloud_4090_preflight.json"),
    )
    parser.add_argument("--minimum-ram-gib", type=float, default=120.0)
    parser.add_argument("--minimum-gpu-memory-mib", type=float, default=23000.0)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def nvidia_csv(query: str) -> list[list[str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    return [[field.strip() for field in row.split(",")] for row in completed.stdout.splitlines() if row.strip()]


def active_compute_processes() -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    return [row.strip() for row in completed.stdout.splitlines() if row.strip() and row.strip() != "No running processes found"]


def memory_gib() -> float:
    try:
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1024**3
    except (AttributeError, ValueError, OSError):
        return 0.0


def main() -> None:
    args = parse_args()
    devices: list[dict[str, Any]] = []
    for row in nvidia_csv("name,memory.total,driver_version"):
        if len(row) != 3:
            continue
        try:
            devices.append({"name": row[0], "memory_total_mib": float(row[1]), "driver_version": row[2]})
        except ValueError:
            continue
    compute_rows = nvidia_csv("compute_mode")
    ram_gib = memory_gib()
    valid_gpu = any("RTX 4090" in item["name"] and item["memory_total_mib"] >= args.minimum_gpu_memory_mib for item in devices)
    active_processes = active_compute_processes()
    passed = valid_gpu and ram_gib >= args.minimum_ram_gib and not active_processes
    payload = {
        "status": "passed" if passed else "failed",
        "requirements": {
            "gpu": "RTX 4090",
            "minimum_gpu_memory_mib": args.minimum_gpu_memory_mib,
            "minimum_host_ram_gib": args.minimum_ram_gib,
            "exclusive_execution_required": True,
            "high_rss_methods_must_be_serial": ["ovc_svgp", "st_svgp", "fsde_svi"],
        },
        "observed": {
            "host_ram_gib": ram_gib,
            "gpus": devices,
            "gpu_compute_modes": compute_rows,
            "active_compute_processes": active_processes,
        },
    }
    output = absolute(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
