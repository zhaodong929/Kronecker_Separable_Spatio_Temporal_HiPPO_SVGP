#!/usr/bin/env python3
"""Run the ICLR 2027 ERA5 mechanism matrix serially with resource guards."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_iclr_era5_transfer_mechanism.py"


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jobs() -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for mt in (16, 32, 64, 128):
        for seed in range(5):
            matrix.append(
                {
                    "mode": "structured_changing",
                    "mt": mt,
                    "ms": 128,
                    "seed": seed,
                    "batch_reference": True,
                }
            )
    for mode in ("structured_fixed", "mean_field_changing", "zero_cross_changing"):
        for seed in range(5):
            matrix.append(
                {
                    "mode": mode,
                    "mt": 128,
                    "ms": 128,
                    "seed": seed,
                    "batch_reference": False,
                }
            )
    return matrix


def failure_signature(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if "Error" in line or "Exception" in line or "RuntimeError" in line:
            return line[:240]
    return lines[-1][:240] if lines else "unknown failure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--theta-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--spectral-mixture-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-gpu-hours", type=float, default=3.0)
    parser.add_argument("--max-memory-mib", type=float, default=10240.0)
    args = parser.parse_args()

    started = time.monotonic()
    failures: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    manifest_path = args.output_root / "suite_manifest.json"
    suite_jobs = jobs()
    manifest: dict[str, Any] = {
        "status": "running",
        "single_worker": True,
        "max_gpu_hours": args.max_gpu_hours,
        "max_memory_mib": args.max_memory_mib,
        "runner": str(RUNNER),
        "runner_sha256": sha256(RUNNER),
        "jobs_planned": len(suite_jobs),
        "records": records,
    }
    write_json(manifest, manifest_path)

    for index, job in enumerate(suite_jobs):
        elapsed_hours = (time.monotonic() - started) / 3600.0
        if elapsed_hours >= args.max_gpu_hours:
            manifest["status"] = "stopped_time_limit"
            break
        seed = job["seed"]
        name = f'{job["mode"]}_ms{job["ms"]}_mt{job["mt"]}_seed{seed}'
        output_dir = args.output_root / name
        result_path = output_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "complete":
                records.append({"name": name, "status": "reused_complete"})
                write_json(manifest, manifest_path)
                continue
        command = [
            sys.executable,
            str(RUNNER),
            "--protocol-npz",
            str(args.protocol_root / f"seed{seed}" / "protocol.npz"),
            "--protocol-json",
            str(args.protocol_root / f"seed{seed}" / "protocol.json"),
            "--data-root",
            str(args.data_root),
            "--theta-json",
            str(args.theta_root / f"seed{seed}" / "result.json"),
            "--spectral-mixture-json",
            str(args.spectral_mixture_json),
            "--output-dir",
            str(output_dir),
            "--mode",
            job["mode"],
            "--seed",
            str(seed),
            "--ms",
            str(job["ms"]),
            "--mt",
            str(job["mt"]),
            "--device",
            args.device,
        ]
        if job["batch_reference"]:
            command.append("--batch-reference")
        output_dir.mkdir(parents=True, exist_ok=True)
        job_started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (output_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
        record = {
            "index": index,
            "name": name,
            "command": command,
            "wall_seconds": time.monotonic() - job_started,
            "returncode": completed.returncode,
        }
        if completed.returncode != 0 or not result_path.exists():
            signature = failure_signature(completed.stdout)
            failures[signature] += 1
            record.update({"status": "failed", "failure_signature": signature})
            records.append(record)
            write_json(manifest, manifest_path)
            if failures[signature] >= 2:
                manifest["status"] = "stopped_repeated_failure"
                break
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        peak = float(result["resources"]["peak_cuda_allocated_mib"])
        record.update({"status": "complete", "peak_cuda_allocated_mib": peak})
        records.append(record)
        write_json(manifest, manifest_path)
        if peak > args.max_memory_mib:
            manifest["status"] = "stopped_memory_limit"
            break
    else:
        manifest["status"] = "complete"

    manifest["elapsed_gpu_hours_upper_bound"] = (time.monotonic() - started) / 3600.0
    manifest["jobs_complete"] = sum(
        record["status"] in {"complete", "reused_complete"} for record in records
    )
    manifest["jobs_failed"] = sum(record["status"] == "failed" for record in records)
    write_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
