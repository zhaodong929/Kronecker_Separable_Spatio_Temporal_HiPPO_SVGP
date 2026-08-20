#!/usr/bin/env python3
"""Verify the RTX 4090 GPU-only shared-batch ERA5 comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


COMPLETE_STATUSES = {"complete"}
GPU_DEVICE_CLASSES = {
    "modern_gpu",
    "a100_official_full",
    "a100_gpflow",
    "a100_routeb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def record_status(record: dict[str, object]) -> str:
    output_dir = Path(str(record["output_dir"]))
    status_path = output_dir / "status.json"
    if not status_path.is_file():
        return "missing"
    try:
        return str(json.loads(status_path.read_text(encoding="utf-8")).get("status"))
    except (json.JSONDecodeError, OSError, TypeError):
        return "invalid"


def expected_artifacts_exist(record: dict[str, object]) -> bool:
    return all(Path(str(path)).is_file() for path in record.get("expected", []))


def main() -> int:
    args = parse_args()
    manifest = args.benchmark_root / "manifests" / "shared_batch_short.jsonl"
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    problems: list[str] = []

    for index, record in enumerate(records):
        method = str(record.get("method", ""))
        device_class = str(record.get("device_class", ""))
        entry = {
            "index": index,
            "method": method,
            "seed": record.get("seed"),
            "device_class": device_class,
        }
        if method == "official_st_vgp_full":
            exclusions.append({**entry, "reason": "documented_rtx4090_oom"})
            continue
        if device_class == "a100_markovflow":
            exclusions.append(
                {
                    **entry,
                    "reason": "legacy_tf24_cusolverDnCreate_failed_on_rtx4090",
                }
            )
            continue
        if device_class == "a100_gpflow_preflight":
            exclusions.append({**entry, "reason": "capacity_preflight_not_comparison_row"})
            continue
        if device_class not in GPU_DEVICE_CLASSES:
            exclusions.append({**entry, "reason": "cpu_or_preparation_not_run_in_gpu_only_policy"})
            continue

        status = record_status(record)
        artifacts_complete = expected_artifacts_exist(record)
        required.append({**entry, "status": status, "artifacts_complete": artifacts_complete})
        if status not in COMPLETE_STATUSES:
            problems.append(f"shared_batch_short[{index}] status:{status}")
        if not artifacts_complete:
            problems.append(f"shared_batch_short[{index}] missing_expected_artifacts")

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": str(args.benchmark_root),
        "verification_status": "VERIFIED_GPU_ONLY_RTX4090" if not problems else "FAILED",
        "scope": "shared_batch_short GPU-only capacity-ladder comparison",
        "required_record_count": len(required),
        "required_complete_count": sum(
            row["status"] in COMPLETE_STATUSES and row["artifacts_complete"]
            for row in required
        ),
        "required_records": required,
        "exclusions": exclusions,
        "problems": problems,
        "policy": {
            "common_hardware": "NVIDIA GeForce RTX 4090",
            "gpu_only": True,
            "markovflow_gpu_status": "excluded_after_actual_cusolverDnCreate_failure",
            "cpu_xlag_rows_included": False,
            "full_st_vgp_oom_rows_included": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verification_status": payload["verification_status"],
                "required_record_count": payload["required_record_count"],
                "required_complete_count": payload["required_complete_count"],
                "problems": problems,
            },
            sort_keys=True,
        )
    )
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
