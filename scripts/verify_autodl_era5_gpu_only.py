#!/usr/bin/env python3
"""Verify the complete RTX 4090 GPU-only ERA5 comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPLETE_STATUSES = {"complete"}
GPU_MANIFEST_POLICIES = {
    "shared_batch_short": {
        "device_classes": {
            "modern_gpu",
            "a100_official_full",
            "a100_gpflow",
            "a100_routeb",
        },
        "exclude_full_stvgp": True,
    },
    "official_long_preflight": {
        "device_classes": {"a100_official_preflight"},
        "exclude_full_stvgp": True,
    },
    "official_long_full": {
        "device_classes": {"a100_official_full"},
        "exclude_full_stvgp": True,
    },
    "online_short": {
        "device_classes": {
            "modern_gpu",
            "modern_gpu_legacy_api",
            "a100_routeb_online",
        },
        "exclude_full_stvgp": False,
    },
    "online_long": {
        "device_classes": {
            "modern_gpu",
            "modern_gpu_legacy_api",
            "a100_routeb_online",
        },
        "exclude_full_stvgp": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def record_status(record: dict[str, Any]) -> str:
    status_path = Path(str(record["output_dir"])) / "status.json"
    if not status_path.is_file():
        return "missing"
    try:
        return str(json.loads(status_path.read_text(encoding="utf-8")).get("status"))
    except (json.JSONDecodeError, OSError, TypeError):
        return "invalid"


def expected_artifacts_exist(record: dict[str, Any]) -> bool:
    try:
        return all(
            Path(str(path)).is_file() and Path(str(path)).stat().st_size > 0
            for path in record.get("expected", [])
        )
    except OSError:
        return False


def exclusion_reason(record: dict[str, Any], policy: dict[str, Any]) -> str | None:
    method = str(record.get("method", ""))
    device_class = str(record.get("device_class", ""))
    if policy["exclude_full_stvgp"] and method.endswith("st_vgp_full"):
        return "documented_rtx4090_full_stvgp_oom"
    if device_class == "a100_markovflow":
        return "tf24_cusolverDnCreate_failed_on_rtx4090"
    if device_class == "a100_gpflow_preflight":
        return "capacity_preflight_not_comparison_row"
    if device_class not in policy["device_classes"]:
        return "not_selected_by_gpu_only_policy"
    return None


def load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    required: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    problems: list[str] = []

    for manifest_name, policy in GPU_MANIFEST_POLICIES.items():
        manifest = args.benchmark_root / "manifests" / f"{manifest_name}.jsonl"
        if not manifest.is_file():
            problems.append(f"missing_manifest:{manifest_name}")
            continue
        for index, record in enumerate(load_records(manifest)):
            entry = {
                "manifest": manifest_name,
                "index": index,
                "method": str(record.get("method", "")),
                "seed": record.get("seed"),
                "device_class": str(record.get("device_class", "")),
            }
            reason = exclusion_reason(record, policy)
            if reason is not None:
                exclusions.append({**entry, "reason": reason})
                continue

            status = record_status(record)
            artifacts_complete = expected_artifacts_exist(record)
            required.append({**entry, "status": status, "artifacts_complete": artifacts_complete})
            if status not in COMPLETE_STATUSES:
                problems.append(f"{manifest_name}[{index}] status:{status}")
            if not artifacts_complete:
                problems.append(f"{manifest_name}[{index}] missing_expected_artifacts")

    payload = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": str(args.benchmark_root),
        "verification_status": "VERIFIED_GPU_ONLY_RTX4090" if not problems else "FAILED",
        "scope": "complete GPU-only RTX 4090 ERA5 matrix",
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
            "included_manifests": list(GPU_MANIFEST_POLICIES),
            "markovflow_gpu_status": "excluded_after_actual_cusolverDnCreate_failure",
            "full_stvgp_gpu_status": "excluded_after_documented_rtx4090_oom",
            "cpu_rows_included": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
