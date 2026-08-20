#!/usr/bin/env python3
"""Prepare and verify a resumable AutoDL ERA5 capacity-ladder recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable


MANIFEST_NAMES = (
    "shared_batch_short.jsonl",
    "official_long_preflight.jsonl",
    "official_long_full.jsonl",
    "online_short.jsonl",
    "online_long.jsonl",
    "efficiency.jsonl",
)
KNOWN_RTX4090_OOM_METHODS = frozenset(
    {"official_st_vgp_full", "official_preflight_st_vgp_full"}
)
COMPLETE_STATUSES = frozenset({"complete", "skipped"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(payload)
    return rows


def _write_records(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _path(value: Any, repo: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else repo / path


def _expected(record: dict[str, Any], repo: Path) -> list[Path]:
    value = record.get("complete", record.get("expected", []))
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return []
    return [_path(item, repo) for item in value]


def _status_path(record: dict[str, Any], repo: Path) -> Path:
    output = _path(record.get("output_dir", repo), repo)
    return _path(record.get("status_path", output / "status.json"), repo)


def _status(record: dict[str, Any], repo: Path) -> str | None:
    try:
        payload = json.loads(_status_path(record, repo).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    value = payload.get("status") if isinstance(payload, dict) else None
    return str(value).lower() if value is not None else None


def _artifacts_complete(record: dict[str, Any], repo: Path) -> bool:
    expected = _expected(record, repo)
    return bool(expected) and all(path.is_file() and path.stat().st_size > 0 for path in expected)


def _tier(record: dict[str, Any]) -> str | None:
    selection = record.get("selection")
    if not isinstance(selection, dict) or selection.get("tier") is None:
        return None
    return str(selection["tier"])


def _is_nonselected_gpflow_preflight(record: dict[str, Any], selected_tier: str) -> bool:
    return (
        str(record.get("kind", "")) == "gpflow_feasibility_preflight"
        and _tier(record) != selected_tier
    )


def _is_nonselected_gpflow_batch(record: dict[str, Any], selected_tier: str) -> bool:
    return str(record.get("method", "")).startswith("gpflow_feasibility_") and _tier(record) != selected_tier


def _is_known_oom_exclusion(record: dict[str, Any]) -> bool:
    return str(record.get("method", "")) in KNOWN_RTX4090_OOM_METHODS


def rewrite(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    destination = args.destination.resolve()
    repo = args.repo.resolve()
    missing = [name for name in MANIFEST_NAMES if not (source / name).is_file()]
    if missing:
        raise ValueError("missing canonical manifests: " + ", ".join(missing))
    for name in MANIFEST_NAMES:
        rows = _records(source / name)
        for row in rows:
            row["cwd"] = str(repo)
            row["recovery_worktree"] = str(repo)
            argv = row.get("argv")
            if (
                isinstance(argv, list)
                and "scripts/run_official_markovflow_stsvgp_era5.py" in argv
                and "--temporal-jitter" not in argv
            ):
                row["argv"] = [
                    *argv,
                    "--temporal-jitter",
                    str(args.markovflow_temporal_jitter),
                ]
                row["recovery_markovflow_temporal_jitter"] = float(
                    args.markovflow_temporal_jitter
                )
        _write_records(destination / name, rows)
    metadata = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "source_manifest_dir": str(source),
        "recovery_manifest_dir": str(destination),
        "recovery_worktree": str(repo),
        "manifests": list(MANIFEST_NAMES),
    }
    (destination / "recovery_manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


def select(args: argparse.Namespace) -> int:
    rows = _records(args.manifest.resolve())
    repo = args.repo.resolve()
    selected_tier = str(args.selected_tier)
    for index, record in enumerate(rows):
        if _is_known_oom_exclusion(record):
            continue
        if _is_nonselected_gpflow_preflight(record, selected_tier):
            continue
        if _is_nonselected_gpflow_batch(record, selected_tier):
            continue
        artifacts_complete = _artifacts_complete(record, repo)
        status = _status(record, repo)
        if artifacts_complete and status in COMPLETE_STATUSES:
            continue
        action = "force" if artifacts_complete else "normal"
        print(f"{index}\t{action}\t{record.get('method', '')}")
    return 0


def verify(args: argparse.Namespace) -> int:
    manifest_dir = args.manifest_dir.resolve()
    benchmark = args.benchmark.resolve()
    selected_tier = str(args.selected_tier)
    required: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    problems: list[str] = []
    manifest_records = 0

    for name in MANIFEST_NAMES:
        path = manifest_dir / name
        if not path.is_file():
            problems.append(f"missing_manifest:{path}")
            continue
        for index, record in enumerate(_records(path)):
            manifest_records += 1
            method = str(record.get("method", ""))
            entry = {"manifest": name, "index": index, "method": method, "seed": record.get("seed")}
            if _is_known_oom_exclusion(record):
                entry["reason"] = "known_rtx4090_full_st_vgp_oom_exclusion"
                allowed.append(entry)
                continue
            if _is_nonselected_gpflow_preflight(record, selected_tier):
                entry["reason"] = "nonselected_gpflow_feasibility_preflight"
                allowed.append(entry)
                continue
            if _is_nonselected_gpflow_batch(record, selected_tier):
                entry["reason"] = "nonselected_gpflow_capacity_tier"
                allowed.append(entry)
                continue
            artifacts_complete = _artifacts_complete(record, benchmark)
            status = _status(record, benchmark)
            entry.update({"artifacts_complete": artifacts_complete, "status": status})
            required.append(entry)
            if not artifacts_complete:
                problems.append(f"{name}[{index}] missing_expected_artifacts")
            if status not in COMPLETE_STATUSES:
                problems.append(f"{name}[{index}] status:{status}")

    audit = benchmark / "audit.json"
    if not audit.is_file() or audit.stat().st_size == 0:
        problems.append("missing_audit")
    report = benchmark / "report"
    if not report.is_dir() or not any(path.is_file() and path.stat().st_size > 0 for path in report.rglob("*")):
        problems.append("missing_report_artifacts")
    if args.require_ncu_audit:
        ncu_audit = benchmark / "ncu_validation"
        if not ncu_audit.is_dir() or not any(path.is_file() and path.stat().st_size > 0 for path in ncu_audit.rglob("*")):
            problems.append("missing_ncu_permission_audit")

    payload = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "verification_status": "VERIFIED_PARTIAL_RTX4090" if not problems else "FAILED",
        "benchmark_root": str(benchmark),
        "selected_gpflow_tier": selected_tier,
        "manifest_records": manifest_records,
        "required_record_count": len(required),
        "required_complete_count": sum(
            row["artifacts_complete"] and row["status"] in COMPLETE_STATUSES for row in required
        ),
        "allowed_exclusions": allowed,
        "required_records": required,
        "problems": problems,
        "policy": {
            "known_rtx4090_oom_methods": sorted(KNOWN_RTX4090_OOM_METHODS),
            "nonselected_gpflow_preflights_allowed": True,
            "ncu_hardware_flops_required": False,
            "audit_zero_failures_required": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("verification_status", "required_record_count", "required_complete_count", "problems")}, sort_keys=True))
    return 0 if not problems else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    rewrite_parser = subparsers.add_parser("rewrite", help="rewrite manifests for the repaired worktree")
    rewrite_parser.add_argument("--source", type=Path, required=True)
    rewrite_parser.add_argument("--destination", type=Path, required=True)
    rewrite_parser.add_argument("--repo", type=Path, required=True)
    rewrite_parser.add_argument("--markovflow-temporal-jitter", type=float, default=1e-6)
    rewrite_parser.set_defaults(handler=rewrite)

    select_parser = subparsers.add_parser("select", help="print incomplete non-excluded records")
    select_parser.add_argument("--manifest", type=Path, required=True)
    select_parser.add_argument("--repo", type=Path, required=True)
    select_parser.add_argument("--selected-tier", required=True)
    select_parser.set_defaults(handler=select)

    verify_parser = subparsers.add_parser("verify", help="verify a publishable partial RTX 4090 bundle")
    verify_parser.add_argument("--manifest-dir", type=Path, required=True)
    verify_parser.add_argument("--benchmark", type=Path, required=True)
    verify_parser.add_argument("--selected-tier", required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    verify_parser.add_argument("--require-ncu-audit", action="store_true")
    verify_parser.set_defaults(handler=verify)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"recovery manifest error: {exc}", file=sys.stderr)
        raise SystemExit(2)
