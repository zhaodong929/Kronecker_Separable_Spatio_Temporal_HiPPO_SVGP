#!/usr/bin/env python3
"""Assess the two clean-process OVC memory audits before formal execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def assess(record: dict[str, Any]) -> dict[str, Any]:
    rows = record.get("rows", [])
    state_matches = bool(rows) and all(
        row.get("fantasy_state_observation_count") == row.get("expected_arrived_observation_count")
        for row in rows
    )
    model_count_ok = bool(rows) and all(row.get("reachable_model_instances") == 1 for row in rows)
    expected = np.asarray([row.get("expected_arrived_observation_count", np.nan) for row in rows], dtype=float)
    storage = np.asarray([row.get("reachable_unique_tensor_storage_bytes", np.nan) for row in rows], dtype=float)
    correlation = float(np.corrcoef(expected, storage)[0, 1]) if len(rows) >= 3 and np.std(storage) > 0 else float("nan")
    return {
        "replicate_id": record.get("replicate_id"),
        "status": "passed" if state_matches and model_count_ok else "failed",
        "fantasy_state_matches_expected": state_matches,
        "single_reachable_model": model_count_ok,
        "state_storage_correlation": correlation,
        "rows": len(rows),
    }


def main() -> None:
    args = parse_args()
    audit_root = absolute(args.audit_root)
    output = absolute(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace OVC memory assessment: {output}")
    assessments = []
    for replicate in (1, 2):
        path = audit_root / f"replicate_{replicate}" / "memory_audit.json"
        if not path.is_file():
            assessments.append({"replicate_id": replicate, "status": "missing", "path": str(path)})
            continue
        assessments.append(assess(json.loads(path.read_text(encoding="utf-8"))))
    passed = len(assessments) == 2 and all(item["status"] == "passed" for item in assessments)
    payload = {
        "status": "passed" if passed else "validation_pending_or_failed",
        "purpose": "OVC can continue only if state growth follows the official exact-fantasy observation count with one reachable model per clean process.",
        "replicates": assessments,
        "interpretation": "This is a retention audit, not a claim that larger grids are better. RSS is retained as evidence but not used as an arbitrary pass/fail capacity threshold.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
