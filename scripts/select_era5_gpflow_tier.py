#!/usr/bin/env python3
"""Select the runnable GPflow inducing tier from seed-0 preflight artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from build_era5_a100_manifests import build_manifests, load_spec
except ImportError:
    from scripts.build_era5_a100_manifests import build_manifests, load_spec


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _nested(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def evaluate_preflight(
    manifest: Path,
    *,
    tier: str,
    expected_candidates: int,
    full_iterations: int,
    preflight_iterations: int,
    max_peak_mib: float,
    max_estimated_seconds: float,
) -> tuple[bool, list[dict[str, Any]]]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        row
        for row in rows
        if row.get("kind") == "gpflow_feasibility_preflight"
        and str(row.get("selection", {}).get("tier")) == tier
    ]
    if len(rows) != expected_candidates:
        raise ValueError(
            f"Expected {expected_candidates} GPflow preflight records for tier {tier}, found {len(rows)}"
        )

    decisions: list[dict[str, Any]] = []
    for row in rows:
        output_dir = Path(str(row["output_dir"]))
        status_path = Path(str(row.get("status_path", output_dir / "status.json")))
        result_path = next(
            (Path(str(path)) for path in row.get("expected", []) if str(path).endswith("result.json")),
            output_dir / "result.json",
        )
        status = _read_json(status_path)
        result = _read_json(result_path)
        status_ok = bool(
            status
            and (
                status.get("status") == "complete"
                or status.get("classification") == "complete"
            )
        )
        finite = bool(result is not None and _finite(result))
        peak_mib = None if result is None else _nested(result, "resources.peak_cuda_allocated_mib")
        wall_seconds = None if status is None else status.get("wall_seconds")
        if wall_seconds is None and result is not None:
            wall_seconds = _nested(result, "timing.process_total_seconds")
        try:
            peak_mib = float(peak_mib)
        except (TypeError, ValueError):
            peak_mib = None
        try:
            wall_seconds = float(wall_seconds)
        except (TypeError, ValueError):
            wall_seconds = None
        estimated_seconds = (
            None
            if wall_seconds is None
            else wall_seconds * float(full_iterations) / float(preflight_iterations)
        )
        accepted = bool(
            status_ok
            and finite
            and peak_mib is not None
            and peak_mib <= max_peak_mib
            and estimated_seconds is not None
            and estimated_seconds <= max_estimated_seconds
        )
        decisions.append(
            {
                "tier": tier,
                "method": row.get("method"),
                "status_path": str(status_path),
                "result_path": str(result_path),
                "status_ok": status_ok,
                "finite": finite,
                "peak_cuda_allocated_mib": peak_mib,
                "estimated_100_step_seconds": estimated_seconds,
                "accepted": accepted,
            }
        )
    return all(row["accepted"] for row in decisions), decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--max-peak-gib", type=float, default=72.0)
    parser.add_argument("--max-full-hours", type=float, default=6.0)
    parser.add_argument("--hardware-class", default="NVIDIA A100")
    args = parser.parse_args()

    spec = load_spec(args.config.resolve())
    gpflow = spec["short_batch"]["gpflow_feasibility"]
    preflight_iterations = int(gpflow["preflight_iterations"])
    base_config = json.loads(
        (args.config.parent / str(spec.get("base_config", "benchmark.json"))).read_text(
            encoding="utf-8"
        )
    )
    full_iterations = int(base_config["gpflow_svgp"]["iterations"])
    manifest = args.manifest_dir / "shared_batch_short.jsonl"
    tier_decisions = []
    selected_tier = None
    for tier in (str(item) for item in gpflow["preflight_tiers"]):
        accepted, decisions = evaluate_preflight(
            manifest,
            tier=tier,
            expected_candidates=len(gpflow["tiers"][tier]),
            full_iterations=full_iterations,
            preflight_iterations=preflight_iterations,
            max_peak_mib=args.max_peak_gib * 1024.0,
            max_estimated_seconds=args.max_full_hours * 3600.0,
        )
        tier_decisions.append({"tier": tier, "accepted": accepted, "records": decisions})
        if accepted and selected_tier is None:
            selected_tier = tier
    if selected_tier is None:
        raise SystemExit("No GPflow inducing tier passed the finite-value and resource preflight")
    outputs = build_manifests(
        spec_path=args.config.resolve(),
        benchmark_root=args.benchmark_root.resolve(),
        output_dir=args.manifest_dir.resolve(),
        gpflow_tier=selected_tier,
        hardware_class=args.hardware_class,
    )
    payload = {
        "schema_version": 2,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "selected_tier": selected_tier,
        "selection_basis": "largest seed-0 tier passing finite-value and resource checks; no test metric was inspected",
        "thresholds": {
            "max_peak_gib": args.max_peak_gib,
            "max_estimated_full_hours": args.max_full_hours,
        },
        "preflight_tiers": tier_decisions,
        "manifests": {key: str(value) for key, value in outputs.items()},
    }
    output = args.benchmark_root / "gpflow_feasibility" / "selected_tier.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
