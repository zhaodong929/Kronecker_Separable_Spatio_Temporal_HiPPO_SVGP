#!/usr/bin/env python3
"""Hash pre-repair formal prediction archives before cloud formalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.formalization import snapshot_archives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("baselines/covid_long_setting_b/reproduction/convergence_repair_v1/frozen_pre_repair_archives.json"),
    )
    parser.add_argument("--verify", action="store_true", help="Verify an existing manifest rather than replacing it.")
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def archive_paths() -> list[Path]:
    roots = (
        ROOT / "results/diagnostics/covid_long_stream_2020_2024_mandatory",
        ROOT / "baselines/covid_long_setting_b/results",
    )
    formal_seed_dirs = {f"seed{seed}" for seed in (5, 6, 7, 8, 9)}
    return [
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("predictions.npz")
        if formal_seed_dirs.intersection(path.parts)
    ]


def main() -> None:
    args = parse_args()
    output = absolute(args.output)
    if args.verify:
        from baselines.covid_long_setting_b.formalization import verify_snapshot

        payload = json.loads(output.read_text(encoding="utf-8"))
        mismatches = verify_snapshot(payload["archives"])
        print(json.dumps({"status": "passed" if not mismatches else "failed", "mismatches": mismatches}, indent=2))
        if mismatches:
            raise SystemExit(1)
        return
    if output.exists():
        raise FileExistsError(f"Refusing to replace frozen-archive manifest: {output}")
    records = snapshot_archives(archive_paths())
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "purpose": "pre-repair archive immutability record; these archives are preliminary evidence unless catalog status permits a final row",
        "archives": records,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "archives": len(records), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
