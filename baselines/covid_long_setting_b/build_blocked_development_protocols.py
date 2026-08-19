#!/usr/bin/env python3
"""Materialise immutable seed-0 blocked development folds for repaired baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.development import build_development_protocols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-npz",
        type=Path,
        default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed0/protocol.npz"),
    )
    parser.add_argument("--formal-json", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("baselines/covid_long_setting_b/results/convergence_repair_v1/development_protocols"),
    )
    args = parser.parse_args()
    formal_json = args.formal_npz.with_suffix(".json") if args.formal_json is None else args.formal_json
    paths = build_development_protocols(
        formal_npz=args.formal_npz,
        formal_json=formal_json,
        output_root=args.output_root,
    )
    print(json.dumps({"status": "complete", "folds": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
