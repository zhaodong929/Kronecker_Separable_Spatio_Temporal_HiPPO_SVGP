#!/usr/bin/env python3
"""Create one locked Task-1 VFE theta file per traffic spatial split."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CALIBRATOR = ROOT / "scripts" / "calibrate_traffic_task1.py"


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", choices=["pems_bay", "metr_la"], nargs="+", default=["pems_bay"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/traffic/raw")
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "results/traffic/protocols")
    parser.add_argument("--theta-root", type=Path, default=ROOT / "results/traffic/locked_theta")
    parser.add_argument("--task1-steps", type=int, default=2016)
    parser.add_argument("--calibration-stride", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--mt", type=int, default=32)
    parser.add_argument("--ms", type=int, default=32)
    parser.add_argument("--rff", type=int, default=256)
    parser.add_argument("--temporal-lengthscale-min", type=float, default=0.003)
    parser.add_argument("--temporal-lengthscale-max", type=float, default=2.0)
    parser.add_argument("--relative-objective-tolerance", type=float, default=0.001)
    parser.add_argument("--plateau-checks", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.theta_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for seed in args.seeds:
            split = args.protocol_root / dataset / f"{dataset}_seed{seed}_spatial_split.json"
            output = args.theta_root / dataset / f"seed{seed}"
            command = [
                sys.executable,
                str(CALIBRATOR),
                "--dataset", dataset,
                "--data-root", str(args.data_root),
                "--split-manifest", str(split),
                "--output", str(output),
                "--task1-steps", str(args.task1_steps),
                "--calibration-stride", str(args.calibration_stride),
                "--iterations", str(args.iterations),
                "--validation-every", str(args.validation_every),
                "--mt", str(args.mt),
                "--ms", str(args.ms),
                "--rff", str(args.rff),
                "--temporal-lengthscale-min", str(args.temporal_lengthscale_min),
                "--temporal-lengthscale-max", str(args.temporal_lengthscale_max),
                "--relative-objective-tolerance", str(args.relative_objective_tolerance),
                "--plateau-checks", str(args.plateau_checks),
                "--device", args.device,
                "--seed", str(seed),
            ]
            row = {"dataset": dataset, "seed": seed, "output": str(output), "command": " ".join(command)}
            if args.dry_run:
                row["status"] = "planned"
            else:
                if not split.exists():
                    raise FileNotFoundError(split)
                row["started_utc"] = datetime.now(timezone.utc).isoformat()
                completed = subprocess.run(command, cwd=ROOT, text=True)
                row.update({"returncode": completed.returncode, "finished_utc": datetime.now(timezone.utc).isoformat(), "status": "complete" if completed.returncode == 0 else "failed"})
                if completed.returncode != 0:
                    rows.append(row)
                    write_csv(rows, args.theta_root / "calibration_status.csv")
                    raise SystemExit(completed.returncode)
            rows.append(row)
            write_csv(rows, args.theta_root / "calibration_status.csv")
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
