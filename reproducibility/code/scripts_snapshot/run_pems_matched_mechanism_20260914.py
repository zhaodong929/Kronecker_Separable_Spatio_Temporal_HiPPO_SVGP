#!/usr/bin/env python3
"""Run the matched PEMS-BAY joint-memory mechanism suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/pems_matched_mechanism_20260914"
THETA_ROOT = ROOT / "results/traffic/formal_locked_sm_q2_road_context_v1/task1_theta"


MODES = {
    "joint_changing": {"method": "hippo", "mechanism_mode": "changing", "zero_cross": False},
    "zero_cross_changing": {"method": "hippo", "mechanism_mode": "changing", "zero_cross": True},
    "joint_fixed_global": {"method": "hippo", "mechanism_mode": "fixed_global", "zero_cross": False},
}


def complete(path: Path) -> bool:
    status = path / "status.json"
    if not status.exists():
        return False
    return json.loads(status.read_text(encoding="utf-8")).get("status") == "complete"


def command(seed: int, output: Path, mode_name: str, device: str) -> list[str]:
    mode = MODES[mode_name]
    values = [
        sys.executable,
        "scripts/run_traffic_routeb.py",
        "--dataset", "pems_bay",
        "--data-root", "data/traffic/raw",
        "--split-manifest", f"results/traffic/protocols/pems_bay/pems_bay_seed{seed}_spatial_split.json",
        "--output", str(output),
        "--protocol", "nowcast",
        "--method", mode["method"],
        "--mechanism-mode", mode["mechanism_mode"],
        "--task1-steps", "2016",
        "--stream-stride", "1",
        "--mt", "128",
        "--ms", "32",
        "--rff", "512",
        "--temporal-evaluator", "scipy_frozen",
        "--mean-feature-mode", "road_context_xlag",
        "--xlag-length", "10",
        "--road-distance-csv", "data/traffic/raw/pems_bay/distances_bay_2017.csv",
        "--theta-json", str(THETA_ROOT / f"seed{seed}" / "theta.json"),
        "--model-seed", str(seed),
        "--device", device,
        "--dtype", "float64",
    ]
    if mode["zero_cross"]:
        values.append("--zero-cross")
    return values


def run(values: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(values, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return int(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--modes", choices=tuple(MODES), nargs="+", default=list(MODES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for seed in args.seeds:
        theta = THETA_ROOT / f"seed{seed}" / "theta.json"
        if not theta.exists():
            raise FileNotFoundError(theta)
        for mode_name in args.modes:
            destination = output / "pems_bay" / "nowcast" / mode_name / f"seed{seed}"
            values = command(seed, destination, mode_name, args.device)
            if complete(destination):
                records.append({"seed": seed, "mode": mode_name, "status": "skipped_complete", "path": str(destination)})
                continue
            if args.dry_run:
                records.append({"seed": seed, "mode": mode_name, "status": "planned", "command": values})
                continue
            rc = run(values, output / "logs" / f"seed{seed}_{mode_name}.log")
            status = "complete" if rc == 0 and complete(destination) else "failed"
            records.append({"seed": seed, "mode": mode_name, "status": status, "returncode": rc, "path": str(destination), "command": values})
            (output / "RUN_STATUS.json").write_text(
                json.dumps(
                    {"updated_utc": datetime.now(timezone.utc).isoformat(), "records": records},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if status != "complete":
                raise SystemExit(f"failed: seed {seed}, {mode_name}; see {output / 'logs'}")
    (output / "RUN_STATUS.json").write_text(
        json.dumps(
            {"updated_utc": datetime.now(timezone.utc).isoformat(), "records": records},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
