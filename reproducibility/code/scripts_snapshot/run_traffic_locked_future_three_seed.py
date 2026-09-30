#!/usr/bin/env python3
"""Run the locked PEMS-BAY Protocol-F experiment with durable chunk checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/traffic/formal_future_locked_sm_q2_road_context_v1"
THETA_ROOT = ROOT / "results/traffic/formal_locked_sm_q2_road_context_v1/task1_theta"
METHODS = {
    "persistence": ("persistence", "changing"),
    "kronhippo_stgp": ("hippo", "changing"),
    "kron_stgp": ("ordinary", "changing"),
}


def complete(path: Path) -> bool:
    status_path = path / "status.json"
    if not status_path.exists():
        return False
    return json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete"


def command(
    *,
    seed: int,
    label: str,
    output: Path,
    chunk_steps: int,
    max_stream_steps: int,
    device: str,
) -> list[str]:
    method, mode = METHODS[label]
    values = [
        sys.executable,
        "scripts/run_traffic_routeb.py",
        "--dataset", "pems_bay",
        "--data-root", "data/traffic/raw",
        "--split-manifest", f"results/traffic/protocols/pems_bay/pems_bay_seed{seed}_spatial_split.json",
        "--output", str(output),
        "--protocol", "forecast",
        "--forecast-horizons", "3", "6", "12",
        "--method", method,
        "--mechanism-mode", mode,
        "--task1-steps", "2016",
        "--stream-stride", "1",
        "--chunk-steps", str(chunk_steps),
        "--model-seed", str(seed),
        "--device", device,
        "--dtype", "float64",
    ]
    if label != "persistence":
        values.extend(
            [
                "--mt", "128",
                "--ms", "32",
                "--rff", "512",
                "--temporal-evaluator", "scipy_frozen",
                "--mean-feature-mode", "road_context_xlag",
                "--xlag-length", "10",
                "--road-distance-csv", "data/traffic/raw/pems_bay/distances_bay_2017.csv",
                "--theta-json", str(THETA_ROOT / f"seed{seed}" / "theta.json"),
            ]
        )
    if max_stream_steps:
        values.extend(["--max-stream-steps", str(max_stream_steps)])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--methods", choices=tuple(METHODS), nargs="+", default=list(METHODS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-steps", type=int, default=2000)
    parser.add_argument("--max-stream-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive for the formal resumable run")

    output = args.output.resolve()
    records: list[dict[str, object]] = []
    for seed in args.seeds:
        theta = THETA_ROOT / f"seed{seed}" / "theta.json"
        if any(label != "persistence" for label in args.methods) and not theta.exists():
            raise FileNotFoundError(theta)
        for label in args.methods:
            destination = output / "pems_bay" / "forecast" / label / f"seed{seed}"
            values = command(
                seed=seed,
                label=label,
                output=destination,
                chunk_steps=args.chunk_steps,
                max_stream_steps=args.max_stream_steps,
                device=args.device,
            )
            if args.dry_run:
                records.append({"seed": seed, "method": label, "status": "planned", "command": values})
                continue
            chunks = 0
            while not complete(destination):
                completed = subprocess.run(values, cwd=ROOT)
                chunks += 1
                if completed.returncode != 0:
                    raise SystemExit(completed.returncode)
                if chunks > 1000:
                    raise RuntimeError(f"Exceeded chunk safety limit for {label} seed {seed}")
            records.append(
                {
                    "seed": seed,
                    "method": label,
                    "status": "complete",
                    "chunks": chunks,
                    "output": str(destination),
                }
            )
            (output / "RUN_STATUS.json").write_text(
                json.dumps(
                    {
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "records": records,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    if args.dry_run:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
