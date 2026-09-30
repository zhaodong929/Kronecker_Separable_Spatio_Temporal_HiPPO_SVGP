#!/usr/bin/env python3
"""Run Graph WaveNet and DCRNN on the locked three-split Protocol-F task."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/traffic/formal_future_locked_sm_q2_road_context_v1"


def complete(path: Path) -> bool:
    status = path / "status.json"
    return status.exists() and json.loads(status.read_text(encoding="utf-8")).get("status") == "complete"


def command(
    *,
    method: str,
    seed: int,
    output: Path,
    device: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    max_stream_origins: int,
) -> list[str]:
    values = [
        sys.executable,
        "scripts/run_traffic_graph_future_baseline.py",
        "--method", method,
        "--seed", str(seed),
        "--split-manifest", f"results/traffic/protocols/pems_bay/pems_bay_seed{seed}_spatial_split.json",
        "--output", str(output),
        "--data-root", "data/traffic/raw",
        "--road-distance-csv", "data/traffic/raw/pems_bay/distances_bay_2017.csv",
        "--device", device,
        "--batch-size", str(batch_size),
        "--max-epochs", str(max_epochs),
        "--patience", str(patience),
        "--training-mask-fraction", "0.2",
    ]
    if max_stream_origins:
        values.extend(["--max-stream-origins", str(max_stream_origins)])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", choices=["graph_wavenet", "dcrnn"], nargs="+", default=["graph_wavenet", "dcrnn"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-stream-origins", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    records: list[dict[str, object]] = []
    for method in args.methods:
        for seed in args.seeds:
            destination = output / "pems_bay" / "forecast" / method / f"seed{seed}"
            values = command(
                method=method,
                seed=seed,
                output=destination,
                device=args.device,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                max_stream_origins=args.max_stream_origins,
            )
            if args.dry_run:
                records.append({"method": method, "seed": seed, "status": "planned", "command": values})
                continue
            if complete(destination):
                records.append({"method": method, "seed": seed, "status": "already_complete", "output": str(destination)})
            else:
                completed = subprocess.run(values, cwd=ROOT)
                if completed.returncode != 0:
                    raise SystemExit(completed.returncode)
                if not complete(destination):
                    raise RuntimeError(f"Run exited without a complete archive: {destination}")
                records.append({"method": method, "seed": seed, "status": "complete", "output": str(destination)})
            (output / "GRAPH_BASELINE_RUN_STATUS.json").write_text(
                json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(), "records": records}, indent=2) + "\n",
                encoding="utf-8",
            )
    if args.dry_run:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
