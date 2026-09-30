#!/usr/bin/env python3
"""Launch the predeclared PEMS-BAY/METR-LA paired-spatial experiment matrix."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_traffic_routeb.py"


def write_status(rows: list[dict[str, object]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command_for(
    *,
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    label: str,
    method: str,
    protocol: str,
    mode: str,
) -> tuple[list[str], Path]:
    split = args.protocol_root / dataset / f"{dataset}_seed{seed}_spatial_split.json"
    output = args.output_root / dataset / args.stage / protocol / label / f"seed{seed}"
    command = [
        sys.executable,
        str(RUNNER),
        "--dataset", dataset,
        "--data-root", str(args.data_root),
        "--split-manifest", str(split),
        "--output", str(output),
        "--protocol", protocol,
        "--method", method,
        "--mechanism-mode", mode,
        "--task1-steps", str(args.task1_steps),
        "--mt", str(args.mt),
        "--ms", str(args.ms),
        "--rff", str(args.rff),
        "--temporal-evaluator", args.temporal_evaluator,
        "--device", args.device,
        "--dtype", args.dtype,
    ]
    if args.max_stream_steps:
        command.extend(["--max-stream-steps", str(args.max_stream_steps)])
    if args.stream_stride != 1:
        command.extend(["--stream-stride", str(args.stream_stride)])
    if protocol == "forecast":
        command.extend(["--forecast-horizons", *[str(value) for value in args.forecast_horizons]])
    if args.theta_root is not None:
        theta = args.theta_root / dataset / f"seed{seed}" / "theta.json"
        if not theta.exists():
            raise FileNotFoundError(f"Missing locked Task-1 theta file: {theta}")
        theta_payload = json.loads(theta.read_text(encoding="utf-8"))
        if not args.allow_unconverged_theta and theta_payload.get("calibration_status") != "converged":
            raise ValueError(
                f"Task-1 theta is not convergence-gated: {theta} "
                f"({theta_payload.get('calibration_status', 'missing status')})"
            )
        command.extend(["--theta-json", str(theta)])
    return command, output


def experiment_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    stages = {args.stage} if args.stage != "all" else {"main", "mechanism", "forecast"}
    for dataset in args.datasets:
        if "main" in stages:
            for seed in args.seeds:
                for label, method, mode in (
                    ("kronhippo_stgp", "hippo", "changing"),
                    ("kron_stgp", "ordinary", "fixed"),
                    ("mean_field_changing", "mean_field", "changing"),
                    ("persistence", "persistence", "fixed"),
                    ("frozen_mean", "frozen_mean", "fixed"),
                ):
                    rows.append({"dataset": dataset, "seed": str(seed), "label": label, "method": method, "protocol": "nowcast", "mode": mode})
        if "mechanism" in stages:
            for seed in args.seeds:
                for label, method, mode in (
                    ("joint_changing", "hippo", "changing"),
                    ("joint_fixed", "hippo", "fixed"),
                    ("decoupled_changing", "mean_field", "changing"),
                    ("decoupled_fixed", "mean_field", "fixed"),
                ):
                    rows.append({"dataset": dataset, "seed": str(seed), "label": label, "method": method, "protocol": "nowcast", "mode": mode})
        if "forecast" in stages:
            for seed in args.seeds:
                for label, method, mode in (
                    ("kronhippo_stgp", "hippo", "changing"),
                    ("kron_stgp", "ordinary", "fixed"),
                    ("persistence", "persistence", "fixed"),
                ):
                    rows.append({"dataset": dataset, "seed": str(seed), "label": label, "method": method, "protocol": "forecast", "mode": mode})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["main", "mechanism", "forecast", "all"], default="main")
    parser.add_argument("--datasets", choices=["pems_bay", "metr_la"], nargs="+", default=["pems_bay"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/traffic/raw")
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "results/traffic/protocols")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/traffic/formal")
    parser.add_argument("--task1-steps", type=int, default=2016)
    parser.add_argument("--theta-root", type=Path, help="Root containing {dataset}/seed{seed}/theta.json files")
    parser.add_argument("--allow-unconverged-theta", action="store_true", help="Development-only override; formal runs should not use this")
    parser.add_argument("--max-stream-steps", type=int, default=0)
    parser.add_argument("--stream-stride", type=int, default=1)
    parser.add_argument("--forecast-horizons", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--mt", type=int, default=32)
    parser.add_argument("--ms", type=int, default=32)
    parser.add_argument("--rff", type=int, default=256)
    parser.add_argument("--temporal-evaluator", choices=["torch_autograd", "scipy_frozen"], default="scipy_frozen")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        for seed in args.seeds:
            split = args.protocol_root / dataset / f"{dataset}_seed{seed}_spatial_split.json"
            if not split.exists():
                raise FileNotFoundError(f"Missing fixed split manifest: {split}")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        commit = "unavailable"
    manifest = {
        "schema_version": 1,
        "protocol": "paired_spatial_streaming",
        "stage": args.stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "tasks": experiment_rows(args),
    }
    suite_root = args.output_root / "suite_manifests"
    suite_root.mkdir(parents=True, exist_ok=True)
    (suite_root / f"{args.stage}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    statuses: list[dict[str, object]] = []
    for task in manifest["tasks"]:
        command, output = command_for(
            args=args,
            dataset=task["dataset"],
            seed=int(task["seed"]),
            label=task["label"],
            method=task["method"],
            protocol=task["protocol"],
            mode=task["mode"],
        )
        status = {**task, "output": str(output), "command": " ".join(command)}
        if args.dry_run:
            status["status"] = "planned"
        else:
            started = datetime.now(timezone.utc).isoformat()
            completed = subprocess.run(command, cwd=ROOT, text=True)
            status.update({"status": "complete" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat()})
            if completed.returncode != 0:
                statuses.append(status)
                write_status(statuses, suite_root / f"{args.stage}_status.csv")
                raise SystemExit(completed.returncode)
        statuses.append(status)
        write_status(statuses, suite_root / f"{args.stage}_status.csv")
        print(json.dumps(status, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
