#!/usr/bin/env python3
"""Freeze the selected PEMS-BAY seed-0 configuration and run missing controls."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_traffic_routeb.py"
DEVELOPMENT_ROOT = ROOT / "results/traffic/seed0_xlag_v1"
OUTPUT_ROOT = ROOT / "results/traffic/seed0_frozen_baselines_v1"
ROAD_DISTANCES = ROOT / "data/traffic/raw/pems_bay/distances_bay_2017.csv"
SPLIT = ROOT / "results/traffic/protocols/pems_bay/pems_bay_seed0_spatial_split.json"


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def freeze() -> tuple[Path, dict[str, object]]:
    selection_path = DEVELOPMENT_ROOT / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected"]
    if selection["formal_stream_used_for_selection"]:
        raise RuntimeError("Refusing to freeze a formal-stream-selected configuration")
    if selected["candidate"] != "spectral_mixture_q2" or selected["calibration_status"] != "converged":
        raise RuntimeError("Expected the converged Task-1-selected spectral_mixture_q2 candidate")

    source_theta = ROOT / str(selected["theta_path"])
    lock_dir = DEVELOPMENT_ROOT / "locked_task1_config"
    lock_dir.mkdir(parents=True, exist_ok=True)
    locked_theta = lock_dir / "theta.json"
    shutil.copy2(source_theta, locked_theta)

    inputs = {
        "speed_h5": ROOT / "data/traffic/raw/pems_bay/PEMS-BAY.h5",
        "coordinates": ROOT / "data/traffic/raw/pems_bay/graph_sensor_locations_bay.csv",
        "road_distances": ROAD_DISTANCES,
        "split_manifest": SPLIT,
        "selection": selection_path,
    }
    implementation = {
        "calibration": ROOT / "scripts/calibrate_traffic_task1.py",
        "development_driver": ROOT / "scripts/run_traffic_seed0_xlag.py",
        "online_runner": RUNNER,
        "mean_features": ROOT / "stvgp_kronecker/data/traffic.py",
        "spatial_kernels": ROOT / "stvgp_kronecker/traffic_spatial_kernels.py",
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    lock = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "dataset": "pems_bay",
        "seed": 0,
        "task1_steps": 2016,
        "selection_boundary": "Task-1 visible-validation sensors only",
        "selection_metric": "Gaussian NLPD, then RMSE",
        "formal_stream_used_for_selection": False,
        "selected_configuration": {
            "spatial_kernel": "spectral_mixture_q2",
            "mean": "28-D road-context L10",
            "calibration_stride": 1,
            "ell_t_hours": 0.5,
            "ms": 32,
            "mt": 128,
            "rff": 512,
        },
        "locked_theta": str(locked_theta.relative_to(ROOT)),
        "locked_theta_sha256": digest(locked_theta),
        "inputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for name, path in inputs.items()},
        "implementation": {name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for name, path in implementation.items()},
        "reused_results": {
            "joint_changing": "results/traffic/seed0_xlag_v1/final_strict_online_seed0",
            "persistence": "results/traffic/formal_deterministic_v1/pems_bay/main/nowcast/persistence/seed0",
        },
        "online_policy": {
            "hyperparameters_frozen_after_task1": True,
            "rff_frequencies_frozen_after_task1": True,
            "posterior_state_updates_only": True,
            "current_hidden_reads_before_prediction": 0,
        },
    }
    (lock_dir / "LOCK.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return locked_theta, lock


def command(label: str, method: str, mode: str, theta: Path, max_stream_steps: int) -> tuple[list[str], Path]:
    output = OUTPUT_ROOT / label
    cmd = [
        sys.executable,
        str(RUNNER),
        "--dataset", "pems_bay",
        "--split-manifest", str(SPLIT),
        "--output", str(output),
        "--protocol", "nowcast",
        "--method", method,
        "--mechanism-mode", mode,
        "--task1-steps", "2016",
        "--stream-stride", "1",
        "--mt", "128",
        "--ms", "32",
        "--rff", "512",
        "--device", "cpu",
        "--dtype", "float64",
    ]
    if method in {"hippo", "mean_field", "ordinary"}:
        cmd.extend([
            "--theta-json", str(theta),
            "--road-distance-csv", str(ROAD_DISTANCES),
            "--temporal-evaluator", "scipy_frozen",
        ])
    if max_stream_steps:
        cmd.extend(["--max-stream-steps", str(max_stream_steps)])
    return cmd, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["kron_stgp", "joint_fixed_global", "decoupled_changing", "decoupled_fixed_global", "spatial_idw"],
        default=["kron_stgp", "joint_fixed_global", "decoupled_changing", "decoupled_fixed_global", "spatial_idw"],
    )
    parser.add_argument("--max-stream-steps", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    theta, lock = freeze()
    task_specs = {
        "kron_stgp": ("ordinary", "fixed"),
        "joint_fixed_global": ("hippo", "fixed_global"),
        "decoupled_changing": ("mean_field", "changing"),
        "decoupled_fixed_global": ("mean_field", "fixed_global"),
        "spatial_idw": ("spatial_idw", "fixed"),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lock": "results/traffic/seed0_xlag_v1/locked_task1_config/LOCK.json",
        "source_commit": lock["source_commit"],
        "tasks": args.tasks,
        "max_stream_steps": args.max_stream_steps,
        "note": "Existing protocol-identical completed archives are reused and are not rerun.",
    }
    (OUTPUT_ROOT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    statuses: list[dict[str, object]] = []
    for label in args.tasks:
        method, mode = task_specs[label]
        cmd, output = command(label, method, mode, theta, args.max_stream_steps)
        status_path = output / "status.json"
        row: dict[str, object] = {"label": label, "method": method, "mode": mode, "output": str(output.relative_to(ROOT))}
        if status_path.exists() and not args.force:
            row["status"] = "reused_complete"
        elif args.dry_run:
            row["status"] = "planned"
            row["command"] = " ".join(cmd)
        else:
            completed = subprocess.run(cmd, cwd=ROOT)
            row["status"] = "complete" if completed.returncode == 0 else "failed"
            row["returncode"] = completed.returncode
            if completed.returncode != 0:
                statuses.append(row)
                write_csv(statuses, OUTPUT_ROOT / "RUN_STATUS.csv")
                raise SystemExit(completed.returncode)
        statuses.append(row)
        write_csv(statuses, OUTPUT_ROOT / "RUN_STATUS.csv")
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
