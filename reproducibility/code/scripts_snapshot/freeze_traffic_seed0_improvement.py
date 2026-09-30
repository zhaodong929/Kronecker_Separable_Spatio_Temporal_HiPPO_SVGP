#!/usr/bin/env python3
"""Freeze the Task-1-selected PEMS-BAY seed-0 configuration and optionally run it."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", type=Path, default=Path("results/traffic/seed0_improvement_v2"))
    parser.add_argument("--run-final", action="store_true")
    args = parser.parse_args()

    development_root = args.development_root
    if not development_root.is_absolute():
        development_root = ROOT / development_root
    development_root = development_root.resolve()

    selection_path = development_root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["completed_through_stage"] != "E" or selection["formal_stream_used_for_selection"]:
        raise RuntimeError("Refusing to freeze an incomplete or formal-stream-selected configuration")
    selected = selection["selected_candidate"]
    source_theta = ROOT / selected["theta_path"]
    locked_dir = development_root / "locked_task1_config"
    locked_dir.mkdir(parents=True, exist_ok=True)
    locked_theta = locked_dir / "theta.json"
    shutil.copy2(source_theta, locked_theta)

    data_paths = {
        "speed_h5": ROOT / "data/traffic/raw/pems_bay/PEMS-BAY.h5",
        "coordinates": ROOT / "data/traffic/raw/pems_bay/graph_sensor_locations_bay.csv",
        "road_distances": ROOT / "data/traffic/raw/pems_bay/distances_bay_2017.csv",
        "split_manifest": ROOT / "results/traffic/protocols/pems_bay/pems_bay_seed0_spatial_split.json",
    }
    implementation_paths = {
        "calibration": ROOT / "scripts/calibrate_traffic_task1.py",
        "development_driver": ROOT / "scripts/run_traffic_seed0_improvement.py",
        "online_runner": ROOT / "scripts/run_traffic_routeb.py",
        "spatial_kernels": ROOT / "stvgp_kronecker/traffic_spatial_kernels.py",
    }
    final_output = development_root / "final_strict_online_seed0"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_traffic_routeb.py"),
        "--dataset", "pems_bay",
        "--split-manifest", str(data_paths["split_manifest"]),
        "--output", str(final_output),
        "--protocol", "nowcast",
        "--method", "hippo",
        "--mechanism-mode", "changing",
        "--task1-steps", "2016",
        "--stream-stride", "1",
        "--mt", str(selected["mt"]),
        "--ms", str(selected["ms"]),
        "--rff", str(selected["rff"]),
        "--theta-json", str(locked_theta),
        "--road-distance-csv", str(data_paths["road_distances"]),
        "--temporal-evaluator", "scipy_frozen",
        "--device", "cpu",
        "--dtype", "float64",
    ]
    lock = {
        "schema_version": 1,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dataset": "pems_bay",
        "seed": 0,
        "task1_steps": 2016,
        "selection_metric": "full Task-1 visible-validation Gaussian NLPD, then RMSE",
        "formal_stream_used_for_selection": False,
        "selected_candidate": selected,
        "locked_theta": str(locked_theta.relative_to(ROOT)),
        "locked_theta_sha256": digest(locked_theta),
        "inputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for name, path in data_paths.items()},
        "implementation": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            for name, path in implementation_paths.items()
        },
        "final_command": command,
        "final_output": str(final_output.relative_to(ROOT)),
        "online_policy": {
            "hyperparameters_frozen_after_task1": True,
            "rff_frequencies_frozen_after_task1": True,
            "posterior_state_updates_only": True,
            "stream_stride": 1,
            "current_hidden_reads_before_prediction": 0,
        },
    }
    lock_path = locked_dir / "LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.run_final:
        subprocess.run(command, cwd=ROOT, check=True)

    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
