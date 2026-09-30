#!/usr/bin/env python3
"""Run missing PEMS-BAY splits for the frozen SM-Q2 road-context configuration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/traffic/formal_locked_sm_q2_road_context_v1"


def complete(path: Path, accepted: set[str]) -> bool:
    status_path = path / "status.json"
    if not status_path.exists():
        return False
    return json.loads(status_path.read_text(encoding="utf-8")).get("status") in accepted


def run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return int(process.returncode)


def calibration_command(seed: int, output: Path, device: str) -> list[str]:
    return [
        sys.executable,
        "scripts/calibrate_traffic_task1.py",
        "--dataset", "pems_bay",
        "--data-root", "data/traffic/raw",
        "--split-manifest", f"results/traffic/protocols/pems_bay/pems_bay_seed{seed}_spatial_split.json",
        "--output", str(output),
        "--task1-steps", "2016",
        "--calibration-stride", "1",
        "--iterations", "250",
        "--validation-every", "25",
        "--mt", "128",
        "--ms", "32",
        "--rff", "512",
        "--fixed-temporal-lengthscale", "0.5",
        "--spatial-kernel", "spectral_mixture",
        "--spatial-mixtures", "2",
        "--mean-feature-mode", "road_context_xlag",
        "--xlag-length", "10",
        "--context-graph-diffusion", "7.448975327393576",
        "--road-distance-csv", "data/traffic/raw/pems_bay/distances_bay_2017.csv",
        "--seed", str(seed),
        "--device", device,
    ]


def routeb_command(
    seed: int,
    theta: Path,
    output: Path,
    method: str,
    mechanism_mode: str,
    device: str,
    *,
    stability_gate: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_traffic_routeb.py",
        "--dataset", "pems_bay",
        "--data-root", "data/traffic/raw",
        "--split-manifest", f"results/traffic/protocols/pems_bay/pems_bay_seed{seed}_spatial_split.json",
        "--output", str(output),
        "--protocol", "nowcast",
        "--method", method,
        "--mechanism-mode", mechanism_mode,
        "--task1-steps", "2016",
        "--stream-stride", "1",
        "--mt", "128",
        "--ms", "32",
        "--rff", "512",
        "--temporal-evaluator", "scipy_frozen",
        "--mean-feature-mode", "road_context_xlag",
        "--xlag-length", "10",
        "--road-distance-csv", "data/traffic/raw/pems_bay/distances_bay_2017.csv",
        "--theta-json", str(theta),
        "--model-seed", str(seed),
        "--device", device,
        "--dtype", "float64",
    ]
    if stability_gate:
        command.extend(["--state-diagnostics", "--divergence-rmse-threshold", "5"])
    return command


def ignnk_command(seed: int, output: Path, device: str) -> list[str]:
    return [
        sys.executable,
        "scripts/run_official_ignnk_traffic.py",
        "--data-root", "data/traffic/raw",
        "--split-manifest", f"results/traffic/protocols/pems_bay/pems_bay_seed{seed}_spatial_split.json",
        "--road-distance-csv", "data/traffic/raw/pems_bay/distances_bay_2017.csv",
        "--output", str(output),
        "--task1-steps", "2016",
        "--seed", str(seed),
        "--device", device,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = args.output.resolve()
    logs = output / "logs"
    records: list[dict[str, object]] = []
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    for seed in args.seeds:
        theta_dir = output / "task1_theta" / f"seed{seed}"
        theta = theta_dir / "theta.json"
        if theta.exists():
            records.append({"seed": seed, "stage": "calibration", "status": "skipped_complete", "path": str(theta_dir)})
        else:
            rc = run(calibration_command(seed, theta_dir, args.device), logs / f"seed{seed}_calibration.log")
            status = "complete" if rc == 0 and theta.exists() else "failed"
            records.append({"seed": seed, "stage": "calibration", "status": status, "returncode": rc, "path": str(theta_dir)})
            if status != "complete":
                continue

        methods = (
            ("kronhippo_stgp", "hippo", "changing", False),
            ("kron_stgp", "ordinary", "changing", False),
            ("joint_fixed_global", "hippo", "fixed_global", False),
            ("decoupled_fixed_global", "mean_field", "fixed_global", False),
            ("decoupled_changing_diagnostic", "mean_field", "changing", True),
        )
        for label, method, mode, stability_gate in methods:
            destination = output / "pems_bay" / "nowcast" / label / f"seed{seed}"
            accepted = {"diverged"} if stability_gate else {"complete"}
            if complete(destination, accepted):
                records.append({"seed": seed, "stage": label, "status": "skipped_complete", "path": str(destination)})
                continue
            rc = run(
                routeb_command(
                    seed,
                    theta,
                    destination,
                    method,
                    mode,
                    args.device,
                    stability_gate=stability_gate,
                ),
                logs / f"seed{seed}_{label}.log",
            )
            status = "complete" if rc == 0 and complete(destination, accepted) else "failed"
            records.append({"seed": seed, "stage": label, "status": status, "returncode": rc, "path": str(destination)})

        destination = output / "pems_bay" / "nowcast" / "ignnk" / f"seed{seed}"
        if complete(destination, {"complete"}):
            records.append({"seed": seed, "stage": "ignnk", "status": "skipped_complete", "path": str(destination)})
        else:
            rc = run(ignnk_command(seed, destination, args.device), logs / f"seed{seed}_ignnk.log")
            status = "complete" if rc == 0 and complete(destination, {"complete"}) else "failed"
            records.append({"seed": seed, "stage": "ignnk", "status": status, "returncode": rc, "path": str(destination)})

        (output / "RUN_STATUS.json").write_text(
            json.dumps({"source_commit": source_commit, "updated_utc": datetime.now(timezone.utc).isoformat(), "records": records}, indent=2) + "\n",
            encoding="utf-8",
        )

    failed = [row for row in records if row["status"] == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} stages failed; see {logs}")


if __name__ == "__main__":
    main()
