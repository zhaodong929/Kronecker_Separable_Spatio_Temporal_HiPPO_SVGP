#!/usr/bin/env python3
"""Run the official ST-SVGP causal-refit adapter in isolated weekly processes.

The legacy JAX stack accumulates shape-specialised executables when all 143
refits share one process.  Each worker below therefore loads the same frozen
Task-1 kernel, likelihood and inducing locations, reconstructs exactly one
legal weekly posterior, and exits.  This file only orchestrates those official
adapter calls and assembles a common archive after every week is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.archive import PredictionArchive
from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--st-python", type=Path, default=Path("baselines/.venvs/st_svgp/bin/python"))
    parser.add_argument("--spatial-inducing", type=int, default=32)
    parser.add_argument("--task1-iterations", type=int, default=300)
    parser.add_argument("--online-inference-steps", type=int, default=5)
    parser.add_argument("--max-weeks", type=int, default=0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"ST-SVGP worker failed (exit {completed.returncode}); see {log}")


def segment_complete(path: Path, week: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as values:
            return (
                np.asarray(values["week_indices"]).shape == (1,)
                and int(np.asarray(values["week_indices"])[0]) == week
                and np.asarray(values["pred_mean"]).shape == (1, 10)
                and np.asarray(values["pred_var"]).shape == (1, 10)
                and np.isfinite(np.asarray(values["pred_mean"])).all()
                and np.isfinite(np.asarray(values["pred_var"])).all()
                and (np.asarray(values["pred_var"]) >= 0.0).all()
            )
    except (KeyError, OSError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    protocol = COVIDSettingBProtocol(args.protocol_npz, args.protocol_json)
    weeks = protocol.online_weeks if args.max_weeks <= 0 else int(args.max_weeks)
    if not 1 <= weeks <= protocol.online_weeks:
        raise ValueError("--max-weeks must be between 1 and the full online horizon")

    output = resolve(args.output_dir)
    state = output / "task1_frozen_state.npz"
    preparation = output / "task1_state_preparation"
    segments = output / "weekly_segments"
    interpreter = resolve(args.st_python)
    adapter = ROOT / "baselines/covid_long_setting_b/adapters/run_st_svgp.py"

    started = time.perf_counter()
    if not state.is_file():
        run(
            [
                str(interpreter), str(adapter), "--protocol-npz", str(resolve(args.protocol_npz)),
                "--output-dir", str(preparation), "--seed", str(args.seed),
                "--spatial-inducing", str(args.spatial_inducing),
                "--task1-iterations", str(args.task1_iterations),
                "--online-inference-steps", str(args.online_inference_steps), "--max-weeks", "1",
                "--write-frozen-task1-state", str(state),
            ],
            preparation / "run.log",
        )
    if not state.is_file():
        raise RuntimeError(f"The Task-1 state was not written: {state}")

    worker_seconds: list[float] = []
    for week in range(weeks):
        segment = segments / f"week_{week:03d}.npz"
        if segment_complete(segment, week):
            continue
        worker_started = time.perf_counter()
        run(
            [
                str(interpreter), str(adapter), "--protocol-npz", str(resolve(args.protocol_npz)),
                "--output-dir", str(output / "worker_logs" / f"week_{week:03d}"),
                "--seed", str(args.seed), "--spatial-inducing", str(args.spatial_inducing),
                "--task1-iterations", str(args.task1_iterations),
                "--online-inference-steps", str(args.online_inference_steps),
                "--frozen-task1-state", str(state), "--segment-start", str(week),
                "--segment-end", str(week + 1), "--segment-output", str(segment),
            ],
            output / "worker_logs" / f"week_{week:03d}.log",
        )
        if not segment_complete(segment, week):
            raise RuntimeError(f"ST-SVGP worker did not write a valid segment: {segment}")
        worker_seconds.append(time.perf_counter() - worker_started)

    archive = PredictionArchive(protocol, method="st_svgp_causal_refit_isolated", seed=args.seed)
    expected_targets = protocol.evaluation_targets()
    for week in range(weeks):
        segment = segments / f"week_{week:03d}.npz"
        with np.load(segment, allow_pickle=False) as values:
            target = np.asarray(values["y_true"], dtype=np.float64)[0]
            mean = np.asarray(values["pred_mean"], dtype=np.float64)[0]
            variance = np.asarray(values["pred_var"], dtype=np.float64)[0]
        if not np.array_equal(target, expected_targets[week]):
            raise RuntimeError(f"Segment target does not match the audited protocol at week {week}")
        archive.append(protocol.week(week), mean, variance)

    audit = archive.write(
        output / "predictions.npz",
        require_complete=weeks == protocol.online_weeks,
        extra_metadata={
            "adapter": "official_aaltoml_st_svgp_isolated_causal_refit",
            "source_commit": "c5b929e1fc07b14ff9671dd1d66b3b8041e2a2ce",
            "spatial_inducing": int(args.spatial_inducing),
            "task1_iterations": int(args.task1_iterations),
            "online_inference_steps": int(args.online_inference_steps),
            "task1_state": "fitted once, then loaded by isolated weekly official-adapter workers",
            "worker_processes": weeks,
        },
    )
    result = {
        "status": "complete" if weeks == protocol.online_weeks else "smoke_complete",
        "method": "ST-SVGP causal refit (isolated workers)",
        "seed": args.seed,
        "weeks": weeks,
        "capacity": {"spatial_inducing": int(args.spatial_inducing)},
        "total_seconds": time.perf_counter() - started,
        "worker_seconds_total": float(np.sum(worker_seconds)),
        "worker_seconds_per_completed_week": float(np.mean(worker_seconds)) if worker_seconds else 0.0,
        "audit": audit,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
