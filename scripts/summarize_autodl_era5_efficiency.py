#!/usr/bin/env python3
"""Aggregate measured AutoDL runtime, memory, state and FLOP metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def nested(payload: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = payload
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return current
    return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def monitor_peak(path: Path, column: str) -> float | None:
    if not path.is_file():
        return None
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                values.append(float(row[column]))
            except (KeyError, TypeError, ValueError):
                continue
    return max(values) if values else None


def calibration_path(benchmark: Path, method: str, seed: int) -> Path | None:
    if method == "routeb_analytic_hippo_rff":
        representation = "analytic_hippo_rff"
    elif method == "routeb_inducing_points":
        representation = "inducing_points"
    elif method.startswith(("bui_", "maddox_", "official_ohsvgp")):
        representation = "inducing_points"
    else:
        return None
    return (
        benchmark
        / "calibration"
        / f"routeb_joint_{representation}"
        / f"seed{seed}"
        / "result.json"
    )


def collect(benchmark: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result_path in sorted((benchmark / "runs").glob("*/*/*/seed*/result.json")):
        relative = result_path.relative_to(benchmark / "runs").parts
        if len(relative) < 5:
            continue
        scope, branch, method, seed_part = relative[:4]
        seed = int(seed_part.removeprefix("seed"))
        payload = read_json(result_path)
        resources = payload.get("resources", {})
        timing = payload.get("timing", {})
        status_path = result_path.parent / "status.json"
        status = read_json(status_path) if status_path.is_file() else {}
        process_seconds = nested(
            payload,
            "timing.process_total_seconds",
            "timing.end_to_end_training_seconds",
            "train_seconds",
        )
        training_seconds = nested(
            payload,
            "timing.end_to_end_training_seconds",
            "timing.training_seconds",
            "train_seconds",
        )
        update_seconds = nested(payload, "timing.stream_update_seconds")
        prediction_seconds = nested(
            payload,
            "timing.stream_prediction_seconds",
            "timing.prediction_seconds",
            "final.prediction_seconds",
        )
        calibration_seconds = 0.0
        calibration = calibration_path(benchmark, method, seed) if branch == "online" else None
        if calibration is not None and calibration.is_file():
            calibration_payload = read_json(calibration)
            calibration_seconds = float(
                nested(calibration_payload, "timing.process_total_seconds") or 0.0
            )
        runner_peak = nested(
            payload,
            "resources.peak_cuda_allocated_mib",
            "resources.peak_gpu_memory_mib",
        )
        row = {
            "scope": scope,
            "branch": branch,
            "method": method,
            "seed": seed,
            "device": resources.get("device"),
            "training_device": resources.get("training_device", resources.get("device")),
            "prediction_device": resources.get("prediction_device", resources.get("device")),
            "dtype": resources.get("dtype"),
            "orchestrator_status": status.get("status", "missing_status"),
            "device_class": status.get("device_class"),
            "legacy": bool(status.get("legacy", False)),
            "orchestrator_wall_seconds": status.get("wall_seconds"),
            "process_seconds": process_seconds,
            "training_seconds": training_seconds,
            "calibration_seconds_charged": calibration_seconds,
            "end_to_end_with_calibration_seconds": (
                float(process_seconds or 0.0) + calibration_seconds
            ),
            "stream_update_seconds": update_seconds,
            "prediction_seconds": prediction_seconds,
            "mean_iteration_seconds": nested(
                payload,
                "timing.mean_iteration_seconds",
                "selection.mean_iteration_seconds",
            ),
            "first_iteration_seconds": nested(
                payload,
                "selection.first_iteration_seconds",
                "timing.first_iteration_seconds",
            ),
            "steady_state_iteration_seconds": nested(
                payload,
                "timing.mean_steady_state_iteration_seconds",
                "selection.mean_steady_state_iteration_seconds",
            ),
            "mean_block_update_seconds": timing.get("mean_block_update_seconds"),
            "first_block_update_seconds": timing.get("first_block_update_seconds"),
            "steady_state_block_update_seconds": timing.get(
                "mean_steady_state_block_update_seconds"
            ),
            "time_to_best_validation_seconds": nested(
                payload,
                "time_to_best_validation_seconds",
                "selection.time_to_best_validation_seconds",
            ),
            "peak_rss_mib": resources.get("peak_rss_mib"),
            "peak_cuda_allocated_mib": runner_peak,
            "peak_cuda_reserved_mib": resources.get("peak_cuda_reserved_mib"),
            "nvidia_smi_peak_total_used_mib": monitor_peak(
                result_path.parent / "nvidia_smi.csv", "memory_used_mib"
            ),
            "nvidia_smi_peak_power_w": monitor_peak(
                result_path.parent / "nvidia_smi.csv", "power_draw_w"
            ),
            "persistent_state_mib": nested(
                payload,
                "resources.persistent_state_mib",
                "resources.persistent_model_state_mib",
            ),
            "serialized_checkpoint_mib": (
                float(resources["serialized_training_checkpoint_bytes"]) / 1024.0**2
                if resources.get("serialized_training_checkpoint_bytes") is not None
                else None
            ),
            "history_replay_mib": (
                float(resources.get("history_replay_buffer_bytes", 0)) / 1024.0**2
            ),
            "estimated_flops": nested(
                payload,
                "resources.estimated_training_flops",
                "resources.estimated_streaming_flops",
            ),
            "flops_scope": resources.get("flops_scope"),
            "rmse": nested(payload, "overall_current_block.rmse", "final.rmse", "rmse"),
            "nll": nested(payload, "overall_current_block.nll", "final.nll", "nll"),
        }
        if row["estimated_flops"] is not None:
            row["estimated_gflops"] = float(row["estimated_flops"]) / 1e9
        else:
            row["estimated_gflops"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = ["scope", "branch", "method", "device", "dtype", "device_class", "legacy"]
    numeric = [
        column
        for column in frame.columns
        if column not in {*keys, "seed", "orchestrator_status", "training_device", "prediction_device", "flops_scope"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["num_seeds"] = int(group.seed.nunique())
        row["complete_seeds"] = int((group.orchestrator_status == "complete").sum())
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy()
            row[f"{column}_mean"] = float(values.mean()) if values.size else np.nan
            row[f"{column}_sd"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = collect(args.benchmark_root)
    summary = summarize(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "efficiency_per_run.csv", index=False)
    summary.to_csv(args.output_dir / "efficiency_summary.csv", index=False)
    payload = {
        "num_runs": int(len(frame)),
        "num_summary_rows": int(len(summary)),
        "warning": (
            "FLOP counts combine backend counters and method-specific analytic estimates; "
            "compare only rows with the same flops_scope. nvidia-smi memory is whole-device "
            "usage, while framework peak allocation is process-level."
        ),
    }
    (args.output_dir / "README.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
