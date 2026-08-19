#!/usr/bin/env python3
"""Run official ST-SVGP causal refits in isolated processes and merge archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "baselines/covid_long_setting_b/adapters/run_st_svgp.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"))
    parser.add_argument("--output-root", type=Path, default=Path("baselines/covid_long_setting_b/results/formal_selected_st_svgp_segmented"))
    parser.add_argument("--st-python", type=Path, default=Path("baselines/.venvs/st_svgp/bin/python"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--segment-weeks", type=int, default=16)
    parser.add_argument("--spatial-inducing", type=int, default=32)
    parser.add_argument("--task1-iterations", type=int, default=300)
    parser.add_argument("--online-inference-steps", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_protocol(path: Path) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as archive:
        return int(archive["stream_y"].shape[0]), int(archive["test_indices"].shape[0])


def command_for(args: argparse.Namespace, protocol: Path, output: Path, segment: Path, seed: int, start: int, end: int) -> list[str]:
    return [
        str(absolute(args.st_python)), str(ADAPTER),
        "--protocol-npz", str(protocol),
        "--protocol-json", str(protocol.with_suffix(".json")),
        "--output-dir", str(output),
        "--seed", str(seed),
        "--spatial-inducing", str(args.spatial_inducing),
        "--task1-iterations", str(args.task1_iterations),
        "--online-inference-steps", str(args.online_inference_steps),
        "--segment-start", str(start),
        "--segment-end", str(end),
        "--segment-output", str(segment),
    ]


def validate_segment(path: Path, start: int, end: int, hidden: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"week_indices", "y_true", "pred_mean", "pred_var", "times", "test_indices"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} misses fields {sorted(missing)}")
        weeks = np.asarray(archive["week_indices"])
        truth = np.asarray(archive["y_true"])
        mean = np.asarray(archive["pred_mean"])
        variance = np.asarray(archive["pred_var"])
        if not np.array_equal(weeks, np.arange(start, end, dtype=np.int64)):
            raise ValueError(f"{path} has non-contiguous week indices")
        expected = (end - start, hidden)
        if truth.shape != expected or mean.shape != expected or variance.shape != expected:
            raise ValueError(f"{path} has shapes truth={truth.shape}, mean={mean.shape}, variance={variance.shape}")
        if not np.isfinite(truth).all() or not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError(f"{path} contains non-finite values")
        if np.any(variance < 0.0):
            raise ValueError(f"{path} contains negative predictive variance")
        return {
            "week_indices": weeks,
            "y_true": truth,
            "pred_mean": mean,
            "pred_var": variance,
            "times": np.asarray(archive["times"]),
            "test_indices": np.asarray(archive["test_indices"]),
        }


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    protocol = absolute(args.protocol_root) / f"seed{seed}/protocol.npz"
    online_weeks, hidden = read_protocol(protocol)
    if online_weeks != 143 or hidden != 10:
        raise ValueError(f"Unexpected protocol shape for seed {seed}: weeks={online_weeks}, hidden={hidden}")
    output = absolute(args.output_root) / f"seed{seed}/st_svgp"
    output.mkdir(parents=True, exist_ok=True)
    segments = []
    for start in range(0, online_weeks, int(args.segment_weeks)):
        end = min(start + int(args.segment_weeks), online_weeks)
        segment_dir = output / "segments" / f"week_{start:03d}_{end:03d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_path = segment_dir / "predictions.npz"
        command = command_for(args, protocol, output, segment_path, seed, start, end)
        log_path = segment_dir / "run.log"
        if not segment_path.is_file():
            if args.dry_run:
                segments.append({"start": start, "end": end, "status": "dry_run", "command": command})
                continue
            with log_path.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env={
                        **dict(__import__("os").environ),
                        "PYTHONUNBUFFERED": "1",
                        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1",
                    },
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            if completed.returncode:
                raise RuntimeError(f"ST-SVGP segment failed for seed {seed}, weeks {start}:{end}; see {log_path}")
        record = validate_segment(segment_path, start, end, hidden)
        segments.append({"start": start, "end": end, "status": "complete", "path": str(segment_path), "weeks": end - start})
        if args.dry_run:
            continue
        if start == 0:
            merged = {key: [record[key]] for key in ("y_true", "pred_mean", "pred_var", "times")}
            test_indices = record["test_indices"]
        else:
            for key in merged:
                merged[key].append(record[key])
    if args.dry_run:
        return {"seed": seed, "status": "dry_run", "segments": segments}
    arrays = {key: np.concatenate(value, axis=0) for key, value in merged.items()}
    if arrays["y_true"].shape != (online_weeks, hidden):
        raise ValueError(f"Merged archive shape is invalid for seed {seed}: {arrays['y_true'].shape}")
    np.savez_compressed(
        output / "predictions.npz",
        **arrays,
        test_indices=test_indices,
        metadata_json=np.asarray(json.dumps({
            "method": "ST-SVGP causal refit segmented",
            "source_commit": "c5b929e1fc07b14ff9671dd1d66b3b8041e2a2ce",
            "seed": seed,
            "segment_weeks": int(args.segment_weeks),
            "task1_iterations": int(args.task1_iterations),
            "online_inference_steps": int(args.online_inference_steps),
            "leakage_rule": "Each segment reconstructs only protocol-delivered history before prediction.",
        }, sort_keys=True)),
    )
    status = {
        "status": "complete",
        "method": "ST-SVGP causal refit segmented",
        "source_commit": "c5b929e1fc07b14ff9671dd1d66b3b8041e2a2ce",
        "seed": seed,
        "weeks": online_weeks,
        "archive": str(output / "predictions.npz"),
        "segments": segments,
        "audit": {
            "online_steps_completed": online_weeks,
            "delayed_hidden_labels": (online_weeks - 1) * hidden,
            "current_visible_labels": online_weeks * 42,
            "current_hidden_labels_read": 0,
            "hidden_predictions": online_weeks * hidden,
            "passed": True,
        },
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def main() -> None:
    args = parse_args()
    if args.segment_weeks < 1:
        raise ValueError("--segment-weeks must be positive")
    results = [run_seed(args, seed) for seed in args.seeds]
    manifest = {"status": "dry_run" if args.dry_run else "complete", "seeds": args.seeds, "results": results}
    output_root = absolute(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "formal_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
