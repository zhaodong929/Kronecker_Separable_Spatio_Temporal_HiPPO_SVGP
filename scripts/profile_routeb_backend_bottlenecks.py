#!/usr/bin/env python3
"""Run matched Route-B CPU/GPU profiles and summarize stage bottlenecks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_iclr_era5_routeb_strict_online.py"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float("nan") if value in {None, ""} else float(value)


def median(values: list[float]) -> float | None:
    finite = [value for value in values if value == value]
    return float(statistics.median(finite)) if finite else None


def optional_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def backend_specs(
    include_cuda: bool,
    representation: str,
) -> list[tuple[str, str, str, str]]:
    result = [
        ("numpy_cpu", "numpy", "cpu", "cpu"),
        ("torch_cpu", "torch", "cpu", "cpu"),
    ]
    if include_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile requested but torch.cuda.is_available() is false")
        result.append(("torch_cuda", "torch", "cuda", "solver"))
        if representation == "analytic_hippo_rff":
            result.append(("torch_cuda_hybrid", "torch", "cuda", "cpu"))
    return result


def run_profile(
    args: argparse.Namespace,
    name: str,
    backend: str,
    device: str,
    temporal_factor_device: str,
    repeat: int,
) -> dict[str, Any]:
    directory = args.output_dir / name / f"repeat{repeat}"
    result_path = directory / "result.json"
    blocks_path = directory / "blocks.csv"
    predictions_path = directory / "predictions.npz"
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RUNNER),
        "--protocol-npz",
        str(args.protocol_npz),
        "--protocol-json",
        str(args.protocol_json),
        "--theta-json",
        str(args.theta_json),
        "--output",
        str(result_path),
        "--blockwise-output",
        str(blocks_path),
        "--predictions-output",
        str(predictions_path),
        "--representation",
        args.representation,
        "--mt",
        str(args.mt),
        "--ms",
        str(args.ms),
        "--rff-sample-size",
        str(args.rff_sample_size),
        "--prediction-chunk-size",
        str(args.prediction_chunk_size),
        "--seed",
        str(args.seed),
        "--max-blocks",
        str(args.max_blocks),
        "--solver-backend",
        backend,
        "--device",
        device,
        "--temporal-factor-device",
        temporal_factor_device,
        "--dtype",
        args.dtype,
    ]
    (directory / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    block_rows = read_csv(blocks_path)
    for row in block_rows:
        row.update(backend=name, repeat=repeat)
    return {"payload": payload, "blocks": block_rows, "directory": str(directory)}


def summarize_run(name: str, repeat: int, payload: dict[str, Any], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    factor = [numeric(row, "factor_preparation_seconds") for row in blocks]
    update = [numeric(row, "update_seconds") for row in blocks]
    prediction = [numeric(row, "prediction_seconds") for row in blocks]
    feature = [numeric(row, "feature_loading_seconds") for row in blocks]
    core = [a + b + c for a, b, c in zip(factor, update, prediction)]
    steady = slice(1, None) if len(blocks) > 1 else slice(None)
    resources = payload["resources"]
    return {
        "backend": name,
        "temporal_factor_device": payload["temporal_factor_device"],
        "repeat": repeat,
        "num_blocks": len(blocks),
        "rmse": payload["overall_current_block"]["rmse"],
        "nll": payload["overall_current_block"]["nll"],
        "process_total_seconds": payload["timing"]["process_total_seconds"],
        "feature_loading_total_seconds": sum(feature),
        "factor_total_seconds": sum(factor),
        "update_total_seconds": sum(update),
        "prediction_total_seconds": sum(prediction),
        "online_core_total_seconds": sum(core),
        "first_block_core_seconds": core[0],
        "steady_block_core_seconds": sum(core[steady]) / len(core[steady]),
        "steady_factor_seconds": sum(factor[steady]) / len(factor[steady]),
        "steady_update_seconds": sum(update[steady]) / len(update[steady]),
        "steady_prediction_seconds": sum(prediction[steady]) / len(prediction[steady]),
        "peak_rss_mib": resources.get("peak_rss_mib"),
        "peak_cuda_allocated_mib": resources.get("peak_cuda_allocated_mib"),
        "peak_cuda_reserved_mib": resources.get("peak_cuda_reserved_mib"),
        "persistent_state_mib": resources.get("persistent_state_mib"),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "rmse",
        "nll",
        "process_total_seconds",
        "feature_loading_total_seconds",
        "factor_total_seconds",
        "update_total_seconds",
        "prediction_total_seconds",
        "online_core_total_seconds",
        "first_block_core_seconds",
        "steady_block_core_seconds",
        "steady_factor_seconds",
        "steady_update_seconds",
        "steady_prediction_seconds",
        "peak_rss_mib",
        "peak_cuda_allocated_mib",
        "peak_cuda_reserved_mib",
        "persistent_state_mib",
    ]
    result = []
    for name in sorted({row["backend"] for row in rows}):
        selected = [row for row in rows if row["backend"] == name]
        record: dict[str, Any] = {
            "backend": name,
            "temporal_factor_device": selected[0]["temporal_factor_device"],
            "repeats": len(selected),
            "num_blocks": selected[0]["num_blocks"],
        }
        for key in keys:
            record[f"median_{key}"] = median(
                [optional_float(row[key]) for row in selected]
            )
        result.append(record)
    reference = next(row for row in result if row["backend"] == "numpy_cpu")
    for row in result:
        for key in (
            "online_core_total_seconds",
            "steady_block_core_seconds",
            "steady_factor_seconds",
            "steady_update_seconds",
            "steady_prediction_seconds",
        ):
            denominator = row[f"median_{key}"]
            reference_value = reference[f"median_{key}"]
            row[f"speedup_vs_numpy_{key}"] = (
                reference_value / denominator
                if reference_value is not None
                and denominator is not None
                and denominator > 0.0
                else None
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--theta-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=["analytic_hippo_rff", "inducing_points"],
        required=True,
    )
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--prediction-chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=["float64"], default="float64")
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    block_rows: list[dict[str, Any]] = []
    for name, backend, device, temporal_factor_device in backend_specs(
        not args.skip_cuda,
        args.representation,
    ):
        for repeat in range(args.repeats):
            completed = run_profile(
                args,
                name,
                backend,
                device,
                temporal_factor_device,
                repeat,
            )
            runs.append(
                summarize_run(name, repeat, completed["payload"], completed["blocks"])
            )
            block_rows.extend(completed["blocks"])
            print(json.dumps(runs[-1]), flush=True)

    summary = aggregate(runs)
    write_csv(args.output_dir / "per_run.csv", runs)
    write_csv(args.output_dir / "per_block.csv", block_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    payload = {
        "protocol": "matched Route-B backend bottleneck profile",
        "algorithm": "finite-DTC structured joint Schur/Kronecker/Sylvester",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
