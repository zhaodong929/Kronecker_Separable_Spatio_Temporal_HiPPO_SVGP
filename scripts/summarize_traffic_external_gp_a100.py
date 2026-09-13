#!/usr/bin/env python3
"""Audit external-GP archives and merge them into the locked PEMS main table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ohsvgp", "maddox_streaming_sgpr", "bui_osgpr", "st_svgp")
LABELS = {
    "ohsvgp": "OHSVGP",
    "maddox_streaming_sgpr": "Maddox StreamingSGPR",
    "bui_osgpr": "Bui OSGPR",
    "st_svgp": "ST-SVGP (causal refit)",
}


def metrics(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    variance = np.maximum(np.asarray(variance, dtype=np.float64).reshape(-1), 1e-10)
    std = np.sqrt(variance)
    z = (y - mean) / std
    crps = std * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
        - 1.0 / math.sqrt(math.pi)
    )
    levels = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    quantiles = np.asarray([0.0627067779, 0.189118426, 0.318639364, 0.45376219, 0.597760126, 0.755415026, 0.934589291, 1.15034938, 1.43953147, 1.95996398])
    coverage = np.asarray([np.mean(np.abs(y - mean) <= value * std) for value in quantiles])
    return {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "crps": float(np.mean(np.maximum(crps, 0.0))),
        "gaussian_nlpd": float(np.mean(0.5 * (np.log(2.0 * np.pi * variance) + z**2))),
        "ece": float(np.mean(np.abs(coverage - levels))),
        "coverage90": float(np.mean(np.abs(y - mean) <= 1.644853627 * std)),
    }


def archive_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["y_true"], dtype=np.float64),
            np.asarray(data["pred_mean"], dtype=np.float64),
            np.asarray(data["pred_var"], dtype=np.float64),
        )


def reference(seed: int) -> tuple[np.ndarray, float]:
    directory = ROOT / f"results/traffic/protocol_n_external_gp/seed{seed}"
    with np.load(directory / "protocol.npz", allow_pickle=False) as data:
        truth = np.asarray(data["stream_y"], dtype=np.float64)[:, np.asarray(data["test_indices"], dtype=int)]
    metadata = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    return truth, float(metadata["target_standardisation"]["scale"])


def delayed_count(method: str, directory: Path) -> int:
    if method in {"ohsvgp", "maddox_streaming_sgpr", "bui_osgpr"}:
        payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        return int(payload["delayed_observation_rows"])
    payload = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    return int(payload["audit"]["delayed_hidden_labels"])


def current_hidden_reads(method: str, directory: Path) -> int:
    if method == "ohsvgp":
        payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        return int(payload["audit"]["current_hidden_labels_read"])
    if method in {"maddox_streaming_sgpr", "bui_osgpr"}:
        payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        return int(payload["current_hidden_labels_read"])
    payload = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    return int(payload["audit"]["current_hidden_labels_read"])


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict[str, object]], methods: tuple[str, ...] = METHODS
) -> list[dict[str, object]]:
    output = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        item: dict[str, object] = {"method": method, "n": len(subset)}
        for metric in ("rmse", "rmse_mph", "crps", "gaussian_nlpd", "ece", "coverage90"):
            values = np.asarray([row[metric] for row in subset], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd"] = float(values.std(ddof=1))
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/traffic/formal_external_gp_a100_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    if any(seed not in (1, 2, 3) for seed in args.seeds):
        raise ValueError("External-GP evaluation is restricted to seeds 1, 2 and 3")
    selected_methods = tuple(dict.fromkeys(args.methods))
    output = args.output or args.input / "paper_ready"
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for method in selected_methods:
        for seed in args.seeds:
            directory = args.input / "pems_bay" / "nowcast" / method / f"seed{seed}"
            y, mean, variance = archive_arrays(directory / "predictions.npz")
            expected_truth, speed_scale = reference(seed)
            passed = bool(
                y.shape == mean.shape == variance.shape == (50100, 65)
                and np.all(np.isfinite(mean))
                and np.all(np.isfinite(variance))
                and np.all(variance > 0.0)
                and np.array_equal(y, expected_truth)
                and delayed_count(method, directory) == 50099 * 65
                and current_hidden_reads(method, directory) == 0
            )
            value = metrics(y, mean, variance)
            value["rmse_mph"] = value["rmse"] * speed_scale
            rows.append({"method": method, "seed": seed, **value})
            audit_rows.append({
                "method": method,
                "seed": seed,
                "status": "passed" if passed else "failed",
                "shape": list(mean.shape),
                "truth_matches_locked_protocol": bool(np.array_equal(y, expected_truth)),
                "delayed_hidden_labels": delayed_count(method, directory),
                "current_hidden_labels_read": current_hidden_reads(method, directory),
            })
    if any(row["status"] != "passed" for row in audit_rows):
        raise SystemExit("At least one external-GP archive failed the common Protocol-N audit")
    summary = aggregate(rows, selected_methods)
    write_csv(rows, output / "external_gp_per_seed.csv")
    write_csv(summary, output / "external_gp_aggregate.csv")
    (output / "EXTERNAL_GP_AUDIT.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "methods": list(selected_methods),
                "seeds": args.seeds,
                "rows": audit_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metric_lines = [
        "# PEMS-BAY Existing-GP Metrics",
        "",
        "Common Protocol-N evaluator. Mean +/- sample standard deviation; lower is better except Coverage90, whose nominal target is 0.90.",
        "",
        "| Method | RMSE | RMSE (mph) | CRPS | Gaussian NLPD | ECE | Coverage90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def metric_cell(metric: str) -> str:
            return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_sd']):.4f}"

        metric_lines.append(
            f"| {LABELS[str(row['method'])]} | {metric_cell('rmse')} | "
            f"{metric_cell('rmse_mph')} | {metric_cell('crps')} | "
            f"{metric_cell('gaussian_nlpd')} | {metric_cell('ece')} | "
            f"{metric_cell('coverage90')} |"
        )
    (output / "external_gp_metrics.md").write_text(
        "\n".join(metric_lines) + "\n", encoding="utf-8"
    )

    if selected_methods != METHODS or args.seeds != [1, 2, 3]:
        print(json.dumps({"status": "passed", "output": str(output), "external": summary}, indent=2))
        return

    existing_path = ROOT / "results/traffic/formal_locked_sm_q2_road_context_v1/aggregate_results_1_2_3.csv"
    with existing_path.open(newline="", encoding="utf-8") as handle:
        existing = list(csv.DictReader(handle))
    combined = existing + [{key: value for key, value in row.items()} for row in summary]
    write_csv(combined, output / "pems_nowcasting_main_table_with_external_gp.csv")
    lines = [
        "# PEMS-BAY Protocol-N Main Table With Existing GP Baselines",
        "",
        "Mean +/- sample standard deviation over the same three fixed spatial splits. Lower is better except Coverage90, whose nominal target is 0.90.",
        "",
        "| Method | RMSE | RMSE (mph) | CRPS | Gaussian NLPD | ECE | Coverage90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    label = {
        "persistence": "Delayed-target last-value persistence",
        "kron_stgp": "Kron-STGP",
        "kronhippo_stgp": "KronHiPPO-STGP",
        "joint_fixed_global": "Joint + fixed-global",
        "decoupled_fixed_global": "Decoupled + fixed-global",
        "ignnk": "IGNNK",
        **LABELS,
    }
    for row in combined:
        def cell(metric: str) -> str:
            mean_key, sd_key = f"{metric}_mean", f"{metric}_sd"
            if mean_key not in row or row[mean_key] in (None, ""):
                return "N/A"
            return f"{float(row[mean_key]):.4f} +/- {float(row[sd_key]):.4f}"
        lines.append(
            f"| {label[str(row['method'])]} | {cell('rmse')} | {cell('rmse_mph')} | "
            f"{cell('crps')} | {cell('gaussian_nlpd')} | {cell('ece')} | {cell('coverage90')} |"
        )
    (output / "pems_nowcasting_main_table_with_external_gp.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "output": str(output), "external": summary}, indent=2))


if __name__ == "__main__":
    main()
