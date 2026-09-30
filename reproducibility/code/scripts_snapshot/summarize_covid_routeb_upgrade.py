#!/usr/bin/env python3
"""Audit and summarize the seed-0 COVID Route B P0-P2 screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from scripts.run_epidemiology_pilot import predictive_metrics
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_epidemiology_pilot import predictive_metrics


def audit_variant(path: Path, protocol_npz: Path, label: str, decision: str) -> dict[str, object]:
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    with np.load(protocol_npz) as protocol, np.load(path / "predictions.npz") as predictions:
        y_true = np.asarray(predictions["y_true"], dtype=np.float64)
        pred_mean = np.asarray(predictions["pred_mean"], dtype=np.float64)
        pred_var = np.asarray(predictions["pred_var"], dtype=np.float64)
        expected = np.asarray(protocol["stream_y"], dtype=np.float64)[:, protocol["test_indices"]]
        saved_test = np.asarray(predictions["test_indices"], dtype=int)
        expected_test = np.asarray(protocol["test_indices"], dtype=int)
    if not np.array_equal(saved_test, expected_test):
        raise ValueError(f"{label}: prediction test indices differ from protocol")
    if not np.allclose(y_true, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label}: prediction labels differ from protocol")
    if not np.isfinite(pred_mean).all() or not np.isfinite(pred_var).all() or (pred_var <= 0.0).any():
        raise ValueError(f"{label}: non-finite prediction or non-positive variance")
    recomputed = predictive_metrics(y_true, pred_mean, pred_var)
    reported = result["overall_current_block"]
    for metric in ("rmse", "nll", "coverage90"):
        if abs(float(recomputed[metric]) - float(reported[metric])) > 1e-10:
            raise ValueError(f"{label}: {metric} disagrees with prediction archive")
    return {
        "phase": label.split(" ", 1)[0],
        "candidate": label,
        "decision": decision,
        "rmse": float(recomputed["rmse"]),
        "nll": float(recomputed["nll"]),
        "coverage90": float(recomputed["coverage90"]),
        "mean_predictive_std": float(recomputed["mean_predictive_std"]),
        "peak_allocated_mib": float(result["resources"]["peak_cuda_allocated_mib"]),
        "peak_reserved_mib": float(result["resources"]["peak_cuda_reserved_mib"]),
        "runtime_seconds": float(result["timing"]["process_total_seconds"]),
        "features": int(result["feature_projection"]["original_dimension"]),
        "task1_rows": int(result.get("task1_posterior_initialization_rows", 0)),
        "delayed_rows": int(result.get("delayed_observation_rows", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    p1 = root.parent / "phase7_p1_seed0"
    p2 = root.parent / "phase8_p2_seed0"
    rows = [
        audit_variant(
            root.parent / "phase5_delayed_history_mean_seed0",
            Path("data/epidemiology/protocol/covid_history_aware/seed0/protocol.npz"),
            "Current-D delayed history mean",
            "reference",
        ),
        audit_variant(
            root.parent / "phase6_task1_init_seed0",
            Path("data/epidemiology/protocol/covid_history_aware/seed0/protocol.npz"),
            "P0 Task-1 posterior init",
            "retain",
        ),
        audit_variant(p1 / "lag4", p1 / "protocol_lag4.npz", "P1a lag4 reduced", "retain_no_regression"),
        audit_variant(p1 / "lag8", p1 / "protocol_lag8.npz", "P1a lag8", "discard"),
        audit_variant(p1 / "state_intercept", p1 / "protocol_state_intercept.npz", "P1b state intercept", "discard"),
        audit_variant(p1 / "state_lag1", p1 / "protocol_state_lag1.npz", "P1b state lag1", "discard"),
        audit_variant(p1 / "state_lag1_growth", p1 / "protocol_state_lag1_growth.npz", "P1b state lag1 growth", "discard"),
        audit_variant(p2 / "neighbour_lag4", p2 / "protocol_neighbour_lag4.npz", "P2 neighbour exposure", "diagnostic_only"),
    ]
    baseline = rows[0]["rmse"]
    for row in rows:
        row["rmse_change_vs_current_d"] = float(row["rmse"]) - baseline
        row["rmse_improvement_percent"] = 100.0 * (baseline - float(row["rmse"])) / baseline
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (args.output_dir / "phase_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "artifact_audit.json").write_text(
        json.dumps({"status": "complete", "variants": len(rows), "prediction_recomputed": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# COVID Route B P0-P2 seed-0 optimization audit",
        "",
        "All rows use float64, Mt=32, Ms=32, RFF=64, fixed Task-1 Q=2 spectral-mixture theta, delayed observations, and full-joint-conditional variance. P0-P2 candidate rows additionally use Task-1 posterior initialization; Current-D is the pre-P0 reference.",
        "",
        "| Candidate | Decision | RMSE | NLL | Coverage90 | Delta RMSE | Runtime (s) | Peak alloc (MiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['decision']} | {row['rmse']:.6f} | {row['nll']:.6f} | "
            f"{row['coverage90']:.4f} | {row['rmse_change_vs_current_d']:+.6f} | "
            f"{row['runtime_seconds']:.3f} | {row['peak_allocated_mib']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "P0 is retained for protocol completeness and its small NLL improvement. P1a lag4 is retained as a no-regression simplification that removes linearly redundant lag-derived columns. P1a lag8 and all state-specific partial-pooling candidates are discarded. P2 neighbour exposure is diagnostic-only: it improves seed-0 RMSE by less than 1% and slightly lowers Coverage90, so it is not promoted.",
            "The per-state audit shows why P2 is not promoted: it improves Hawaii and Pennsylvania but worsens North Dakota, the state contributing the largest share of squared error, and slightly worsens Oklahoma.",
            "",
            "The COVID archive remains a 52-location, 52-calibration-week, 39-online-week feasibility pilot.",
        ]
    )
    (args.output_dir / "phase_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str((args.output_dir / "phase_summary.md").resolve())}, indent=2))


if __name__ == "__main__":
    main()
