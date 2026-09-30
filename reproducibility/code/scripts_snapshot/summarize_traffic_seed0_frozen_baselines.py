#!/usr/bin/env python3
"""Audit and summarize frozen PEMS-BAY seed-0 baseline results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/traffic/seed0_frozen_baselines_v1"


SOURCES = (
    ("Delayed-target last-value persistence", "reused", ROOT / "results/traffic/formal_deterministic_v1/pems_bay/main/nowcast/persistence/seed0"),
    ("Spatial IDW (8 neighbours)", "new", OUTPUT / "spatial_idw"),
    ("Kron-STGP (point temporal)", "new", OUTPUT / "kron_stgp"),
    ("KronHiPPO-STGP (joint + changing)", "reused", ROOT / "results/traffic/seed0_xlag_v1/final_strict_online_seed0"),
    ("Joint + fixed-global", "new", OUTPUT / "joint_fixed_global"),
    ("Decoupled + fixed-global", "new", OUTPUT / "decoupled_fixed_global"),
)


def load_row(label: str, provenance: str, path: Path) -> dict[str, object]:
    payload = json.loads((path / "result.json").read_text(encoding="utf-8"))
    final = payload["result"]["final"]
    archive = np.load(path / "predictions.npz")
    y = np.asarray(archive["y"])
    mean = np.asarray(archive["mean"])
    variance = np.asarray(archive["variance"])
    guard = payload["result"]["guard"]
    expected = (50_100, 65)
    passed = (
        y.shape == expected
        and mean.shape == expected
        and variance.shape == expected
        and np.all(np.isfinite(y))
        and np.all(np.isfinite(mean))
        and np.all(np.isfinite(variance))
        and np.all(variance > 0.0)
        and guard["current_hidden_reads_before_prediction"] == 0
        and guard["current_hidden_reveals"] == expected[0]
        and guard["current_visible_reads"] == expected[0]
        and guard["unique_delayed_hidden_absorptions"] == expected[0] - 1
    )
    return {
        "method": label,
        "provenance": provenance,
        "status": "complete_passed" if passed else "audit_failed",
        "rmse": final["rmse"],
        "crps": final["crps"],
        "gaussian_nlpd": final["gaussian_nlpd"],
        "ece": final["ece"],
        "coverage90": final["coverage90"],
        "rmse_mph": final["rmse_speed"],
        "archive_shape": str(y.shape),
        "current_hidden_reads": guard["current_hidden_reads_before_prediction"],
        "result_path": str(path.relative_to(ROOT)),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_point_row() -> dict[str, object]:
    path = ROOT / "results/traffic/seed0_external_baselines_v1/ignnk"
    payload = json.loads((path / "result.json").read_text(encoding="utf-8"))
    archive = np.load(path / "predictions.npz")
    expected = (50_100, 65)
    passed = (
        np.asarray(archive["y"]).shape == expected
        and np.asarray(archive["mean"]).shape == expected
        and np.all(np.isfinite(archive["y"]))
        and np.all(np.isfinite(archive["mean"]))
        and payload["causal_audit"]["current_hidden_reads_before_prediction"] == 0
    )
    return {
        "method": "IGNNK",
        "provenance": "new official-core adaptation",
        "status": "complete_passed" if passed else "audit_failed",
        **payload["metrics"],
        "best_task1_validation_rmse": payload["best_validation_rmse"],
        "result_path": str(path.relative_to(ROOT)),
    }


def load_stability_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = OUTPUT / "stability_audit"
    for name in ("seed0_repeat_a", "seed1_rff"):
        path = root / name
        payload = json.loads((path / "result.json").read_text(encoding="utf-8"))
        result = payload["result"]
        with (path / "per_step_metrics.csv").open(newline="", encoding="utf-8") as handle:
            trace = list(csv.DictReader(handle))
        onset = result["divergence_onset"]
        final_row = trace[-1]
        rows.append(
            {
                "run": name,
                "model_seed": payload["args"]["model_seed"],
                "status": result["stability_status"],
                "onset_step": onset["step"],
                "onset_time_index": onset["time_index"],
                "onset_timestamp": onset["timestamp"],
                "onset_rmse": onset["rmse"],
                "mean_predictive_std": final_row["mean_predictive_std"],
                "kt_condition_number": final_row["kt_condition_number"],
                "beta_cov_condition_number": final_row["beta_cov_condition_number"],
                "beta_mean_norm": final_row["beta_mean_norm"],
                "beta_natural_norm": final_row["beta_natural_norm"],
                "gp_mean_state_norm": final_row["gp_mean_state_norm"],
                "gp_information_norm": final_row["gp_information_norm"],
                "result_path": str(path.relative_to(ROOT)),
            }
        )
    return rows


def write_report(
    rows: list[dict[str, object]],
    point_row: dict[str, object],
    stability_rows: list[dict[str, object]],
) -> None:
    failed = json.loads((OUTPUT / "DECOUPLED_CHANGING_GATE.json").read_text(encoding="utf-8"))
    lines = [
        "# Frozen PEMS-BAY Seed-0 Baseline Report",
        "",
        "Configuration: Task-1-selected SM-Q2 residual kernel with the 28-D road-context L10 mean; Task 1 has 2,016 samples. No formal-stream metric was used for configuration selection.",
        "",
        "| Method | Provenance | RMSE | CRPS | NLPD | ECE | Coverage90 | Audit |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['provenance']} | {row['rmse']:.4f} | {row['crps']:.4f} | "
            f"{row['gaussian_nlpd']:.4f} | {row['ece']:.4f} | {row['coverage90']:.4f} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Deterministic external baseline",
            "",
            "| Method | RMSE | MAE | RMSE (mph) | MAE (mph) | Probabilistic metrics | Audit |",
            "|---|---:|---:|---:|---:|---|---|",
            f"| {point_row['method']} | {point_row['rmse']:.4f} | {point_row['mae']:.4f} | "
            f"{point_row['rmse_speed']:.4f} | {point_row['mae_speed']:.4f} | N/A | {point_row['status']} |",
        ]
    )
    metrics = failed["metrics"]
    lines.extend(
        [
            "",
            "## Stability-gated condition",
            "",
            f"Decoupled + changing failed the 256-step canary (RMSE {metrics['rmse']:.3f}, Gaussian NLPD {metrics['gaussian_nlpd']:.1f}). "
            "Its complete stream was not launched. This is retained as a numerical-stability result, not a competitive score.",
            "",
            "| RFF seed | First RMSE > 5 step | Time | Predictive std | Beta natural norm | GP information norm |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in stability_rows:
        lines.append(
            f"| {row['model_seed']} | {row['onset_step']} | {row['onset_timestamp']} | "
            f"{float(row['mean_predictive_std']):.4f} | {float(row['beta_natural_norm']):.1f} | "
            f"{float(row['gp_information_norm']):.1f} |"
        )
    lines.extend(
        [
            "",
            "The failure repeats under two deterministic RFF seeds. Predictive standard deviation remains near 0.678 while beta and GP information-state norms grow sharply, so the observed failure is posterior-mean/state drift rather than variance collapse. This remains a stability diagnostic, not proof that joint coupling is universally necessary.",
            "",
            "## Interpretation boundary",
            "",
            "The persistence control uses only the legally revealed previous held-out value, y_H,t-1. "
            "KronHiPPO-STGP and the adapted IGNNK receive the same delayed held-out history; all methods have zero current-hidden reads before prediction.",
            "",
            "Joint + fixed-global and decoupled + fixed-global use a temporal basis constructed over the complete known timestamp grid but never use future target values. They are offline representation references, not online changing-coordinate methods.",
            "",
            "The gain is strongly associated with the changing-coordinate HiPPO representation, while this comparison does not isolate the benefit of the transfer approximation itself. "
            "A transfer-cost claim requires repeated changing-coordinate transfer versus all-seen recomputation in the same current basis.",
            "",
            "IGNNK has passed its official-core adapter gate and is reported separately because it is deterministic. KITS, OHSVGP and Maddox/OVC remain outside the result table until their remaining gates pass.",
        ]
    )
    (OUTPUT / "SEED0_FROZEN_BASELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_mechanism(rows: list[dict[str, object]]) -> None:
    chosen = [row for row in rows if row["method"] in {
        "KronHiPPO-STGP (joint + changing)",
        "Joint + fixed-global",
        "Decoupled + fixed-global",
    }]
    labels = [
        "Joint\nChanging",
        "Joint\nFixed-global",
        "Decoupled\nFixed-global",
    ]
    colors = ["#276FBF", "#8A8F98", "#C46A3A"]
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for axis, metric, title in zip(axes, ("rmse", "gaussian_nlpd"), ("RMSE", "Gaussian NLPD")):
        values = [float(row[metric]) for row in chosen]
        bars = axis.bar(labels, values, color=colors, width=0.65)
        axis.set_ylabel(title)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("PEMS-BAY seed-0 mechanism controls", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT / "fig_seed0_mechanism_controls.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "fig_seed0_mechanism_controls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    missing = [str(path) for _, _, path in SOURCES if not (path / "result.json").exists()]
    if missing:
        raise FileNotFoundError("Missing complete results: " + ", ".join(missing))
    rows = [load_row(*source) for source in SOURCES]
    point_row = load_point_row()
    stability_rows = load_stability_rows()
    write_csv(rows, OUTPUT / "seed0_frozen_baseline_table.csv")
    write_csv([point_row], OUTPUT / "seed0_deterministic_external_table.csv")
    write_csv(stability_rows, OUTPUT / "decoupled_changing_stability_audit.csv")
    write_report(rows, point_row, stability_rows)
    plot_mechanism(rows)
    if any(row["status"] != "complete_passed" for row in rows) or point_row["status"] != "complete_passed":
        raise SystemExit("At least one complete archive failed audit")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
