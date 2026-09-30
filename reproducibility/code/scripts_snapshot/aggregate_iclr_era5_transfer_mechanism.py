#!/usr/bin/env python3
"""Aggregate the ICLR 2027 mechanism suite and draw the 2x2 paper figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODES = (
    "structured_changing",
    "structured_fixed",
    "mean_field_changing",
    "zero_cross_changing",
)
METRICS = ("rmse", "crps", "nll", "ece", "coverage90")
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#4D4D4D",
}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def load_result(root: Path, mode: str, mt: int, seed: int) -> dict[str, Any]:
    path = root / f"{mode}_ms128_mt{mt}_seed{seed}" / "result.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Incomplete result: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--long-log-root", type=Path, required=True)
    parser.add_argument("--solver-csv", type=Path, required=True)
    parser.add_argument("--paper-figure-dir", type=Path)
    args = parser.parse_args()
    output = args.result_root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)

    ablation_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    main_by_seed: dict[int, dict[str, float]] = {}
    for mode in MODES:
        values = {metric: [] for metric in METRICS}
        for seed in range(5):
            result = load_result(args.result_root, mode, 128, seed)
            row = {"mode": mode, "seed": seed, **result["aggregate"]}
            per_seed_rows.append(row)
            for metric in METRICS:
                values[metric].append(float(result["aggregate"][metric]))
            if mode == "structured_changing":
                main_by_seed[seed] = {
                    metric: float(result["aggregate"][metric]) for metric in METRICS
                }
        aggregate: dict[str, Any] = {"mode": mode, "seeds": 5}
        for metric in METRICS:
            aggregate[f"{metric}_mean"] = float(np.mean(values[metric]))
            aggregate[f"{metric}_sd"] = sample_sd(values[metric])
        ablation_rows.append(aggregate)

    for row in per_seed_rows:
        base = main_by_seed[int(row["seed"])]
        for metric in METRICS:
            row[f"delta_{metric}_vs_structured_changing"] = float(row[metric]) - base[metric]
    write_csv(ablation_rows, output / "ablation_summary.csv")
    write_csv(per_seed_rows, output / "ablation_per_seed.csv")

    capacity_seed_rows: list[dict[str, Any]] = []
    for mt in (16, 32, 64, 128):
        for seed in range(5):
            result = load_result(args.result_root, "structured_changing", mt, seed)
            block_path = (
                args.result_root
                / f"structured_changing_ms128_mt{mt}_seed{seed}"
                / "blocks.csv"
            )
            blocks = pd.read_csv(block_path)
            steady = blocks[blocks["block_id"] > 0]
            capacity_seed_rows.append(
                {
                    "mt": mt,
                    "seed": seed,
                    "conditional_trace_ratio": float(steady["conditional_trace_ratio"].mean()),
                    "natural_R_beta_u_relerr": float(steady["reference_R_beta_u_relerr"].mean()),
                    "natural_B_relerr": float(steady["reference_B_temporal_relerr"].mean()),
                    "posterior_mean_relerr": float(steady["reference_posterior_mean_relerr"].mean()),
                    "posterior_cov_relerr": float(steady["reference_posterior_cov_relerr"].mean()),
                    "predictive_mean_relerr": float(steady["reference_predictive_mean_relerr"].mean()),
                    "predictive_variance_relerr": float(steady["reference_predictive_variance_relerr"].mean()),
                    "predictive_gaussian_kl": float(steady["reference_predictive_gaussian_kl"].mean()),
                    "rmse": float(result["aggregate"]["rmse"]),
                }
            )
    write_csv(capacity_seed_rows, output / "transfer_capacity_per_seed.csv")
    capacity = pd.DataFrame(capacity_seed_rows)
    capacity_rows: list[dict[str, Any]] = []
    gap_means: list[float] = []
    for mt, group in capacity.groupby("mt", sort=True):
        row: dict[str, Any] = {"mt": int(mt), "seeds": len(group)}
        for column in capacity.columns:
            if column in {"mt", "seed"}:
                continue
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd"] = float(group[column].std(ddof=1))
        gap_means.append(row["predictive_gaussian_kl_mean"])
        capacity_rows.append(row)
    write_csv(capacity_rows, output / "transfer_capacity_summary.csv")

    slopes = []
    for seed, group in capacity.groupby("seed"):
        ordered = group.sort_values("mt")
        gap = np.maximum(ordered["predictive_gaussian_kl"].to_numpy(), 1e-16)
        slope = float(np.polyfit(np.log(ordered["mt"].to_numpy()), np.log(gap), 1)[0])
        slopes.append({"seed": int(seed), "log_gap_slope": slope})
    write_csv(slopes, output / "transfer_capacity_slopes.csv")
    monotone = bool(np.all(np.diff(np.asarray(gap_means)) <= 0.0))
    negative_slopes = sum(row["log_gap_slope"] < 0.0 for row in slopes)
    transfer_claim_allowed = monotone and negative_slopes >= 4

    long_frames = []
    state_constant = True
    for seed in range(5):
        frame = pd.read_csv(args.long_log_root / f"seed{seed}" / "blocks.csv")
        if len(frame) != 171:
            raise ValueError(f"seed{seed} has {len(frame)} long-stream blocks, expected 171")
        frame["seed"] = seed
        state_constant &= frame["persistent_state_bytes"].nunique() == 1
        long_frames.append(frame)
    long_data = pd.concat(long_frames, ignore_index=True)
    history_summary = (
        long_data.groupby("block_id")
        .agg(
            persistent_state_bytes_mean=("persistent_state_bytes", "mean"),
            persistent_state_bytes_sd=("persistent_state_bytes", "std"),
            update_seconds_mean=("update_seconds", "mean"),
            update_seconds_sd=("update_seconds", "std"),
        )
        .reset_index()
    )
    history_summary.to_csv(output / "history_scaling_summary.csv", index=False)

    solver = pd.read_csv(args.solver_csv)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(solver["Ms"], solver["dense_time_s"], "o-", color=COLORS["vermillion"], label="Dense")
    ax.plot(solver["Ms"], solver["sylvester_time_s"], "s-", color=COLORS["blue"], label="Sylvester")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_s=M_t$")
    ax.set_ylabel("Solve time (s)")
    ax.set_title("(a) Solver scaling", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[0, 1]
    ax.plot(solver["Ms"], solver["rel_err_solution"], "o-", color=COLORS["green"], label="Solution")
    ax.plot(solver["Ms"], solver["rel_err_qform"], "s--", color=COLORS["purple"], label="Quadratic form")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_s=M_t$")
    ax.set_ylabel("Relative error")
    ax.set_title("(b) Dense agreement", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    cap = pd.DataFrame(capacity_rows).sort_values("mt")
    ax = axes[1, 0]
    x = cap["mt"].to_numpy()
    predictive_gap = np.maximum(
        cap["predictive_gaussian_kl_mean"].to_numpy(), 1e-16
    )
    predictive_sd = cap["predictive_gaussian_kl_sd"].to_numpy()
    conditional_gap = np.maximum(
        cap["conditional_trace_ratio_mean"].to_numpy(), 1e-16
    )
    conditional_sd = cap["conditional_trace_ratio_sd"].to_numpy()
    ax.errorbar(
        x,
        conditional_gap,
        yerr=conditional_sd,
        marker="s",
        linestyle="--",
        capsize=2.5,
        color=COLORS["green"],
        label=r"Conditional trace ratio",
    )
    ax.errorbar(
        x,
        predictive_gap,
        yerr=predictive_sd,
        marker="o",
        capsize=2.5,
        color=COLORS["blue"],
        label=r"Predictive Gaussian KL",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, labels=[str(int(value)) for value in x])
    ax.set_xlabel(r"Temporal capacity $M_t$")
    ax.set_ylabel("Dimensionless gap")
    ax.set_title("(c) Transfer capacity", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    ax = axes[1, 1]
    blocks = history_summary["block_id"].to_numpy()
    memory = history_summary["persistent_state_bytes_mean"].to_numpy() / 1024.0**2
    update_ms = history_summary["update_seconds_mean"].to_numpy() * 1000.0
    memory_line = ax.plot(blocks, memory, color=COLORS["blue"], label="State memory")[0]
    ax.set_xlabel("Online block")
    ax.set_ylabel("Persistent state (MiB)", color=COLORS["blue"])
    ax.tick_params(axis="y", labelcolor=COLORS["blue"])
    twin = ax.twinx()
    update_line = twin.plot(blocks[1:], update_ms[1:], color=COLORS["orange"], alpha=0.85, label="Update time")[0]
    twin.scatter([blocks[0]], [update_ms[0]], color=COLORS["gray"], s=16, zorder=3, label="Warm-up")
    twin.set_ylabel("Update time (ms)", color=COLORS["orange"])
    twin.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax.set_title("(d) History-length scaling", loc="left", fontweight="bold")
    ax.legend([memory_line, update_line], ["State memory", "Update time"], frameon=False, fontsize=8, loc="upper right")

    for axis in axes.ravel():
        axis.spines["top"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    pdf = output / "iclr2027_mechanism_2x2.pdf"
    png = output / "iclr2027_mechanism_2x2.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if args.paper_figure_dir is not None:
        args.paper_figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, args.paper_figure_dir / pdf.name)
        shutil.copy2(png, args.paper_figure_dir / png.name)

    report = {
        "status": "complete",
        "five_seed_cells": 7,
        "ablation_rows": ablation_rows,
        "transfer_capacity": {
            "aggregate_gap_monotone_nonincreasing": monotone,
            "negative_seed_slopes": negative_slopes,
            "claim_transfer_error_decreases_allowed": transfer_claim_allowed,
            "slopes": slopes,
        },
        "history_scaling": {
            "five_seeds": True,
            "blocks_per_seed": 171,
            "state_bytes_constant_within_each_seed": state_constant,
            "warmup_block_excluded_from_steady_curve": True,
        },
        "solver_source": str(args.solver_csv.resolve()),
        "long_log_source": str(args.long_log_root.resolve()),
        "figure_pdf": str(pdf.resolve()),
        "figure_png": str(png.resolve()),
    }
    (output / "aggregate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
