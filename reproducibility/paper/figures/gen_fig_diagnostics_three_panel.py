#!/usr/bin/env python3
"""Render the main transfer/solver diagnostic from locked aggregate CSVs."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TRANSFER = PAPER / "transfer_capacity_summary.csv"
EXPERIMENT_B = HERE / "experiment_b_summary.csv"
SOLVER = PAPER / "structured_dense_solver_benchmark.csv"

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
GRID = "#D9D9D9"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.75)


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.4,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.4,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "legend.fontsize": 8.2,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.8,
        "savefig.dpi": 400,
    })

    transfer = pd.read_csv(TRANSFER).sort_values("mt")
    experiment_b = pd.read_csv(EXPERIMENT_B).sort_values(["mt", "variant"])
    solver = pd.read_csv(SOLVER).sort_values("Ms")
    if transfer["mt"].astype(int).tolist() != [16, 32, 64, 128]:
        raise ValueError("Unexpected capacities in transfer summary")
    if solver["Ms"].astype(int).tolist() != [8, 16, 32, 48, 64]:
        raise ValueError("Unexpected capacities in solver benchmark")
    if sorted(experiment_b["mt"].astype(int).unique().tolist()) != [16, 32, 64, 128]:
        raise ValueError("Unexpected capacities in Experiment-B summary")
    if set(experiment_b["variant"]) != {"conditional_transport", "identity_reuse"}:
        raise ValueError("Experiment-B summary must contain both transfer variants")

    fig, axes = plt.subplots(1, 3, figsize=(10.15, 2.45), constrained_layout=True)
    x = transfer["mt"].to_numpy()

    axes[0].errorbar(
        x, np.maximum(transfer["conditional_trace_ratio_mean"], 1e-16),
        yerr=transfer["conditional_trace_ratio_sd"], color=GREEN,
        marker="s", capsize=2.0,
    )
    axes[0].set(xscale="log", yscale="log", xlabel=r"Temporal capacity $M_t$",
                ylabel="Conditional trace ratio")
    axes[0].set_xticks(x, labels=[str(v) for v in x])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_title("(a) One-step projection discrepancy", loc="left")

    conditional = experiment_b[experiment_b["variant"] == "conditional_transport"]
    identity = experiment_b[experiment_b["variant"] == "identity_reuse"]
    kl = np.maximum(conditional["predictive_gaussian_kl_to_reference_mean"].to_numpy(), 1e-16)
    axes[1].errorbar(
        x, kl, yerr=conditional["predictive_gaussian_kl_to_reference_sample_sd"],
        color=BLUE, marker="o", linestyle="-", capsize=2.0,
        label="Conditional transport",
    )
    identity_kl = np.maximum(identity["predictive_gaussian_kl_to_reference_mean"].to_numpy(), 1e-16)
    axes[1].errorbar(
        x, identity_kl, yerr=identity["predictive_gaussian_kl_to_reference_sample_sd"],
        color=ORANGE, marker="s", linestyle="--", capsize=2.0,
        label="Identity reuse",
    )
    axes[1].set(xscale="log", yscale="log", xlabel=r"Temporal capacity $M_t$",
                ylabel="Predictive Gaussian KL")
    axes[1].set_xticks(x, labels=[str(v) for v in x])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_title("(b) Predictive KL", loc="left")
    axes[1].legend(loc="lower left", fontsize=7.2, frameon=False)

    m = solver["Ms"].to_numpy()
    axes[2].plot(m, solver["dense_time_s"], color=ORANGE, marker="o", label="Dense")
    axes[2].plot(m, solver["sylvester_time_s"], color=BLUE, marker="s",
                 label="Schur--Sylvester")
    axes[2].set(yscale="log", xlabel=r"$M_s=M_t$", ylabel="Solve time (s)")
    axes[2].set_title("(c) Solver scaling", loc="left")
    axes[2].legend(loc="upper left")
    axes[2].annotate(r"$1036.7\times$", xy=(64, solver["sylvester_time_s"].iloc[-1]),
                     xytext=(-5, 22), textcoords="offset points", ha="right",
                     fontsize=8.2, fontweight="bold", color=BLUE)

    for ax in axes:
        style(ax)

    fig.savefig(HERE / "mechanism_diagnostics_three_panel.pdf", bbox_inches="tight")
    fig.savefig(HERE / "mechanism_diagnostics_three_panel.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
