#!/usr/bin/env python3
"""Reformat the audited transfer diagnostics into separate panels.

No experimental values are generated here. Both panels are read from the
machine-readable five-seed aggregate. The solver panel remains the existing
audited vector figure and is cropped only by LaTeX at inclusion time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent
SUMMARY = ROOT / "transfer_capacity_summary.csv"

BLUE = "#0072B2"
GREEN = "#009E73"
GRID = "#D9D9D9"


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)


def save_panel(name: str, y: np.ndarray, sd: np.ndarray, color: str,
               marker: str, title: str, ylabel: str,
               annotate_endpoint: bool = False,
               annotate_nonmonotone: bool = False) -> None:
    x = np.array([16, 32, 64, 128])
    fig, ax = plt.subplots(figsize=(3.0, 2.55), constrained_layout=True)
    ax.errorbar(x, y, yerr=sd, marker=marker, linestyle="-", capsize=2.2,
                linewidth=1.8, markersize=4.5, color=color)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, labels=[str(value) for value in x])
    ax.set_xlabel(r"Temporal capacity $M_t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    if annotate_nonmonotone:
        ax.annotate("non-monotone", xy=(32, y[1]), xytext=(3, 8),
                    textcoords="offset points", fontsize=7.4, color="#4D4D4D")
    if annotate_endpoint:
        ax.annotate(r"$1.27\times10^{-4}$", xy=(128, y[-1]),
                    xytext=(-4, -13), textcoords="offset points", ha="right",
                    fontsize=7.4, color=color)
    style_axis(ax)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.dpi": 300,
    })
    cap = pd.read_csv(SUMMARY).sort_values("mt")
    if cap["mt"].astype(int).tolist() != [16, 32, 64, 128]:
        raise ValueError("Unexpected temporal capacities in audited summary")
    save_panel(
        "one_step_projection_panel",
        np.maximum(cap["conditional_trace_ratio_mean"].to_numpy(), 1e-16),
        cap["conditional_trace_ratio_sd"].to_numpy(),
        GREEN, "s", "(a) One-step projection discrepancy",
        "Conditional trace ratio",
    )
    save_panel(
        "accumulated_transfer_panel",
        np.maximum(cap["predictive_gaussian_kl_mean"].to_numpy(), 1e-16),
        cap["predictive_gaussian_kl_sd"].to_numpy(),
        BLUE, "o", "(b) Accumulated transfer discrepancy",
        "Predictive Gaussian KL",
        annotate_endpoint=True,
        annotate_nonmonotone=True,
    )


if __name__ == "__main__":
    main()
