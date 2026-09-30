#!/usr/bin/env python3
"""Render Appendix Figure 7 after moving solver scaling to main Figure 3."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter


PAPER = Path(__file__).resolve().parents[1]
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRID = "#D9D9D9"


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.0, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    transfer = pd.read_csv(PAPER / "transfer_capacity_summary.csv").sort_values("mt")
    summary = pd.read_csv(PAPER / "experiment_b_summary.csv").sort_values(["mt", "variant"])
    mt = transfer["mt"].to_numpy(dtype=int)
    if mt.tolist() != [16, 32, 64, 128]:
        raise ValueError("Unexpected temporal capacities")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), constrained_layout=False,
                             gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    y = np.maximum(transfer["conditional_trace_ratio_mean"].to_numpy(), 1e-16)
    sd = transfer["conditional_trace_ratio_sd"].to_numpy()
    ax.errorbar(mt, y, yerr=sd, color=BLUE, marker="o", markersize=3.0,
                linewidth=1.2, capsize=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(mt, [str(v) for v in mt])
    ax.set_xlabel(r"Temporal capacity $M_t$")
    ax.set_ylabel("Conditional-projection discrepancy")
    ax.set_title("(a) One-step projection discrepancy", loc="left", fontsize=8.0, pad=3)

    ax = axes[1]
    for variant, color, ls, marker in (("conditional_transport", BLUE, "-", "o"),
                                       ("identity_reuse", ORANGE, "--", "s")):
        part = summary[summary["variant"] == variant].sort_values("mt")
        mean = np.maximum(part["predictive_gaussian_kl_to_reference_mean"].to_numpy(), 1e-16)
        sd = part["predictive_gaussian_kl_to_reference_sample_sd"].to_numpy()
        ax.errorbar(mt, mean, yerr=sd, color=color, linestyle=ls, marker=marker,
                    markersize=3.0, linewidth=1.2, capsize=2.0,
                    label="Conditional transport" if variant == "conditional_transport" else "Identity reuse")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(mt, [str(v) for v in mt])
    ax.set_xlabel(r"Temporal capacity $M_t$")
    ax.set_ylabel("Predictive Gaussian KL")
    ax.set_title("(b) Predictive KL", loc="left", fontsize=8.0, pad=3)
    ax.legend(loc="lower left", frameon=False, handlelength=1.7, borderpad=0.2,
              labelspacing=0.25, handletextpad=0.4)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", which="major", color=GRID, linewidth=0.45, alpha=0.25)
        ax.grid(axis="y", which="minor", visible=False)
        ax.tick_params(direction="out", length=2.4, width=0.55, pad=2)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())

    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.84)
    out_pdf = PAPER / "figures/mechanism_diagnostics_two_panel.pdf"
    out_png = PAPER / "figures/mechanism_diagnostics_two_panel.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Appendix Figure 7 PDF: {out_pdf}")
    print(f"Appendix Figure 7 PNG: {out_png}")


if __name__ == "__main__":
    main()
