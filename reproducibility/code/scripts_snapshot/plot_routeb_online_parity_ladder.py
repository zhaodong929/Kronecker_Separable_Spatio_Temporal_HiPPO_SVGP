#!/usr/bin/env python3
"""Plot blockwise diagnostics from the Route-B online parity ladder."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "global_fixed": "#0072B2",
    "cumulative_changing": "#D55E00",
    "local_block": "#009E73",
}
LABELS = {
    "global_fixed": "Fixed global basis",
    "cumulative_changing": "Cumulative changing basis",
    "local_block": "Local reset basis",
}


def grouped(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        data.groupby(["basis_mode", "block_id"], as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
        .fillna(0.0)
    )


def draw_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    *,
    log_scale: bool = False,
) -> None:
    summary = grouped(data, metric)
    for mode in COLORS:
        subset = summary[summary["basis_mode"] == mode]
        x = subset["block_id"].to_numpy(dtype=float) + 1.0
        mean = subset["mean"].to_numpy(dtype=float)
        std = subset["std"].to_numpy(dtype=float)
        if log_scale:
            mean = np.maximum(mean, 1e-16)
        ax.plot(x, mean, color=COLORS[mode], linewidth=2.0, label=LABELS[mode])
        lower = mean - std
        upper = mean + std
        if log_scale:
            lower = np.maximum(lower, 1e-16)
        if not log_scale:
            ax.fill_between(x, lower, upper, color=COLORS[mode], alpha=0.16, linewidth=0)
    ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.55)
    ax.set_xlabel("Online block")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, int(data["block_id"].max()) + 1)
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.75)
    if log_scale:
        ax.set_yscale("log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
    draw_metric(
        axes[0, 0], data, "relative_state_mean_error", r"Relative state-mean error", log_scale=True
    )
    draw_metric(
        axes[0, 1],
        data,
        "relative_prediction_mean_error",
        r"Relative prediction-mean error",
        log_scale=True,
    )
    draw_metric(axes[1, 0], data, "online_minus_batch_rmse", "Online - batch RMSE")
    draw_metric(axes[1, 1], data, "online_minus_batch_nll", "Online - batch NLL")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("KronHiPPO-STGP online parity ladder", fontsize=14, y=1.095)
    fig.savefig(args.outdir / "online_parity_ladder.png", bbox_inches="tight")
    fig.savefig(args.outdir / "online_parity_ladder.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    for mode in COLORS:
        subset = data[data["basis_mode"] == mode]
        updates = subset.groupby("block_id")["online_update_seconds"].mean()
        batches = subset.groupby("block_id")["batch_recompute_seconds"].mean()
        x = updates.index.to_numpy(dtype=float) + 1.0
        axes[0].plot(x, updates, color=COLORS[mode], linewidth=2.0, label=f"{LABELS[mode]}: online")
        axes[0].plot(x, batches, color=COLORS[mode], linewidth=1.5, linestyle="--", label=f"{LABELS[mode]}: batch")
        state_mib = subset.groupby("block_id")["persistent_state_bytes"].mean() / (1024.0**2)
        if mode == "global_fixed":
            axes[1].plot(x, state_mib, color="#222222", linewidth=2.0)
            axes[1].annotate(
                f"All modes: {float(state_mib.iloc[-1]):.2f} MiB",
                xy=(x[-1], float(state_mib.iloc[-1])),
                xytext=(-8, 12),
                textcoords="offset points",
                ha="right",
                fontsize=9,
            )
    axes[0].set_xlabel("Online block")
    axes[0].set_ylabel("Posterior update time (s)")
    axes[0].grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axes[0].legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
    )
    axes[1].set_xlabel("Online block")
    axes[1].set_ylabel("Persistent model state (MiB)")
    axes[1].grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.75)
    fig.savefig(args.outdir / "online_parity_runtime_state.png", bbox_inches="tight")
    fig.savefig(args.outdir / "online_parity_runtime_state.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
