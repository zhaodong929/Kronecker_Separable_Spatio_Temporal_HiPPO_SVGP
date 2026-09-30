#!/usr/bin/env python3
"""Generate figures for the 2026-07-13 weekly group-meeting report."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_READY = ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready"
XLAG = PAPER_READY / "xlag_lag_kernel_validation"
ICLR = PAPER_READY / "ICLR Formal experiment"
FOUR_STEP = ICLR / "four_step_priority_experiments"
BATCH = ICLR / "batch_structured_joint_hippo_stvgp"
OUT = ICLR / "weekly_group_meeting_2026-07-13"
FIG = OUT / "figures"


COLORS = {
    "neutral": "#66727A",
    "mean": "#B8C0C5",
    "routeb": "#277DA1",
    "routeb_light": "#86BBD8",
    "sparse": "#8D99AE",
    "exact": "#2A9D8F",
    "joint": "#E76F51",
    "accent": "#F4A261",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 300}),
    ]:
        fig.savefig(FIG / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def annotate_points(ax: plt.Axes, x: np.ndarray, y: np.ndarray, fmt: str = ".3f") -> None:
    for xv, yv in zip(x, y):
        ax.annotate(format(float(yv), fmt), (xv, yv), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.5)


def plot_lag_and_kernel() -> None:
    lag = pd.read_csv(XLAG / "lag_length_validation_results.csv")
    kernel = pd.read_csv(XLAG / "kernel_ablation_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
    ax = axes[0]
    ax.plot(lag["lag_length"], lag["val_rmse"], "o-", color=COLORS["neutral"], lw=1.6, ms=4, label="Validation")
    ax.plot(lag["lag_length"], lag["test_rmse"], "s-", color=COLORS["routeb"], lw=1.6, ms=4, label="Test")
    ax.axvline(10, color=COLORS["accent"], lw=1, ls="--")
    ax.text(9.72, 0.19035, "selected $L=10$", ha="right", va="bottom", color=COLORS["accent"], fontsize=7)
    ax.set_xlabel("X-lag length, $L$")
    ax.set_ylabel("RMSE")
    ax.set_title("a  Lag-length validation", loc="left", fontweight="bold")
    ax.set_xticks(lag["lag_length"])
    ax.grid(axis="y", color="#D9DEE2", lw=0.6)
    ax.legend(loc="lower left")

    show = kernel[
        ((kernel["method"] == "X-lag") & kernel["kernel_type"].isin(["rbf", "matern32"]))
        | ((kernel["method"] == "X-lag") & (kernel["kernel_type"] == "spectral_mixture") & (kernel["num_mixtures"] == 4))
        | (kernel["method"] == "recursive Y-lag")
        | (kernel["method"] == "oracle Y-lag")
    ].copy()
    labels = ["X-lag\nRBF", "X-lag\nMatern-\n3/2", "X-lag\nSM\n($Q=4$)", "Recursive\nY-lag", "Oracle\nY-lag"]
    values = show["test_rmse"].to_numpy()
    colors = [COLORS["routeb"], COLORS["sparse"], COLORS["accent"], COLORS["neutral"], COLORS["joint"]]
    x = np.arange(len(values))
    axes[1].bar(x, values, color=colors, width=0.72)
    axes[1].set_xticks(x, labels, fontsize=6.8)
    axes[1].set_ylabel("Test RMSE")
    axes[1].set_title("b  Kernel and rollout protocol", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#D9DEE2", lw=0.6)
    axes[1].set_ylim(0, 0.285)
    annotate_points(axes[1], x, values)
    save(fig, "lag_kernel_validation")


def plot_residual_benchmark() -> None:
    labels = [
        "X-lag mean only",
        "Matched sparse STVGP\n$M_t=8, M_s=64$",
        "Online structured-joint\nHiPPO-STVGP $8,64$",
        "Exact separable STVGP\ntwo-stage",
    ]
    rmse = np.array([0.3305, 0.2382, 0.1899, 0.0843])
    nll = np.array([0.3027, 0.1212, -0.0350, -1.0920])
    colors = [COLORS["mean"], COLORS["sparse"], COLORS["routeb"], COLORS["exact"]]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), constrained_layout=True)
    axes[0].barh(y, rmse, color=colors, height=0.68)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Test RMSE (lower is better)")
    axes[0].set_title("a  Residual-model comparison", loc="left", fontweight="bold")
    axes[0].grid(axis="x", color="#D9DEE2", lw=0.6)
    for yi, val in zip(y, rmse):
        axes[0].text(val + 0.006, yi, f"{val:.4f}", va="center", fontsize=7)

    axes[1].barh(y, nll, color=colors, height=0.68)
    axes[1].axvline(0, color="#69757D", lw=0.8)
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Test NLL (lower is better)")
    axes[1].set_title("b  Predictive density", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#D9DEE2", lw=0.6)
    for yi, val in zip(y, nll):
        offset = 0.035 if val >= 0 else -0.035
        axes[1].text(val + offset, yi, f"{val:.4f}", va="center", ha="left" if val >= 0 else "right", fontsize=7)
    save(fig, "residual_benchmark")


def plot_capacity_path() -> None:
    df = pd.read_csv(FOUR_STEP / "step1_routeb_capacity_summary.csv")
    labels = ["8,64", "16,64", "16,128", "32,128", "32,256"]
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), constrained_layout=True)
    axes[0].plot(x, df["rmse"], "o-", color=COLORS["routeb"], lw=1.8, ms=4.5)
    axes[0].axhline(0.0843, color=COLORS["exact"], ls="--", lw=1, label="Exact two-stage")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("a  Accuracy", loc="left", fontweight="bold")
    axes[0].legend(fontsize=6.5)
    annotate_points(axes[0], x, df["rmse"].to_numpy(), ".3f")

    axes[1].plot(x, df["nll"], "o-", color=COLORS["routeb"], lw=1.8, ms=4.5)
    axes[1].axhline(-1.0920, color=COLORS["exact"], ls="--", lw=1)
    axes[1].set_ylabel("NLL")
    axes[1].set_title("b  Calibration", loc="left", fontweight="bold")
    annotate_points(axes[1], x, df["nll"].to_numpy(), ".3f")

    axes[2].plot(x, df["runtime_per_block"], "o-", color=COLORS["accent"], lw=1.8, ms=4.5)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Seconds per block (log scale)")
    axes[2].set_title("c  Compute cost", loc="left", fontweight="bold")

    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_xlabel("$M_t, M_s$")
        ax.grid(axis="y", color="#D9DEE2", lw=0.6)
    save(fig, "capacity_path")


def plot_batch_and_joint() -> None:
    batch = pd.read_csv(BATCH / "batch_structured_joint_comparison.csv")
    selected = batch[batch["method"].isin([
        "Online Route B block-average Mt=32 Ms=128",
        "Online Route B final block Mt=32 Ms=128",
        "Batch structured-joint HiPPO-STVGP Mt=32 Ms=128",
        "Exact STVGP two-stage upper bound",
    ])].copy()
    order = [
        "Online Route B block-average Mt=32 Ms=128",
        "Online Route B final block Mt=32 Ms=128",
        "Batch structured-joint HiPPO-STVGP Mt=32 Ms=128",
        "Exact STVGP two-stage upper bound",
    ]
    selected["method"] = pd.Categorical(selected["method"], order, ordered=True)
    selected = selected.sort_values("method")
    labels = ["Online\nblock average", "Online\nfinal block", "Batch\nfull history", "Exact STVGP\nupper bound"]
    colors = [COLORS["routeb_light"], COLORS["routeb"], COLORS["accent"], COLORS["exact"]]
    x = np.arange(len(selected))

    joint = pd.read_csv(FOUR_STEP / "step3_joint_stvgp_summary.csv")
    jx = np.arange(len(joint))
    jlabels = ["Two-stage\nexact STVGP", "Joint $\\beta$-GP\nexact STVGP"]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), constrained_layout=True)
    axes[0].bar(x, selected["rmse"], color=colors, width=0.7)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("RMSE")
    axes[0].set_ylim(0, 0.18)
    axes[0].set_title("a  Online versus batch", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#D9DEE2", lw=0.6)
    annotate_points(axes[0], x, selected["rmse"].to_numpy(), ".4f")
    axes[0].annotate(
        "fair final-state comparison",
        xy=(1.5, 0.151),
        xytext=(1.5, 0.174),
        ha="center",
        fontsize=6.5,
        arrowprops={"arrowstyle": "-[,widthB=2.7", "lw": 0.8, "color": COLORS["neutral"]},
    )

    bars = axes[1].bar(jx, joint["rmse"], color=[COLORS["exact"], COLORS["joint"]], width=0.62)
    axes[1].set_xticks(jx, jlabels)
    axes[1].set_ylabel("Final-block RMSE")
    axes[1].set_ylim(0, 0.1)
    axes[1].set_title("b  Benefit of joint coupling", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#D9DEE2", lw=0.6)
    annotate_points(axes[1], jx, joint["rmse"].to_numpy(), ".4f")
    improvement = 100 * (joint.iloc[0]["rmse"] - joint.iloc[1]["rmse"]) / joint.iloc[0]["rmse"]
    axes[1].text(0.5, 0.094, f"{improvement:.1f}% RMSE reduction", ha="center", color=COLORS["joint"], fontsize=7, fontweight="bold")
    for bar in bars:
        bar.set_edgecolor("white")
    save(fig, "batch_online_joint")


def export_source_data() -> None:
    rows = [
        {"figure": "residual_benchmark", "setting": "X-lag mean only", "rmse": 0.3305, "nll": 0.3027},
        {"figure": "residual_benchmark", "setting": "Matched sparse STVGP Mt8 Ms64", "rmse": 0.2382, "nll": 0.1212},
        {"figure": "residual_benchmark", "setting": "Online structured-joint HiPPO-STVGP Mt8 Ms64", "rmse": 0.1899, "nll": -0.0350},
        {"figure": "residual_benchmark", "setting": "Exact separable STVGP two-stage", "rmse": 0.0843, "nll": -1.0920},
        {"figure": "residual_benchmark", "setting": "Joint exact STVGP", "rmse": 0.0695, "nll": -1.2115},
    ]
    pd.DataFrame(rows).to_csv(OUT / "weekly_report_source_data.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_lag_and_kernel()
    plot_residual_benchmark()
    plot_capacity_path()
    plot_batch_and_joint()
    export_source_data()
    print(OUT)


if __name__ == "__main__":
    main()
