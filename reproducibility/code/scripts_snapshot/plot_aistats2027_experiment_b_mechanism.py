#!/usr/bin/env python3
"""Plot the completed Experiment-B mechanism diagnostic from archived CSVs.

The script performs no experiment execution and does not alter source values.
Panel (a) aggregates the five archived seeds at each online block using the
sample standard deviation. Panel (b) reads the archived aggregate RMSE fields
directly from ``experiment_b_summary.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd


RUN_RELATIVE = (
    Path("results")
    / "aistats2027_controlled_controls_20260916T163843Z"
    / "aggregate"
)


def locate_repo_root() -> Path:
    candidates = (
        Path(__file__).resolve().parents[1],
        Path("/home/zd929/projects/stvgp_kronecker"),
        Path(r"\\wsl.localhost\Ubuntu-24.04\home\zd929\projects\stvgp_kronecker"),
    )
    for candidate in candidates:
        if (candidate / RUN_RELATIVE / "experiment_b_summary.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not locate the archived Experiment-B aggregate directory")


REPO_ROOT = locate_repo_root()
AGGREGATE_DIR = (
    REPO_ROOT
    / "results"
    / "aistats2027_controlled_controls_20260916T163843Z"
    / "aggregate"
)
DEFAULT_BLOCKWISE = AGGREGATE_DIR / "experiment_b_blockwise_kl.csv"
DEFAULT_SUMMARY = AGGREGATE_DIR / "experiment_b_summary.csv"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "aistats2027_experiment_b_mechanism"
)

VARIANTS = ("conditional_transport", "identity_reuse")
LABELS = {
    "conditional_transport": "Conditional transport",
    "identity_reuse": "Identity reuse",
}
COLORS = {
    "conditional_transport": "#0072B2",
    "identity_reuse": "#D55E00",
}
MARKERS = {
    "conditional_transport": "o",
    "identity_reuse": "s",
}
CAPACITIES = (16, 32, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blockwise", type=Path, default=DEFAULT_BLOCKWISE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_blockwise(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source, float_precision="round_trip")
    require_columns(
        frame,
        {
            "block_id",
            "mt",
            "reference_predictive_gaussian_kl",
            "seed",
            "variant",
        },
        source,
    )
    selected = frame.loc[
        (frame["mt"] == 128) & frame["variant"].isin(VARIANTS)
    ].copy()
    if set(selected["variant"]) != set(VARIANTS):
        raise ValueError("Mt=128 blockwise data do not contain both variants")
    for variant in VARIANTS:
        part = selected.loc[selected["variant"] == variant]
        seeds = sorted(part["seed"].astype(int).unique().tolist())
        if seeds != [0, 1, 2, 3, 4]:
            raise ValueError(f"Unexpected seeds for {variant}: {seeds}")
        counts = part.groupby("block_id", sort=True)["seed"].nunique()
        if not (counts == 5).all():
            raise ValueError(f"Every {variant} block must contain five seeds")
        if part.duplicated(["block_id", "seed"]).any():
            raise ValueError(f"Duplicate block/seed rows for {variant}")
        warmup = part.loc[part["block_id"] == 0, "reference_predictive_gaussian_kl"]
        if len(warmup) != 5 or not np.all(warmup.to_numpy(dtype=float) == 0.0):
            raise ValueError(f"Expected five zero-KL warm-up rows for {variant}")

    plotted = selected.loc[selected["block_id"] != 0]
    if (plotted["reference_predictive_gaussian_kl"] <= 0).any():
        raise ValueError("Non-warm-up KL values must be positive for logarithmic plotting")
    audit = (
        plotted.groupby(["block_id", "variant"], sort=True)[
            "reference_predictive_gaussian_kl"
        ]
        .agg(mean="mean", sample_sd="std", seeds="count")
        .reset_index()
    )
    if not (audit["seeds"] == 5).all():
        raise ValueError("Unexpected seed count in blockwise aggregation")
    if (audit["mean"] - audit["sample_sd"] <= 0).any():
        raise ValueError("A mean-minus-SD band reaches a non-positive value on the log axis")
    return audit


def load_summary(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source, float_precision="round_trip")
    require_columns(
        frame,
        {"mt", "rmse_mean", "rmse_sample_sd", "splits", "variant"},
        source,
    )
    selected = frame.loc[
        frame["variant"].isin(VARIANTS) & frame["mt"].isin(CAPACITIES),
        ["mt", "variant", "rmse_mean", "rmse_sample_sd", "splits"],
    ].copy()
    if len(selected) != len(CAPACITIES) * len(VARIANTS):
        raise ValueError("Expected exactly eight RMSE summary rows")
    if selected.duplicated(["mt", "variant"]).any():
        raise ValueError("Duplicate capacity/variant rows in RMSE summary")
    if not (selected["splits"] == 5).all():
        raise ValueError("Every RMSE summary row must aggregate five splits")
    for variant in VARIANTS:
        capacities = sorted(
            selected.loc[selected["variant"] == variant, "mt"].astype(int).tolist()
        )
        if capacities != list(CAPACITIES):
            raise ValueError(f"Unexpected capacities for {variant}: {capacities}")
    return selected.sort_values(["mt", "variant"]).reset_index(drop=True)


def print_audit(blockwise: pd.DataFrame, summary: pd.DataFrame) -> None:
    panel_a = blockwise.pivot(
        index="block_id", columns="variant", values=["mean", "sample_sd"]
    )
    panel_a.columns = [
        f"{LABELS[variant]} {stat}" for stat, variant in panel_a.columns
    ]
    panel_a = panel_a.reset_index()
    panel_b = summary.pivot(
        index="mt", columns="variant", values=["rmse_mean", "rmse_sample_sd"]
    )
    panel_b.columns = [
        f"{LABELS[variant]} {stat}" for stat, variant in panel_b.columns
    ]
    panel_b = panel_b.reset_index()

    print("\nPanel (a) plotted values: Mt=128, five-seed mean +/- sample SD")
    print(panel_a.to_string(index=False, float_format=lambda value: f"{value:.10g}"))
    print("\nPanel (b) plotted values: archived RMSE mean +/- sample SD")
    print(panel_b.to_string(index=False, float_format=lambda value: f"{value:.10g}"))


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.0, width=0.7, pad=2.5)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.38)


def plot(blockwise: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.6,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "lines.linewidth": 1.55,
            "lines.markersize": 3.6,
            "axes.linewidth": 0.7,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.25, 4.35))
    figure_handles = []

    for variant in VARIANTS:
        part = blockwise.loc[blockwise["variant"] == variant].sort_values("block_id")
        x = part["block_id"].to_numpy(dtype=int)
        mean = part["mean"].to_numpy(dtype=float)
        sd = part["sample_sd"].to_numpy(dtype=float)
        line = axes[0].plot(
            x,
            mean,
            color=COLORS[variant],
            marker=MARKERS[variant],
            markevery=2,
            markerfacecolor="white",
            markeredgewidth=0.8,
            linestyle="-",
            label=LABELS[variant],
            zorder=3,
        )[0]
        axes[0].fill_between(
            x,
            mean - sd,
            mean + sd,
            color=COLORS[variant],
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        figure_handles.append(line)

    axes[0].set_yscale("log")
    axes[0].set_xlim(1, 18)
    axes[0].set_xticks([1, 6, 12, 18])
    axes[0].set_xlabel("Online block")
    axes[0].set_ylabel(
        "Predictive Gaussian KL to\nrecomputed current-basis reference"
    )
    axes[0].set_title("(a) Distributional mismatch", loc="left", pad=3.0)
    axes[0].yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    style_axis(axes[0])

    for variant in VARIANTS:
        part = summary.loc[summary["variant"] == variant].sort_values("mt")
        x = part["mt"].to_numpy(dtype=int)
        mean = part["rmse_mean"].to_numpy(dtype=float)
        sd = part["rmse_sample_sd"].to_numpy(dtype=float)
        axes[1].errorbar(
            x,
            mean,
            yerr=sd,
            color=COLORS[variant],
            marker=MARKERS[variant],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linestyle="-",
            capsize=2.2,
            capthick=0.8,
            elinewidth=0.9,
            zorder=3,
        )

    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(CAPACITIES, labels=[str(value) for value in CAPACITIES])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel(r"Temporal capacity $M_t$")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("(b) Predictive consequence", loc="left", pad=3.0)
    style_axis(axes[1])

    fig.legend(
        figure_handles,
        [LABELS[variant] for variant in VARIANTS],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=2,
        handlelength=2.2,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.225, right=0.975, bottom=0.095, top=0.925, hspace=0.52)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "experiment_b_mechanism.pdf"
    png_path = output_dir / "experiment_b_mechanism.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.025, dpi=400)
    plt.close(fig)

    blockwise.to_csv(
        output_dir / "experiment_b_mechanism_panel_a_audit.csv",
        index=False,
        float_format="%.17g",
    )
    summary.to_csv(
        output_dir / "experiment_b_mechanism_panel_b_audit.csv",
        index=False,
        float_format="%.17g",
    )
    print(f"\nVector PDF: {pdf_path}")
    print(f"PNG preview: {png_path}")


def main() -> None:
    args = parse_args()
    blockwise = load_blockwise(args.blockwise)
    summary = load_summary(args.summary)
    print_audit(blockwise, summary)
    plot(blockwise, summary, args.output_dir)


if __name__ == "__main__":
    main()
