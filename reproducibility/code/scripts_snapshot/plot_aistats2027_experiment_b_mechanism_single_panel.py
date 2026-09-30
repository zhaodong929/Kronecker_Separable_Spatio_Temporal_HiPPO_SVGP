#!/usr/bin/env python3
"""Render the compact single-panel Experiment-B mechanism figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd


RUN_RELATIVE = Path(
    "results/aistats2027_controlled_controls_20260916T163843Z/aggregate"
)


def locate_repo_root() -> Path:
    candidates = (
        Path(__file__).resolve().parents[1],
        Path("/home/zd929/projects/stvgp_kronecker"),
        Path(r"\\wsl.localhost\Ubuntu-24.04\home\zd929\projects\stvgp_kronecker"),
    )
    for candidate in candidates:
        if (candidate / RUN_RELATIVE / "experiment_b_blockwise_kl.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not locate the archived Experiment-B aggregate")


REPO_ROOT = locate_repo_root()
SOURCE = REPO_ROOT / RUN_RELATIVE / "experiment_b_blockwise_kl.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output/aistats2027_experiment_b_mechanism_single_panel"
VARIANTS = ("conditional_transport", "identity_reuse")
LABELS = {
    "conditional_transport": "Conditional transport",
    "identity_reuse": "Identity reuse",
}
COLORS = {"conditional_transport": "#0072B2", "identity_reuse": "#D55E00"}
LINESTYLES = {"conditional_transport": "-", "identity_reuse": "--"}
MARKERS = {"conditional_transport": "o", "identity_reuse": "s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def aggregate(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source, float_precision="round_trip")
    required = {
        "block_id",
        "mt",
        "reference_predictive_gaussian_kl",
        "seed",
        "variant",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")
    selected = frame.loc[
        (frame["mt"] == 128) & frame["variant"].isin(VARIANTS)
    ].copy()
    for variant in VARIANTS:
        part = selected.loc[selected["variant"] == variant]
        if sorted(part["seed"].astype(int).unique()) != [0, 1, 2, 3, 4]:
            raise ValueError(f"Expected five seeds for {variant}")
        counts = part.groupby("block_id")["seed"].nunique()
        if not (counts == 5).all():
            raise ValueError(f"Expected five splits at every block for {variant}")
        warmup = part.loc[part["block_id"] == 0, "reference_predictive_gaussian_kl"]
        if len(warmup) != 5 or not np.all(warmup.to_numpy() == 0.0):
            raise ValueError(f"Warm-up block is not five zero-KL rows for {variant}")
    plotted = selected.loc[selected["block_id"] != 0].copy()
    if (plotted["reference_predictive_gaussian_kl"] <= 0).any():
        raise ValueError("Non-warm-up KL must be positive on a log axis")
    audit = (
        plotted.groupby(["block_id", "variant"], sort=True)[
            "reference_predictive_gaussian_kl"
        ]
        .agg(mean="mean", sample_sd="std", splits="count")
        .reset_index()
    )
    if set(audit["block_id"]) != set(range(1, 19)):
        raise ValueError("Expected online blocks 1 through 18 after warm-up exclusion")
    if not (audit["splits"] == 5).all():
        raise ValueError("Every plotted point must aggregate five splits")
    if (audit["mean"] - audit["sample_sd"] <= 0).any():
        raise ValueError("Mean-minus-SD band is non-positive on the log axis")
    return audit


def print_audit(audit: pd.DataFrame) -> None:
    table = audit.pivot(
        index="block_id", columns="variant", values=["mean", "sample_sd"]
    )
    table.columns = [
        f"{LABELS[variant]} {stat}" for stat, variant in table.columns
    ]
    table = table.reset_index()
    print("\nExperiment-B single-panel audit: Mt=128, block 0 excluded")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.10g}"))


def render(audit: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.1,
            "axes.labelsize": 8.25,
            "xtick.labelsize": 7.25,
            "ytick.labelsize": 7.25,
            "legend.fontsize": 7.25,
            "legend.frameon": False,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.0,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axis = plt.subplots(figsize=(2.95, 1.85))
    handles = []
    for variant in VARIANTS:
        part = audit.loc[audit["variant"] == variant].sort_values("block_id")
        x = part["block_id"].to_numpy(dtype=int)
        mean = part["mean"].to_numpy(dtype=float)
        sd = part["sample_sd"].to_numpy(dtype=float)
        line = axis.plot(
            x,
            mean,
            color=COLORS[variant],
            linestyle=LINESTYLES[variant],
            linewidth=1.2,
            marker=MARKERS[variant],
            markevery=4,
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.7,
            label=LABELS[variant],
            zorder=3,
        )[0]
        axis.fill_between(
            x,
            mean - sd,
            mean + sd,
            color=COLORS[variant],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        handles.append(line)

    axis.set_yscale("log")
    axis.set_xlim(1, 18)
    axis.set_xticks([1, 6, 12, 18])
    axis.set_xlabel("Online block", labelpad=1.5)
    axis.set_ylabel("Predictive KL", labelpad=2.0)
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    axis.yaxis.set_minor_formatter(NullFormatter())
    lower = float((audit["mean"] - audit["sample_sd"]).min())
    upper = float((audit["mean"] + audit["sample_sd"]).max())
    axis.set_ylim(lower * 0.70, upper * 1.45)
    axis.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.4, alpha=0.14)
    axis.grid(axis="y", which="minor", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=2.5, width=0.6, pad=2.0)
    axis.legend(
        handles=handles,
        labels=[LABELS[variant] for variant in VARIANTS],
        loc="center right",
        bbox_to_anchor=(0.98, 0.53),
        handlelength=1.8,
        handletextpad=0.45,
        borderpad=0.25,
        labelspacing=0.25,
        frameon=False,
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.225, top=0.98)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "experiment_b_mechanism_single_panel.pdf"
    png = output_dir / "experiment_b_mechanism_single_panel.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=400)
    plt.close(fig)
    audit.to_csv(output_dir / "experiment_b_mechanism_single_panel_audit.csv", index=False, float_format="%.17g")
    print(f"\nVector PDF: {pdf}")
    print(f"PNG preview: {png}")


def main() -> None:
    args = parse_args()
    audit = aggregate(args.source)
    print_audit(audit)
    render(audit, args.output_dir)


if __name__ == "__main__":
    main()
