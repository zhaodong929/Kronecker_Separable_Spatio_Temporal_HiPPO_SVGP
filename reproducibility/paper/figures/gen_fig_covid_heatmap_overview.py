#!/usr/bin/env python3
"""Render a readable COVID jurisdiction-by-week protocol heatmap."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(r"\\wsl.localhost\Ubuntu-24.04\home\zd929\projects\stvgp_kronecker")
PROTOCOL = REPO / "data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed5"
OUT = Path(__file__).resolve().parent


def main() -> None:
    metadata = json.loads((PROTOCOL / "protocol.json").read_text(encoding="utf-8"))
    with np.load(PROTOCOL / "protocol.npz") as data:
        standardized = np.vstack([data["calibration_y"], data["stream_y"]]).astype(float)
    dates = np.asarray(metadata["raw_dates"], dtype="datetime64[D]")
    names = np.asarray(metadata["location_names"], dtype=str)
    standardization = metadata["target_standardization"]
    target = standardized * float(standardization["scale"]) + float(standardization["mean"])
    # Stable display order: neighbouring trajectories are grouped by their
    # full-stream profile, while labels are deliberately sparse.
    profile = target - target.mean(axis=0, keepdims=True)
    order = np.argsort(np.argsort(profile, axis=1).mean(axis=0))
    ordered_names = names[order]
    ordered_target = target[:, order]

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.2,
    })
    # Give the overview enough vertical room to remain legible when stacked
    # above the trajectory panels in the appendix composite.
    fig, ax = plt.subplots(figsize=(6.6, 2.90))
    vmax = float(np.quantile(ordered_target, 0.99))
    image = ax.imshow(ordered_target.T, aspect="auto", interpolation="nearest",
                      cmap="viridis", vmin=0.0, vmax=vmax)
    ax.axvspan(-0.5, 51.5, color="white", alpha=0.08, linewidth=0)
    ax.axvline(51.5, color="#202020", linewidth=0.85)
    ax.annotate("", xy=(-0.5, 1.10), xytext=(51.5, 1.10),
                xycoords=("data", "axes fraction"),
                arrowprops={"arrowstyle": "<->", "color": "#3c3c3c", "lw": 0.65})
    ax.annotate("", xy=(51.5, 1.10), xytext=(194.5, 1.10),
                xycoords=("data", "axes fraction"),
                arrowprops={"arrowstyle": "<->", "color": "#3c3c3c", "lw": 0.65})
    ax.text(25.5, 1.125, "Task 1: 52-week calibration", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=7.3, color="#303030")
    ax.text(123.0, 1.125, "Strict-online stream: 143 weeks", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=7.3, color="#303030")
    year_starts = [i for i, date in enumerate(dates) if str(date)[5:] == "01-01"]
    ticks = [0, *year_starts, len(dates) - 1]
    ax.set_xticks(ticks, [str(dates[i])[:4] for i in ticks])
    ax.set_xlim(-0.5, len(dates) - 0.5)
    display = {"Connecticut", "Nevada"}
    positions = [i for i, name in enumerate(ordered_names) if name in display]
    ax.set_yticks(positions, [ordered_names[i] for i in positions])
    ax.set_ylabel("Jurisdiction (selected labels)")
    ax.set_xlabel("Weekly endpoint")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=3, pad=2)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("log(1 + weekly admissions per 100k)", fontsize=7.5, labelpad=3)
    cbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.18, top=0.78)
    fig.savefig(OUT / "dataset_overview_covid_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "dataset_overview_covid_heatmap.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
