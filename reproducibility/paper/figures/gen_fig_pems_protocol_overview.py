#!/usr/bin/env python3
"""Render the PEMS-BAY spatial protocol overview for the appendix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import contextily as ctx


ROOT = Path(r"\\wsl.localhost\Ubuntu-24.04\home\zd929\projects\stvgp_kronecker")
COORDS = ROOT / "data/traffic/raw/pems_bay/graph_sensor_locations_bay.csv"
SPLIT = ROOT / "results/traffic/protocols/pems_bay/pems_bay_seed0_spatial_split.json"
OUT = Path(__file__).resolve().parent


def main() -> None:
    rows = []
    with COORDS.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            rows.append((row[0], float(row[1]), float(row[2])))

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    held_out = set(split["split"]["heldout_indices"])

    lon = [row[2] for row in rows]
    lat = [row[1] for row in rows]
    visible = [i not in held_out for i in range(len(rows))]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "legend.fontsize": 7.0,
    })

    fig, ax = plt.subplots(figsize=(5.25, 3.15))
    ax.set_facecolor("#f4f2ed")
    ax.set_xlim(min(lon) - 0.035, max(lon) + 0.035)
    ax.set_ylim(min(lat) - 0.025, max(lat) + 0.025)
    # A pale geographic basemap provides the DCRNN-style spatial context while
    # keeping the sensor split visually dominant. Attribution is added below.
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldGrayCanvas,
            crs="EPSG:4326",
            alpha=0.58,
            attribution=False,
            reset_extent=False,
        )
        ctx.add_attribution(
            ax,
            "Tiles © Esri — Esri, DeLorme, NAVTEQ",
            font_size=5.2,
            color="#77736d",
        )
    except Exception as exc:  # pragma: no cover - network fallback for reruns
        print(f"Basemap download unavailable ({exc}); using a pale background.")
    ax.scatter(
        [lon[i] for i, keep in enumerate(visible) if keep],
        [lat[i] for i, keep in enumerate(visible) if keep],
        s=10,
        color="#34383d",
        alpha=0.88,
        linewidths=0,
        label="Visible (260)",
        zorder=2,
    )
    ax.scatter(
        [lon[i] for i, keep in enumerate(visible) if not keep],
        [lat[i] for i, keep in enumerate(visible) if not keep],
        s=17,
        color="#D55E00",
        alpha=0.95,
        edgecolors="#fffaf2",
        linewidths=0.25,
        label="Held-out (65)",
        zorder=3,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#30343b",
                   markeredgewidth=0, markersize=4.3, label="Visible (260)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#D55E00",
                   markeredgecolor="#fffaf2", markeredgewidth=0.25, markersize=4.8,
                   label="Held-out (65)"),
        ],
        loc="lower left", frameon=True, facecolor="#ffffff", edgecolor="none",
        handletextpad=0.35, borderpad=0.35,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / "dataset_overview_pems.pdf", bbox_inches="tight")
    fig.savefig(OUT / "dataset_overview_pems.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
