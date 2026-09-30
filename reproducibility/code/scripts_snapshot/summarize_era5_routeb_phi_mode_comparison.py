#!/usr/bin/env python
"""Summarize ERA5 Route B base-vs-lag_ar diagnostic runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_summary(run_dir: Path, phi_mode: str) -> list[dict[str, Any]]:
    rows = read_csv(run_dir / "era5_routeb_summary.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["phi_mode"] = row.get("phi_mode") or phi_mode
        item["run_dir"] = str(run_dir)
        out.append(item)
    return out


def plot_metric(rows: list[dict[str, Any]], outdir: Path, metric: str, eval_mode: str) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = ["no_transfer", "mean_field", "structured_joint"]
    phi_modes = ["base", "lag_ar"]
    width = 0.36
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for offset, phi_mode in [(-width / 2, "base"), (width / 2, "lag_ar")]:
        vals = []
        for method in methods:
            found = [
                row
                for row in rows
                if row.get("method") == method and row.get("eval_mode") == eval_mode and row.get("phi_mode") == phi_mode
            ]
            vals.append(float(found[0][metric]) if found and found[0].get(metric) not in (None, "") else np.nan)
        ax.bar(x + offset, vals, width=width, label=phi_mode)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"ERA5 Route B Phi-mode diagnostic: {eval_mode} {metric}")
    if metric == "coverage90":
        ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / f"era5_routeb_phi_mode_{eval_mode}_{metric}.png", dpi=180)
    plt.close(fig)


def write_report(rows: list[dict[str, Any]], outdir: Path, args: argparse.Namespace) -> None:
    def fmt(row: dict[str, Any], metric: str) -> str:
        return f"{float(row[metric]):.4f}" if row.get(metric) not in (None, "") else "NA"

    lines = [
        "# ERA5 RouteB-AR Diagnostic",
        "",
        "RouteB-AR is a forecasting diagnostic, not the paper's main Route B method.",
        "It keeps the Route B core formulas unchanged and only augments the linear feature matrix Phi with lag features.",
        "",
        "Lag features are `y_{t-1,s}`, `y_{t-2,s}`, and `y_{t-1,s}-y_{t-2,s}`.",
        "For future-block evaluation, the lag features are built only from observations before the predicted block; labels inside the future block are not read.",
        "",
        "The goal is to check whether weak future performance is caused by the original Phi lacking autoregressive information.",
        "",
        "## Output files",
        "",
        "- `era5_routeb_phi_mode_combined_summary.csv`",
        "- `plots/era5_routeb_phi_mode_future_rmse.png`",
        "- `plots/era5_routeb_phi_mode_future_nll.png`",
        "- `plots/era5_routeb_phi_mode_future_coverage90.png`",
        "- `plots/era5_routeb_phi_mode_seen_history_rmse.png`",
        "- `plots/era5_routeb_phi_mode_seen_history_nll.png`",
        "",
        "## Future diagnostic",
        "",
        "| phi_mode | method | RMSE | NLL | Cov90 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("eval_mode") == "future":
            lines.append(f"| {row['phi_mode']} | {row['method']} | {fmt(row, 'rmse')} | {fmt(row, 'nll')} | {fmt(row, 'coverage90')} |")
    lines.extend(["", "## Seen-history retention", "", "| phi_mode | method | RMSE | NLL | Cov90 |", "|---|---|---:|---:|---:|"])
    for row in rows:
        if row.get("eval_mode") == "seen_history":
            lines.append(f"| {row['phi_mode']} | {row['method']} | {fmt(row, 'rmse')} | {fmt(row, 'nll')} | {fmt(row, 'coverage90')} |")
    (outdir / "era5_routeb_phi_mode_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--lag-ar-dir", required=True)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    rows = load_summary(Path(args.base_dir), "base") + load_summary(Path(args.lag_ar_dir), "lag_ar")
    write_csv(rows, outdir / "era5_routeb_phi_mode_combined_summary.csv")
    for eval_mode in ["future", "seen_history"]:
        for metric in ["rmse", "nll", "coverage90"]:
            plot_metric(rows, outdir, metric, eval_mode)
    write_report(rows, outdir, args)


if __name__ == "__main__":
    main()
