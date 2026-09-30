#!/usr/bin/env python
"""Summarize observed-vs-extended future-basis ERA5 Route B diagnostics."""

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


def load_summary(run_dir: Path, mode: str) -> list[dict[str, Any]]:
    rows = read_csv(run_dir / "era5_routeb_summary.csv")
    out = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["future_basis_mode"] = row.get("future_basis_mode") or mode
        item["run_dir"] = str(run_dir)
        out.append(item)
    return out


def plot_metric(rows: list[dict[str, Any]], outdir: Path, metric: str) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    modes = ["observed", "extended"]
    vals = []
    labels = []
    for mode in modes:
        found = [
            row
            for row in rows
            if row.get("method") == "structured_joint"
            and row.get("eval_mode") == "future"
            and row.get("future_basis_mode") == mode
        ]
        if found:
            labels.append(mode)
            vals.append(float(found[0][metric]))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(np.arange(len(labels)), vals)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric)
    ax.set_title(f"Structured Route B future basis diagnostic: {metric}")
    if metric == "coverage90":
        ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / f"era5_future_basis_structured_joint_{metric}.png", dpi=180)
    plt.close(fig)


def write_report(rows: list[dict[str, Any]], outdir: Path) -> None:
    def fmt(row: dict[str, Any], metric: str) -> str:
        return f"{float(row[metric]):.4f}" if row.get(metric) not in (None, "") else "NA"

    future = [
        row
        for row in rows
        if row.get("method") == "structured_joint" and row.get("eval_mode") == "future"
    ]
    lines = [
        "# ERA5 Horizon-Extended Future Basis Diagnostic",
        "",
        "This diagnostic does not change the Route B core formulas, old-likelihood transfer, R_beta_u transfer, Schur complement, or Sylvester/Du solves.",
        "",
        "`observed` keeps the current future prediction basis. `extended` uses the future time coordinates and prediction interval length, transfers the current old-likelihood statistics to a temporary prediction basis ending at `end(B_{n+1})`, and then predicts `B_{n+1}`.",
        "",
        "The extended state is temporary and is not used for online training. No `B_{n+1}` labels are used before prediction. Unlike RouteB-AR, this diagnostic does not add lagged label features.",
        "",
        "## Structured Joint Future Summary",
        "",
        "| future_basis_mode | RMSE | NLL | Cov90 | ECE | Avg var | avg_nu_star |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(future, key=lambda r: r.get("future_basis_mode", "")):
        lines.append(
            f"| {row['future_basis_mode']} | {fmt(row, 'rmse')} | {fmt(row, 'nll')} | {fmt(row, 'coverage90')} | "
            f"{fmt(row, 'ece')} | {fmt(row, 'avg_predictive_variance')} | {fmt(row, 'avg_nu_star')} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "- `plots/era5_future_basis_structured_joint_rmse.png`",
            "- `plots/era5_future_basis_structured_joint_nll.png`",
            "- `plots/era5_future_basis_structured_joint_coverage90.png`",
            "- `plots/era5_future_basis_structured_joint_ece.png`",
            "- `plots/era5_future_basis_structured_joint_avg_predictive_variance.png`",
            "- `plots/era5_future_basis_structured_joint_avg_nu_star.png`",
        ]
    )
    (outdir / "era5_future_basis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-dir", required=True)
    parser.add_argument("--extended-dir", required=True)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    rows = load_summary(Path(args.observed_dir), "observed") + load_summary(Path(args.extended_dir), "extended")
    write_csv(rows, outdir / "era5_future_basis_combined_summary.csv")
    for metric in ["rmse", "nll", "coverage90", "ece", "avg_predictive_variance", "avg_nu_star"]:
        plot_metric(rows, outdir, metric)
    write_report(rows, outdir)


if __name__ == "__main__":
    main()
