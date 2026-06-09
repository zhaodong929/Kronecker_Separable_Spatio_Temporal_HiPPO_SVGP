#!/usr/bin/env python
"""Generate a PDF report for the ERA5 lag-AR diagnostic.

The report uses retained experiment results only:
- the full aggressive-tuning summary table;
- the retained lag_ar_diagnostic_32_256 run;
- the previous non-AR Matérn-3/2 rich_v4 per-location predictions for comparison.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "tmp_era5_routeb_aggressive_tuning_attempt"
SUMMARY_CSV = OUTDIR / "aggressive_tuning_summary.csv"
LAG_AR_CSV = OUTDIR / "lag_ar_diagnostic_32_256" / "era5_routeb_per_location_predictions.csv"
NON_AR_CSV = (
    ROOT
    / "results"
    / "experiments_era5_ohsvgp_heldout_fullspace"
    / "paper_ready"
    / "kernel_family_fullgp_wide_diagnostic"
    / "matern32_rich_v4_fullgp_wide"
    / "era5_routeb_per_location_predictions.csv"
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 140,
        }
    )


def save_summary_bar(summary: pd.DataFrame, outdir: Path) -> tuple[Path, Path]:
    labels = summary["attempt"].str.replace(" ", "\n", regex=False).str.replace("_", "\n", regex=False)
    x = np.arange(len(summary))
    colors = ["#376795" if v != "effective" else "#2f7d4f" for v in summary["verdict"]]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), constrained_layout=True)
    axes[0].bar(x, summary["rmse"], color=colors)
    axes[0].axhline(0.2, color="#a33a3a", linestyle="--", linewidth=1.2, label="RMSE target 0.2")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("ERA5 Route B aggressive diagnostic: RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].legend()

    axes[1].bar(x, summary["nll"], color=colors)
    axes[1].axhline(0.0, color="#777777", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("NLL")
    axes[1].set_title("ERA5 Route B aggressive diagnostic: NLL")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")

    png = outdir / "aggressive_tuning_full_comparison.png"
    pdf = outdir / "aggressive_tuning_full_comparison.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def load_location(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["eval_mode"] == "seen_history"].copy()
    if frame.empty:
        raise ValueError(f"No seen_history rows in {path}")
    frame = frame.sort_values(["time_index", "location_index"])
    return frame


def save_single_location_plot(non_ar: pd.DataFrame, lag_ar: pd.DataFrame, outdir: Path) -> tuple[Path, Path]:
    rows = [
        ("Non-AR Rich-$\\Phi$ + Matérn-3/2 32/256", non_ar, "#d95f02"),
        ("RouteB-AR lag-$\\Phi$ + Matérn-3/2 32/256", lag_ar, "#2c7fb8"),
    ]
    loc = int(lag_ar["location_index"].iloc[0])
    lat = float(lag_ar["latitude"].iloc[0])
    lon = float(lag_ar["longitude"].iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True, constrained_layout=True)
    for ax, (title, frame, color) in zip(axes, rows):
        x = frame["actual_time"].to_numpy(dtype=float)
        y = frame["y_true"].to_numpy(dtype=float)
        mean = frame["pred_mean"].to_numpy(dtype=float)
        std = np.sqrt(np.maximum(frame["pred_var_y"].to_numpy(dtype=float), 1e-12))
        half = 1.6448536269514722 * std
        ax.plot(x, y, color="black", linewidth=1.1, label="ERA5 target")
        ax.plot(x, mean, color=color, linewidth=1.6, label="prediction mean")
        ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.16, label="90% interval")
        ax.set_ylabel("scaled value")
        ax.set_title(title)
        ax.grid(True, alpha=0.18, linewidth=0.7)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("task_2 time index")
    fig.suptitle(f"Single-location ERA5 seen-history diagnostic, location {loc} (lat={lat:.2f}, lon={lon:.2f})", y=1.02)
    png = outdir / f"lag_ar_single_location_loc{loc}.png"
    pdf = outdir / f"lag_ar_single_location_loc{loc}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def add_title_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, "ERA5 Route B Lag-AR Diagnostic", fontsize=19, weight="bold")
    paragraphs = [
        "Goal: test whether Rich-Phi + Matérn-3/2 + 32/256 can be improved without changing the Route B structured posterior formulas.",
        "Protocol: all attempts use task_1 calibration and task_2 held-out seen-history evaluation. The retained effective setting appends lagged target features to Phi and keeps the existing Route B runner.",
        "Key result: ordinary kernel/capacity/lengthscale tuning did not reduce RMSE materially. The lag-AR diagnostic reduced RMSE to 0.1229 and NLL to -0.1714, indicating that missing autoregressive temporal covariates are the dominant bottleneck.",
    ]
    y = 0.86
    for paragraph in paragraphs:
        fig.text(0.08, y, textwrap.fill(paragraph, 95), fontsize=11, va="top")
        y -= 0.11
    fig.text(0.08, 0.50, "Interpretation", fontsize=14, weight="bold")
    interpretation = (
        "This is not evidence that minor hyperparameter tuning solves the ERA5 fit. "
        "Instead, the effective result comes from changing the covariate information: "
        "lagged observations give the linear component direct access to short-term local dynamics. "
        "Therefore RouteB-AR should be reported as a diagnostic or extension, while the original non-AR Rich-Phi result remains the fair baseline for the stated feature set."
    )
    fig.text(0.08, 0.46, textwrap.fill(interpretation, 95), fontsize=11, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    display = summary[["attempt", "rmse", "nll", "coverage90", "verdict"]].copy()
    display["rmse"] = display["rmse"].map(lambda x: f"{x:.4f}")
    display["nll"] = display["nll"].map(lambda x: f"{x:.4f}")
    display["coverage90"] = display["coverage90"].map(lambda x: f"{x:.4f}")
    display.columns = ["Attempt", "RMSE", "NLL", "Cov90", "Verdict"]
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.44, 0.12, 0.12, 0.12, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        if row > 0 and display.iloc[row - 1]["Verdict"] == "effective":
            cell.set_facecolor("#e8f4ec")
    ax.set_title("Aggressive tuning attempt summary", fontsize=15, weight="bold", loc="left", pad=18)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold", loc="left")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_CSV)
    non_ar = load_location(NON_AR_CSV)
    lag_ar = load_location(LAG_AR_CSV)
    bar_png, _ = save_summary_bar(summary, OUTDIR)
    loc_png, _ = save_single_location_plot(non_ar, lag_ar, OUTDIR)
    pdf_path = OUTDIR / "era5_routeb_lag_ar_diagnostic_report.pdf"
    with PdfPages(pdf_path) as pdf:
        add_title_page(pdf)
        add_table_page(pdf, summary)
        add_image_page(pdf, bar_png, "RMSE and NLL comparison")
        add_image_page(pdf, loc_png, "Single-location prediction effect")
    print(pdf_path)


if __name__ == "__main__":
    main()
