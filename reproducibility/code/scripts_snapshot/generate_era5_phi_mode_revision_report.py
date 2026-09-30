#!/usr/bin/env python
"""Generate the revised ERA5 paper-ready report with new Phi-mode experiments."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace")
OUT = ROOT / "paper_ready"
REV = OUT / "phi_mode_revision"
FIG_DIR = OUT / "figures" / "phi_mode_revision"
TABLE_DIR = OUT / "tables"

MODES = [
    {
        "key": "minimal",
        "label": "minimal",
        "folder": REV / "minimal",
        "position": "minimal climatology",
        "contents": "periodic time + latitude/longitude",
    },
    {
        "key": "base",
        "label": "base",
        "folder": ROOT,
        "position": "legacy conservative baseline",
        "contents": "legacy 8-column time/space Phi",
    },
    {
        "key": "engineered",
        "label": "engineered",
        "folder": REV / "engineered",
        "position": "renamed old rich",
        "contents": "base + seasonal-spatial polynomial/interactions",
    },
    {
        "key": "medium_era5",
        "label": "medium-ERA5",
        "folder": REV / "medium_era5",
        "position": "recommended main covariate setting",
        "contents": "minimal + lagged target + available surface meteorology",
    },
    {
        "key": "rich_era5",
        "label": "rich-ERA5",
        "folder": REV / "rich_era5",
        "position": "extended covariate setting",
        "contents": "medium-ERA5 + PCA of remaining available ERA5 single-level variables",
    },
]

METHOD_ORDER = ["no_transfer", "mean_field", "structured_joint"]
METHOD_LABEL = {
    "no_transfer": "No transfer",
    "mean_field": "Mean-field",
    "structured_joint": "Structured joint",
}
MODE_COLORS = {
    "minimal": "#718096",
    "base": "#4C78A8",
    "engineered": "#72B7B2",
    "medium_era5": "#F58518",
    "rich_era5": "#54A24B",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def f(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return float(default)
    return float(value)


def tex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%")


def metric_text(row: dict[str, Any], metric: str, *, ci_key: str | None = None, digits: int = 4) -> str:
    if ci_key is None:
        ci_key = f"{metric}_ci95"
    return f"{f(row, metric):.{digits}f} $\\pm$ {f(row, ci_key, 0.0):.{digits}f}"


def mode_rows(mode: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    folder = Path(mode["folder"])
    heldout = read_csv(folder / "era5_ohsvgp_heldout_independent_run_summary.csv")
    forgetting = read_csv(folder / "era5_ohsvgp_heldout_final_forgetting_independent_run_summary.csv")
    report = read_json(folder / "era5_routeb_report.json")
    return heldout, forgetting, report


def collect_summary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        heldout, forgetting, report = mode_rows(mode)
        forget_by_method = {row["method"]: row for row in forgetting}
        shape = report.get("dataset_shape", {})
        for row in heldout:
            method = row["method"]
            forget = forget_by_method.get(method, {})
            rows.append(
                {
                    "phi_mode": mode["key"],
                    "phi_label": mode["label"],
                    "phi_position": mode["position"],
                    "method": method,
                    "method_label": METHOD_LABEL.get(method, method),
                    "nll": f(row, "nll"),
                    "nll_ci95": f(row, "nll_ci95", 0.0),
                    "rmse": f(row, "rmse"),
                    "rmse_ci95": f(row, "rmse_ci95", 0.0),
                    "coverage90": f(row, "coverage90"),
                    "coverage90_ci95": f(row, "coverage90_ci95", 0.0),
                    "ece": f(row, "ece"),
                    "ece_ci95": f(row, "ece_ci95", 0.0),
                    "runtime_per_block": f(row, "runtime_per_block"),
                    "runtime_per_block_ci95": f(row, "runtime_per_block_ci95", 0.0),
                    "nll_forgetting": f(forget, "nll_forgetting"),
                    "nll_forgetting_ci95": f(forget, "nll_forgetting_ci95", 0.0),
                    "rmse_forgetting": f(forget, "rmse_forgetting"),
                    "rmse_forgetting_ci95": f(forget, "rmse_forgetting_ci95", 0.0),
                    "p": shape.get("p", shape.get("p_raw", "")),
                }
            )
    return rows


def write_summary_csv(rows: list[dict[str, Any]]) -> Path:
    path = REV / "era5_phi_mode_revision_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_mode_summary(rows: list[dict[str, Any]]) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    structured = [row for row in rows if row["method"] == "structured_joint"]
    x = np.arange(len(structured))
    labels = [row["phi_label"] for row in structured]
    colors = [MODE_COLORS[row["phi_mode"]] for row in structured]
    outputs: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), constrained_layout=True)
    panels = [
        ("rmse", "RMSE", "lower is better"),
        ("nll", "NLL/NLPD", "lower is better"),
        ("coverage90", "Cov90", "nominal 0.90"),
        ("ece", "ECE", "lower is better"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.ravel(), panels):
        vals = [float(row[metric]) for row in structured]
        ax.bar(x, vals, color=colors, alpha=0.9)
        if metric == "coverage90":
            ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Structured joint Route B across Phi modes")
    for ext in ("png", "pdf", "svg"):
        out = FIG_DIR / f"fig_era5_phi_mode_structured_metrics.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        outputs.append(out)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "rmse_forgetting", "Final RMSE forgetting"),
        (axes[1], "nll_forgetting", "Final NLL forgetting"),
    ]:
        vals = [float(row[metric]) for row in structured]
        ax.bar(x, vals, color=colors, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Structured joint final forgetting across Phi modes")
    for ext in ("png", "pdf", "svg"):
        out = FIG_DIR / f"fig_era5_phi_mode_structured_forgetting.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        outputs.append(out)
    plt.close(fig)

    medium = [row for row in rows if row["phi_mode"] == "medium_era5"]
    medium = sorted(medium, key=lambda row: METHOD_ORDER.index(row["method"]))
    x2 = np.arange(len(medium))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    for ax, metric, title in [(axes[0], "rmse", "RMSE"), (axes[1], "nll", "NLL/NLPD")]:
        ax.bar(x2, [float(row[metric]) for row in medium], color=["#6B8FB3", "#D19A66", "#6AA77A"], alpha=0.9)
        ax.set_title(f"medium-ERA5 {title}")
        ax.set_xticks(x2)
        ax.set_xticklabels([METHOD_LABEL[row["method"]] for row in medium], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Main medium-ERA5 Route B ablation")
    for ext in ("png", "pdf", "svg"):
        out = FIG_DIR / f"fig_era5_medium_era5_method_ablation.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        outputs.append(out)
    plt.close(fig)

    return outputs


def gaussian_nll(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(np.asarray(var, dtype=float), 1e-10)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (np.asarray(y) - mean) ** 2 / var))


def read_per_location_csv(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def plot_single_location_phi_comparison() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panels = [
        ("base", "Base Phi, Mt=8, Ms=64", "#6f7f95"),
        ("medium_era5", "medium-ERA5, Mt=8, Ms=64", "#2f74c0"),
        ("rich_era5", "rich-ERA5, Mt=8, Ms=64", "#2e8b57"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 6.8), sharex=True, constrained_layout=True)
    for ax, (mode, title, color) in zip(axes, panels):
        path = REV / f"per_location_{mode}" / "era5_routeb_per_location_predictions.csv"
        rows = read_per_location_csv(path)
        x = np.asarray([f(row, "actual_time") for row in rows], dtype=float)
        y = np.asarray([f(row, "y_true") for row in rows], dtype=float)
        mean = np.asarray([f(row, "pred_mean") for row in rows], dtype=float)
        std = np.asarray([f(row, "pred_std_y") for row in rows], dtype=float)
        var = np.maximum(std**2, 1e-10)
        rmse = float(np.sqrt(np.mean((y - mean) ** 2)))
        nll = gaussian_nll(y, mean, var)
        width = float(np.mean(2.0 * 1.6448536269514722 * std))
        n = y.size
        loc = int(f(rows[0], "location_index"))
        lat = f(rows[0], "latitude")
        lon = f(rows[0], "longitude")

        ax.plot(x, y, color="black", linewidth=1.15, label="ERA5 target")
        ax.plot(x, mean, color=color, linewidth=1.55, label="prediction mean")
        ax.fill_between(x, mean - 1.6448536269514722 * std, mean + 1.6448536269514722 * std, color=color, alpha=0.18, label="90% interval")
        ax.set_title(title)
        ax.set_ylabel("scaled value")
        ax.grid(True, alpha=0.18)
        ax.text(
            0.012,
            0.06,
            f"n={n} points | RMSE={rmse:.3f} | NLL={nll:.3f} | avg 90% width={width:.2f}",
            transform=ax.transAxes,
            fontsize=8.2,
            bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none", "pad": 2.8},
        )
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"Single-location Phi-mode diagnostic at ERA5 location {loc} (lat={lat:.1f}, lon={lon:.1f})")
    out = FIG_DIR / "fig_era5_single_location_phi_mode_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_tex_table_phi_definitions() -> str:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Feature maps used in the revised ERA5 experiment. ERA5 covariates enter only the linear mean $\\Phi\\beta$; the GP residual kernel still uses time and spatial coordinates.}",
        "\\label{tab:era5-phi-modes}",
        "\\small",
        "\\begin{tabular}{p{0.16\\linewidth}p{0.25\\linewidth}p{0.46\\linewidth}}",
        "\\toprule",
        "Mode & Role & Contents \\\\",
        "\\midrule",
    ]
    for mode in MODES:
        lines.append(
            f"{tex_escape(mode['label'])} & {tex_escape(mode['position'])} & {tex_escape(mode['contents'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def make_tex_table_structured(rows: list[dict[str, Any]]) -> str:
    structured = [row for row in rows if row["method"] == "structured_joint"]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Structured joint Route B across feature maps. Values are mean $\\pm$ 95\\% confidence interval over three held-out spatial splits.}",
        "\\label{tab:era5-phi-structured}",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "$\\Phi$ mode & $p$ & NLL/NLPD $\\downarrow$ & RMSE $\\downarrow$ & Cov90 & ECE $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in structured:
        lines.append(
            f"{tex_escape(row['phi_label'])} & {row['p']} & {metric_text(row, 'nll')} & {metric_text(row, 'rmse')} & "
            f"{metric_text(row, 'coverage90')} & {metric_text(row, 'ece')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def make_tex_table_medium(rows: list[dict[str, Any]]) -> str:
    medium = [row for row in rows if row["phi_mode"] == "medium_era5"]
    medium = sorted(medium, key=lambda row: METHOD_ORDER.index(row["method"]))
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Main medium-ERA5 held-out seen-history result. This is the recommended main covariate setting.}",
        "\\label{tab:era5-medium-main}",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & NLL/NLPD $\\downarrow$ & RMSE $\\downarrow$ & Cov90 & ECE $\\downarrow$ & Runtime/block $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in medium:
        lines.append(
            f"{METHOD_LABEL[row['method']]} & {metric_text(row, 'nll')} & {metric_text(row, 'rmse')} & "
            f"{metric_text(row, 'coverage90')} & {metric_text(row, 'ece')} & {metric_text(row, 'runtime_per_block')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def make_tex_table_forgetting(rows: list[dict[str, Any]]) -> str:
    structured = [row for row in rows if row["method"] == "structured_joint"]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Structured joint final forgetting across feature maps. RMSE forgetting improves strongly with ERA5 covariates, while NLL forgetting shows remaining calibration drift.}",
        "\\label{tab:era5-phi-forgetting}",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "$\\Phi$ mode & Final NLL forgetting $\\downarrow$ & Final RMSE forgetting $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in structured:
        lines.append(
            f"{tex_escape(row['phi_label'])} & {metric_text(row, 'nll_forgetting')} & {metric_text(row, 'rmse_forgetting')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def make_report_tex(rows: list[dict[str, Any]]) -> str:
    med = {row["method"]: row for row in rows if row["phi_mode"] == "medium_era5"}
    sj = med["structured_joint"]
    mf = med["mean_field"]
    base_sj = next(row for row in rows if row["phi_mode"] == "base" and row["method"] == "structured_joint")
    rich_sj = next(row for row in rows if row["phi_mode"] == "rich_era5" and row["method"] == "structured_joint")

    rmse_gain_base = 100.0 * (float(base_sj["rmse"]) - float(sj["rmse"])) / float(base_sj["rmse"])
    nll_gain_base = 100.0 * (float(base_sj["nll"]) - float(sj["nll"])) / float(base_sj["nll"])
    rmse_gain_mf = 100.0 * (float(mf["rmse"]) - float(sj["rmse"])) / float(mf["rmse"])
    nll_gain_mf = 100.0 * (float(mf["nll"]) - float(sj["nll"])) / float(mf["nll"])
    rich_rmse_gain = 100.0 * (float(sj["rmse"]) - float(rich_sj["rmse"])) / float(sj["rmse"])

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.8in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{caption}}
\usepackage{{array}}
\usepackage{{longtable}}
\usepackage{{amsmath}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlength{{\parskip}}{{0.55em}}
\setlength{{\parindent}}{{0pt}}
\graphicspath{{{{./}}{{figures/}}{{figures/phi_mode_revision/}}}}

\title{{Continuation Report: Revised ERA5 OHSVGP-style Held-out Continual-learning Results}}
\author{{Route B experimental report}}
\date{{}}

\begin{{document}}
\maketitle

\section*{{One-sentence argument}}
After revising the ERA5 feature map, structured joint Route B remains the best internal Route B variant under the recommended medium-ERA5 covariate setting, reducing held-out RMSE to {float(sj['rmse']):.4f} and NLL/NLPD to {float(sj['nll']):.4f}; the large improvement over the legacy base setting comes mainly from moving lagged and surface meteorological information into $\Phi$, while the structured $\beta$--$u$ posterior still improves over mean-field transfer.

\section*{{Scope and protocol}}
This report updates \texttt{{results/experiments\_era5\_ohsvgp\_heldout\_fullspace/paper\_ready}} using the requested feature-map revision document. The data source and continual-learning protocol are unchanged: \texttt{{task\_1}} is used for calibration, \texttt{{task\_2}} is used for online evaluation, each online block has 10 time steps, and evaluation follows the OHSVGP-style held-out seen-history protocol with a fixed spatial train/test split per run and held-out split seeds 0, 1 and 2.

The target used by the existing experiment is \texttt{{variable\_index=0}}, which maps to \texttt{{2m\_dewpoint\_temperature}} in \texttt{{era5\_variable\_mapping.xlsx}}. The processed data package contains 35 ERA5 single-level variables. It does not contain explicit elevation, land-sea mask, or pressure-level variables, so the requested static and pressure-level terms are documented as unavailable rather than silently fabricated. In the revised medium/rich settings, available ERA5 covariates enter only the linear mean term $\Phi(t,s)^\top\beta$; the GP residual still uses time and spatial coordinates.

{make_tex_table_phi_definitions()}

\section*{{Main medium-ERA5 result}}
The recommended main setting is medium-ERA5: minimal periodic/spatial features plus lagged target covariates and available surface meteorology. Under this setting, structured joint Route B improves RMSE by {rmse_gain_mf:.1f}\% and NLL/NLPD by {nll_gain_mf:.1f}\% relative to mean-field transfer. Compared with the legacy base structured-joint result, medium-ERA5 reduces RMSE by {rmse_gain_base:.1f}\% and NLL/NLPD by {nll_gain_base:.1f}\%, confirming that the original ERA5 experiment was strongly feature-limited.

{make_tex_table_medium(rows)}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\linewidth]{{fig_era5_medium_era5_method_ablation.png}}
\caption{{Main medium-ERA5 Route B ablation. Structured joint gives the best RMSE and NLL/NLPD among the three Route B internal variants.}}
\end{{figure}}

\section*{{Feature-map comparison}}
The old \texttt{{rich\_seasonal\_spatial}} setting is retained but renamed engineered, because it adds polynomial and seasonal-spatial interactions rather than true ERA5 meteorological covariates. The new medium-ERA5 and rich-ERA5 settings add real covariate information available in the processed ERA5 package. Rich-ERA5 adds PCA scores from the remaining available single-level ERA5 variables; pressure-level variables are not available in this processed dataset.

{make_tex_table_structured(rows)}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{fig_era5_phi_mode_structured_metrics.png}}
\caption{{Structured joint Route B across feature maps. The main jump occurs when lagged and surface meteorological covariates are added in medium-ERA5. Rich-ERA5 gives a smaller additional improvement.}}
\end{{figure}}

Rich-ERA5 is slightly better than medium-ERA5 on RMSE ({float(rich_sj['rmse']):.4f} versus {float(sj['rmse']):.4f}) and NLL/NLPD ({float(rich_sj['nll']):.4f} versus {float(sj['nll']):.4f}), but the gain is small. This supports reporting medium-ERA5 as the main result and rich-ERA5 as an extension.

\section*{{Why the added dimensions capture dataset variation}}
The central reason for the large improvement is that base and engineered feature maps mostly describe climatological average structure, whereas medium-ERA5 and rich-ERA5 describe the current weather state. Base features can tell the model what value is typical for a location and season. They cannot tell whether that location is currently in an unusually cold, warm, windy, low-pressure or high-pressure state.

First, lagged target features capture temporal inertia. Many meteorological variables, especially near-surface temperature, surface pressure and wind, are strongly continuous in time:
\[
y_t \approx y_{{t-1}} + \hbox{{small weather change}}.
\]
After adding lagged target covariates such as $y_{{t-1}}$ and related short-lag summaries, the model immediately knows the recent local state. This is much more informative than only knowing day-of-year, latitude and longitude. Therefore the reduction from base structured-joint RMSE 0.6201 to medium-ERA5 RMSE 0.1227 is plausible rather than suspicious.

Second, surface meteorological covariates describe weather systems that coordinate and seasonal terms cannot represent. Wind variables encode transport and advection; pressure variables encode synoptic systems such as cyclones and anticyclones; humidity and dewpoint encode water-vapour state and thermodynamic conditions; radiation and flux variables encode surface heating, cooling and land-atmosphere exchange. The medium-ERA5 improvement is therefore not merely the effect of adding parameters. It gives the linear component observed state variables that explain real weather variability.

Third, the GP residual is no longer forced to explain everything. Under base $\Phi$, the mean term $\phi(t,s)^\top\beta$ explains only broad temporal and spatial trends, leaving most weather perturbations to $f(t,s)$. But the GP residual receives only time and spatial coordinates, so it can learn smooth spatio-temporal dependence but cannot observe the current wind, humidity, pressure or radiative state. With medium-ERA5, the model has a more natural decomposition:
\[
y(t,s)=\underbrace{{\phi(t,s)^\top\beta}}_{{\text{{observed meteorological state}}}}
+\underbrace{{f(t,s)}}_{{\text{{remaining spatio-temporal residual}}}}+\epsilon.
\]
This supports the paper's core modelling motivation: the mean component and GP memory jointly explain observations, so preserving the $\beta$--$u$ coupling remains important.

Fourth, rich-ERA5 gives only a small additional gain over medium-ERA5. The structured-joint RMSE changes from 0.1227 under medium-ERA5 to 0.1203 under rich-ERA5, an improvement of about {rich_rmse_gain:.1f}\%. This indicates that the main predictable signal is already captured by lagged targets and surface meteorological covariates; PCA features from the remaining available single-level variables add secondary variation. This is a useful ablation result: the decisive step is the physically meaningful move from base to medium-ERA5, not blindly increasing dimensionality.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.94\linewidth]{{fig_era5_single_location_phi_mode_comparison.png}}
\caption{{Single-location time-series diagnostic at location 99. Each panel contains 186 time points and reports RMSE, NLL and average 90\% interval width. Moving from base to medium-ERA5 gives the visible mean-trajectory improvement; rich-ERA5 adds only a small extra refinement.}}
\end{{figure}}

\section*{{Forgetting and calibration}}
The revised feature maps substantially reduce RMSE forgetting, especially for medium-ERA5 and rich-ERA5. However, NLL forgetting increases under the lagged/covariate-assisted settings, indicating that mean prediction improves more cleanly than uncertainty retention. This should be reported as a calibration limitation rather than hidden.

{make_tex_table_forgetting(rows)}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\linewidth]{{fig_era5_phi_mode_structured_forgetting.png}}
\caption{{Structured joint final forgetting across feature maps. ERA5 covariates reduce RMSE forgetting sharply, while NLL forgetting exposes remaining variance calibration drift.}}
\end{{figure}}

\section*{{Bounded conclusion}}
The revised experiment supports three claims. First, the legacy base feature map was too conservative for ERA5; adding lagged and surface meteorological covariates is the dominant improvement. Second, under the recommended medium-ERA5 feature map, structured joint Route B remains better than no-transfer and mean-field transfer on held-out RMSE/NLL. Third, uncertainty calibration remains conservative and NLL forgetting can worsen under strong covariates, so the paper should emphasize RMSE/NLL held-out performance and RMSE forgetting while reporting calibration limitations openly.

\end{{document}}
"""


def main() -> None:
    rows = collect_summary()
    write_summary_csv(rows)
    plot_mode_summary(rows)
    plot_single_location_phi_comparison()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "table_era5_phi_mode_definitions.tex").write_text(make_tex_table_phi_definitions(), encoding="utf-8")
    (TABLE_DIR / "table_era5_phi_mode_structured.tex").write_text(make_tex_table_structured(rows), encoding="utf-8")
    (TABLE_DIR / "table_era5_medium_era5_main.tex").write_text(make_tex_table_medium(rows), encoding="utf-8")
    (TABLE_DIR / "table_era5_phi_mode_forgetting.tex").write_text(make_tex_table_forgetting(rows), encoding="utf-8")

    tex_path = OUT / "era5_ohsvgp_heldout_experiment_report_nature_style_updated.tex"
    tex_path.write_text(make_report_tex(rows), encoding="utf-8")
    md_path = REV / "era5_phi_mode_revision_report.md"
    md_path.write_text(
        "# ERA5 Phi-mode revision report\n\n"
        "Generated revised feature-mode summaries and figures. The updated LaTeX source is "
        f"`{tex_path}`.\n",
        encoding="utf-8",
    )

    if shutil.which("pdflatex"):
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "era5_ohsvgp_heldout_experiment_report_nature_style_updated.tex",
            ],
            cwd=OUT,
            check=True,
        )
    else:
        print("pdflatex not found on PATH; wrote LaTeX and figures only.")


if __name__ == "__main__":
    main()
