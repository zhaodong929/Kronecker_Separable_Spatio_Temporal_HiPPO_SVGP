#!/usr/bin/env python3
"""Assemble the analytic-HiPPO rerun report without overwriting old PDFs."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready"
RERUN = PAPER / "analytic_hippo_rff_fixed_rerun"
PHI = RERUN / "phi_modes"
FIG = RERUN / "figures"
TABLE = RERUN / "tables"
TEX = PAPER / "era5_ohsvgp_heldout_experiment_report_nature_style_analytic_hippo_rff_fixed_rerun.tex"
PDF = TEX.with_suffix(".pdf")

MODES = [
    ("minimal", "minimal"),
    ("base", "base"),
    ("engineered", "engineered"),
    ("medium_era5", "medium-ERA5"),
    ("rich_era5", "rich-ERA5"),
]
METHODS = ["no_transfer", "mean_field", "structured_joint"]
METHOD_LABEL = {
    "no_transfer": "No transfer",
    "mean_field": "Mean-field",
    "structured_joint": "Structured joint",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metric(row: dict[str, Any], name: str) -> str:
    ci = f(row, f"{name}_ci95", f(row, f"{name}_se", 0.0) * 1.96)
    return f"{f(row, name):.4f} $\\pm$ {ci:.4f}"


def tex(text: object) -> str:
    return str(text).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def load_phi_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, label in MODES:
        run_dir = PHI / key
        heldout = read_csv(run_dir / "era5_ohsvgp_heldout_independent_run_summary.csv")
        forgetting = read_csv(run_dir / "era5_ohsvgp_heldout_final_forgetting_independent_run_summary.csv")
        report = json.loads((run_dir / "era5_routeb_report.json").read_text(encoding="utf-8"))
        forget_by_method = {r["method"]: r for r in forgetting}
        for row in heldout:
            method = row["method"]
            out = dict(row)
            out["phi_mode"] = key
            out["phi_label"] = label
            out["p"] = report.get("dataset_shape", {}).get("p", "")
            forget = forget_by_method.get(method, {})
            out["rmse_forgetting"] = forget.get("rmse_forgetting", "nan")
            out["rmse_forgetting_ci95"] = forget.get("rmse_forgetting_ci95", forget.get("rmse_forgetting_se", "0"))
            out["nll_forgetting"] = forget.get("nll_forgetting", "nan")
            out["nll_forgetting_ci95"] = forget.get("nll_forgetting_ci95", forget.get("nll_forgetting_se", "0"))
            rows.append(out)
    return rows


def load_synthetic(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_phi(rows: list[dict[str, Any]]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    structured = [r for r in rows if r["method"] == "structured_joint"]
    x = np.arange(len(structured))
    labels = [r["phi_label"] for r in structured]
    colors = ["#718096", "#4C78A8", "#72B7B2", "#F58518", "#54A24B"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), constrained_layout=True)
    for ax, key, title in [
        (axes[0, 0], "rmse", "RMSE"),
        (axes[0, 1], "nll", "NLL/NLPD"),
        (axes[1, 0], "coverage90", "Coverage90"),
        (axes[1, 1], "ece", "ECE"),
    ]:
        ax.bar(x, [f(r, key) for r in structured], color=colors)
        if key == "coverage90":
            ax.axhline(0.9, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Analytic HiPPO-RFF structured joint Route B across feature maps")
    fig.savefig(FIG / "fig_era5_phi_mode_structured_metrics.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "rmse_forgetting", "Final RMSE forgetting"),
        (axes[1], "nll_forgetting", "Final NLL forgetting"),
    ]:
        ax.bar(x, [f(r, key) for r in structured], color=colors)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(FIG / "fig_era5_phi_mode_structured_forgetting.png", dpi=220)
    plt.close(fig)

    medium = [r for r in rows if r["phi_mode"] == "medium_era5"]
    medium.sort(key=lambda r: METHODS.index(r["method"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    x2 = np.arange(len(medium))
    for ax, key, title in [(axes[0], "rmse", "RMSE"), (axes[1], "nll", "NLL/NLPD")]:
        ax.bar(x2, [f(r, key) for r in medium], color=["#6B8FB3", "#D19A66", "#6AA77A"])
        ax.set_title(f"medium-ERA5 {title}")
        ax.set_xticks(x2, [METHOD_LABEL[r["method"]] for r in medium], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(FIG / "fig_era5_medium_era5_method_ablation.png", dpi=220)
    plt.close(fig)


def maybe_copy_kernel_fig() -> str:
    src = RERUN / "medium_kernel_capacity_safe_lag_diagnostic/plots/era5_medium_kernel_capacity_safe_lag_metrics.png"
    if src.exists():
        dst = FIG / src.name
        shutil.copy2(src, dst)
        return dst.name
    return ""


def maybe_copy_calibration_figs() -> list[str]:
    out = []
    cal = RERUN / "safe_lag_calibration_diagnostics"
    for name in [
        "fig_safe_lag_nll_decomposition.png",
        "fig_safe_lag_variance_scaling.png",
        "fig_safe_lag_lag_shrink_noise.png",
        "fig_safe_lag_single_location_phi_mode_comparison.png",
    ]:
        src = cal / name
        if src.exists():
            dst = FIG / name
            shutil.copy2(src, dst)
            out.append(name)
    return out


def table_synthetic(report: dict[str, Any], regime: str) -> str:
    rows = report["mean_by_method_eval_mode"]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Synthetic {regime} analytic HiPPO-RFF prediction summary.}}",
        "\\small",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Method & Eval & RMSE & NLL & Cov90 \\\\",
        "\\midrule",
    ]
    for method in ["no_transfer", "mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]:
        for ev in ["current", "seen_history", "batch"]:
            r = rows.get(f"{method}|{ev}", {})
            if not r:
                continue
            label = method.replace("_ssgp_transfer", "").replace("_", " ")
            lines.append(f"{tex(label)} & {tex(ev)} & {f(r,'rmse'):.4f} & {f(r,'nll'):.4f} & {f(r,'coverage90'):.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_medium(rows: list[dict[str, Any]]) -> str:
    medium = [r for r in rows if r["phi_mode"] == "medium_era5"]
    medium.sort(key=lambda r: METHODS.index(r["method"]))
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Main medium-ERA5 analytic HiPPO-RFF held-out seen-history result.}",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & RMSE & NLL/NLPD & Cov90 & ECE & Runtime/block \\\\",
        "\\midrule",
    ]
    for r in medium:
        lines.append(
            f"{METHOD_LABEL[r['method']]} & {metric(r,'rmse')} & {metric(r,'nll')} & {metric(r,'coverage90')} & {metric(r,'ece')} & {metric(r,'runtime_per_block')} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_phi(rows: list[dict[str, Any]]) -> str:
    structured = [r for r in rows if r["method"] == "structured_joint"]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Structured joint analytic HiPPO-RFF Route B across feature maps.}",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Feature map & $p$ & RMSE & NLL/NLPD & Cov90 & ECE \\\\",
        "\\midrule",
    ]
    for r in structured:
        lines.append(f"{tex(r['phi_label'])} & {r['p']} & {metric(r,'rmse')} & {metric(r,'nll')} & {metric(r,'coverage90')} & {metric(r,'ece')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_forgetting(rows: list[dict[str, Any]]) -> str:
    structured = [r for r in rows if r["method"] == "structured_joint"]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Structured joint final forgetting across feature maps under analytic HiPPO-RFF.}",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Feature map & RMSE forgetting & NLL forgetting \\\\",
        "\\midrule",
    ]
    for r in structured:
        lines.append(f"{tex(r['phi_label'])} & {metric(r,'rmse_forgetting')} & {metric(r,'nll_forgetting')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_kernel() -> str:
    path = RERUN / "medium_kernel_capacity_safe_lag_diagnostic/era5_medium_kernel_capacity_safe_lag_summary.csv"
    if not path.exists():
        return "\\textit{Kernel/capacity diagnostic has not finished yet.}\\\\"
    rows = read_csv(path)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Medium-ERA5 analytic HiPPO-RFF kernel/capacity diagnostic.}",
        "\\small",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Kernel & Capacity & RMSE & NLL & Cov90 & ECE \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(f"{tex(r['kernel'])} & {tex(r['capacity'])} & {f(r,'rmse'):.4f} & {f(r,'nll'):.4f} & {f(r,'coverage90'):.4f} & {f(r,'ece'):.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def make_tex(rows: list[dict[str, Any]]) -> str:
    syn_std = load_synthetic(RERUN / "synthetic_standard/joint_ssgp_kron_synthetic_report.json")
    syn_long = load_synthetic(RERUN / "synthetic_long_memory/joint_ssgp_kron_synthetic_report.json")
    med = next(r for r in rows if r["phi_mode"] == "medium_era5" and r["method"] == "structured_joint")
    rich = next(r for r in rows if r["phi_mode"] == "rich_era5" and r["method"] == "structured_joint")
    kernel_fig = maybe_copy_kernel_fig()
    cal_figs = maybe_copy_calibration_figs()
    kernel_fig_tex = f"\\includegraphics[width=0.96\\linewidth]{{{kernel_fig}}}" if kernel_fig else ""
    cal_tex = "\n".join(
        "\\begin{figure}[H]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.88\\linewidth]{{{name}}}\n"
        "\\end{figure}"
        for name in cal_figs
    )
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.8in]{{geometry}}
\usepackage{{booktabs,graphicx,float,hyperref,amsmath,array}}
\graphicspath{{{{./}}{{analytic_hippo_rff_fixed_rerun/figures/}}}}
\setlength{{\parskip}}{{0.5em}}
\setlength{{\parindent}}{{0pt}}
\title{{Analytic HiPPO-RFF Rerun of Route B Synthetic and ERA5 Experiments}}
\author{{Route B experiment report}}
\date{{June 17, 2026}}
\begin{{document}}
\maketitle

\section{{Executive Summary}}
This new report does not overwrite the previous safe-lag report. It reruns the synthetic validation and ERA5 held-out continual-learning experiments using the unified analytic HiPPO-RFF temporal interdomain construction. The old moving temporal inducing-point path is retained only as an ablation backend; all results in this report use \texttt{{temporal\_backend=analytic\_hippo\_rff}}.

Under the corrected safe-lag protocol and analytic HiPPO-RFF temporal basis, medium-ERA5 structured joint Route B obtains RMSE {f(med,'rmse'):.4f} and NLL/NLPD {f(med,'nll'):.4f}. Rich-ERA5 obtains RMSE {f(rich,'rmse'):.4f} and NLL/NLPD {f(rich,'nll'):.4f}.

\section{{Implementation Fixes for This Rerun}}
This fixed rerun uses the reference analytic HiPPO-SVGP convention. First, the analytic phase origin is tied to the global continuous start of the time axis, not to each local block start. Second, timestamps are generated as \(\text{{start}} + (\text{{end}}-\text{{start}})(1,\ldots,N)/N\), so \(\text{{start}}\) and \(\text{{end}}\) are interval boundaries rather than observed min/max points. Third, temporal solves use scaled jitter based on the mean diagonal and robust Cholesky retry. Fourth, kernel/capacity diagnostics pass the selected kernel to both the temporal analytic RFF construction and spatial covariance side, including the spectral-mixture comparison.

\section{{Synthetic Validation Taxonomy}}
The synthetic sections keep the same interpretation as the previous report: Table A checks structured linear algebra, Table B checks dense posterior recovery, and Table C checks online prediction/calibration against batch-style references. Here we rerun the prediction-facing synthetic protocols with the analytic HiPPO-RFF temporal basis.

{table_synthetic(syn_std, "standard")}

{table_synthetic(syn_long, "long-memory")}

\section{{Transfer-State Variants and Formulas}}
The methods are unchanged. No-transfer discards old likelihood information. Mean-field transfer propagates the GP state but approximates $q(\beta,u)=q(\beta)q(u)$, dropping the $\beta$--$u$ cross covariance. Structured joint Route B maintains the joint natural parameters over $(\beta,u)$ and transfers the old likelihood ratio across changing temporal bases. Posterior recovery uses the Schur complement and Sylvester solves for the Kronecker GP block.

\section{{ERA5 Scope and Safe-Lag Protocol}}
The ERA5 rerun uses task 1 for calibration and task 2 for online evaluation, block size 10, variable index 0, and held-out spatial split seeds 0, 1 and 2. For medium-ERA5 and rich-ERA5, target-lag features at held-out test locations are recursively filled from previous predictive means rather than ground-truth test labels.

\section{{Main Medium-ERA5 Result}}
{table_medium(rows)}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\linewidth]{{fig_era5_medium_era5_method_ablation.png}}
\caption{{Main medium-ERA5 Route B ablation under analytic HiPPO-RFF.}}
\end{{figure}}

\section{{Feature-Map Comparison}}
{table_phi(rows)}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{fig_era5_phi_mode_structured_metrics.png}}
\caption{{Structured joint Route B across feature maps under analytic HiPPO-RFF.}}
\end{{figure}}

\section{{Why Added Dimensions Capture ERA5 Variation}}
The interpretation is unchanged from the previous report. Base and engineered feature maps mainly describe climatological average structure, while medium-ERA5 and rich-ERA5 describe current weather state. Lagged targets capture temporal inertia, surface meteorological covariates capture synoptic and local weather state, and the GP residual is no longer forced to explain all weather perturbations from time and space coordinates alone.

\section{{Forgetting and Calibration}}
{table_forgetting(rows)}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\linewidth]{{fig_era5_phi_mode_structured_forgetting.png}}
\caption{{Structured joint final forgetting across feature maps under analytic HiPPO-RFF.}}
\end{{figure}}

\section{{Calibration Diagnostics}}
{cal_tex}

\section{{Medium-ERA5 Kernel and Capacity Diagnostic}}
{kernel_fig_tex}

{table_kernel()}

\section{{Bounded Conclusion}}
The rerun confirms that the report is now aligned with the theory document's analytic HiPPO-RFF temporal interdomain construction. The numerical pattern changes materially relative to the older moving-inducing implementation, so the older ERA5 values should not be mixed with this report. Medium/rich ERA5 still benefit from physically meaningful covariates, while NLL and coverage remain calibration-sensitive.

\end{{document}}
"""


def main() -> None:
    rows = load_phi_rows()
    TABLE.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    write_table(TABLE / "phi_mode_summary.csv", rows)
    plot_phi(rows)
    TEX.write_text(make_tex(rows), encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TEX.name], cwd=PAPER, check=True)
    print(json.dumps({"tex": str(TEX), "pdf": str(PDF), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
