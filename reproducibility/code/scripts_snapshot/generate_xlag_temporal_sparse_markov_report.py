#!/usr/bin/env python3
"""Append the X-lag and temporal-sparse Markov study to the staged report."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "legend.frameon": False,
    }
)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7A5195"
GREY = "#7D8790"
GRID = "#D8DDE1"


def parse_peak_rss(path: Path) -> float:
    if not path.exists():
        return float("nan")
    for line in path.read_text(errors="ignore").splitlines():
        if "Maximum resident set size (kbytes):" in line:
            return float(line.rsplit(":", 1)[1].strip()) / 1024.0
    return float("nan")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def pm(mean: Any, sd: Any, digits: int = 4) -> str:
    return f"{float(mean):.{digits}f} $\\pm$ {float(sd):.{digits}f}"


def esc(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def result_row(path: Path, *, method: str, condition: str, mt: float) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "method": method,
        "condition": condition,
        "mt": mt,
        "ms": int(payload["num_spatial_inducing"]),
        "seed": int(payload["seed"]),
        "rmse": float(payload["rmse"]),
        "nll": float(payload["nll"]),
        "coverage90": float(payload["coverage90"]),
        "avg_std": float(payload["mean_predictive_std"]),
        "runtime_sec": float(payload["train_seconds"]),
        "peak_rss_mb": parse_peak_rss(path.parent / "resource_usage.txt"),
        "iterations": int(payload["iterations"]),
        "parameter_source": "Bayes-Newton/Objax training",
    }


def routeb_row(path: Path, *, condition: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    final = payload["final"]
    status_path = path.parent / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    return {
        "method": "Structured-joint Route B",
        "condition": condition,
        "mt": int(final["mt"]),
        "ms": int(final["ms"]),
        "seed": int(final["heldout_split_seed"]),
        "rmse": float(final["rmse"]),
        "nll": float(final["nll"]),
        "coverage90": float(final["coverage90"]),
        "avg_std": float(final["avg_std"]),
        "runtime_sec": float(final["block_incremental_runtime_sec"]),
        "peak_rss_mb": parse_peak_rss(path.parent / "resource_usage.txt"),
        "iterations": 0,
        "parameter_source": status.get("parameter_source", "matched temporal-sparse run"),
    }


def collect(root: Path) -> pd.DataFrame:
    phase_h = root / "phase_h_xlag_temporal_sparse_markov"
    phase_d = root / "phase_d_joint_xlag_controlled"
    rows: list[dict[str, Any]] = []
    for seed in range(3):
        for ms in (64, 128):
            rows.append(
                result_row(
                    phase_h / f"seed{seed}/full_markov_no_xlag_Ms{ms}/result.json",
                    method="Official ST-SVGP",
                    condition="No X-lag",
                    mt=np.nan,
                )
            )
            rows.append(
                result_row(
                    phase_d / f"seed{seed}/official_st_svgp_Ms{ms}/result.json",
                    method="Official ST-SVGP",
                    condition="Learned X-lag",
                    mt=np.nan,
                )
            )
        for mt, ms in ((8, 64), (32, 128)):
            for slug, condition in (("no_xlag", "No X-lag"), ("xlag", "Learned X-lag")):
                rows.append(
                    result_row(
                        phase_h / f"seed{seed}/temporal_sparse_{slug}_Mt{mt}_Ms{ms}/result.json",
                        method="Temporal-inducing Markov ST-SVGP",
                        condition=condition,
                        mt=mt,
                    )
                )
                rows.append(
                    routeb_row(
                        phase_h / f"seed{seed}/routeb_{slug}_Mt{mt}_Ms{ms}/run_metadata.json",
                        condition=condition,
                    )
                )
    frame = pd.DataFrame(rows)
    expected = 36
    if len(frame) != expected or frame.seed.nunique() != 3:
        raise RuntimeError(f"Expected {expected} complete rows over three seeds, found {len(frame)}")
    return frame


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(["method", "condition", "mt", "ms"], dropna=False):
        row = dict(zip(["method", "condition", "mt", "ms"], key))
        row["num_seeds"] = int(group.seed.nunique())
        for metric in ["rmse", "nll", "coverage90", "avg_std", "runtime_sec", "peak_rss_mb", "iterations"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        records.append(row)
    return pd.DataFrame(records)


def plot_full_ablation(summary: pd.DataFrame, figures: Path) -> None:
    frame = summary.loc[summary.method == "Official ST-SVGP"].copy()
    order = [(64, "No X-lag"), (64, "Learned X-lag"), (128, "No X-lag"), (128, "Learned X-lag")]
    rows = [frame.loc[(frame.ms == ms) & (frame.condition == condition)].iloc[0] for ms, condition in order]
    labels = ["No X-lag\n$M_s=64$", "X-lag\n$M_s=64$", "No X-lag\n$M_s=128$", "X-lag\n$M_s=128$"]
    colors = [GREY, BLUE, GREY, BLUE]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), constrained_layout=True)
    for ax, metric, ylabel, letter in [
        (axes[0], "rmse", "Test RMSE", "a"),
        (axes[1], "nll", "Test NLL", "b"),
    ]:
        values = [row[f"{metric}_mean"] for row in rows]
        errors = [row[f"{metric}_sd"] for row in rows]
        ax.bar(x, values, yerr=errors, color=colors, edgecolor="white", linewidth=0.5, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)
    save_figure(fig, figures / "figure5_official_xlag_ablation")


def plot_temporal_budget(summary: pd.DataFrame, figures: Path) -> None:
    frame = summary.loc[summary.method != "Official ST-SVGP"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.55), constrained_layout=True)
    conditions = ["No X-lag", "Learned X-lag"]
    colors = [ORANGE, GREEN, ORANGE, GREEN]
    labels = ["Markov ext.\n$(8,64)$", "Route B\n$(8,64)$", "Markov ext.\n$(32,128)$", "Route B\n$(32,128)$"]
    for row_index, condition in enumerate(conditions):
        order = [
            ("Temporal-inducing Markov ST-SVGP", 8, 64),
            ("Structured-joint Route B", 8, 64),
            ("Temporal-inducing Markov ST-SVGP", 32, 128),
            ("Structured-joint Route B", 32, 128),
        ]
        rows = [
            frame.loc[
                (frame.condition == condition)
                & (frame.method == method)
                & (frame.mt == mt)
                & (frame.ms == ms)
            ].iloc[0]
            for method, mt, ms in order
        ]
        for col_index, (metric, ylabel) in enumerate((("rmse", "Test RMSE"), ("nll", "Test NLL"))):
            ax = axes[row_index, col_index]
            values = [row[f"{metric}_mean"] for row in rows]
            errors = [row[f"{metric}_sd"] for row in rows]
            x = np.arange(len(rows))
            ax.bar(x, values, yerr=errors, color=colors, edgecolor="white", linewidth=0.5, capsize=2)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel(ylabel)
            ax.set_title(condition, loc="left", fontsize=8, fontweight="bold")
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            letter = chr(ord("a") + row_index * 2 + col_index)
            ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)
    save_figure(fig, figures / "figure6_matched_temporal_budget")


def full_rows(summary: pd.DataFrame) -> str:
    frame = summary.loc[summary.method == "Official ST-SVGP"].sort_values(["ms", "condition"], ascending=[True, False])
    return "\n".join(
        f"{esc(row.condition)} & Full Markov (186 states) & {int(row.ms)} & "
        f"{pm(row.rmse_mean, row.rmse_sd)} & {pm(row.nll_mean, row.nll_sd)} & "
        f"{pm(row.coverage90_mean, row.coverage90_sd, 3)} & {row.runtime_sec_mean / 60.0:.1f} \\\\"
        for _, row in frame.iterrows()
    )


def matched_rows(summary: pd.DataFrame) -> str:
    methods = ["Temporal-inducing Markov ST-SVGP", "Structured-joint Route B"]
    rows = []
    for condition in ("No X-lag", "Learned X-lag"):
        for mt, ms in ((8, 64), (32, 128)):
            for method in methods:
                row = summary.loc[
                    (summary.condition == condition)
                    & (summary.method == method)
                    & (summary.mt == mt)
                    & (summary.ms == ms)
                ].iloc[0]
                temporal = "Markov inducing pairs" if method.startswith("Temporal") else "Analytic HiPPO"
                rows.append(
                    f"{esc(condition)} & {esc(method)} & {temporal} & {mt} & {ms} & "
                    f"{pm(row.rmse_mean, row.rmse_sd)} & {pm(row.nll_mean, row.nll_sd)} & "
                    f"{pm(row.coverage90_mean, row.coverage90_sd, 3)} \\\\"
                )
    return "\n".join(rows)


def row_at(summary: pd.DataFrame, method: str, condition: str, mt: float, ms: int) -> pd.Series:
    subset = summary.loc[
        (summary.method == method)
        & (summary.condition == condition)
        & (summary.ms == ms)
    ]
    if np.isnan(mt):
        subset = subset.loc[subset.mt.isna()]
    else:
        subset = subset.loc[subset.mt == mt]
    return subset.iloc[0]


def build_section(summary: pd.DataFrame) -> str:
    full64_no = row_at(summary, "Official ST-SVGP", "No X-lag", np.nan, 64)
    full64_x = row_at(summary, "Official ST-SVGP", "Learned X-lag", np.nan, 64)
    full128_no = row_at(summary, "Official ST-SVGP", "No X-lag", np.nan, 128)
    full128_x = row_at(summary, "Official ST-SVGP", "Learned X-lag", np.nan, 128)

    comparisons = []
    for condition in ("No X-lag", "Learned X-lag"):
        for mt, ms in ((8, 64), (32, 128)):
            markov = row_at(summary, "Temporal-inducing Markov ST-SVGP", condition, mt, ms)
            routeb = row_at(summary, "Structured-joint Route B", condition, mt, ms)
            comparisons.append(
                f"For {condition.lower()} at $({mt},{ms})$, the Markov extension and Route B obtained "
                f"RMSE {markov.rmse_mean:.4f} and {routeb.rmse_mean:.4f}, respectively "
                f"(Route B minus Markov: {routeb.rmse_mean - markov.rmse_mean:+.4f})."
            )

    return rf"""
\section{{Phase H: X-lag ablation and finite temporal Markov states}}
\textbf{{Question 1: what does X-lag contribute to official ST-SVGP?}} The paper-faithful spatially sparse model was rerun on the same three 800/200 spatial splits with fixed spatial inducing coordinates. Both versions predict original scaled $y$. The no-X-lag version uses a zero mean; the learned-X-lag version fits $y=\Phi_X\beta+f+\epsilon$ by alternating closed-form $\beta$ coordinate updates with the official Bayes--Newton/Objax GP updates. No target lag $y_{{t-1}}$ is included.

\begin{{table}}[H]\centering\scriptsize
\caption{{X-lag ablation within paper-faithful full-temporal ST-SVGP, mean $\pm$ SD over three paired spatial splits. Spatial inducing coordinates are fixed and shared within each $M_s$. Runtime is training wall time in the legacy JAX environment.}}
\begin{{tabular}}{{lllrrrr}}\toprule
Mean/covariates & Temporal representation & $M_s$ & RMSE & NLL & Cov$_{{90}}$ & Minutes \\\midrule
{full_rows(summary)}
\bottomrule\end{{tabular}}\end{{table}}

\begin{{figure}}[H]\centering\includegraphics[width=0.92\linewidth]{{figures/figure5_official_xlag_ablation.pdf}}
\caption{{Official full-temporal ST-SVGP with and without learned X-lag covariates. Error bars show one SD over three paired spatial splits. Both variants retain all 186 temporal Markov states and predict original $y$.}}
\end{{figure}}

Learned X-lag changed RMSE from {full64_no.rmse_mean:.4f} to {full64_x.rmse_mean:.4f} at $M_s=64$ and from {full128_no.rmse_mean:.4f} to {full128_x.rmse_mean:.4f} at $M_s=128$. The corresponding RMSE reductions were {full64_no.rmse_mean - full64_x.rmse_mean:.4f} and {full128_no.rmse_mean - full128_x.rmse_mean:.4f}. This is an ablation of the additive exogenous mean within the official full-temporal architecture; it does not turn the target into a precomputed residual.

\textbf{{Question 2: can a small $M_t$ preserve the Markov construction?}} It is not possible to set $M_t\ll N_t$ and remain exactly the paper's ST-SVGP, because that model defines a temporal state at every observed time. We instead use Bayes-Newton's \texttt{{SparseMarkovVariationalGP}} to place states at $M_t$ temporal inducing locations. Adjacent inducing states retain state-space Markov transitions, and observations are conditionally projected through their neighbouring inducing-state pair. This preserves Markov transition structure but changes the variational approximation. We therefore call it the \emph{{temporal-inducing Markov ST-SVGP extension}}, not official or paper-faithful ST-SVGP.

\fcolorbox{{accent}}{{soft}}{{\parbox{{0.94\linewidth}}{{\textbf{{Interpretation boundary.}} Full-temporal ST-SVGP has $N_t=186$ temporal states and no $M_t$ hyperparameter. The extension below has $M_t=8$ or $32$ finite temporal inducing states. Only the extension can enter a same-$(M_t,M_s)$ comparison with Route B.}}}}

\textbf{{Matched design.}} The extension and Route B use the same splits, original-$y$ target, fixed spatial inducing coordinates, product Mat\'ern-3/2 spatial kernel, and $(M_t,M_s)$ budget. The comparison is run both without X-lag and with the same non-target X-lag design ($L=10$). For each seed, condition and capacity, the temporal-inducing Markov model first learns kernel and noise parameters with Bayes--Newton/Objax; those fitted parameters are then frozen and supplied to Route B. This controls the numerical kernel in the posterior comparison, but the parameter source favours the Markov extension and does not constitute a method-optimized leaderboard. Posterior inference remains method-specific: inducing-pair CVI for the Markov extension and a closed-form structured joint Gaussian posterior for Route B.

\begin{{table}}[H]\centering\scriptsize
\caption{{Matched temporal-budget comparison, mean $\pm$ SD over three paired spatial splits. The temporal-inducing Markov rows are an extension implemented with official Bayes-Newton machinery, not paper-faithful ST-SVGP.}}
\begin{{tabularx}}{{\textwidth}}{{llYccrrr}}\toprule
Mean & Method & Temporal state & $M_t$ & $M_s$ & RMSE & NLL & Cov$_{{90}}$ \\\midrule
{matched_rows(summary)}
\bottomrule\end{{tabularx}}\end{{table}}

\begin{{figure}}[H]\centering\includegraphics[width=0.96\linewidth]{{figures/figure6_matched_temporal_budget.pdf}}
\caption{{Same-$(M_t,M_s)$ comparison of the temporal-inducing Markov extension and structured-joint Route B. Rows separate the no-X-lag and learned-X-lag conditions; error bars show one SD across paired spatial splits.}}
\end{{figure}}

{' '.join(comparisons)} Conditional on the Markov-trained shared hyperparameters, these comparisons test the analytic HiPPO representation against another finite temporal Markov approximation under a matched nominal state count. They do not replace the full-temporal ST-SVGP upper benchmark. Nominal $M_t$ equality also does not imply identical latent state dimension or per-state computational cost: a Mat\'ern state carries derivative components and the sparse Markov implementation operates on neighbouring state pairs.

\subsection*{{Phase H limitations}}
The learned-X-lag mechanisms are aligned at the observation-model level but are not identical inference algorithms: official Bayes-Newton alternates a point estimate of $\beta$, whereas Route B retains joint Gaussian uncertainty over $(\beta,u)$. Route B does not receive an independent outer-loop hyperparameter optimization in Table 6; a symmetric method-specific tuning study is therefore required before making an architecture ranking. The temporal-inducing extension uses linearly spaced temporal inducing locations; optimizing or adaptively placing them is left for a separate ablation. Runtime remains backend-specific and should not be interpreted as an algorithmic speed ranking.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    frame = collect(root)
    summary = aggregate(frame)
    frame.to_csv(tables / "table5_xlag_temporal_sparse_all_seeds.csv", index=False)
    summary.to_csv(tables / "table5_xlag_temporal_sparse_summary.csv", index=False)
    plot_full_ablation(summary, figures)
    plot_temporal_budget(summary, figures)

    base_tex = root / "unified_stvgp_routeb_staged_experiment_report_xlag_revised.tex"
    if not base_tex.exists():
        raise FileNotFoundError(base_tex)
    source = base_tex.read_text(encoding="utf-8")
    marker = r"\section{Conclusions}"
    if marker not in source:
        raise RuntimeError("Could not find the Conclusions insertion point")
    source = source.replace(marker, build_section(summary) + "\n" + marker, 1)
    source = source.replace(
        "Exact joint $\\beta$--GP coupling remains the strongest evidence for the theoretical contribution; the next technical priority is to transfer that ceiling into the finite analytic HiPPO representation.",
        "Exact joint $\\beta$--GP coupling remains the strongest evidence for the theoretical contribution. Phase H adds the missing X-lag ablation and a finite temporal Markov comparator: the paper-faithful model remains the full-temporal benchmark, whereas same-$M_t$ claims are restricted to the explicitly labelled temporal-inducing extension. The next technical priority is to transfer the exact joint ceiling into the finite analytic HiPPO representation.",
        1,
    )
    tex_path = root / "unified_stvgp_routeb_staged_experiment_report_xlag_temporal_sparse_revised.tex"
    tex_path.write_text(source, encoding="utf-8")

    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        windows_pdflatex = Path(
            "/mnt/c/Users/zhaod/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe"
        )
        if windows_pdflatex.exists():
            pdflatex = str(windows_pdflatex)
    if pdflatex is None:
        raise FileNotFoundError("pdflatex was not found in WSL or the configured MiKTeX path")

    for pass_index in (1, 2):
        completed = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (root / f"pdflatex_temporal_sparse_pass{pass_index}.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout[-4000:])

    manifest = {
        "report": str(tex_path.with_suffix(".pdf")),
        "base_report_preserved": str(base_tex.with_suffix(".pdf")),
        "num_rows": int(len(frame)),
        "num_seeds": int(frame.seed.nunique()),
        "conditions": sorted(frame.condition.unique().tolist()),
        "temporal_sparse_label": "Temporal-inducing Markov ST-SVGP extension",
    }
    (root / "phase_h_xlag_temporal_sparse_markov/manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
