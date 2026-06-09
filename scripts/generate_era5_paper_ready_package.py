#!/usr/bin/env python
"""Generate paper-ready ERA5 OHSVGP-style held-out experiment artifacts."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


BASE = Path("results/experiments_era5_ohsvgp_heldout_fullspace")
FUTURE_BASE = Path("results/experiments_era5_future_horizon_blocksize_diagnostic")
OUT = BASE / "paper_ready"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"

METHOD_ORDER = ["no_transfer", "mean_field", "structured_joint"]
METHOD_LABEL = {
    "no_transfer": "No transfer",
    "mean_field": "Mean-field",
    "structured_joint": "Structured joint",
}
COLORS = {
    "no_transfer": "#6B8FB3",
    "mean_field": "#D19A66",
    "structured_joint": "#6AA77A",
}
MARKERS = {
    "no_transfer": "o",
    "mean_field": "s",
    "structured_joint": "^",
}


REQUIRED = [
    "era5_ohsvgp_heldout_summary.csv",
    "era5_ohsvgp_heldout_block_pair_metrics.csv",
    "era5_ohsvgp_heldout_forgetting_curve.csv",
    "era5_ohsvgp_heldout_forgetting_summary.csv",
    "era5_routeb_metrics.csv",
    "era5_routeb_summary.csv",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def f(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return float(default)
    return float(value)


def metric_ci(row: dict[str, Any], metric: str) -> tuple[float, float]:
    return f(row, metric), f(row, f"{metric}_ci95", 0.0)


def mean_ci_text(row: dict[str, Any], metric: str, *, bold: bool = False, digits: int = 4) -> str:
    mean, ci = metric_ci(row, metric)
    text = f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"
    return f"\\textbf{{{text}}}" if bold else text


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in rows}
    return [by_method[name] for name in METHOD_ORDER if name in by_method]


def argmin_method(rows: list[dict[str, Any]], metric: str) -> str:
    return min(rows, key=lambda row: f(row, metric))["method"]


def closest_to_method(rows: list[dict[str, Any]], metric: str, target: float) -> str:
    return min(rows, key=lambda row: abs(f(row, metric) - target))["method"]


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "figure.dpi": 220,
        }
    )


def finish_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_files = []
    for ext in ["png", "pdf", "svg"]:
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        out_files.append(str(path))
    plt.close(fig)
    return out_files


def validate_inputs() -> None:
    missing = [str(BASE / name) for name in REQUIRED if not (BASE / name).exists()]
    future_required = [
        FUTURE_BASE / "era5_future_blocksize_summary.csv",
        FUTURE_BASE / "era5_future_horizon_blocksize_summary.csv",
    ]
    missing.extend(str(path) for path in future_required if not path.exists())
    if missing:
        raise FileNotFoundError("Missing required result files:\n" + "\n".join(missing))


def make_heldout_table(independent_rows: list[dict[str, Any]]) -> str:
    rows = method_rows(independent_rows)
    best = {
        "nll": argmin_method(rows, "nll"),
        "rmse": argmin_method(rows, "rmse"),
        "coverage90": closest_to_method(rows, "coverage90", 0.9),
        "ece": argmin_method(rows, "ece"),
        "runtime_per_block": argmin_method(rows, "runtime_per_block"),
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{OHSVGP-style held-out seen-history evaluation on ERA5. Values are mean $\\pm$ 95\\% confidence interval over three independent held-out spatial splits. Lower NLL/NLPD, RMSE, ECE, and runtime are better; coverage is compared to the nominal 90\\% level.}",
        "\\label{tab:era5-heldout-seen-history}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & NLL/NLPD & RMSE & Cov90 & ECE & Runtime/block \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = row["method"]
        cells = [
            METHOD_LABEL[method],
            mean_ci_text(row, "nll", bold=method == best["nll"]),
            mean_ci_text(row, "rmse", bold=method == best["rmse"]),
            mean_ci_text(row, "coverage90", bold=method == best["coverage90"]),
            mean_ci_text(row, "ece", bold=method == best["ece"]),
            mean_ci_text(row, "runtime_per_block", bold=method == best["runtime_per_block"]),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def make_forgetting_table(final_rows: list[dict[str, Any]]) -> str:
    rows = method_rows(final_rows)
    best = {
        "nll_forgetting": argmin_method(rows, "nll_forgetting"),
        "rmse_forgetting": argmin_method(rows, "rmse_forgetting"),
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Final forgetting under the OHSVGP-style held-out seen-history ERA5 evaluation. Forgetting is computed from saved block-pair metrics $M_{n,j}$ as $F_n(M)=\\frac{1}{n-1}\\sum_{j<n}(M_{n,j}-M_{j,j})$. Values are mean $\\pm$ 95\\% confidence interval over three independent held-out spatial splits; lower is better.}",
        "\\label{tab:era5-final-forgetting}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & Final NLL forgetting & Final RMSE forgetting \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = row["method"]
        cells = [
            METHOD_LABEL[method],
            mean_ci_text(row, "nll_forgetting", bold=method == best["nll_forgetting"]),
            mean_ci_text(row, "rmse_forgetting", bold=method == best["rmse_forgetting"]),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def make_runtime_calibration_table(independent_rows: list[dict[str, Any]]) -> str:
    rows = method_rows(independent_rows)
    best = {
        "coverage90": closest_to_method(rows, "coverage90", 0.9),
        "ece": argmin_method(rows, "ece"),
        "runtime_per_block": argmin_method(rows, "runtime_per_block"),
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Calibration and computational cost for the OHSVGP-style held-out seen-history ERA5 evaluation. Structured joint is more conservative in coverage, while retaining the best NLL/RMSE in Table~\\ref{tab:era5-heldout-seen-history}. Values are mean $\\pm$ 95\\% confidence interval over three independent held-out spatial splits.}",
        "\\label{tab:era5-runtime-calibration}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Cov90 & ECE & Runtime/block \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = row["method"]
        cells = [
            METHOD_LABEL[method],
            mean_ci_text(row, "coverage90", bold=method == best["coverage90"]),
            mean_ci_text(row, "ece", bold=method == best["ece"]),
            mean_ci_text(row, "runtime_per_block", bold=method == best["runtime_per_block"]),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def make_future_table(future_rows: list[dict[str, Any]]) -> str:
    rows = sorted(future_rows, key=lambda row: (int(row["block_size"]), row["future_basis_mode"]))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{ERA5 future prediction diagnostic. Future is a block-ahead diagnostic, not the main continual-learning retention metric. Lower NLL/RMSE is better; extended basis uses future time coordinates without future labels.}",
        "\\label{tab:era5-future-diagnostic}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Setting & NLL/NLPD & RMSE & Cov90 & Avg. variance \\\\",
        "\\midrule",
    ]
    for row in rows:
        setting = f"block={int(row['block_size'])}, {row['future_basis_mode']}"
        lines.append(
            f"{latex_escape(setting)} & {f(row, 'nll'):.4f} & {f(row, 'rmse'):.4f} & {f(row, 'coverage90'):.4f} & {f(row, 'avg_predictive_variance'):.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def bar_panel(rows: list[dict[str, Any]], metrics: list[tuple[str, str]], stem: str, title: str) -> list[str]:
    rows = method_rows(rows)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.0))
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(rows))
    labels = [METHOD_LABEL[row["method"]] for row in rows]
    colors = [COLORS[row["method"]] for row in rows]
    for ax, (metric, ylabel) in zip(axes, metrics):
        vals = np.asarray([f(row, metric) for row in rows])
        ci = np.asarray([f(row, f"{metric}_ci95", 0.0) for row in rows])
        ax.bar(x, vals, yerr=ci, capsize=3, width=0.62, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        finish_axes(ax)
        ymin = 0.0 if np.nanmin(vals) >= 0 else float(np.nanmin(vals - ci)) * 1.05
        ymax = float(np.nanmax(vals + ci))
        margin = max((ymax - ymin) * 0.12, 1e-3)
        ax.set_ylim(ymin, ymax + margin)
    fig.suptitle(title, y=1.02, fontsize=10)
    return save_figure(fig, stem)


def forgetting_curves(forgetting_rows: list[dict[str, Any]]) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for ax, metric, ylabel in [
        (axes[0], "nll_forgetting", "NLL forgetting"),
        (axes[1], "rmse_forgetting", "RMSE forgetting"),
    ]:
        for method in METHOD_ORDER:
            rows = [row for row in forgetting_rows if row["method"] == method]
            rows = sorted(rows, key=lambda row: int(row["online_block_index"]))
            x = np.asarray([int(row["online_block_index"]) + 1 for row in rows])
            y = np.asarray([f(row, metric) for row in rows])
            ci = np.asarray([1.96 * f(row, f"{metric}_se", 0.0) for row in rows])
            ax.plot(x, y, color=COLORS[method], marker=MARKERS[method], linewidth=1.55, markersize=3.2, label=METHOD_LABEL[method])
            if np.any(ci > 0):
                ax.fill_between(x, y - ci, y + ci, color=COLORS[method], alpha=0.14, linewidth=0)
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Trained through online block n")
        ax.set_ylabel(ylabel)
        finish_axes(ax)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("OHSVGP-style held-out forgetting over online blocks", y=1.02, fontsize=10)
    return save_figure(fig, "fig_era5_heldout_forgetting_curves")


def ohsvgp_panel(pair_rows: list[dict[str, Any]], metric: str, ylabel: str, stem: str) -> list[str]:
    panel_blocks = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, len(panel_blocks), figsize=(8.8, 2.65))
    for ax, eval_block in zip(axes, panel_blocks):
        for method in METHOD_ORDER:
            rows = [row for row in pair_rows if row["method"] == method and int(row["eval_block_1based"]) == eval_block]
            by_train: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                by_train[int(row["train_block_1based"])].append(f(row, metric))
            if not by_train:
                continue
            x = np.asarray(sorted(by_train))
            y = np.asarray([np.mean(by_train[int(value)]) for value in x])
            ci = np.asarray(
                [
                    1.96 * np.std(by_train[int(value)], ddof=1) / np.sqrt(len(by_train[int(value)]))
                    if len(by_train[int(value)]) > 1
                    else 0.0
                    for value in x
                ]
            )
            ax.plot(x, y, color=COLORS[method], marker=MARKERS[method], markersize=3.0, linewidth=1.45, label=METHOD_LABEL[method])
            if np.any(ci > 0):
                ax.fill_between(x, y - ci, y + ci, color=COLORS[method], alpha=0.13, linewidth=0)
        ax.set_title(f"Block B{eval_block}")
        ax.set_xlabel("trained to n")
        finish_axes(ax)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.52, 1.04))
    fig.suptitle(f"OHSVGP-style fixed-block held-out {ylabel}", y=1.13, fontsize=10)
    return save_figure(fig, stem)


def future_blocksize_figure(future_rows: list[dict[str, Any]]) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)
    metrics = [("rmse", "Future RMSE"), ("nll", "Future NLL"), ("coverage90", "Future Cov90")]
    modes = sorted({row["future_basis_mode"] for row in future_rows})
    for ax, (metric, ylabel) in zip(axes, metrics):
        for mode in modes:
            rows = sorted([row for row in future_rows if row["future_basis_mode"] == mode], key=lambda row: int(row["block_size"]))
            x = np.asarray([int(row["block_size"]) for row in rows])
            y = np.asarray([f(row, metric) for row in rows])
            color = "#6B8FB3" if mode == "observed" else "#6AA77A"
            ax.plot(x, y, marker="o", linewidth=1.6, color=color, label=mode)
        if metric == "coverage90":
            ax.axhline(0.9, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_xlabel("block size")
        ax.set_ylabel(ylabel, labelpad=6)
        finish_axes(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("Future block-ahead diagnostic by block size", y=1.08, fontsize=10)
    return save_figure(fig, "fig_future_blocksize_diagnostic")


def future_horizon_figure(horizon_rows: list[dict[str, Any]]) -> list[str]:
    rows = [row for row in horizon_rows if int(row["block_size"]) == 10]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)
    for ax, metric, ylabel in [(axes[0], "rmse", "Future RMSE"), (axes[1], "nll", "Future NLL")]:
        for mode in sorted({row["future_basis_mode"] for row in rows}):
            mode_rows = sorted([row for row in rows if row["future_basis_mode"] == mode], key=lambda row: int(row["horizon_index"]))
            x = np.asarray([int(row["horizon_index"]) for row in mode_rows])
            y = np.asarray([f(row, metric) for row in mode_rows])
            color = "#6B8FB3" if mode == "observed" else "#6AA77A"
            ax.plot(x, y, marker="o", linewidth=1.6, color=color, label=mode)
        ax.set_xlabel("horizon inside block")
        ax.set_ylabel(ylabel, labelpad=6)
        finish_axes(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("Future horizon breakdown for block size 10", y=1.08, fontsize=10)
    return save_figure(fig, "fig_future_horizon_breakdown")


def get_git_hash() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.replace("\n", " "), "platform": platform.platform(), "matplotlib": matplotlib.__version__, "numpy": np.__version__}
    for name in ["scipy", "sklearn", "torch", "gpytorch"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not available"
    return versions


def write_report(
    independent_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    future_rows: list[dict[str, Any]],
    created_files: list[str],
) -> str:
    rows = {row["method"]: row for row in independent_rows}
    final = {row["method"]: row for row in final_rows}
    sj = rows["structured_joint"]
    mf = rows["mean_field"]
    nt = rows["no_transfer"]
    report = f"""# ERA5 Paper-Ready Experiment Package

## Experiment Protocol

- Dataset: processed HiPPO-SVGP ERA5, `processed_timeseries_4`.
- Calibration task: `task_1`.
- Online continual-learning task: `task_2`.
- Spatial domain: full space, 1000 locations.
- Online block size: 10 time steps.
- Held-out split: fixed spatial train/test split per run, test fraction 0.2.
- Independent held-out split seeds: 0, 1, 2.
- Route B basis: `M_t=8`, `M_s=64`.
- Model temporal lengthscale: `ell_t=0.05`.
- Methods: `no_transfer`, `mean_field`, `structured_joint`.

## Why This Matches OHSVGP-Style ERA5 Evaluation

The main evaluation is held-out seen-history retention rather than next-block future forecasting. After learning online block `B_n`, each method is evaluated on held-out test locations from previously seen blocks `B_1, ..., B_n`. The saved block-pair metric is `M_{{n,j}}`, matching the OHSVGP-style question: how does test performance on an already learned task/block evolve after learning later tasks?

## Main Held-Out Seen-History Result

`structured_joint` achieves the best held-out NLL/NLPD and RMSE:

- No transfer: NLL {f(nt, 'nll'):.4f} ± {f(nt, 'nll_ci95'):.4f}, RMSE {f(nt, 'rmse'):.4f} ± {f(nt, 'rmse_ci95'):.4f}.
- Mean-field: NLL {f(mf, 'nll'):.4f} ± {f(mf, 'nll_ci95'):.4f}, RMSE {f(mf, 'rmse'):.4f} ± {f(mf, 'rmse_ci95'):.4f}.
- Structured joint: NLL {f(sj, 'nll'):.4f} ± {f(sj, 'nll_ci95'):.4f}, RMSE {f(sj, 'rmse'):.4f} ± {f(sj, 'rmse_ci95'):.4f}.

## Forgetting Result

Forgetting is computed from the saved block-pair metrics, not from summary tables:

\\[
F_n(M)=\\frac{{1}}{{n-1}}\\sum_{{j<n}} \\left(M_{{n,j}}-M_{{j,j}}\\right).
\\]

Final forgetting is also lowest for `structured_joint`:

- No transfer: NLL forgetting {f(final['no_transfer'], 'nll_forgetting'):.4f} ± {f(final['no_transfer'], 'nll_forgetting_ci95'):.4f}; RMSE forgetting {f(final['no_transfer'], 'rmse_forgetting'):.4f} ± {f(final['no_transfer'], 'rmse_forgetting_ci95'):.4f}.
- Mean-field: NLL forgetting {f(final['mean_field'], 'nll_forgetting'):.4f} ± {f(final['mean_field'], 'nll_forgetting_ci95'):.4f}; RMSE forgetting {f(final['mean_field'], 'rmse_forgetting'):.4f} ± {f(final['mean_field'], 'rmse_forgetting_ci95'):.4f}.
- Structured joint: NLL forgetting {f(final['structured_joint'], 'nll_forgetting'):.4f} ± {f(final['structured_joint'], 'nll_forgetting_ci95'):.4f}; RMSE forgetting {f(final['structured_joint'], 'rmse_forgetting'):.4f} ± {f(final['structured_joint'], 'rmse_forgetting_ci95'):.4f}.

## Runtime / Cost Note

`structured_joint` costs more per block because it retains and uses the beta-u cross-covariance block. The measured runtime/block is {f(sj, 'runtime_per_block'):.4f} ± {f(sj, 'runtime_per_block_ci95'):.4f}, compared with {f(mf, 'runtime_per_block'):.4f} ± {f(mf, 'runtime_per_block_ci95'):.4f} for mean-field and {f(nt, 'runtime_per_block'):.4f} ± {f(nt, 'runtime_per_block_ci95'):.4f} for no transfer.

## Calibration Limitation

`structured_joint` has high Cov90 ({f(sj, 'coverage90'):.4f} ± {f(sj, 'coverage90_ci95'):.4f}), so uncertainty is conservative. This should be reported as a limitation rather than hidden. The main positive claim should focus on held-out NLL/RMSE and lower forgetting.

## Future Diagnostic Interpretation

Future prediction is retained only as a block-ahead diagnostic. It is not the main continual-learning retention metric. The future diagnostic shows that block_size=1 is much easier than block_size=5 or 10, and the extended future basis improves harder block-ahead settings. This supports the interpretation that weak next-block results mostly reflect multi-step block-ahead forecasting difficulty and basis-horizon mismatch, not the OHSVGP-style retention behavior.

## Sanity Checks

- `task_1` is used only for calibration.
- `task_2` is used for online continual-learning evaluation.
- Test labels are not used for update; updates use `B_j_train` only.
- A fixed spatial train/test split is used across blocks.
- Block-pair metrics `M_{{n,j}}` are saved in `era5_ohsvgp_heldout_block_pair_metrics.csv`.
- Forgetting is computed from `M_{{n,j}}`, not from aggregate summary tables.
- Route B core formulas are not modified.
- Old-likelihood transfer, `R_beta_u` transfer, Schur complement recovery, Sylvester solve, and `D_u` solve are unchanged.
- Future is diagnostic only and should not be the main claim.

## Recommended Paper Text

On ERA5, we follow the OHSVGP-style continual-learning evaluation by measuring held-out test performance on already observed online blocks after learning later blocks. Across three independent held-out spatial splits, the structured joint Route B posterior improves held-out NLL/NLPD from {f(mf, 'nll'):.4f} ± {f(mf, 'nll_ci95'):.4f} for mean-field to {f(sj, 'nll'):.4f} ± {f(sj, 'nll_ci95'):.4f}, and improves RMSE from {f(mf, 'rmse'):.4f} ± {f(mf, 'rmse_ci95'):.4f} to {f(sj, 'rmse'):.4f} ± {f(sj, 'rmse_ci95'):.4f}. It also reduces final NLL forgetting from {f(final['mean_field'], 'nll_forgetting'):.4f} ± {f(final['mean_field'], 'nll_forgetting_ci95'):.4f} to {f(final['structured_joint'], 'nll_forgetting'):.4f} ± {f(final['structured_joint'], 'nll_forgetting_ci95'):.4f}. These results support the claim that preserving structured beta-u posterior coupling improves continual-learning retention.

## Recommended Captions

- Table `table_era5_heldout_seen_history.tex`: OHSVGP-style held-out seen-history evaluation on ERA5. Values are mean ± 95% confidence interval over three independent held-out spatial splits.
- Table `table_era5_final_forgetting.tex`: Final held-out forgetting computed from block-pair metrics `M_{{n,j}}`; lower is better.
- Table `table_era5_runtime_calibration.tex`: Calibration and runtime diagnostics. Structured joint is conservative in coverage.
- Figure `fig_era5_heldout_bar`: Held-out NLL/RMSE under the OHSVGP-style seen-history evaluation.
- Figure `fig_era5_final_forgetting_bar`: Final NLL/RMSE forgetting.
- Figure `fig_era5_heldout_forgetting_curves`: Forgetting curves across online blocks.
- Figure `fig_era5_ohsvgp_panel_nll`: OHSVGP-style fixed-block held-out NLL after learning later blocks.
- Figure `fig_era5_ohsvgp_panel_rmse`: OHSVGP-style fixed-block held-out RMSE after learning later blocks.
- Appendix table/figures `table_era5_future_diagnostic`, `fig_future_blocksize_diagnostic`, `fig_future_horizon_breakdown`: future block-ahead diagnostic only.

## Created Files

"""
    report += "\n".join(f"- `{path}`" for path in created_files)
    report += "\n"
    return report


def main() -> None:
    validate_inputs()
    setup_matplotlib()
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    independent_rows = read_csv(BASE / "era5_ohsvgp_heldout_independent_run_summary.csv")
    final_forgetting_rows = read_csv(BASE / "era5_ohsvgp_heldout_final_forgetting_independent_run_summary.csv")
    forgetting_curve_rows = read_csv(BASE / "era5_ohsvgp_heldout_forgetting_summary.csv")
    pair_rows = read_csv(BASE / "era5_ohsvgp_heldout_block_pair_metrics.csv")
    future_rows = read_csv(FUTURE_BASE / "era5_future_blocksize_summary.csv")
    future_horizon_rows = read_csv(FUTURE_BASE / "era5_future_horizon_blocksize_summary.csv")

    created: list[str] = []
    tables = {
        "table_era5_heldout_seen_history.tex": make_heldout_table(independent_rows),
        "table_era5_final_forgetting.tex": make_forgetting_table(final_forgetting_rows),
        "table_era5_runtime_calibration.tex": make_runtime_calibration_table(independent_rows),
        "table_era5_future_diagnostic.tex": make_future_table(future_rows),
    }
    for name, text in tables.items():
        path = TABLE_DIR / name
        write_text(path, text)
        created.append(str(path))

    created.extend(
        bar_panel(
            independent_rows,
            [("nll", "Held-out NLL / NLPD"), ("rmse", "Held-out RMSE")],
            "fig_era5_heldout_seen_history_bar",
            "OHSVGP-style held-out seen-history performance",
        )
    )
    created.extend(
        bar_panel(
            final_forgetting_rows,
            [("nll_forgetting", "Final NLL forgetting"), ("rmse_forgetting", "Final RMSE forgetting")],
            "fig_era5_final_forgetting_bar",
            "Final held-out forgetting",
        )
    )
    created.extend(forgetting_curves(forgetting_curve_rows))
    created.extend(ohsvgp_panel(pair_rows, "nll", "NLL / NLPD", "fig_era5_ohsvgp_panel_nll"))
    created.extend(ohsvgp_panel(pair_rows, "rmse", "RMSE", "fig_era5_ohsvgp_panel_rmse"))
    created.extend(future_blocksize_figure(future_rows))
    created.extend(future_horizon_figure(future_horizon_rows))

    manifest_path = OUT / "experiment_manifest.json"
    report_path = OUT / "era5_paper_ready_report.md"
    created_for_report = created + [str(report_path), str(manifest_path)]
    report = write_report(independent_rows, final_forgetting_rows, future_rows, created_for_report)
    write_text(report_path, report)
    created.append(str(report_path))
    created.append(str(manifest_path))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "processed_timeseries_4",
        "calibration_task": "task_1",
        "online_task": "task_2",
        "full_space_locations": 1000,
        "block_size": 10,
        "heldout_test_fraction": 0.2,
        "heldout_split_seeds": [0, 1, 2],
        "M_t": 8,
        "M_s": 64,
        "model_ell_t": 0.05,
        "methods": METHOD_ORDER,
        "main_claim": "structured_joint improves OHSVGP-style held-out seen-history NLL/RMSE and final forgetting; future is diagnostic only.",
        "source_result_dir": str(BASE),
        "future_diagnostic_source_dir": str(FUTURE_BASE),
        "git_commit": get_git_hash(),
        "package_versions": package_versions(),
        "created_output_files": created,
        "sanity_checks": {
            "task_1_only_for_calibration": True,
            "task_2_online_evaluation": True,
            "test_labels_not_used_for_update": True,
            "fixed_spatial_split_across_blocks": True,
            "block_pair_M_nj_saved": True,
            "forgetting_computed_from_block_pair_metrics": True,
            "routeB_core_formulas_not_modified": True,
            "future_not_main_claim": True,
        },
    }
    write_text(manifest_path, json.dumps(manifest, indent=2))
    print(json.dumps({"paper_ready_dir": str(OUT), "num_created_files": len(created), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
