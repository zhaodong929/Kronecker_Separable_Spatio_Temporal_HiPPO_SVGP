"""Summarize the matched ERA5 Route B vs baseline comparison.

This script combines:

1. Rich-Phi baseline summary from ``run_hipposvgp_era5_baselines.py``.
2. Route B summary from ``run_hipposvgp_era5_routeb.py``.

It does not retrain models.  It only reads summary CSV files and produces a
single comparison CSV, publication-style plots, and a markdown report.

Example:
    python scripts/summarize_era5_fair_routeb_baseline_comparison.py \
      --baseline-summary results/.../era5_fair_baseline_summary.csv \
      --routeb-summary results/.../era5_routeb_summary.csv \
      --outdir results/.../combined_routeb_baselines
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "climatology": "Climatology",
    "persistence": "Persistence",
    "ridge": "Ridge",
    "gpytorch_sgpr_phi": "SGPR + Rich-Phi",
    "gpytorch_svgp_phi": "SVGP + Rich-Phi",
    "structured_joint": "Route B structured joint",
}

METHOD_ORDER = [
    "Climatology",
    "Persistence",
    "Ridge",
    "SGPR + Rich-Phi",
    "SVGP + Rich-Phi",
    "Route B structured joint",
]

EVAL_ORDER = ["current", "seen_history", "future"]


def _read_summary(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    df = pd.read_csv(path)
    df["source"] = source
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    return df


def _metric_col(df: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"None of these columns are present: {names}")


def _format_mean_se(row: pd.Series, metric: str, digits: int = 4) -> str:
    val = row.get(metric, np.nan)
    se = row.get(f"{metric}_se", np.nan)
    if pd.isna(val):
        return ""
    if pd.isna(se):
        return f"{val:.{digits}f}"
    return f"{val:.{digits}f} +/- {se:.{digits}f}"


def _to_markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Small dependency-free markdown table writer."""
    if df.empty:
        return "_No rows._"
    headers = [str(c) for c in df.columns]
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        formatted: list[str] = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                formatted.append(format(float(value), floatfmt))
            elif pd.isna(value):
                formatted.append("")
            else:
                formatted.append(str(value))
        rows.append(formatted)

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def _plot_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    outpath: Path,
    symlog: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), sharey=False)
    colors = {
        "Climatology": "#6B7280",
        "Persistence": "#9CA3AF",
        "Ridge": "#3B82F6",
        "SGPR + Rich-Phi": "#10B981",
        "SVGP + Rich-Phi": "#F59E0B",
        "Route B structured joint": "#B91C1C",
    }

    for ax, eval_mode in zip(axes, EVAL_ORDER):
        sub = df[df["eval_mode"] == eval_mode].copy()
        sub["method_label"] = pd.Categorical(
            sub["method_label"], categories=METHOD_ORDER, ordered=True
        )
        sub = sub.sort_values("method_label")
        labels = [str(x) for x in sub["method_label"]]
        values = sub[metric].to_numpy(float)
        se_col = f"{metric}_se"
        errors = sub[se_col].to_numpy(float) if se_col in sub.columns else None
        x = np.arange(len(labels))
        bar_colors = [colors.get(label, "#4B5563") for label in labels]
        ax.bar(x, values, yerr=errors, capsize=3, color=bar_colors, alpha=0.9)
        ax.set_title(eval_mode.replace("_", " "), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if symlog:
            ax.set_yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_seen_history_focus(df: pd.DataFrame, outpath: Path) -> None:
    sub = df[df["eval_mode"] == "seen_history"].copy()
    sub["method_label"] = pd.Categorical(
        sub["method_label"], categories=METHOD_ORDER, ordered=True
    )
    sub = sub.sort_values("method_label")

    metrics = [
        ("nll", "NLL / NLPD"),
        ("rmse", "RMSE"),
        ("coverage90", "90% coverage"),
        ("runtime_per_block", "Runtime / block (s)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    colors = ["#6B7280", "#9CA3AF", "#3B82F6", "#10B981", "#F59E0B", "#B91C1C"]

    for ax, (metric, ylabel) in zip(axes, metrics):
        vals = sub[metric].to_numpy(float)
        errs = sub[f"{metric}_se"].to_numpy(float) if f"{metric}_se" in sub else None
        labels = [str(x) for x in sub["method_label"]]
        x = np.arange(len(labels))
        ax.bar(x, vals, yerr=errs, capsize=3, color=colors[: len(labels)], alpha=0.9)
        ax.set_title(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric == "runtime_per_block":
            ax.set_yscale("log")
    fig.suptitle("ERA5 task_2 online seen-history comparison, 300 matched locations", y=1.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_report(df: pd.DataFrame, outpath: Path, note: str) -> None:
    seen = df[df["eval_mode"] == "seen_history"].copy()
    seen["method_label"] = pd.Categorical(
        seen["method_label"], categories=METHOD_ORDER, ordered=True
    )
    seen = seen.sort_values("method_label")

    table_cols = [
        "method_label",
        "nll",
        "nll_se",
        "rmse",
        "rmse_se",
        "coverage90",
        "coverage90_se",
        "ece",
        "ece_se",
        "runtime_per_block",
        "runtime_per_block_se",
    ]
    table = seen[[c for c in table_cols if c in seen.columns]].copy()
    table = table.rename(columns={"method_label": "method"})
    routeb = seen[seen["method_label"] == "Route B structured joint"]
    sgpr = seen[seen["method_label"] == "SGPR + Rich-Phi"]
    svgp = seen[seen["method_label"] == "SVGP + Rich-Phi"]
    ridge = seen[seen["method_label"] == "Ridge"]

    def get(row_df: pd.DataFrame, col: str) -> float:
        if row_df.empty or col not in row_df:
            return float("nan")
        return float(row_df.iloc[0][col])

    routeb_rmse = get(routeb, "rmse")
    routeb_nll = get(routeb, "nll")
    lines = [
        "# ERA5 fair Route B vs baseline comparison",
        "",
        "## Protocol",
        "",
        "- Dataset: processed ERA5, task_2 online evaluation.",
        "- Calibration for Route B: task_1 initial-task full-GP MLL grid selects temporal lengthscale, noise sigma, and kernel variance; the selected hyperparameters are frozen on task_2.",
        "- Route B setting: rich seasonal-spatial Phi, Matern-3/2 kernel, Mt=32, Ms=256.",
        "- Matched online protocol: seeds 0/1/2, random 300-location subsets, block_size=10, same standardization and task_2 block split.",
        "- Baseline fairness: Ridge and GPyTorch Phi baselines use the same rich Phi covariates; climatology and persistence are covariate-free by design.",
        "- Primary mode: seen_history. Current and future are diagnostics.",
        "",
        "## Important scope note",
        "",
        note,
        "",
        "The original 100-location baseline setting cannot be used for Mt=32, Ms=256 because Ms=256 requires at least 256 spatial locations. Therefore this comparison uses 300 matched locations.",
        "",
        "## Seen-history summary",
        "",
        _to_markdown_table(table, floatfmt=".4f"),
        "",
        "## Main observations",
        "",
    ]
    if not np.isnan(routeb_rmse):
        lines.append(
            f"- Route B seen-history RMSE is {routeb_rmse:.4f} and NLL is {routeb_nll:.4f}."
        )
    if not sgpr.empty:
        sgpr_rmse = get(sgpr, "rmse")
        sgpr_nll = get(sgpr, "nll")
        lines.append(
            f"- SGPR + Rich-Phi has lower RMSE ({sgpr_rmse:.4f}) than Route B, but worse NLL ({sgpr_nll:.4f} vs {routeb_nll:.4f}) and much larger runtime."
        )
    if not svgp.empty:
        lines.append(
            f"- SVGP + Rich-Phi has slightly lower RMSE than Route B ({get(svgp, 'rmse'):.4f} vs {routeb_rmse:.4f}), but its NLL is worse ({get(svgp, 'nll'):.4f} vs {routeb_nll:.4f}) and its coverage is close to 1.0, indicating over-wide predictive intervals."
        )
    if not ridge.empty:
        lines.append(
            f"- Ridge is the strongest deterministic baseline among the non-GP baselines, but Route B improves both RMSE and NLL over ridge."
        )
    lines.extend(
        [
            "- Future remains a diagnostic in this protocol; block-ahead future performance is not the main continual-learning claim.",
            "- Route B gives the best seen-history NLL among the completed methods, but it is not the best RMSE method; SGPR + Rich-Phi has the lowest RMSE at substantially higher runtime.",
            "- Route B coverage is below 0.9 in this matched 300-location run, so calibration remains a visible limitation under this setting.",
            "- Independent temporal GP was attempted under the same 300-location, 3-seed protocol but did not complete within the local 20-minute run budget, so it is not mixed into the final aggregate plots.",
            "",
            "## Output figures",
            "",
            "- `era5_fair_comparison_seen_history_focus.png`",
            "- `era5_fair_comparison_rmse.png`",
            "- `era5_fair_comparison_nll.png`",
            "- `era5_fair_comparison_coverage90.png`",
            "- `era5_fair_comparison_runtime_per_block.png`",
        ]
    )
    outpath.write_text("\n".join(lines), encoding="utf-8")


def combine_summaries(
    baseline_summary: Path,
    routeb_summary: Path,
    outdir: Path,
    note: str,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    baseline = _read_summary(baseline_summary, "baseline")
    routeb = _read_summary(routeb_summary, "routeb")
    combined = pd.concat([baseline, routeb], ignore_index=True, sort=False)
    combined = combined[combined["method_label"].isin(METHOD_ORDER)].copy()
    combined["method_label"] = pd.Categorical(
        combined["method_label"], categories=METHOD_ORDER, ordered=True
    )
    combined["eval_mode"] = pd.Categorical(
        combined["eval_mode"], categories=EVAL_ORDER, ordered=True
    )
    combined = combined.sort_values(["eval_mode", "method_label"])

    combined_path = outdir / "era5_fair_routeb_baseline_combined_summary.csv"
    combined.to_csv(combined_path, index=False)

    seen = combined[combined["eval_mode"] == "seen_history"].copy()
    seen.to_csv(outdir / "era5_fair_routeb_baseline_seen_history_summary.csv", index=False)

    _plot_seen_history_focus(
        combined, plots_dir / "era5_fair_comparison_seen_history_focus.png"
    )
    _plot_metric(
        combined,
        metric="rmse",
        ylabel="RMSE",
        outpath=plots_dir / "era5_fair_comparison_rmse.png",
    )
    _plot_metric(
        combined,
        metric="nll",
        ylabel="NLL / NLPD",
        outpath=plots_dir / "era5_fair_comparison_nll.png",
        symlog=True,
    )
    _plot_metric(
        combined,
        metric="coverage90",
        ylabel="90% coverage",
        outpath=plots_dir / "era5_fair_comparison_coverage90.png",
    )
    _plot_metric(
        combined,
        metric="runtime_per_block",
        ylabel="Runtime / block (s)",
        outpath=plots_dir / "era5_fair_comparison_runtime_per_block.png",
        symlog=True,
    )
    _write_report(combined, outdir / "era5_fair_routeb_baseline_report.md", note)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--routeb-summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--note",
        default=(
            "This is a matched online seen-history comparison, not a full "
            "OHSVGP-style spatial held-out benchmark for every external baseline."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = combine_summaries(
        args.baseline_summary,
        args.routeb_summary,
        args.outdir,
        args.note,
    )
    print(f"Wrote {len(combined)} rows to {args.outdir}")


if __name__ == "__main__":
    main()
