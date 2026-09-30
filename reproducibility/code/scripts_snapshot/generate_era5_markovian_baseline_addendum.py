"""Generate an ERA5 baseline-suite addendum report.

This script combines the existing ERA5 fair-baseline outputs, Route B outputs,
same-kernel/frozen-hyperparameter SGPR/SVGP outputs when available, and the
external-online-GP manifest. It intentionally distinguishes runnable local
results from downloaded-but-not-yet-integrated external reference code.

Example:

    uv run --no-sync python scripts/generate_era5_markovian_baseline_addendum.py \
      --base-dir results/experiments_era5_fair_routeb_baseline_comparison_300loc_richphi \
      --output-dir results/experiments_era5_fair_routeb_baseline_comparison_300loc_richphi/next_steps_markovian_baselines/integrated_report
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


METRICS = ["nll", "rmse", "coverage90", "ece", "runtime_per_block"]


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_existing_results(base_dir: Path) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}

    fair = _read_csv(base_dir / "combined_routeb_baselines" / "era5_fair_routeb_baseline_combined_summary.csv")
    if fair is None:
        fair = _read_csv(base_dir / "era5_fair_baseline_summary.csv")
    if fair is not None:
        fair = fair.copy()
        fair["result_group"] = "existing_300loc_fair_baseline"
        results["existing_300loc_fair_baseline"] = fair

    smoke = _read_csv(base_dir / "next_steps_markovian_baselines" / "smoke_same_kernel" / "smoke_same_kernel_summary.csv")
    if smoke is not None:
        smoke = smoke.copy()
        smoke["result_group"] = "same_kernel_frozen_tiny_smoke"
        results["same_kernel_frozen_tiny_smoke"] = smoke

    same_kernel_candidates = [
        base_dir / "next_steps_markovian_baselines" / "same_kernel_frozen_300loc" / "era5_same_kernel_frozen_summary.csv",
        base_dir / "next_steps_markovian_baselines" / "same_kernel_frozen_1000loc" / "era5_same_kernel_frozen_summary.csv",
    ]
    for candidate in same_kernel_candidates:
        df = _read_csv(candidate)
        if df is not None:
            key = candidate.parent.name
            df = df.copy()
            df["result_group"] = key
            results[key] = df

    return results


def _format_mean_se(row: pd.Series, metric: str) -> str:
    value = row.get(metric, np.nan)
    se = row.get(f"{metric}_se", np.nan)
    if pd.isna(value):
        return ""
    if pd.notna(se):
        return f"{value:.4f} ± {se:.4f}"
    return f"{value:.4f}"


def _seen_history_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["eval_mode"].astype(str).eq("seen_history")].copy()
    if sub.empty:
        return sub
    cols = ["method"]
    for optional in ["phi_mode", "gp_kernel_type", "gp_hyperparam_fit_mode"]:
        if optional in sub.columns:
            cols.append(optional)
    out = sub[cols].copy()
    for metric in METRICS:
        if metric in sub.columns:
            out[metric] = sub.apply(lambda r: _format_mean_se(r, metric), axis=1)
    return out


def _plot_metric_bars(df: pd.DataFrame, outdir: Path, prefix: str) -> list[Path]:
    paths: list[Path] = []
    sub = df[df["eval_mode"].astype(str).eq("seen_history")].copy()
    if sub.empty:
        return paths
    label_col = "method"
    if "phi_mode" in sub.columns:
        sub["_label"] = sub["method"].astype(str) + "\n" + sub["phi_mode"].astype(str)
        label_col = "_label"
    for metric in ["nll", "rmse", "coverage90", "runtime_per_block"]:
        if metric not in sub.columns:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(sub)), 3.4))
        x = np.arange(len(sub))
        y = sub[metric].to_numpy(dtype=float)
        err_col = f"{metric}_se"
        yerr = sub[err_col].to_numpy(dtype=float) if err_col in sub.columns else None
        ax.bar(x, y, yerr=yerr, capsize=3, color="#4C78A8", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sub[label_col].astype(str), rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"Seen-history {metric}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = outdir / f"{prefix}_{metric}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_markdown(
    outdir: Path,
    results: dict[str, pd.DataFrame],
    manifest: dict,
    figure_paths: list[Path],
) -> Path:
    out = outdir / "era5_markovian_baseline_addendum.md"
    lines: list[str] = []
    lines.append("# ERA5 baseline-suite addendum: same-kernel, same-hyperparameter and online GP references\n")
    lines.append("## Purpose\n")
    lines.append(
        "This addendum records the next baseline step after the current ERA5 Route B report. "
        "The goal is to separate two questions: whether Route B loses RMSE because the previous "
        "SGPR/SVGP baselines used a different kernel/hyperparameter protocol, and whether a full "
        "online-GP baseline suite is available under the same held-out seen-history evaluation.\n"
    )
    lines.append("## Fair protocol target\n")
    lines.append(
        "- Calibration: task_1 initial full-GP MLL grid selects temporal lengthscale, observation noise variance, and kernel variance.\n"
        "- Online evaluation: task_2 held-out seen-history blocks, with test labels excluded from updates.\n"
        "- Route B reference: Rich-v3 Phi + Matérn-3/2 + Mt=32 + Ms=256.\n"
        "- Main metrics: NLL/NLPD, RMSE, Cov90, ECE, runtime/block, and forgetting.\n"
    )

    lines.append("## External online GP sources\n")
    repos = manifest.get("repositories", [])
    ext_rows = []
    for repo in repos:
        ext_rows.append(
            {
                "repo": repo.get("name"),
                "intended_methods": ", ".join(repo.get("intended_methods", [])),
                "download_status": repo.get("download_status"),
                "integration": repo.get("unified_era5_runner_status"),
            }
        )
    if ext_rows:
        lines.append(pd.DataFrame(ext_rows).to_markdown(index=False))
        lines.append("")
    lines.append(
        "Important: downloaded external code is only provenance/reference material until it is wrapped by the local "
        "`OnlineBaseline` interface and evaluated with the same ERA5 split, standardization, block split, and metrics.\n"
    )

    for name, df in results.items():
        lines.append(f"## Result group: `{name}`\n")
        table = _seen_history_table(df)
        if table.empty:
            lines.append("No seen-history rows were found.\n")
            continue
        lines.append(table.to_markdown(index=False))
        lines.append("")
        if name == "same_kernel_frozen_tiny_smoke":
            lines.append(
                "This group is a tiny smoke test only. It validates the code path for Matérn-3/2 + frozen hyperparameters, "
                "but it should not be used as a paper result because it used a very small subset and very few training iterations.\n"
            )

    if figure_paths:
        lines.append("## Generated figures\n")
        for path in figure_paths:
            lines.append(f"- `{path}`")
        lines.append("")

    lines.append("## Pending full benchmark commands\n")
    lines.append(
        "The requested full setting is 1000 locations x 3 seeds. In the current session, long WSL execution was blocked by "
        "the desktop approval/usage limit after the tiny smoke test. The following commands are the intended reruns once "
        "execution is available again.\n"
    )
    lines.append("```bash")
    lines.append(
        "uv run --no-sync python scripts/run_hipposvgp_era5_baselines.py "
        "--root data/era5/processed_timeseries_4 --tasks task_2 --calibration-tasks task_1 "
        "--split all --seeds 0 1 2 --block-size 10 --phi-mode rich_v3 "
        "--methods sgpr_phi svgp_phi --eval-modes seen_history "
        "--gp-kernel-type matern32 --gp-hyperparam-fit-mode routeb_initial_task_fullgp_grid "
        "--gp-training-iterations 200 --gp-learning-rate 0.01 --gp-inducing-points 64 "
        "--gp-minibatch-size 1024 --gp-inducing-init random "
        "--outdir results/experiments_era5_fair_routeb_baseline_comparison_300loc_richphi/next_steps_markovian_baselines/same_kernel_frozen_1000loc "
        "--output-prefix era5_same_kernel_frozen"
    )
    lines.append(
        "uv run --no-sync python scripts/run_hipposvgp_era5_baselines.py "
        "--root data/era5/processed_timeseries_4 --tasks task_2 --calibration-tasks task_1 "
        "--split all --seeds 0 1 2 --block-size 10 --phi-mode rich_v3_lag_ar "
        "--methods sgpr_phi svgp_phi --eval-modes seen_history "
        "--gp-kernel-type matern32 --gp-hyperparam-fit-mode routeb_initial_task_fullgp_grid "
        "--gp-training-iterations 200 --gp-learning-rate 0.01 --gp-inducing-points 64 "
        "--gp-minibatch-size 1024 --gp-inducing-init random "
        "--outdir results/experiments_era5_fair_routeb_baseline_comparison_300loc_richphi/next_steps_markovian_baselines/same_kernel_frozen_lagphi_1000loc "
        "--output-prefix era5_same_kernel_frozen_lagphi"
    )
    lines.append("```")

    lines.append("## Current interpretation\n")
    lines.append(
        "The local code now supports same-kernel and frozen-hyperparameter SGPR/SVGP baselines. "
        "However, Markovian GP, s2VGP/STVGP, OVC and OHSVGP-style baselines are not yet paper-ready comparisons: "
        "their source code has been collected, but the local ERA5 wrapper and no-leakage held-out seen-history evaluation still need implementation.\n"
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _write_pdf(md_path: Path, outdir: Path, results: dict[str, pd.DataFrame], figure_paths: list[Path]) -> Path:
    pdf_path = outdir / "era5_markovian_baseline_addendum.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(
            0.08,
            0.95,
            "ERA5 baseline-suite addendum",
            fontsize=18,
            weight="bold",
            va="top",
        )
        body = (
            "This addendum documents the same-kernel/frozen-hyperparameter SGPR/SVGP extension "
            "and the external online-GP baseline sources. Full 1000-location x 3-seed execution "
            "was not completed in this session because WSL long-run execution became unavailable "
            "after the tiny smoke test."
        )
        fig.text(0.08, 0.89, textwrap.fill(body, 88), fontsize=10, va="top")
        y = 0.80
        for name, df in results.items():
            table = _seen_history_table(df)
            if table.empty:
                continue
            fig.text(0.08, y, name, fontsize=12, weight="bold", va="top")
            y -= 0.03
            show = table.head(8).copy()
            txt = show.to_string(index=False)
            fig.text(0.08, y, txt, fontsize=7.2, family="monospace", va="top")
            y -= 0.18
            if y < 0.18:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11))
                y = 0.94
        fig.text(0.08, 0.08, f"Markdown report: {md_path}", fontsize=8, color="#444444")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for path in figure_paths:
            if not path.exists():
                continue
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(path.name)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="results/experiments_era5_fair_routeb_baseline_comparison_300loc_richphi",
    )
    parser.add_argument(
        "--manifest",
        default="baselines/external/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/experiments_era5_fair_routeb_baseline_comparison_300loc_richphi/"
            "next_steps_markovian_baselines/integrated_report"
        ),
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    results = _load_existing_results(base_dir)
    all_figures: list[Path] = []
    for name, df in results.items():
        all_figures.extend(_plot_metric_bars(df, outdir, name))

    md = _write_markdown(outdir, results, manifest, all_figures)
    pdf = _write_pdf(md, outdir, results, all_figures)

    print(f"Wrote {md}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
