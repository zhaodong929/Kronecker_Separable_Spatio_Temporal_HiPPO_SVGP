"""Build the integrated next-step Route B diagnostic report.

The script reads existing outputs and the newly saved Route B pointwise
diagnostic run.  It does not retrain models.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Z90 = 1.6448536269514722


def gaussian_nll(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(np.asarray(var, dtype=float), 1e-10)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (np.asarray(y) - mean) ** 2 / var))


def metric_dict(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    var = np.maximum(np.asarray(var, dtype=float), 1e-10)
    half = Z90 * np.sqrt(var)
    return {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "mae": float(np.mean(np.abs(y - mean))),
        "nll": gaussian_nll(y, mean, var),
        "coverage90": float(np.mean((y >= mean - half) & (y <= mean + half))),
        "avg_var": float(np.mean(var)),
        "avg_std": float(np.mean(np.sqrt(var))),
        "avg_width90": float(np.mean(2.0 * half)),
        "num_points": int(y.size),
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_variance_decomposition(routeb_summary: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    cols = [
        "avg_sigma2",
        "avg_nu_star",
        "avg_u_posterior_term",
        "avg_beta_schur_term",
    ]
    rows = []
    for _, row in routeb_summary.iterrows():
        rows.append(
            {
                "eval_mode": row["eval_mode"],
                "sigma2": row["avg_sigma2"],
                "nu_star": row["avg_nu_star"],
                "u_posterior": row["avg_u_posterior_term"],
                "beta_schur": row["avg_beta_schur_term"],
                "total_predictive_variance": row["avg_predictive_variance"],
                "coverage90": row["coverage90"],
                "nll": row["nll"],
                "rmse": row["rmse"],
            }
        )
    decomp = pd.DataFrame(rows)
    write_csv(decomp, outdir / "data" / "routeb_variance_decomposition.csv")

    plot_df = decomp.set_index("eval_mode")[["sigma2", "nu_star", "u_posterior", "beta_schur"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bottom = np.zeros(len(plot_df))
    colors = ["#4B5563", "#D97706", "#2563EB", "#059669"]
    for col, color in zip(plot_df.columns, colors):
        vals = plot_df[col].to_numpy(float)
        ax.bar(plot_df.index, vals, bottom=bottom, label=col, color=color, alpha=0.88)
        bottom += vals
    ax.set_ylabel("Average variance contribution")
    ax.set_title("Route B predictive variance decomposition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, outdir / "plots" / "routeb_variance_decomposition.png")
    return decomp


def build_block_diagnostics(metrics: pd.DataFrame, block_pairs: pd.DataFrame, routeb_dir: Path, outdir: Path) -> None:
    current = metrics[metrics["eval_mode"] == "current"].copy()
    if not current.empty:
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        for seed, group in current.groupby("seed"):
            ax.plot(group["block_id"], group["beta_u_coupling_ratio"], lw=1.6, alpha=0.75, label=f"seed {seed}")
        ax.set_xlabel("Online block index")
        ax.set_ylabel(r"$\|R_{\beta u}\|_F / \sqrt{\|R_{\beta\beta}\|_F\|R_{uu}\|_F}$")
        ax.set_title("Route B beta-u coupling ratio by block")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        save_fig(fig, outdir / "plots" / "routeb_coupling_ratio_by_block.png")

    if not block_pairs.empty:
        bp = block_pairs.copy()
        bp["nu_bin"] = pd.qcut(bp["avg_nu_star"], q=min(4, bp["avg_nu_star"].nunique()), duplicates="drop")
        nu_summary = (
            bp.groupby("nu_bin", observed=True)
            .agg(
                avg_nu_star=("avg_nu_star", "mean"),
                rmse=("rmse", "mean"),
                nll=("nll", "mean"),
                coverage90=("coverage90", "mean"),
                avg_var=("avg_predictive_variance", "mean"),
                rows=("rmse", "size"),
            )
            .reset_index()
        )
        nu_summary["nu_bin"] = nu_summary["nu_bin"].astype(str)
        write_csv(nu_summary, outdir / "data" / "block_level_nu_star_bin_summary.csv")

        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
        x = np.arange(len(nu_summary))
        for ax, metric, ylabel in zip(axes, ["rmse", "nll", "coverage90"], ["RMSE", "NLL", "Cov90"]):
            ax.bar(x, nu_summary[metric], color="#B45309", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Q{i+1}" for i in x], rotation=0)
            ax.set_xlabel(r"$\nu_*$ quantile bin")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Block-level sparse conditional residual diagnostic")
        fig.tight_layout()
        save_fig(fig, outdir / "plots" / "routeb_nu_star_bin_diagnostic.png")

        fc_path = routeb_dir / "era5_routeb_forgetting_curve.csv"
        if fc_path.exists():
            fc = pd.read_csv(fc_path)
            write_csv(fc, outdir / "data" / "routeb_forgetting_curve.csv")
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
            x_col = "train_block_id" if "train_block_id" in fc.columns else "online_block_index"
            for ax, metric, ylabel in zip(axes, ["rmse_forgetting", "nll_forgetting"], ["RMSE forgetting", "NLL forgetting"]):
                for seed, group in fc.groupby("seed"):
                    ax.plot(group[x_col], group[metric], lw=1.6, alpha=0.75, label=f"seed {seed}")
                ax.set_xlabel("Trained up to block n")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25)
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle("Route B seen-history forgetting curves")
            fig.tight_layout()
            save_fig(fig, outdir / "plots" / "routeb_forgetting_curves.png")


def build_pointwise_diagnostics(pointwise: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seen = pointwise[pointwise["eval_mode"] == "seen_history"].copy()
    if seen.empty:
        return pd.DataFrame(), pd.DataFrame()
    seen["abs_error"] = np.abs(seen["y_true"] - seen["pred_mean"])
    seen["sq_error"] = (seen["y_true"] - seen["pred_mean"]) ** 2
    half = Z90 * np.sqrt(np.maximum(seen["pred_var_y"].to_numpy(float), 1e-10))
    seen["covered90"] = (
        (seen["y_true"].to_numpy(float) >= seen["pred_mean"].to_numpy(float) - half)
        & (seen["y_true"].to_numpy(float) <= seen["pred_mean"].to_numpy(float) + half)
    )
    seen["is_peak_abs_y_top10"] = seen["y_true"].abs() >= seen["y_true"].abs().quantile(0.90)

    loc = (
        seen.groupby(["seed", "location_index", "latitude", "longitude"])
        .agg(
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            mae=("abs_error", "mean"),
            coverage90=("covered90", "mean"),
            avg_var=("pred_var_y", "mean"),
            avg_abs_y=("y_true", lambda x: float(np.mean(np.abs(x)))),
        )
        .reset_index()
    )
    write_csv(loc, outdir / "data" / "location_level_seen_history_diagnostics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    sc0 = axes[0].scatter(loc["longitude"], loc["latitude"], c=loc["rmse"], s=18, cmap="viridis")
    axes[0].set_title("Location-wise RMSE")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(sc0, ax=axes[0], fraction=0.046)
    sc1 = axes[1].scatter(loc["longitude"], loc["latitude"], c=loc["coverage90"], s=18, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Location-wise 90% coverage")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(sc1, ax=axes[1], fraction=0.046)
    fig.suptitle("Route B location/time coverage diagnostic")
    fig.tight_layout()
    save_fig(fig, outdir / "plots" / "routeb_location_error_coverage_scatter.png")

    time = (
        seen.groupby(["seed", "time_index"])
        .agg(
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            coverage90=("covered90", "mean"),
            avg_var=("pred_var_y", "mean"),
        )
        .reset_index()
    )
    write_csv(time, outdir / "data" / "time_level_seen_history_diagnostics.csv")
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.0), sharex=True)
    for seed, group in time.groupby("seed"):
        axes[0].plot(group["time_index"], group["rmse"], alpha=0.75, label=f"seed {seed}")
        axes[1].plot(group["time_index"], group["coverage90"], alpha=0.75)
        axes[2].plot(group["time_index"], group["avg_var"], alpha=0.75)
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("Cov90")
    axes[2].set_ylabel("Avg var")
    axes[2].set_xlabel("Time index")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Route B time-wise diagnostics")
    fig.tight_layout()
    save_fig(fig, outdir / "plots" / "routeb_timewise_error_coverage.png")

    rows = []
    for label, sub in [("all_seen_history", seen), ("peak_abs_y_top10", seen[seen["is_peak_abs_y_top10"]])]:
        m = metric_dict(sub["y_true"], sub["pred_mean"], sub["pred_var_y"])
        m["subset"] = label
        rows.append(m)
    peak = pd.DataFrame(rows)
    write_csv(peak, outdir / "data" / "peak_subset_metrics.csv")

    seen["var_bin"] = pd.qcut(seen["pred_var_y"], q=5, duplicates="drop")
    var_bins = (
        seen.groupby("var_bin", observed=True)
        .apply(lambda g: pd.Series(metric_dict(g["y_true"], g["pred_mean"], g["pred_var_y"])))
        .reset_index()
    )
    var_bins["var_bin"] = var_bins["var_bin"].astype(str)
    write_csv(var_bins, outdir / "data" / "pointwise_predictive_variance_bin_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    x = np.arange(len(var_bins))
    for ax, metric, ylabel in zip(axes, ["rmse", "nll", "coverage90"], ["RMSE", "NLL", "Cov90"]):
        ax.bar(x, var_bins[metric], color="#1D4ED8", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i+1}" for i in x])
        ax.set_xlabel("Predictive variance bin")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Pointwise predictive-variance bin diagnostic")
    fig.tight_layout()
    save_fig(fig, outdir / "plots" / "routeb_pointwise_variance_bin_diagnostic.png")
    return loc, peak


def build_calibration(pointwise: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    seen = pointwise[pointwise["eval_mode"] == "seen_history"].copy()
    if seen.empty:
        return pd.DataFrame()
    min_t, max_t = int(seen["time_index"].min()), int(seen["time_index"].max())
    cutoff = min_t + int(0.2 * (max_t - min_t + 1))
    calib = seen[seen["time_index"] <= cutoff]
    eval_df = seen[seen["time_index"] > cutoff]
    y_c, m_c, v_c = calib["y_true"].to_numpy(), calib["pred_mean"].to_numpy(), calib["pred_var_y"].to_numpy()
    y_e, m_e, v_e = eval_df["y_true"].to_numpy(), eval_df["pred_mean"].to_numpy(), eval_df["pred_var_y"].to_numpy()
    tau_grid = np.array([0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2])
    alpha_grid = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0])
    rows = []

    def add_row(name: str, alpha: float, tau2: float) -> None:
        rows.append(
            {
                "calibration": name,
                "alpha": alpha,
                "tau2": tau2,
                "calibration_nll": gaussian_nll(y_c, m_c, alpha * v_c + tau2),
                "calibration_cov90": metric_dict(y_c, m_c, alpha * v_c + tau2)["coverage90"],
                **{f"eval_{k}": v for k, v in metric_dict(y_e, m_e, alpha * v_e + tau2).items()},
                **{f"all_{k}": v for k, v in metric_dict(seen["y_true"], seen["pred_mean"], alpha * seen["pred_var_y"] + tau2).items()},
            }
        )

    add_row("base_uncalibrated", 1.0, 0.0)
    best_alpha = min(alpha_grid, key=lambda a: gaussian_nll(y_c, m_c, a * v_c))
    add_row("variance_scale_nll", float(best_alpha), 0.0)
    best_tau = min(tau_grid, key=lambda t: gaussian_nll(y_c, m_c, v_c + t))
    add_row("additive_tau2_nll", 1.0, float(best_tau))
    best = None
    for alpha in alpha_grid:
        for tau2 in tau_grid:
            score = gaussian_nll(y_c, m_c, alpha * v_c + tau2)
            if best is None or score < best[0]:
                best = (score, float(alpha), float(tau2))
    assert best is not None
    add_row("joint_alpha_tau2_nll", best[1], best[2])
    # Target-coverage diagnostic: closest calibration coverage to 0.9.
    best_cov = None
    for alpha in alpha_grid:
        for tau2 in tau_grid:
            cov = metric_dict(y_c, m_c, alpha * v_c + tau2)["coverage90"]
            score = abs(cov - 0.9)
            if best_cov is None or score < best_cov[0]:
                best_cov = (score, float(alpha), float(tau2))
    add_row("joint_alpha_tau2_cov90", best_cov[1], best_cov[2])
    out = pd.DataFrame(rows)
    out["calibration_time_cutoff"] = cutoff
    write_csv(out, outdir / "data" / "routeb_posthoc_calibration_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    labels = out["calibration"].str.replace("_", "\n")
    x = np.arange(len(out))
    for ax, col, ylabel in zip(axes, ["eval_nll", "eval_coverage90", "eval_avg_width90"], ["Eval NLL", "Eval Cov90", "Eval width90"]):
        ax.bar(x, out[col], color="#7C3AED", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if col == "eval_coverage90":
            ax.axhline(0.9, ls="--", lw=1.2, color="black")
    fig.suptitle("Post-hoc variance calibration diagnostic")
    fig.tight_layout()
    save_fig(fig, outdir / "plots" / "routeb_posthoc_calibration.png")
    return out


def build_future_horizon(future_horizon: pd.DataFrame, outdir: Path) -> None:
    if future_horizon.empty:
        return
    write_csv(future_horizon, outdir / "data" / "routeb_future_horizon_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for ax, metric, ylabel in zip(axes, ["rmse", "nll", "coverage90"], ["RMSE", "NLL", "Cov90"]):
        ax.errorbar(
            future_horizon["horizon_index"],
            future_horizon[metric],
            yerr=future_horizon.get(f"{metric}_se"),
            marker="o",
            lw=1.8,
            color="#DC2626",
        )
        ax.set_xlabel("Horizon inside future block")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle("Future block-ahead horizon breakdown")
    fig.tight_layout()
    save_fig(fig, outdir / "plots" / "routeb_future_horizon_breakdown.png")


def build_lag_ar(lag_summary_path: Path, outdir: Path) -> pd.DataFrame:
    if not lag_summary_path.exists():
        return pd.DataFrame()
    lag = pd.read_csv(lag_summary_path)
    write_csv(lag, outdir / "data" / "lag_ar_aggressive_tuning_summary.csv")
    keep = lag[lag["attempt"].isin(["baseline expanded-grid", "fixed ell_t 0.025", "lag_ar diagnostic 32/256"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    x = np.arange(len(keep))
    labels = keep["attempt"].str.replace(" ", "\n")
    for ax, metric, ylabel in zip(axes, ["rmse", "nll", "coverage90"], ["RMSE", "NLL", "Cov90"]):
        ax.bar(x, keep[metric], color=["#6B7280", "#2563EB", "#059669"], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Short-lag Rich-Phi diagnostic")
    fig.tight_layout()
    save_fig(fig, outdir / "plots" / "lag_ar_diagnostic_comparison.png")
    return lag


def build_baseline_plot(combined_path: Path, outdir: Path) -> pd.DataFrame:
    if not combined_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(combined_path)
    seen = df[df["eval_mode"] == "seen_history"].copy()
    write_csv(seen, outdir / "data" / "fair_baseline_seen_history_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    labels = seen["method_label"].str.replace(" ", "\n")
    x = np.arange(len(seen))
    for ax, metric, ylabel in zip(axes, ["rmse", "nll", "runtime_per_block"], ["RMSE", "NLL", "Runtime/block"]):
        ax.bar(x, seen[metric], color="#374151", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if metric == "runtime_per_block":
            ax.set_yscale("log")
    fig.suptitle("Completed fair-feature ERA5 baseline comparison")
    fig.tight_layout()
    save_fig(fig, outdir / "plots" / "fair_baseline_seen_history_rmse_nll_runtime.png")
    return seen


def build_verification(verification_path: Path, outdir: Path) -> pd.DataFrame:
    if not verification_path.exists():
        return pd.DataFrame()
    data = json.loads(verification_path.read_text(encoding="utf-8"))
    table = data["checks"]["routeB_cross_covariance_dense_diagnostic"]["table"]
    df = pd.DataFrame(table)
    write_csv(df, outdir / "data" / "dense_reference_cross_covariance_validation.csv")
    return df


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def table_tex(df: pd.DataFrame, columns: list[str], caption: str, label: str, max_rows: int = 12) -> str:
    sub = df[columns].head(max_rows).copy()
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{latex_escape(caption)}}}", f"\\label{{{label}}}"]
    lines.append("\\begin{tabular}{" + "l" * len(columns) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(latex_escape(c) for c in columns) + " \\\\")
    lines.append("\\midrule")
    for _, row in sub.iterrows():
        vals = []
        for c in columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.4f}")
            else:
                vals.append(latex_escape(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def write_reports(
    outdir: Path,
    decomp: pd.DataFrame,
    calibration: pd.DataFrame,
    lag: pd.DataFrame,
    baseline: pd.DataFrame,
    verification: pd.DataFrame,
) -> None:
    md = outdir / "routeb_next_steps_integrated_report.md"
    lines = [
        "# Route B next-step diagnostics and improvement report",
        "",
        "## Scope",
        "",
        "This report fixes the current strongest non-AR ERA5 Route B configuration: Rich-v3 Phi, Matern-3/2 kernel, Mt=32, Ms=256. Hyperparameters are selected on task_1 by the initial-task full-GP MLL grid and frozen on task_2. The report then adds diagnostics, post-hoc calibration, the previously effective short-lag Phi diagnostic, fair-feature baseline comparison, and dense-reference mechanism validation.",
        "",
        "## Step 1. Fixed-model diagnostics",
        "",
        "The seen-history variance is dominated by sparse conditional residual nu_star rather than posterior parameter terms. This confirms that the current ERA5 limitation is not only the beta-u Schur recovery, but also residual allocation/coverage of the inducing representation.",
        "",
        decomp.to_string(index=False),
        "",
        "## Step 2. Post-hoc calibration diagnostic",
        "",
        "Calibration changes only the predictive variance used for scoring; it does not change the Route B posterior mean or core update. The goal is to raise under-coverage while monitoring NLL.",
        "",
        calibration[["calibration", "alpha", "tau2", "calibration_nll", "calibration_cov90", "eval_rmse", "eval_nll", "eval_coverage90", "eval_avg_width90"]].to_string(index=False),
        "",
        "## Step 3. Short-lag Rich-Phi diagnostic",
        "",
        "The previous lag_ar diagnostic is reused because it was already the only attempt that reached the RMSE target range. It adds short-lag target covariates to Phi and therefore tests whether ERA5 short-term local dynamics should be represented deterministically rather than left entirely to the sparse GP residual.",
        "",
        lag[["attempt", "rmse", "nll", "coverage90", "verdict"]].to_string(index=False) if not lag.empty else "Lag-AR summary was not found.",
        "",
        "## Step 4. Completed fair-feature baseline comparison",
        "",
        "The completed comparison uses matched data, matched location subsets, matched online block split, and Rich-Phi covariates for Ridge/SGPR/SVGP. It is not yet a full Markovian/s2VGP baseline suite.",
        "",
        baseline[["method_label", "rmse", "nll", "coverage90", "runtime_per_block"]].to_string(index=False) if not baseline.empty else "Fair baseline summary was not found.",
        "",
        "## Step 5. Dense-reference structured-coupling validation",
        "",
        "This is a finite-dimensional Gaussian posterior validation, not a natural GP data-generating truth. It verifies that Route B recovers the dense beta-u posterior coupling to numerical precision, while mean-field necessarily misses S_beta_u.",
        "",
        verification.to_string(index=False) if not verification.empty else "Verification table was not found.",
        "",
        "## Main conclusion",
        "",
        "The current non-AR Route B configuration is competitive in seen-history NLL but not RMSE. Diagnostics point to local short-term dynamics and calibration as the main practical bottlenecks. The short-lag Rich-Phi diagnostic gives the strongest improvement and should be formalized as the next practical Route B variant, while Markovian GP / s2VGP / sparse Markovian baselines remain necessary for a reviewer-ready comparison.",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")

    tex = outdir / "routeb_next_steps_integrated_report.tex"
    tex_lines = [
        "\\documentclass[11pt]{article}",
        "\\usepackage[margin=0.8in]{geometry}",
        "\\usepackage{graphicx}",
        "\\usepackage{booktabs}",
        "\\usepackage{amsmath}",
        "\\usepackage{hyperref}",
        "\\usepackage{float}",
        "\\title{Route B Next-Step Diagnostics and Improvement Report}",
        "\\author{}",
        "\\date{}",
        "\\begin{document}",
        "\\maketitle",
        "\\section{Scope}",
        "We fix the current strongest non-AR ERA5 Route B configuration: Rich-v3 $\\Phi$, Matern-3/2 kernel, $M_t=32$, $M_s=256$. Hyperparameters are selected on task 1 by an initial-task full-GP MLL grid and then frozen on task 2. The report adds fixed-model diagnostics, post-hoc calibration, the short-lag $\\Phi$ diagnostic, the completed fair-feature baseline comparison, and dense-reference mechanism validation.",
        "\\section{Step 1: Fixed-model diagnostics}",
        "The seen-history variance is dominated by the sparse conditional residual $\\nu_*$ rather than posterior parameter terms. This indicates that local residual representation and calibration are stronger bottlenecks than the Schur/Sylvester posterior recovery itself.",
        table_tex(decomp, ["eval_mode", "sigma2", "nu_star", "u_posterior", "beta_schur", "total_predictive_variance", "rmse", "nll", "coverage90"], "Route B variance decomposition.", "tab:variance"),
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.82\\linewidth]{plots/routeb_variance_decomposition.png}\\caption{Predictive variance decomposition.}\\end{figure}",
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.82\\linewidth]{plots/routeb_nu_star_bin_diagnostic.png}\\caption{Block-level $\\nu_*$ bin diagnostic.}\\end{figure}",
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.82\\linewidth]{plots/routeb_location_error_coverage_scatter.png}\\caption{Location-wise RMSE and coverage diagnostics.}\\end{figure}",
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.82\\linewidth]{plots/routeb_coupling_ratio_by_block.png}\\caption{Beta-u coupling ratio across online blocks.}\\end{figure}",
        "\\section{Step 2: Post-hoc calibration}",
        "Calibration modifies only the scoring variance, not the Route B posterior mean or update formula. It is used to test whether under-coverage can be repaired without damaging NLL.",
        table_tex(calibration, ["calibration", "alpha", "tau2", "eval_rmse", "eval_nll", "eval_coverage90", "eval_avg_width90"], "Post-hoc variance calibration.", "tab:calibration"),
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.86\\linewidth]{plots/routeb_posthoc_calibration.png}\\caption{Post-hoc calibration trade-off.}\\end{figure}",
        "\\section{Step 3: Short-lag Rich-Phi diagnostic}",
        "The lag-AR diagnostic adds short-lag target covariates to $\\Phi$. It tests whether short-term ERA5 dynamics should be captured by deterministic/local covariates instead of forcing the sparse residual GP to learn all local fluctuations.",
        table_tex(lag, ["attempt", "rmse", "nll", "coverage90", "verdict"], "Lag-AR diagnostic attempts.", "tab:lag", max_rows=10) if not lag.empty else "Lag-AR table not available.",
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.86\\linewidth]{plots/lag_ar_diagnostic_comparison.png}\\caption{Short-lag $\\Phi$ diagnostic comparison.}\\end{figure}",
        "\\section{Step 4: Completed fair-feature baseline comparison}",
        "The completed comparison uses matched data, matched location subsets, matched online block split, and Rich-$\\Phi$ covariates for Ridge/SGPR/SVGP. It is not yet a full Markovian/s2VGP baseline suite.",
        table_tex(baseline, ["method_label", "rmse", "nll", "coverage90", "runtime_per_block"], "Completed seen-history fair-feature baseline comparison.", "tab:baseline") if not baseline.empty else "Baseline table not available.",
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.86\\linewidth]{plots/fair_baseline_seen_history_rmse_nll_runtime.png}\\caption{Completed fair-feature baseline comparison.}\\end{figure}",
        "\\section{Step 5: Dense-reference structured-coupling validation}",
        "This verifies finite-dimensional Gaussian posterior recovery. The dense reference is the exact posterior of the same finite-dimensional approximation, not the unknown data-generating truth.",
        table_tex(verification, ["quantity", "routeB_error", "mean_field_error"], "Dense-reference beta-u coupling validation.", "tab:dense", max_rows=12) if not verification.empty else "Verification table not available.",
        "\\section{Conclusion}",
        "The current non-AR Route B setting is strongest in seen-history NLL but does not dominate RMSE. Diagnostics identify local short-term dynamics and calibration as the main practical bottlenecks. The short-lag Rich-$\\Phi$ diagnostic is the most effective route to RMSE improvement. For reviewer-ready comparison, Markovian GP, s2VGP, SVGP-Matern, and sparse Markovian GP baselines remain important next baselines.",
        "\\end{document}",
    ]
    tex.write_text("\n".join(tex_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routeb-dir", type=Path, required=True)
    parser.add_argument("--baseline-combined", type=Path, required=True)
    parser.add_argument("--lag-summary", type=Path, required=True)
    parser.add_argument("--verification-json", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "data").mkdir(exist_ok=True)
    (args.outdir / "plots").mkdir(exist_ok=True)

    routeb_summary = pd.read_csv(args.routeb_dir / "era5_routeb_summary.csv")
    metrics = pd.read_csv(args.routeb_dir / "era5_routeb_metrics.csv")
    block_pairs = pd.read_csv(args.routeb_dir / "era5_routeb_block_pair_metrics.csv")
    pointwise = pd.read_csv(args.routeb_dir / "era5_routeb_per_location_predictions.csv")
    future_horizon = pd.read_csv(args.routeb_dir / "era5_routeb_future_horizon_summary.csv")

    # Keep raw data provenance in the integrated folder.
    copy_if_exists(args.routeb_dir / "era5_routeb_summary.csv", args.outdir / "data" / "raw_routeb_summary.csv")
    copy_if_exists(args.routeb_dir / "era5_routeb_metrics.csv", args.outdir / "data" / "raw_routeb_metrics.csv")
    copy_if_exists(args.routeb_dir / "era5_routeb_block_pair_metrics.csv", args.outdir / "data" / "raw_routeb_block_pair_metrics.csv")
    copy_if_exists(args.routeb_dir / "era5_routeb_future_horizon_summary.csv", args.outdir / "data" / "raw_routeb_future_horizon_summary.csv")

    decomp = build_variance_decomposition(routeb_summary, args.outdir)
    build_block_diagnostics(metrics, block_pairs, args.routeb_dir, args.outdir)
    build_pointwise_diagnostics(pointwise, args.outdir)
    calibration = build_calibration(pointwise, args.outdir)
    build_future_horizon(future_horizon, args.outdir)
    lag = build_lag_ar(args.lag_summary, args.outdir)
    copy_if_exists(Path("results/tmp_era5_routeb_aggressive_tuning_attempt/lag_ar_single_location_loc99.png"), args.outdir / "plots" / "lag_ar_single_location_loc99.png")
    baseline = build_baseline_plot(args.baseline_combined, args.outdir)
    verification = build_verification(args.verification_json, args.outdir)
    write_reports(args.outdir, decomp, calibration, lag, baseline, verification)
    print(f"Wrote integrated next-step report to {args.outdir}")


if __name__ == "__main__":
    main()
