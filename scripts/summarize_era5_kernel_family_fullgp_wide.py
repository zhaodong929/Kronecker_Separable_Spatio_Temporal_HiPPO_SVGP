"""Summarize fair ERA5 kernel-family diagnostic with wide full-GP MLL fitting.

All rows use task_1 initial-task full-GP MLL grid selection for temporal
lengthscale, observation noise, and kernel variance before task_2 online
evaluation. The script reads existing outputs only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready")
OUTDIR = ROOT / "kernel_family_fullgp_wide_diagnostic"
PLOT_DIR = OUTDIR / "plots"
LOC_IDX = 99

RUNS = [
    ("RBF", "base 8/64", "rbf_base_fullgp_wide", "base", "8/64"),
    ("Matérn-3/2", "base 8/64", "matern32_base_fullgp_wide", "base", "8/64"),
    ("RBF", "rich_v3 8/128", "rbf_rich_v3_fullgp_wide", "rich_v3", "8/128"),
    ("Matérn-3/2", "rich_v3 8/128", "matern32_rich_v3_fullgp_wide", "rich_v3", "8/128"),
    ("RBF", "rich_v4 32/256", "rbf_rich_v4_fullgp_wide", "rich_v4", "32/256"),
    ("Matérn-3/2", "rich_v4 32/256", "matern32_rich_v4_fullgp_wide", "rich_v4", "32/256"),
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def read_args(folder: Path) -> dict[str, object]:
    path = folder / "era5_routeb_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    args = report.get("args", {})
    return args if isinstance(args, dict) else {}


def read_summary(folder: Path) -> pd.Series:
    df = pd.read_csv(folder / "era5_routeb_summary.csv")
    rows = df[(df["method"] == "structured_joint") & (df["eval_mode"] == "seen_history")]
    if rows.empty:
        raise ValueError(f"No structured_joint seen_history row in {folder}")
    return rows.iloc[0]


def build_summary() -> pd.DataFrame:
    rows = []
    for kernel_label, table_label, run_dir, phi_version, capacity in RUNS:
        folder = OUTDIR / run_dir
        row = read_summary(folder)
        args = read_args(folder)
        rows.append(
            {
                "kernel": kernel_label,
                "phi_capacity": table_label,
                "run_dir": run_dir,
                "phi_version": phi_version,
                "capacity": capacity,
                "mt": int(args["mt"]),
                "ms": int(args["ms"]),
                "selected_ell_t": float(row["selected_ell_t"]),
                "selected_sigma": float(np.sqrt(row["avg_sigma2"])),
                "selected_sigma2": float(row["avg_sigma2"]),
                "selected_kernel_variance": float(args["kernel_variance"]),
                "rmse": float(row["rmse"]),
                "nll": float(row["nll"]),
                "coverage90": float(row["coverage90"]),
                "ece": float(row["ece"]),
                "avg_nu_star": float(row["avg_nu_star"]),
                "avg_predictive_variance": float(row["avg_predictive_variance"]),
                "avg_std": float(row["avg_std"]),
                "runtime_per_block": float(row["runtime_per_block"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "era5_kernel_family_fullgp_wide_summary.csv", index=False)
    return out


def plot_metrics(summary: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [("rmse", "RMSE"), ("nll", "NLL"), ("coverage90", "Cov90"), ("avg_predictive_variance", "Avg var")]
    x_labels = ["base\n8/64", "rich_v3\n8/128", "rich_v4\n32/256"]
    colors = {"RBF": "#718096", "Matérn-3/2": "#c05621"}
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.1), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        centers = np.arange(len(x_labels))
        width = 0.34
        for offset, kernel in [(-width / 2, "RBF"), (width / 2, "Matérn-3/2")]:
            vals = []
            for phi in ["base", "rich_v3", "rich_v4"]:
                vals.append(float(summary[(summary["kernel"] == kernel) & (summary["phi_version"] == phi)][metric].iloc[0]))
            ax.bar(centers + offset, vals, width=width, color=colors[kernel], alpha=0.92, label=kernel)
        if metric == "coverage90":
            ax.axhline(0.90, color="black", linewidth=0.8, linestyle="--", alpha=0.55)
        ax.set_title(title)
        ax.set_xticks(centers)
        ax.set_xticklabels(x_labels)
        ax.grid(True, axis="y", alpha=0.22)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Kernel-family diagnostic with task-1 full-GP MLL hyperparameter fitting")
    fig.savefig(PLOT_DIR / "era5_kernel_family_fullgp_wide_metrics.png", dpi=240)
    fig.savefig(PLOT_DIR / "era5_kernel_family_fullgp_wide_metrics.pdf")
    plt.close(fig)


def plot_hyperparams(summary: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [("selected_ell_t", "selected ell_t"), ("selected_sigma", "selected sigma"), ("selected_kernel_variance", "selected kernel variance")]
    x_labels = ["base\n8/64", "rich_v3\n8/128", "rich_v4\n32/256"]
    colors = {"RBF": "#718096", "Matérn-3/2": "#c05621"}
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), constrained_layout=True)
    centers = np.arange(len(x_labels))
    width = 0.34
    for ax, (metric, title) in zip(axes, metrics):
        for offset, kernel in [(-width / 2, "RBF"), (width / 2, "Matérn-3/2")]:
            vals = []
            for phi in ["base", "rich_v3", "rich_v4"]:
                vals.append(float(summary[(summary["kernel"] == kernel) & (summary["phi_version"] == phi)][metric].iloc[0]))
            ax.bar(centers + offset, vals, width=width, color=colors[kernel], alpha=0.92, label=kernel)
        ax.set_title(title)
        ax.set_xticks(centers)
        ax.set_xticklabels(x_labels)
        ax.grid(True, axis="y", alpha=0.22)
    axes[0].legend(loc="best")
    fig.suptitle("Selected task-1 full-GP MLL hyperparameters")
    fig.savefig(PLOT_DIR / "era5_kernel_family_fullgp_wide_hyperparams.png", dpi=240)
    fig.savefig(PLOT_DIR / "era5_kernel_family_fullgp_wide_hyperparams.pdf")
    plt.close(fig)


def plot_single_location() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    choices = [
        ("Matérn-3/2 base 8/64", OUTDIR / "matern32_base_fullgp_wide", "#a0aec0"),
        ("Matérn-3/2 rich_v3 8/128", OUTDIR / "matern32_rich_v3_fullgp_wide", "#dd6b20"),
        ("Matérn-3/2 rich_v4 32/256", OUTDIR / "matern32_rich_v4_fullgp_wide", "#9c4221"),
    ]
    fig, axes = plt.subplots(len(choices), 1, figsize=(9.0, 6.0), sharex=True, constrained_layout=True)
    for ax, (label, folder, color) in zip(axes, choices):
        path = folder / f"per_location_loc{LOC_IDX}.csv"
        if not path.exists():
            path = folder / "era5_routeb_per_location_predictions.csv"
        df = pd.read_csv(path)
        part = df[(df["method"] == "structured_joint") & (df["eval_mode"] == "seen_history")].sort_values("time_index")
        x = part["actual_time"].to_numpy(dtype=float)
        y = part["y_true"].to_numpy(dtype=float)
        mean = part["pred_mean"].to_numpy(dtype=float)
        std = part["pred_std_y"].to_numpy(dtype=float)
        ax.plot(x, y, color="black", linewidth=1.0, label="ERA5 target")
        ax.plot(x, mean, color=color, linewidth=1.5, label="prediction mean")
        ax.fill_between(x, mean - 1.645 * std, mean + 1.645 * std, color=color, alpha=0.17, label="90% interval")
        ax.set_title(label)
        ax.set_ylabel("scaled value")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"Full-GP wide-grid Matérn-3/2 single-location diagnostic, loc {LOC_IDX}")
    fig.savefig(PLOT_DIR / "era5_kernel_family_fullgp_wide_matern_single_location_loc99.png", dpi=240)
    fig.savefig(PLOT_DIR / "era5_kernel_family_fullgp_wide_matern_single_location_loc99.pdf")
    plt.close(fig)


def write_markdown(summary: pd.DataFrame) -> None:
    cols = [
        "kernel",
        "phi_capacity",
        "selected_ell_t",
        "selected_sigma",
        "selected_kernel_variance",
        "rmse",
        "nll",
        "coverage90",
        "ece",
        "avg_nu_star",
        "avg_predictive_variance",
        "runtime_per_block",
    ]
    lines = [
        "# ERA5 kernel-family diagnostic with wide full-GP MLL fitting",
        "",
        "All rows use task_1 initial-task full-GP marginal likelihood grid selection for ell_t, observation noise sigma, and kernel variance. The selected hyperparameters are frozen before task_2 online evaluation.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in summary[cols].iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{float(v):.4f}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "Observation: the wide full-GP MLL grid selects a very small observation noise sigma=0.05 and kernel variance=0.25 in every row. This improves some NLL values for Matérn-3/2 but makes coverage lower than in the earlier conservative setting. Matérn-3/2 remains stronger than RBF under the same fitting rule.",
        ]
    )
    (OUTDIR / "era5_kernel_family_fullgp_wide_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    summary = build_summary()
    plot_metrics(summary)
    plot_hyperparams(summary)
    plot_single_location()
    write_markdown(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

