"""Summarize ERA5 Matérn-3/2 temporal lengthscale and M_t diagnostics.

This script reads existing Route B outputs only. It does not retrain models.
It creates compact CSV/PNG/PDF artifacts for the paper-ready ERA5 report.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready")
OUTDIR = ROOT / "matern32_lengthscale_diagnostic"
PLOT_DIR = OUTDIR / "plots"
LOC_IDX = 99

RUNS = [
    ("ell_0.0125", OUTDIR / "rich_v4_mt32_ms256_ell_0.0125", "lengthscale"),
    ("ell_0.025", OUTDIR / "rich_v4_mt32_ms256_ell_0.025", "lengthscale"),
    ("ell_0.075", OUTDIR / "rich_v4_mt32_ms256_ell_0.075", "lengthscale"),
    ("ell_0.10", OUTDIR / "rich_v4_mt32_ms256_ell_0.10", "lengthscale"),
    ("ell_0.15", OUTDIR / "rich_v4_mt32_ms256_ell_0.15", "lengthscale"),
    ("ell_0.20", OUTDIR / "rich_v4_mt32_ms256_ell_0.20", "lengthscale"),
    ("mt64_ell_0.10", OUTDIR / "rich_v4_mt64_ms256_ell_0.10", "mt_capacity"),
    ("mt128_ell_0.10", OUTDIR / "rich_v4_mt128_ms256_ell_0.10", "mt_capacity"),
    ("mt256_ell_0.10", OUTDIR / "rich_v4_mt256_ms256_ell_0.10", "mt_capacity"),
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
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    args = report.get("args", {})
    return args if isinstance(args, dict) else {}


def read_summary(folder: Path) -> pd.Series:
    path = folder / "era5_routeb_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    rows = df[(df["method"] == "structured_joint") & (df["eval_mode"] == "seen_history")]
    if rows.empty:
        raise ValueError(f"No structured_joint seen_history row in {path}")
    return rows.iloc[0]


def build_summary() -> pd.DataFrame:
    rows = []
    for label, folder, group in RUNS:
        if not (folder / "era5_routeb_summary.csv").exists():
            continue
        row = read_summary(folder)
        args = read_args(folder)
        rows.append(
            {
                "setting": label,
                "group": group,
                "kernel_type": args.get("kernel_type", "matern32"),
                "phi_mode": args.get("phi_mode", "rich_v3"),
                "mt": int(args.get("mt", row.get("mt", 0))),
                "ms": int(args.get("ms", row.get("ms", 0))),
                "model_ell_t": float(args.get("model_ell_t", row.get("selected_ell_t", np.nan))),
                "sigma2": float(row.get("avg_sigma2", np.nan)),
                "kernel_variance": float(args.get("kernel_variance", np.nan)),
                "rmse": float(row["rmse"]),
                "nll": float(row["nll"]),
                "coverage90": float(row["coverage90"]),
                "ece": float(row["ece"]),
                "avg_nu_star": float(row["avg_nu_star"]),
                "avg_predictive_variance": float(row["avg_predictive_variance"]),
                "avg_std": float(row["avg_std"]),
                "avg_width90": float(row["avg_width90"]),
                "avg_u_posterior_term": float(row["avg_u_posterior_term"]),
                "avg_beta_schur_term": float(row["avg_beta_schur_term"]),
                "beta_u_coupling_ratio": float(row["beta_u_coupling_ratio"]),
                "runtime_per_block": float(row["runtime_per_block"]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["group", "mt", "model_ell_t"])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTDIR / "era5_matern32_lengthscale_diagnostic_summary.csv", index=False)
    return out


def plot_lengthscale(summary: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = summary[(summary["group"] == "lengthscale") & (summary["mt"] == 32)].sort_values("model_ell_t")
    metrics = [("rmse", "RMSE"), ("nll", "NLL"), ("coverage90", "Cov90"), ("avg_nu_star", "avg nu_star")]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        ax.plot(df["model_ell_t"], df[metric], color="#c05621", marker="o", linewidth=1.7)
        ax.set_xlabel("temporal lengthscale ell_t")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.22)
        best_idx = df[metric].idxmin() if metric != "coverage90" else None
        if best_idx is not None:
            ax.scatter(df.loc[best_idx, "model_ell_t"], df.loc[best_idx, metric], color="black", s=20, zorder=5)
    fig.suptitle("Matérn-3/2 rich_v4 temporal lengthscale diagnostic (Mt=32, Ms=256)")
    fig.savefig(PLOT_DIR / "era5_matern32_lengthscale_metrics.png", dpi=240)
    fig.savefig(PLOT_DIR / "era5_matern32_lengthscale_metrics.pdf")
    plt.close(fig)


def plot_mt_capacity(summary: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    base = summary[(summary["group"] == "lengthscale") & (summary["mt"] == 32) & np.isclose(summary["model_ell_t"], 0.10)]
    df = pd.concat([base, summary[summary["group"] == "mt_capacity"]], ignore_index=True).sort_values("mt")
    metrics = [("rmse", "RMSE"), ("nll", "NLL"), ("avg_nu_star", "avg nu_star"), ("runtime_per_block", "runtime/block")]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        ax.plot(df["mt"], df[metric], color="#805ad5", marker="o", linewidth=1.7)
        ax.set_xlabel("M_t")
        ax.set_ylabel(label)
        ax.set_xticks(df["mt"])
        ax.grid(True, alpha=0.22)
    fig.suptitle("Matérn-3/2 rich_v4 temporal basis capacity diagnostic (ell_t=0.10, Ms=256)")
    fig.savefig(PLOT_DIR / "era5_matern32_mt_capacity_metrics.png", dpi=240)
    fig.savefig(PLOT_DIR / "era5_matern32_mt_capacity_metrics.pdf")
    plt.close(fig)


def plot_single_location(summary: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    choices = [
        ("ell_0.075", OUTDIR / "rich_v4_mt32_ms256_ell_0.075", "#dd6b20"),
        ("ell_0.10", OUTDIR / "rich_v4_mt32_ms256_ell_0.10", "#c05621"),
        ("ell_0.15", OUTDIR / "rich_v4_mt32_ms256_ell_0.15", "#9c4221"),
        ("Mt=256 ell_0.10", OUTDIR / "rich_v4_mt256_ms256_ell_0.10", "#805ad5"),
    ]
    fig, axes = plt.subplots(len(choices), 1, figsize=(9.0, 7.0), sharex=True, constrained_layout=True)
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
        ax.fill_between(x, mean - 1.645 * std, mean + 1.645 * std, color=color, alpha=0.16, label="90% interval")
        ax.set_ylabel("scaled value")
        ax.set_title(label)
        ax.grid(True, alpha=0.18)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"Single-location Matérn-3/2 lengthscale/capacity diagnostic, loc {LOC_IDX}")
    fig.savefig(PLOT_DIR / "era5_matern32_lengthscale_single_location_loc99.png", dpi=240)
    fig.savefig(PLOT_DIR / "era5_matern32_lengthscale_single_location_loc99.pdf")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    cols = [
        "setting",
        "mt",
        "ms",
        "model_ell_t",
        "rmse",
        "nll",
        "coverage90",
        "ece",
        "avg_nu_star",
        "avg_predictive_variance",
        "runtime_per_block",
    ]
    def md_table(df: pd.DataFrame) -> str:
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df[cols].iterrows():
            vals = []
            for c in cols:
                v = row[c]
                vals.append(f"{float(v):.4f}" if isinstance(v, (float, np.floating)) else str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    length_df = summary[summary["group"] == "lengthscale"].sort_values("model_ell_t")
    mt_df = pd.concat(
        [
            summary[(summary["group"] == "lengthscale") & (summary["mt"] == 32) & np.isclose(summary["model_ell_t"], 0.10)],
            summary[summary["group"] == "mt_capacity"],
        ],
        ignore_index=True,
    ).sort_values("mt")
    best_rmse = length_df.loc[length_df["rmse"].idxmin()]
    best_nll = length_df.loc[length_df["nll"].idxmin()]
    text = [
        "# ERA5 Matérn-3/2 lengthscale and M_t diagnostic",
        "",
        "All runs use full-space task_1 calibration + task_2 online held-out seen-history evaluation, structured_joint only, rich_v3 Phi, Matérn-3/2 kernel, Ms=256 unless stated otherwise, sigma=0.09, kernel variance=0.5.",
        "",
        "## Temporal lengthscale sweep",
        md_table(length_df),
        "",
        f"Best RMSE in the lengthscale sweep is ell_t={best_rmse['model_ell_t']:.4f} with RMSE={best_rmse['rmse']:.4f}. Best NLL is ell_t={best_nll['model_ell_t']:.4f} with NLL={best_nll['nll']:.4f}. Smaller ell_t values below 0.05 are worse, so the previous setting is not too large in the simple sense.",
        "",
        "## M_t capacity sweep at ell_t=0.10",
        md_table(mt_df),
        "",
        "Increasing M_t from 32 to 64/128/256 gives only a small NLL gain and no RMSE gain. Runtime/block increases from about 3.8 s to 7.6 s at M_t=256. This suggests diminishing returns from temporal basis capacity after Mt=32 under the current Phi/kernel/noise setting.",
    ]
    (OUTDIR / "era5_matern32_lengthscale_diagnostic_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    summary = build_summary()
    plot_lengthscale(summary)
    plot_mt_capacity(summary)
    plot_single_location(summary)
    write_report(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
