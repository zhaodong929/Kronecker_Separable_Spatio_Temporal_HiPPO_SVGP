"""Summarize ERA5 Route B Matérn-3/2 kernel diagnostics.

Reads existing RBF reference runs and Matérn-3/2 Route B runs and creates
compact tables/figures for the paper-ready report. Does not train models.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready")
OUTDIR = ROOT / "kernel_family_diagnostic"
PLOT_DIR = OUTDIR / "plots"
LOC_IDX = 99

EXPERIMENTS = [
    ("rbf_base", ROOT / "phi_residual_allocation/base_current", "rbf", "base", "8/64"),
    ("matern32_base", OUTDIR / "matern32_base", "matern32", "base", "8/64"),
    ("rbf_rich_v3", ROOT / "phi_residual_allocation/rich_v3_fullgp", "rbf", "rich_v3", "8/128"),
    ("matern32_rich_v3", OUTDIR / "matern32_rich_v3", "matern32", "rich_v3", "8/128"),
    ("rbf_rich_v4", ROOT / "phi_residual_allocation/rich_v4_fullgp", "rbf", "rich_v4", "32/256"),
    ("matern32_rich_v4", OUTDIR / "matern32_rich_v4", "matern32", "rich_v4", "32/256"),
]


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
    for label, folder, kernel, phi_version, capacity in EXPERIMENTS:
        row = read_summary(folder)
        args = read_args(folder)
        rows.append(
            {
                "setting": label,
                "kernel_type": kernel,
                "phi_version": phi_version,
                "capacity": capacity,
                "mt": int(row.get("mt", args.get("mt", 0))),
                "ms": int(row.get("ms", args.get("ms", 0))),
                "selected_ell_t": float(row.get("selected_ell_t", np.nan)),
                "avg_sigma2": float(row["avg_sigma2"]),
                "kernel_variance": float(args.get("kernel_variance", row.get("kernel_variance", np.nan))),
                "spatial_lengthscale": float(args.get("spatial_lengthscale", np.nan)),
                "ard_lat": float(args.get("spatial_ard_lengthscales", [np.nan, np.nan])[0]) if args.get("spatial_ard_lengthscales") else np.nan,
                "ard_lon": float(args.get("spatial_ard_lengthscales", [np.nan, np.nan])[1]) if args.get("spatial_ard_lengthscales") else np.nan,
                "rmse": float(row["rmse"]),
                "nll": float(row["nll"]),
                "coverage90": float(row["coverage90"]),
                "ece": float(row["ece"]),
                "avg_nu_star": float(row["avg_nu_star"]),
                "avg_predictive_variance": float(row["avg_predictive_variance"]),
                "avg_std": float(row["avg_std"]),
                "avg_u_posterior_term": float(row["avg_u_posterior_term"]),
                "avg_beta_schur_term": float(row["avg_beta_schur_term"]),
                "beta_u_coupling_ratio": float(row["beta_u_coupling_ratio"]),
                "runtime_per_block": float(row["runtime_per_block"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "era5_kernel_family_diagnostic_summary.csv", index=False)
    return out


def plot_summary(summary: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [("rmse", "RMSE"), ("nll", "NLL"), ("coverage90", "Cov90"), ("avg_nu_star", "avg nu_star")]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), constrained_layout=True)
    colors = {"rbf": "#718096", "matern32": "#c05621"}
    x_labels = ["base\n8/64", "rich_v3\n8/128", "rich_v4\n32/256"]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        width = 0.32
        centers = np.arange(3)
        for offset, kernel in zip([-width / 2, width / 2], ["rbf", "matern32"]):
            vals = []
            for phi_version in ["base", "rich_v3", "rich_v4"]:
                row = summary[(summary["kernel_type"] == kernel) & (summary["phi_version"] == phi_version)]
                vals.append(float(row[metric].iloc[0]))
            ax.bar(centers + offset, vals, width=width, label=kernel, color=colors[kernel], alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(centers)
        ax.set_xticklabels(x_labels)
        ax.grid(True, axis="y", alpha=0.2)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("ERA5 Matérn-3/2 kernel diagnostic")
    fig.savefig(PLOT_DIR / "era5_kernel_family_metrics.png", dpi=220)
    fig.savefig(PLOT_DIR / "era5_kernel_family_metrics.pdf")
    plt.close(fig)


def plot_single_location() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, folder, kernel, phi_version, capacity in EXPERIMENTS:
        if kernel != "matern32" or phi_version not in {"rich_v3", "rich_v4"}:
            continue
        path = folder / "per_location_loc99.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[(df["method"] == "structured_joint") & (df["eval_mode"] == "seen_history")].copy()
        df["kernel_type"] = kernel
        df["phi_version"] = phi_version
        df["capacity"] = capacity
        df["label"] = f"Matérn-3/2, {phi_version}, {capacity}"
        rows.append(df)
    if not rows:
        return
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows.to_csv(OUTDIR / "era5_kernel_family_matern32_loc99_plot_data.csv", index=False)
    colors = {"rich_v3": "#c05621", "rich_v4": "#805ad5"}
    panels = [("rich_v3", "8/128"), ("rich_v4", "32/256")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.0, 4.8), sharex=True, constrained_layout=True)
    for ax, (phi_version, capacity) in zip(axes, panels):
        part = all_rows[(all_rows["phi_version"] == phi_version) & (all_rows["capacity"] == capacity)].sort_values("time_index")
        x = part["actual_time"].to_numpy(dtype=float)
        y = part["y_true"].to_numpy(dtype=float)
        mean = part["pred_mean"].to_numpy(dtype=float)
        std = part["pred_std_y"].to_numpy(dtype=float)
        ax.plot(x, y, color="black", linewidth=1.0, label="ERA5 target")
        ax.plot(x, mean, color=colors[phi_version], linewidth=1.6, label="prediction mean")
        ax.fill_between(x, mean - 1.645 * std, mean + 1.645 * std, color=colors[phi_version], alpha=0.17, label="90% interval")
        ax.set_title(f"Matérn-3/2, {phi_version}, {capacity}")
        ax.set_ylabel("scaled value")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"Single-location Matérn-3/2 diagnostic, loc {LOC_IDX}")
    fig.savefig(PLOT_DIR / "era5_kernel_family_matern32_single_location_loc99.png", dpi=220)
    fig.savefig(PLOT_DIR / "era5_kernel_family_matern32_single_location_loc99.pdf")
    plt.close(fig)


def write_markdown(summary: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for c in df.columns:
                v = row[c]
                vals.append(f"{float(v):.4f}" if isinstance(v, (float, np.floating)) else str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    cols = ["kernel_type", "phi_version", "capacity", "rmse", "nll", "coverage90", "ece", "avg_nu_star", "avg_predictive_variance", "runtime_per_block"]
    text = [
        "# ERA5 Matérn-3/2 kernel diagnostic",
        "",
        "This diagnostic compares the previous RBF/SE kernel against Matérn-3/2. Route B update formulas are unchanged.",
        "",
        md_table(summary[cols]),
        "",
        "Main observation: Matérn-3/2 improves the Rich Phi settings, especially rich_v4, but makes Base Phi worse. This suggests that a rougher residual prior helps only after the deterministic seasonal-spatial structure has been moved into Phi.",
    ]
    (OUTDIR / "era5_kernel_family_diagnostic_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    plot_summary(summary)
    plot_single_location()
    write_markdown(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
