"""Summarize ERA5 Phi residual-allocation ablation outputs.

This script reads existing Route B ERA5 outputs under
`paper_ready/phi_residual_allocation`, creates compact CSV tables and figures,
and writes a short markdown report. It does not train models.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready")
OUTDIR = ROOT / "phi_residual_allocation"
PLOT_DIR = OUTDIR / "plots"
LOC_IDX = 99

EXPERIMENTS = [
    {
        "version": "base",
        "folder": "base_current",
        "phi": "base Phi",
        "gp_params": "current fitted",
        "capacity": "8/64",
        "mt": 8,
        "ms": 64,
    },
    {
        "version": "rich_v1",
        "folder": "rich_v1_fullgp",
        "phi": "current rich",
        "gp_params": "initial-task full-GP MLL",
        "capacity": "8/64",
        "mt": 8,
        "ms": 64,
    },
    {
        "version": "rich_v2",
        "folder": "rich_v2_fullgp",
        "phi": "rich_v1 + seasonal-spatial interactions",
        "gp_params": "initial-task full-GP MLL",
        "capacity": "8/64",
        "mt": 8,
        "ms": 64,
    },
    {
        "version": "rich_v3",
        "folder": "rich_v3_fullgp",
        "phi": "rich_v2 + 8 spatial RBF features",
        "gp_params": "initial-task full-GP MLL",
        "capacity": "8/128",
        "mt": 8,
        "ms": 128,
    },
    {
        "version": "rich_v3_mt16",
        "folder": "rich_v3_mt16_ms128_fullgp",
        "phi": "rich_v2 + 8 spatial RBF features",
        "gp_params": "initial-task full-GP MLL",
        "capacity": "16/128",
        "mt": 16,
        "ms": 128,
    },
    {
        "version": "rich_v4",
        "folder": "rich_v4_fullgp",
        "phi": "rich_v3",
        "gp_params": "initial-task full-GP MLL",
        "capacity": "32/256",
        "mt": 32,
        "ms": 256,
    },
]


def read_summary(folder: str) -> pd.Series:
    path = OUTDIR / folder / "era5_routeb_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    mask = (df["method"] == "structured_joint") & (df["eval_mode"] == "seen_history")
    row = df.loc[mask].iloc[0]
    return row


def read_run_args(folder: str) -> dict[str, object]:
    path = OUTDIR / folder / "era5_routeb_report.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    args = report.get("args", {})
    return args if isinstance(args, dict) else {}


def read_final_forgetting(folder: str) -> dict[str, float]:
    path = OUTDIR / folder / "era5_routeb_forgetting_curve_summary.csv"
    if not path.exists():
        return {"final_rmse_forgetting": np.nan, "final_nll_forgetting": np.nan}
    df = pd.read_csv(path)
    if df.empty:
        return {"final_rmse_forgetting": np.nan, "final_nll_forgetting": np.nan}
    row = df.sort_values("online_block_index").iloc[-1]
    return {
        "final_rmse_forgetting": float(row["rmse_forgetting"]),
        "final_nll_forgetting": float(row["nll_forgetting"]),
    }


def build_summary_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exp in EXPERIMENTS:
        row = read_summary(exp["folder"])
        args = read_run_args(exp["folder"])
        forget = read_final_forgetting(exp["folder"])
        rows.append(
            {
                "version": exp["version"],
                "phi": exp["phi"],
                "gp_params": exp["gp_params"],
                "capacity": exp["capacity"],
                "mt": exp["mt"],
                "ms": exp["ms"],
                "selected_ell_t": float(row.get("selected_ell_t", np.nan)),
                "avg_sigma2": float(row["avg_sigma2"]),
                "kernel_variance": float(args.get("kernel_variance", np.nan)),
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
                **forget,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "era5_phi_residual_allocation_summary.csv", index=False)
    return out


def plot_summary(summary: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    labels = summary["version"].tolist()
    x = np.arange(len(labels))

    metrics = [
        ("rmse", "RMSE", "#2b6cb0"),
        ("nll", "NLL", "#c05621"),
        ("coverage90", "Cov90", "#2f855a"),
        ("ece", "ECE", "#805ad5"),
        ("final_rmse_forgetting", "RMSE forgetting", "#4a5568"),
        ("final_nll_forgetting", "NLL forgetting", "#dd6b20"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    for ax, (metric, title, color) in zip(axes.ravel(), metrics):
        ax.bar(x, summary[metric], color=color, alpha=0.88)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.2)
    fig.suptitle("ERA5 Phi residual-allocation ablation")
    fig.savefig(PLOT_DIR / "era5_phi_residual_allocation_metrics.png", dpi=220)
    fig.savefig(PLOT_DIR / "era5_phi_residual_allocation_metrics.pdf")
    plt.close(fig)

    decomp_metrics = [
        ("avg_sigma2", "sigma2", "#2b6cb0"),
        ("avg_nu_star", "nu_star", "#c05621"),
        ("avg_predictive_variance", "total predictive variance", "#2f855a"),
        ("beta_u_coupling_ratio", "beta-u coupling ratio", "#805ad5"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.4), constrained_layout=True)
    for ax, (metric, title, color) in zip(axes, decomp_metrics):
        ax.plot(x, summary[metric], marker="o", linewidth=2.0, color=color)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Variance decomposition and coupling under richer Phi")
    fig.savefig(PLOT_DIR / "era5_phi_residual_allocation_decomposition.png", dpi=220)
    fig.savefig(PLOT_DIR / "era5_phi_residual_allocation_decomposition.pdf")
    plt.close(fig)


def load_location_predictions() -> pd.DataFrame:
    parts = []
    for exp in EXPERIMENTS:
        path = OUTDIR / exp["folder"] / "per_location_loc99.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[(df["method"] == "structured_joint") & (df["eval_mode"] == "seen_history")].copy()
        df["version"] = exp["version"]
        df["label"] = f"{exp['version']} ({exp['capacity']})"
        parts.append(df)
    if not parts:
        raise ValueError("No per-location prediction files found")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUTDIR / "era5_phi_residual_allocation_loc99_plot_data.csv", index=False)
    return out


def plot_single_location() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_location_predictions()
    versions = ["base", "rich_v1", "rich_v2", "rich_v3", "rich_v3_mt16", "rich_v4"]
    colors = {
        "base": "#718096",
        "rich_v1": "#2b6cb0",
        "rich_v2": "#2f855a",
        "rich_v3": "#c05621",
        "rich_v3_mt16": "#d69e2e",
        "rich_v4": "#805ad5",
    }
    fig, axes = plt.subplots(len(versions), 1, figsize=(9.0, 9.5), sharex=True, constrained_layout=True)
    for ax, version in zip(axes, versions):
        part = df[df["version"] == version].sort_values("time_index")
        x = part["actual_time"].to_numpy(dtype=float)
        y = part["y_true"].to_numpy(dtype=float)
        mean = part["pred_mean"].to_numpy(dtype=float)
        std = part["pred_std_y"].to_numpy(dtype=float)
        ax.plot(x, y, color="black", linewidth=1.0, label="ERA5 target")
        ax.plot(x, mean, color=colors[version], linewidth=1.6, label="prediction mean")
        ax.fill_between(x, mean - 1.645 * std, mean + 1.645 * std, color=colors[version], alpha=0.17, label="90% interval")
        ax.set_title(part["label"].iloc[0])
        ax.set_ylabel("scaled value")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"Single-location time series under Phi residual-allocation ablation, loc {LOC_IDX}")
    fig.savefig(PLOT_DIR / "era5_phi_residual_allocation_single_location_loc99.png", dpi=220)
    fig.savefig(PLOT_DIR / "era5_phi_residual_allocation_single_location_loc99.pdf")
    plt.close(fig)


def write_markdown(summary: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for c in df.columns:
                v = row[c]
                if isinstance(v, (float, np.floating)):
                    vals.append(f"{float(v):.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# ERA5 Phi residual-allocation ablation",
        "",
        "Protocol: task_1 calibration, task_2 online seen-history evaluation, full 1000-location grid, block_size=10, structured_joint only, streaming Sylvester prediction. Rich variants use initial-task full-GP MLL grid on a task_1 subset with max_time=20 and max_locations=30.",
        "",
        "The full-GP MLL grid was ell_t in {0.03, 0.05, 0.1}, noise in {0.3, 0.5, 0.8}, and kernel variance in {0.5, 1.0}. The selected value for all rich variants was ell_t=0.1, sigma2=0.09, kernel variance=0.5.",
        "",
        md_table(summary),
        "",
        "Key finding: making Phi richer is much more effective than only increasing sparse GP capacity. The best result here is rich_v3 with 8/128 capacity; rich_v4 reduces nu_star and total variance slightly further but worsens RMSE/NLL and is substantially slower.",
    ]
    (OUTDIR / "era5_phi_residual_allocation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {
        "location_index": LOC_IDX,
        "fullgp_mll_grid": {
            "ell_t": [0.03, 0.05, 0.1],
            "noise": [0.3, 0.5, 0.8],
            "kernel_variance": [0.5, 1.0],
            "max_time": 20,
            "max_locations": 30,
        },
        "experiments": EXPERIMENTS,
    }
    (OUTDIR / "era5_phi_residual_allocation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary_table()
    plot_summary(summary)
    plot_single_location()
    write_markdown(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
