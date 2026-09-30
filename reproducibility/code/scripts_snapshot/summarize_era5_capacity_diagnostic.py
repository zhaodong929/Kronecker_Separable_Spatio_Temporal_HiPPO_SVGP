"""Summarize ERA5 Route B inducing-basis capacity diagnostics.

The script consumes existing Route B ERA5 outputs and creates compact tables
and figures for the paper-ready report. It does not train models.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready")
OUTDIR = ROOT / "capacity_diagnostics_streaming"
PLOT_DIR = OUTDIR / "plots"
LOC_IDX = 99


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pick_seen_history_structured(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["method"] == "structured_joint") & (df["eval_mode"] == "seen_history")].copy()
    if out.empty:
        raise ValueError("No structured_joint seen_history rows found")
    return out


def save_capacity_sweep_summary() -> pd.DataFrame:
    sweep = read_csv(OUTDIR / "base_phi_sweep" / "era5_routeb_capacity_by_config_summary.csv")
    rows = pick_seen_history_structured(sweep)
    cols = [
        "mt",
        "ms",
        "rmse",
        "nll",
        "coverage90",
        "ece",
        "avg_nu_star",
        "avg_sigma2",
        "avg_u_posterior_term",
        "avg_beta_schur_term",
        "avg_predictive_variance",
        "avg_std",
        "avg_width90",
        "runtime_per_block",
        "beta_u_coupling_ratio",
    ]
    rows = rows[cols].sort_values(["ms", "mt"]).reset_index(drop=True)
    rows.to_csv(OUTDIR / "era5_capacity_sweep_base_phi_summary.csv", index=False)
    return rows


def save_key_comparison() -> pd.DataFrame:
    base_sweep = save_capacity_sweep_summary()
    original = base_sweep[(base_sweep["mt"] == 8) & (base_sweep["ms"] == 64)].iloc[0].copy()
    selected_base = base_sweep[(base_sweep["mt"] == 32) & (base_sweep["ms"] == 256)].iloc[0].copy()
    rich = pick_seen_history_structured(
        read_csv(OUTDIR / "rich_phi_selected_mt32_ms256" / "era5_routeb_summary.csv")
    ).iloc[0]

    def normalize(
        row: pd.Series,
        label: str,
        phi_mode: str,
        *,
        default_mt: int | None = None,
        default_ms: int | None = None,
    ) -> dict[str, float | str]:
        return {
            "setting": label,
            "phi_mode": phi_mode,
            "mt": int(row["mt"]) if "mt" in row.index else int(default_mt or 0),
            "ms": int(row["ms"]) if "ms" in row.index else int(default_ms or 0),
            "rmse": float(row["rmse"]),
            "nll": float(row["nll"]),
            "coverage90": float(row["coverage90"]),
            "ece": float(row["ece"]),
            "avg_nu_star": float(row["avg_nu_star"]),
            "avg_sigma2": float(row["avg_sigma2"]),
            "avg_u_posterior_term": float(row["avg_u_posterior_term"]),
            "avg_beta_schur_term": float(row["avg_beta_schur_term"]),
            "avg_predictive_variance": float(row["avg_predictive_variance"]),
            "avg_std": float(row["avg_std"]),
            "avg_width90": float(row["avg_width90"]),
            "runtime_per_block": float(row["runtime_per_block"]),
            "beta_u_coupling_ratio": float(row["beta_u_coupling_ratio"]),
        }

    comparison = pd.DataFrame(
        [
            normalize(original, "Base Phi, original capacity", "base"),
            normalize(selected_base, "Base Phi, selected capacity", "base"),
            normalize(rich, "Rich Phi, selected capacity", "rich_seasonal_spatial", default_mt=32, default_ms=256),
        ]
    )
    comparison.to_csv(OUTDIR / "era5_capacity_key_comparison.csv", index=False)
    return comparison


def plot_capacity_trends(base_sweep: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [f"Mt={int(r.mt)}, Ms={int(r.ms)}" for r in base_sweep.itertuples()]
    x = np.arange(len(base_sweep))
    fig, ax1 = plt.subplots(figsize=(7.6, 4.2))
    ax1.plot(x, base_sweep["avg_nu_star"], marker="o", color="#2b6cb0", linewidth=2.0, label="avg nu_star")
    ax1.set_ylabel("avg nu_star", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(True, axis="y", alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(x, base_sweep["nll"], marker="s", color="#c05621", linewidth=2.0, label="NLL")
    ax2.set_ylabel("NLL", color="#c05621")
    ax2.tick_params(axis="y", labelcolor="#c05621")
    fig.suptitle("ERA5 Base Phi capacity diagnostic: sparse residual and NLL")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "era5_capacity_nu_star_nll_trend.png", dpi=220)
    fig.savefig(PLOT_DIR / "era5_capacity_nu_star_nll_trend.pdf")
    plt.close(fig)

    metrics = ["rmse", "nll", "avg_nu_star", "avg_predictive_variance", "runtime_per_block"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.5), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        ax.bar(x, base_sweep[metric], color="#4c78a8")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.18)
    fig.savefig(PLOT_DIR / "era5_capacity_sweep_metrics.png", dpi=220)
    fig.savefig(PLOT_DIR / "era5_capacity_sweep_metrics.pdf")
    plt.close(fig)


def iter_prediction_chunks(path: Path, chunksize: int = 200_000) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk


def load_original_location() -> pd.DataFrame:
    path = ROOT / "single_location_predictions" / "per_location_predictions.csv"
    parts = []
    for chunk in iter_prediction_chunks(path):
        mask = (
            (chunk["method"] == "structured_joint")
            & (chunk["eval_mode"] == "seen_history")
            & (chunk["phi_mode"] == "base")
            & (chunk["location_index"] == LOC_IDX)
            & (chunk["mt"] == 8)
            & (chunk["ms"] == 64)
        )
        if mask.any():
            parts.append(chunk.loc[mask].copy())
    if not parts:
        raise ValueError("Could not find original loc99 prediction rows")
    out = pd.concat(parts, ignore_index=True)
    out["setting"] = "Base Phi, Mt=8, Ms=64"
    return out


def load_selected_location() -> pd.DataFrame:
    base = read_csv(OUTDIR / "base_phi_selected_mt32_ms256" / "base_mt32_ms256_loc99_predictions.csv")
    base["setting"] = "Base Phi, Mt=32, Ms=256"
    rich = read_csv(OUTDIR / "rich_phi_selected_mt32_ms256" / "rich_mt32_ms256_loc99_predictions.csv")
    rich["setting"] = "Rich Phi, Mt=32, Ms=256"
    return pd.concat([base, rich], ignore_index=True)


def plot_single_location_capacity() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([load_original_location(), load_selected_location()], ignore_index=True)
    df = df.sort_values(["setting", "time_index"])
    df.to_csv(OUTDIR / "era5_capacity_single_location_loc99_plot_data.csv", index=False)

    settings = [
        ("Base Phi, Mt=8, Ms=64", "#718096"),
        ("Base Phi, Mt=32, Ms=256", "#2b6cb0"),
        ("Rich Phi, Mt=32, Ms=256", "#2f855a"),
    ]
    fig, axes = plt.subplots(len(settings), 1, figsize=(9.0, 7.2), sharex=True, constrained_layout=True)
    for ax, (setting, color) in zip(axes, settings):
        part = df[df["setting"] == setting].sort_values("time_index")
        x = part["actual_time"].to_numpy(dtype=float)
        y = part["y_true"].to_numpy(dtype=float)
        mean = part["pred_mean"].to_numpy(dtype=float)
        std = part["pred_std_y"].to_numpy(dtype=float)
        ax.plot(x, y, color="black", linewidth=1.2, label="ERA5 target")
        ax.plot(x, mean, color=color, linewidth=1.8, label="prediction mean")
        ax.fill_between(x, mean - 1.645 * std, mean + 1.645 * std, color=color, alpha=0.18, label="90% interval")
        ax.set_title(setting)
        ax.set_ylabel("scaled value")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle("Single-location capacity diagnostic at ERA5 location 99")
    fig.savefig(PLOT_DIR / "era5_capacity_single_location_loc99.png", dpi=220)
    fig.savefig(PLOT_DIR / "era5_capacity_single_location_loc99.pdf")
    plt.close(fig)


def write_markdown(base_sweep: pd.DataFrame, comparison: pd.DataFrame) -> None:
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
        "# ERA5 Route B inducing-basis capacity diagnostic",
        "",
        "This diagnostic keeps the ERA5 task protocol fixed: task_1 calibration, task_2 online seen-history evaluation, full 1000-location grid, block_size=10, model ell_t=0.05, and structured_joint only. The Route B core formulas are unchanged.",
        "",
        "## Base Phi capacity sweep",
        "",
        md_table(base_sweep),
        "",
        "## Key comparison",
        "",
        md_table(comparison),
        "",
        "## Interpretation",
        "",
        "- After replacing dense prediction with streaming Sylvester prediction, the larger Base Phi configurations including Ms=256 and Mt=32 run stably.",
        "- Increasing spatial capacity from Ms=64 to Ms=128/256 reduces avg_nu_star and improves NLL under Base Phi. The marginal gain beyond Ms=128 is small but consistent for NLL and nu_star.",
        "- The selected setting is Mt=32, Ms=256 because it gives the lowest Base Phi NLL and avg_nu_star in the completed sweep. Its RMSE is not the best, so the selection is capacity/calibration-oriented rather than RMSE-only.",
        "- Rich Phi at the selected Mt=32, Ms=256 capacity gives a large additional RMSE/NLL improvement and lowers avg predictive variance mainly through a smaller sigma2 estimate. However, avg_nu_star remains the same as Base Phi at the same capacity, so rich features help explain structured mean/residual signal but do not solve sparse residual variance by themselves.",
    ]
    (OUTDIR / "era5_capacity_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base_sweep = save_capacity_sweep_summary()
    comparison = save_key_comparison()
    plot_capacity_trends(base_sweep)
    plot_single_location_capacity()
    write_markdown(base_sweep, comparison)
    metadata = {
        "selected_setting": {"mt": 32, "ms": 256, "reason": "lowest Base Phi NLL and avg_nu_star after streaming Sylvester prediction"},
        "failed_or_infeasible": [],
        "location_index": LOC_IDX,
    }
    (OUTDIR / "era5_capacity_diagnostic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(OUTDIR / "era5_capacity_diagnostic_report.md")


if __name__ == "__main__":
    main()
