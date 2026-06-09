#!/usr/bin/env python
"""Plot latent-vs-observation predictive intervals and variance decomposition.

The input is the strict per-location prediction CSV produced by
``scripts/run_hipposvgp_era5_routeb.py --save-per-location-predictions``.
The observation interval uses ``pred_var_y``. The latent interval subtracts
``observation_noise_variance`` and clips at a small positive value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Z90 = 1.6448536269514722


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_path", required=True, type=Path)
    parser.add_argument("--summary_paths", nargs="+", type=Path, default=[])
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--location_index", type=int, default=None)
    parser.add_argument("--location_strategy", choices=["median_rmse", "high_rmse", "high_variance"], default="median_rmse")
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--format", default="png,pdf")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, fmt: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in [part.strip() for part in fmt.split(",") if part.strip()]:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=450 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def select_location(df: pd.DataFrame, args: argparse.Namespace) -> tuple[int, dict[str, float | int | str]]:
    if args.location_index is not None:
        loc = int(args.location_index)
    else:
        sub = df[
            (df["eval_mode"] == "seen_history")
            & (df["method"] == "structured_joint")
            & (df["phi_mode"] == "base")
        ].copy()
        max_block = sub["train_block_id"].max()
        sub = sub[sub["train_block_id"] == max_block]
        sub["sq_error"] = (sub["pred_mean"] - sub["y_true"]) ** 2
        grouped = sub.groupby("location_index", as_index=False).agg(
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            avg_variance=("pred_var_y", "mean"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
        )
        if args.location_strategy == "high_rmse":
            row = grouped.sort_values("rmse", ascending=False).iloc[0]
        elif args.location_strategy == "high_variance":
            row = grouped.sort_values("avg_variance", ascending=False).iloc[0]
        else:
            median = grouped["rmse"].median()
            row = grouped.iloc[(grouped["rmse"] - median).abs().argsort().iloc[0]]
        loc = int(row["location_index"])
    first = df[df["location_index"] == loc].iloc[0]
    return loc, {
        "location_index": loc,
        "selection_strategy": "manual" if args.location_index is not None else args.location_strategy,
        "lat": float(first["latitude"]),
        "lon": float(first["longitude"]),
    }


def plot_latent_vs_observation(df: pd.DataFrame, args: argparse.Namespace, loc: int, meta: dict[str, float | int | str]) -> list[str]:
    sub = df[
        (df["eval_mode"] == "seen_history")
        & (df["method"] == "structured_joint")
        & (df["phi_mode"] == "base")
        & (df["location_index"] == loc)
    ].copy()
    max_block = sub["train_block_id"].max()
    sub = sub[sub["train_block_id"] == max_block].sort_values("time_index")
    sub["latent_var"] = np.maximum(sub["pred_var_y"] - sub["observation_noise_variance"], 1e-10)
    obs_std = np.sqrt(sub["pred_var_y"])
    latent_std = np.sqrt(sub["latent_var"])

    fig, ax = plt.subplots(figsize=(6.8, 2.9), constrained_layout=True)
    ax.plot(sub["time_index"], sub["y_true"], color="black", linewidth=1.2, label="Ground truth")
    ax.plot(sub["time_index"], sub["pred_mean"], color="#59A14F", linewidth=1.25, label="Predictive mean")
    ax.fill_between(
        sub["time_index"],
        sub["pred_mean"] - Z90 * obs_std,
        sub["pred_mean"] + Z90 * obs_std,
        color="#59A14F",
        alpha=0.13,
        linewidth=0,
        label="Observation 90% interval",
    )
    ax.fill_between(
        sub["time_index"],
        sub["pred_mean"] - Z90 * latent_std,
        sub["pred_mean"] + Z90 * latent_std,
        color="#4C78A8",
        alpha=0.20,
        linewidth=0,
        label="Latent 90% interval",
    )
    for boundary in range(args.block_size, int(sub["time_index"].max()) + 1, args.block_size):
        ax.axvline(boundary, color="0.78", linestyle="--", linewidth=0.65, zorder=0)
    ax.set_title(f"Latent vs observation intervals at location {loc} (lat {meta['lat']:.2f}, lon {meta['lon']:.2f})")
    ax.set_xlabel("time index")
    ax.set_ylabel("standardized target")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.30))
    return save_figure(fig, args.output_dir, f"single_location_latent_vs_observation_interval_loc{loc:04d}", args.format)


def write_variance_decomposition(summary_paths: list[Path], output_dir: Path) -> tuple[str, str]:
    rows = []
    for path in summary_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            if str(row.get("eval_mode")) != "seen_history":
                continue
            method = str(row.get("method"))
            phi_mode = str(row.get("phi_mode", "base"))
            if not (
                (phi_mode == "base" and method in {"no_transfer", "mean_field", "structured_joint"})
                or (phi_mode == "rich_seasonal_spatial" and method == "structured_joint")
            ):
                continue
            rows.append(
                {
                    "setting": f"{phi_mode} {method}",
                    "phi_mode": phi_mode,
                    "method": method,
                    "sigma2": float(row.get("avg_sigma2", np.nan)),
                    "avg_nu_star": float(row.get("avg_nu_star", np.nan)),
                    "avg_u_posterior_term": float(row.get("avg_u_posterior_term", np.nan)),
                    "avg_beta_schur_term": float(row.get("avg_beta_schur_term", np.nan)),
                    "avg_total_var": float(row.get("avg_predictive_variance", row.get("avg_var", np.nan))),
                    "avg_std": float(row.get("avg_std", np.nan)),
                    "coverage90": float(row.get("coverage90", np.nan)),
                    "nll": float(row.get("nll", np.nan)),
                    "rmse": float(row.get("rmse", np.nan)),
                    "source": str(path),
                }
            )
    out = pd.DataFrame(rows)
    csv_path = output_dir / "era5_variance_decomposition_seen_history.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)

    if not out.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
        labels = out["setting"].str.replace("rich_seasonal_spatial", "rich").str.replace("structured_joint", "structured").tolist()
        x = np.arange(len(out))
        bottom = np.zeros(len(out))
        colors = ["#BAB0AC", "#4C78A8", "#59A14F", "#B07AA1"]
        for col, color, label in [
            ("sigma2", colors[0], "sigma2"),
            ("avg_nu_star", colors[1], "nu_star"),
            ("avg_u_posterior_term", colors[2], "u posterior"),
            ("avg_beta_schur_term", colors[3], "beta/Schur"),
        ]:
            vals = out[col].fillna(0).to_numpy(float)
            ax.bar(x, vals, bottom=bottom, color=color, label=label)
            bottom += vals
        ax.scatter(x, out["avg_total_var"], color="black", s=18, zorder=3, label="reported total var")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("average predictive variance")
        ax.set_title("ERA5 seen-history variance decomposition")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
        fig_path = save_figure(fig, output_dir, "era5_variance_decomposition_seen_history", "png,pdf")[0]
    else:
        fig_path = ""
    return str(csv_path), fig_path


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions_path)
    loc, meta = select_location(df, args)
    outputs = plot_latent_vs_observation(df, args, loc, meta)
    variance_csv, variance_fig = write_variance_decomposition(args.summary_paths, args.output_dir)
    metadata = {
        **meta,
        "predictions_path": str(args.predictions_path),
        "summary_paths": [str(path) for path in args.summary_paths],
        "output_files": outputs + ([variance_fig] if variance_fig else []),
        "variance_decomposition_csv": variance_csv,
    }
    (args.output_dir / "variance_diagnostic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
