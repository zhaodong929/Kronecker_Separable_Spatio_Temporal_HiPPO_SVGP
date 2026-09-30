"""Plot single-location ERA5 time-dependency diagnostics from saved predictions.

This script only reads existing prediction snapshots. It does not retrain any
model, update online states, or use future labels beyond values already saved in
evaluation artifacts.

Expected snapshot files are ``.npz`` files containing fields such as:
``coords``, ``y``, ``mean``, ``variance``, ``t_index`` and ``time_value``.
The current Route B map-snapshot writer uses names like
``structured_joint_seen_history_task_2_snapshot_t010.npz``.

Example commands:

python scripts/plot_single_location_time_dependency.py \
  --input_dir outputs/stvgp_kronecker_maps/routeB_task1_calibration_task2_online_fullspace/test \
  --output_dir results/figures/single_location \
  --eval_mode seen_history \
  --location_strategy median_rmse \
  --methods no_transfer mean_field structured_joint \
  --phi_modes base \
  --interval 90 \
  --format both \
  --save_csv

python scripts/plot_single_location_time_dependency.py \
  --input_dir results/experiments_era5 \
  --output_dir results/figures/single_location \
  --location_strategy median_rmse \
  --methods structured_joint \
  --phi_modes base rich \
  --interval 90 \
  --format both \
  --save_csv

python scripts/plot_single_location_time_dependency.py \
  --input_dir outputs/stvgp_kronecker_maps/routeB_task1_calibration_task2_online_fullspace/test \
  --output_dir results/figures/single_location \
  --eval_mode future \
  --location_strategy median_rmse \
  --methods structured_joint \
  --basis_modes observed extended \
  --block_size 10 \
  --interval 90 \
  --format both \
  --save_csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "no_transfer": "No transfer",
    "mean_field": "Mean-field",
    "mean_field_ssgp_transfer": "Mean-field",
    "structured_joint": "Structured joint",
    "structured_joint_ssgp_transfer": "Structured joint",
}

METHOD_COLORS = {
    "no_transfer": "#4C78A8",
    "mean_field": "#D99058",
    "mean_field_ssgp_transfer": "#D99058",
    "structured_joint": "#59A14F",
    "structured_joint_ssgp_transfer": "#59A14F",
}

BASIS_COLORS = {
    "observed": "#4C78A8",
    "extended": "#E15759",
}

PHI_LABELS = {
    "base": "base Phi",
    "rich": "rich seasonal-spatial Phi",
    "rich_seasonal_spatial": "rich seasonal-spatial Phi",
}


@dataclass
class SnapshotRecord:
    path: Path
    method: str
    eval_mode: str
    phi_mode: str
    basis_mode: str
    t_index: int
    time_value: float
    frame: pd.DataFrame


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
            "axes.linewidth": 0.8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-quality single-location ERA5 time-dependency "
            "diagnostics from saved prediction snapshots."
        )
    )
    parser.add_argument("--input_dir", type=Path, default=None)
    parser.add_argument("--predictions_path", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--location_index", type=int, default=None)
    parser.add_argument(
        "--location_strategy",
        choices=["manual", "median_rmse", "high_rmse", "high_variance"],
        default="median_rmse",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["seen_history", "future", "current"],
        default="seen_history",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["structured_joint", "mean_field", "no_transfer"],
    )
    parser.add_argument("--phi_modes", nargs="+", default=["base"])
    parser.add_argument("--basis_modes", nargs="+", default=["observed", "extended"])
    parser.add_argument("--interval", choices=["std", "90"], default="90")
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--origin_block", type=int, default=None)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--format", default="both", help="png, pdf, both, or comma-separated values such as png,pdf")
    return parser.parse_args()


def normalize_method(value: str) -> str:
    value = value.strip()
    aliases = {
        "Route B": "structured_joint",
        "route_b": "structured_joint",
        "structured_joint_ssgp_transfer": "structured_joint",
        "mean-field": "mean_field",
        "mean_field_ssgp_transfer": "mean_field",
    }
    return aliases.get(value, value)


def normalize_phi_mode(value: str) -> str:
    value = value.strip().lower()
    if value in {"rich", "rich_phi", "rich-seasonal-spatial", "rich_seasonal_spatial"}:
        return "rich_seasonal_spatial"
    return value


def infer_metadata(path: Path) -> tuple[str, str, str, str, int | None]:
    name = path.name
    lower = str(path).lower()
    eval_mode = "unknown"
    for candidate in ("seen_history", "future", "current"):
        if candidate in name:
            eval_mode = candidate
            break

    method = name
    if eval_mode != "unknown":
        method = name.split(f"_{eval_mode}_", maxsplit=1)[0]
    method = normalize_method(method)

    phi_mode = "rich_seasonal_spatial" if "rich" in lower else "base"
    basis_mode = "extended" if "extended" in lower else "observed"

    match = re.search(r"_t(\d+)\.npz$", name)
    t_index = int(match.group(1)) if match else None
    return method, eval_mode, phi_mode, basis_mode, t_index


def first_existing_key(data: np.lib.npyio.NpzFile, candidates: Iterable[str]) -> str | None:
    for key in candidates:
        if key in data.files:
            return key
    return None


def load_snapshot(path: Path) -> SnapshotRecord | None:
    method, eval_mode, phi_mode, basis_mode, parsed_t_index = infer_metadata(path)
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - defensive path reporting
        print(f"Warning: could not load {path}: {exc}")
        return None

    coords_key = first_existing_key(data, ["coords", "spatial_coords", "locations"])
    y_key = first_existing_key(data, ["y", "y_true", "truth", "target"])
    mean_key = first_existing_key(data, ["mean", "pred_mean", "mu", "prediction"])
    var_key = first_existing_key(data, ["variance", "pred_var", "var", "predictive_variance"])
    std_key = first_existing_key(data, ["std", "pred_std", "predictive_std"])

    if coords_key is None or y_key is None or mean_key is None:
        return None

    coords = np.asarray(data[coords_key], dtype=float)
    y = np.asarray(data[y_key], dtype=float).reshape(-1)
    mean = np.asarray(data[mean_key], dtype=float).reshape(-1)
    if coords.ndim != 2 or coords.shape[0] != y.shape[0] or y.shape[0] != mean.shape[0]:
        print(f"Warning: incompatible array shapes in {path}")
        return None

    if var_key is not None:
        variance = np.asarray(data[var_key], dtype=float).reshape(-1)
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)
    elif std_key is not None:
        std = np.asarray(data[std_key], dtype=float).reshape(-1)
        variance = std**2
    else:
        std = np.full_like(y, np.nan, dtype=float)
        variance = np.full_like(y, np.nan, dtype=float)

    t_index_key = first_existing_key(data, ["t_index", "time_index"])
    time_value_key = first_existing_key(data, ["time_value", "time", "t"])
    t_index = int(np.asarray(data[t_index_key]).item()) if t_index_key else parsed_t_index
    if t_index is None:
        t_index = 0
    time_value = float(np.asarray(data[time_value_key]).item()) if time_value_key else float(t_index)

    frame = pd.DataFrame(
        {
            "location_index": np.arange(y.shape[0], dtype=int),
            "lat": coords[:, 0],
            "lon": coords[:, 1] if coords.shape[1] > 1 else np.nan,
            "y": y,
            "mean": mean,
            "variance": variance,
            "std": std,
            "error": mean - y,
            "abs_error": np.abs(mean - y),
            "t_index": t_index,
            "time_value": time_value,
            "method": method,
            "eval_mode": eval_mode,
            "phi_mode": phi_mode,
            "basis_mode": basis_mode,
            "source_file": str(path),
        }
    )
    return SnapshotRecord(
        path=path,
        method=method,
        eval_mode=eval_mode,
        phi_mode=phi_mode,
        basis_mode=basis_mode,
        t_index=t_index,
        time_value=time_value,
        frame=frame,
    )


def load_all_snapshots(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    paths = sorted(input_dir.rglob("*.npz"))
    records = []
    for path in paths:
        record = load_snapshot(path)
        if record is not None:
            records.append(record.frame)
    if not records:
        raise SystemExit(
            f"No usable prediction snapshots were found under {input_dir}. "
            "Expected .npz files with coords, y, mean and optionally variance/std."
        )
    df = pd.concat(records, ignore_index=True)
    df = df.sort_values(["eval_mode", "method", "phi_mode", "basis_mode", "t_index", "location_index"])
    unavailable = sorted(set(p.name for p in paths) - set(Path(s).name for s in df["source_file"].unique()))
    if unavailable:
        warnings.append(f"{len(unavailable)} npz file(s) were skipped because required arrays were missing.")
    return df, warnings


def filter_rows(
    df: pd.DataFrame,
    *,
    eval_mode: str | None = None,
    methods: list[str] | None = None,
    phi_modes: list[str] | None = None,
    basis_modes: list[str] | None = None,
) -> pd.DataFrame:
    out = df
    if eval_mode is not None:
        out = out[out["eval_mode"] == eval_mode]
    if methods:
        normalized = {normalize_method(m) for m in methods}
        out = out[out["method"].isin(normalized)]
    if phi_modes:
        normalized_phi = {normalize_phi_mode(m) for m in phi_modes}
        out = out[out["phi_mode"].isin(normalized_phi)]
    if basis_modes:
        out = out[out["basis_mode"].isin(set(basis_modes))]
    return out.copy()


def select_location(df: pd.DataFrame, args: argparse.Namespace) -> tuple[int, dict[str, float | int | str]]:
    if args.location_index is not None:
        loc = int(args.location_index)
        strategy = "manual"
    else:
        candidate = df.copy()
        candidate["sq_error"] = (candidate["mean"] - candidate["y"]) ** 2
        grouped = candidate.groupby("location_index", as_index=False).agg(
            rmse=("sq_error", lambda x: float(math.sqrt(float(np.mean(x))))),
            avg_variance=("variance", "mean"),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
        if grouped.empty:
            raise SystemExit("No rows are available for location selection.")
        if args.location_strategy == "high_rmse":
            row = grouped.sort_values("rmse", ascending=False).iloc[0]
        elif args.location_strategy == "high_variance":
            row = grouped.sort_values("avg_variance", ascending=False).iloc[0]
        else:
            median_rmse = grouped["rmse"].median()
            row = grouped.iloc[(grouped["rmse"] - median_rmse).abs().argsort().iloc[0]]
        loc = int(row["location_index"])
        strategy = args.location_strategy

    loc_rows = df[df["location_index"] == loc]
    if loc_rows.empty:
        raise SystemExit(f"location_index={loc} is not present in the loaded predictions.")
    first = loc_rows.iloc[0]
    metadata = {
        "location_index": loc,
        "selection_strategy": strategy,
        "lat": float(first["lat"]),
        "lon": float(first["lon"]),
    }
    return loc, metadata


def interval_multiplier(interval: str) -> float:
    return 1.0 if interval == "std" else 1.645


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, fmt: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    formats = ["png", "pdf"] if fmt == "both" else [part.strip() for part in fmt.split(",") if part.strip()]
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=450 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def add_block_boundaries(ax: plt.Axes, values: pd.Series, block_size: int) -> None:
    if block_size <= 0 or values.empty:
        return
    lo = int(values.min())
    hi = int(values.max())
    for boundary in range((lo // block_size + 1) * block_size, hi + 1, block_size):
        ax.axvline(boundary, color="0.75", linestyle="--", linewidth=0.7, zorder=0)


def plot_methods_over_time(
    df: pd.DataFrame,
    args: argparse.Namespace,
    loc: int,
    loc_meta: dict[str, float | int | str],
) -> tuple[list[str], pd.DataFrame, list[str]]:
    warnings: list[str] = []
    plot_df = filter_rows(
        df,
        eval_mode=args.eval_mode,
        methods=args.methods,
        phi_modes=args.phi_modes,
        basis_modes=["observed"],
    )
    plot_df = plot_df[plot_df["location_index"] == loc]
    if plot_df.empty:
        warnings.append(f"No rows available for eval_mode={args.eval_mode} at location {loc}.")
        return [], plot_df, warnings

    requested = [normalize_method(m) for m in args.methods]
    available = set(plot_df["method"])
    missing = [m for m in requested if m not in available]
    if missing:
        warnings.append(
            "Missing per-location predictions for requested method(s): "
            + ", ".join(missing)
            + ". The figure uses available saved predictions only."
        )

    k = interval_multiplier(args.interval)
    fig, ax = plt.subplots(figsize=(6.7, 2.6), constrained_layout=True)
    truth = (
        plot_df[["t_index", "time_value", "y"]]
        .drop_duplicates()
        .sort_values("t_index")
    )
    ax.plot(truth["t_index"], truth["y"], color="black", linewidth=1.3, marker="o", markersize=2.7, label="Ground truth")
    for method in requested:
        method_df = plot_df[plot_df["method"] == method].sort_values("t_index")
        if method_df.empty:
            continue
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, "#666666")
        ax.plot(method_df["t_index"], method_df["mean"], color=color, linewidth=1.25, marker="o", markersize=2.4, label=label)
        if method_df["std"].notna().any():
            lower = method_df["mean"] - k * method_df["std"]
            upper = method_df["mean"] + k * method_df["std"]
            ax.fill_between(method_df["t_index"], lower, upper, color=color, alpha=0.15, linewidth=0)
        else:
            warnings.append(f"Uncertainty is unavailable for method={method}; bands were not drawn.")
    add_block_boundaries(ax, plot_df["t_index"], args.block_size)
    phi_text = ", ".join(PHI_LABELS.get(normalize_phi_mode(p), normalize_phi_mode(p)) for p in args.phi_modes)
    ax.set_title(
        f"ERA5 {args.eval_mode}: location {loc} "
        f"(lat {loc_meta['lat']:.2f}, lon {loc_meta['lon']:.2f}); {phi_text}"
    )
    ax.set_xlabel("time index")
    ax.set_ylabel("standardized ERA5 target")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    stem = f"single_location_{args.eval_mode}_methods_loc{loc:04d}"
    paths = save_figure(fig, args.output_dir, stem, args.format)
    return paths, plot_df, warnings


def plot_base_vs_rich_phi(
    df: pd.DataFrame,
    args: argparse.Namespace,
    loc: int,
    loc_meta: dict[str, float | int | str],
) -> tuple[list[str], pd.DataFrame, list[str]]:
    warnings: list[str] = []
    plot_df = filter_rows(
        df,
        eval_mode="seen_history",
        methods=["structured_joint"],
        phi_modes=["base", "rich_seasonal_spatial"],
        basis_modes=["observed"],
    )
    plot_df = plot_df[plot_df["location_index"] == loc]
    available_phi = set(plot_df["phi_mode"])
    if not {"base", "rich_seasonal_spatial"}.intersection(available_phi):
        warnings.append("No structured_joint base/rich Phi rows were available for Plot 2.")
        return [], plot_df, warnings
    if "rich_seasonal_spatial" not in available_phi:
        warnings.append("Rich Phi per-location predictions were not found; Plot 2 contains only available base Phi data.")
    if "base" not in available_phi:
        warnings.append("Base Phi per-location predictions were not found; Plot 2 contains only available rich Phi data.")

    phi_order = [p for p in ["base", "rich_seasonal_spatial"] if p in available_phi]
    k = interval_multiplier(args.interval)
    fig, axes = plt.subplots(len(phi_order), 1, figsize=(6.7, 2.45 * len(phi_order)), sharex=True, constrained_layout=True)
    if len(phi_order) == 1:
        axes = [axes]
    for ax, phi_mode in zip(axes, phi_order):
        panel = plot_df[plot_df["phi_mode"] == phi_mode].sort_values("t_index")
        ax.plot(panel["t_index"], panel["y"], color="black", linewidth=1.2, marker="o", markersize=2.5, label="Ground truth")
        color = "#59A14F" if phi_mode == "base" else "#B07AA1"
        ax.plot(panel["t_index"], panel["mean"], color=color, linewidth=1.25, marker="o", markersize=2.4, label="Structured joint")
        if panel["std"].notna().any():
            ax.fill_between(panel["t_index"], panel["mean"] - k * panel["std"], panel["mean"] + k * panel["std"], color=color, alpha=0.16, linewidth=0)
        else:
            warnings.append(f"Uncertainty is unavailable for phi_mode={phi_mode}; bands were not drawn.")
        add_block_boundaries(ax, panel["t_index"], args.block_size)
        ax.set_ylabel("target")
        ax.set_title(PHI_LABELS.get(phi_mode, phi_mode))
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time index")
    fig.suptitle(
        f"ERA5 single-location Phi diagnostic: location {loc} "
        f"(lat {loc_meta['lat']:.2f}, lon {loc_meta['lon']:.2f})",
        y=1.02,
        fontsize=9,
    )
    stem = f"single_location_base_vs_rich_phi_loc{loc:04d}"
    paths = save_figure(fig, args.output_dir, stem, args.format)
    return paths, plot_df, warnings


def plot_future_horizon(
    df: pd.DataFrame,
    args: argparse.Namespace,
    loc: int,
    loc_meta: dict[str, float | int | str],
) -> tuple[list[str], pd.DataFrame, list[str]]:
    warnings: list[str] = []
    plot_df = filter_rows(
        df,
        eval_mode="future",
        methods=["structured_joint"],
        phi_modes=args.phi_modes,
        basis_modes=args.basis_modes,
    )
    plot_df = plot_df[plot_df["location_index"] == loc]
    if plot_df.empty:
        warnings.append("No future rows were available for Plot 3.")
        return [], plot_df, warnings

    if args.origin_block is None:
        origin_blocks = ((plot_df["t_index"] // args.block_size) - 1).clip(lower=0)
        counts = origin_blocks.value_counts().sort_values(ascending=False)
        origin = int(counts.index[0])
    else:
        origin = int(args.origin_block)
    start = (origin + 1) * args.block_size
    end = start + args.block_size
    panel = plot_df[(plot_df["t_index"] >= start) & (plot_df["t_index"] < end)].copy()
    if panel.empty:
        warnings.append(
            f"No future rows matched origin_block={origin}; using all available future rows instead."
        )
        panel = plot_df.copy()
    panel["horizon"] = (panel["t_index"] - panel["t_index"].min()) + 1

    requested_basis = set(args.basis_modes)
    available_basis = set(panel["basis_mode"])
    missing = sorted(requested_basis - available_basis)
    if missing:
        warnings.append(
            "Missing future basis mode(s): "
            + ", ".join(missing)
            + ". The figure uses available saved predictions only."
        )

    k = interval_multiplier(args.interval)
    fig, ax = plt.subplots(figsize=(5.2, 2.7), constrained_layout=True)
    truth = panel[["horizon", "y"]].drop_duplicates().sort_values("horizon")
    ax.plot(truth["horizon"], truth["y"], color="black", linewidth=1.25, marker="o", markersize=3.0, label="Ground truth")
    for basis_mode in args.basis_modes:
        basis_df = panel[panel["basis_mode"] == basis_mode].sort_values("horizon")
        if basis_df.empty:
            continue
        color = BASIS_COLORS.get(basis_mode, "#666666")
        ax.plot(basis_df["horizon"], basis_df["mean"], color=color, linewidth=1.25, marker="o", markersize=2.8, label=f"{basis_mode} basis")
        if basis_df["std"].notna().any():
            ax.fill_between(basis_df["horizon"], basis_df["mean"] - k * basis_df["std"], basis_df["mean"] + k * basis_df["std"], color=color, alpha=0.16, linewidth=0)
        else:
            warnings.append(f"Uncertainty is unavailable for basis_mode={basis_mode}; bands were not drawn.")
    ax.set_title(
        f"ERA5 block-ahead future diagnostic: location {loc}; origin block {origin}"
    )
    ax.set_xlabel("horizon inside next block")
    ax.set_ylabel("standardized ERA5 target")
    ax.legend(loc="best")
    stem = f"single_location_future_horizon_observed_vs_extended_loc{loc:04d}_origin{origin:03d}"
    paths = save_figure(fig, args.output_dir, stem, args.format)
    return paths, panel, warnings


def maybe_plot_aggregate_horizon(input_dir: Path, output_dir: Path, fmt: str) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    candidates = sorted(input_dir.rglob("*future*horizon*summary*.csv"))
    if not candidates:
        parent_results = Path("results")
        candidates = sorted(parent_results.rglob("*future*horizon*summary*.csv")) if parent_results.exists() else []
    if not candidates:
        warnings.append("No future horizon summary CSV was found for optional aggregate horizon plot.")
        return [], warnings

    rows = []
    for path in candidates:
        try:
            data = pd.read_csv(path)
        except Exception:
            continue
        if "horizon" not in data.columns or "rmse" not in data.columns:
            continue
        data = data.copy()
        data["source"] = path.parent.name
        rows.append(data)
    if not rows:
        warnings.append("Future horizon CSV files were found, but none had horizon/rmse columns.")
        return [], warnings
    df = pd.concat(rows, ignore_index=True)
    metric_df = df.groupby("horizon", as_index=False)["rmse"].mean().sort_values("horizon")
    fig, ax = plt.subplots(figsize=(4.4, 2.5), constrained_layout=True)
    ax.plot(metric_df["horizon"], metric_df["rmse"], color="#4C78A8", marker="o", linewidth=1.3)
    ax.set_title("Aggregate future RMSE by horizon")
    ax.set_xlabel("horizon")
    ax.set_ylabel("RMSE")
    paths = save_figure(fig, output_dir, "single_location_optional_aggregate_future_rmse_by_horizon", fmt)
    return paths, warnings


def write_csv(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def _norm_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {
        "lat": "latitude",
        "lon": "longitude",
        "y": "y_true",
        "mean": "pred_mean",
        "variance": "pred_var_y",
        "std": "pred_std_y",
        "t_index": "time_index",
        "time_value": "actual_time",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    required = [
        "eval_mode",
        "method",
        "phi_mode",
        "basis_mode",
        "time_index",
        "location_index",
        "latitude",
        "longitude",
        "y_true",
        "pred_mean",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Prediction file is missing required column(s): {', '.join(missing)}")
    if "pred_var_y" not in df.columns:
        if "pred_std_y" in df.columns:
            df["pred_var_y"] = np.asarray(df["pred_std_y"], dtype=float) ** 2
        elif "latent_var" in df.columns and "observation_noise_variance" in df.columns:
            df["pred_var_y"] = np.asarray(df["latent_var"], dtype=float) + np.asarray(df["observation_noise_variance"], dtype=float)
        else:
            raise SystemExit(
                "Prediction file must contain pred_var_y, pred_std_y, or latent_var + observation_noise_variance."
            )
    df["pred_var_y"] = np.maximum(pd.to_numeric(df["pred_var_y"], errors="coerce"), 1e-10)
    df["pred_std_y"] = np.sqrt(df["pred_var_y"])
    df["method"] = df["method"].map(normalize_method)
    df["phi_mode"] = df["phi_mode"].map(normalize_phi_mode)
    df["basis_mode"] = df["basis_mode"].fillna("observed")
    for col in ["time_index", "location_index"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    if "train_block_id" not in df.columns and "block_id" in df.columns:
        df["train_block_id"] = df["block_id"]
    if "block_id" not in df.columns and "train_block_id" in df.columns:
        df["block_id"] = df["train_block_id"]
    if "horizon" in df.columns:
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
    return df


def _final_block_rows(df: pd.DataFrame, *, eval_mode: str, methods: list[str], phi_modes: list[str]) -> pd.DataFrame:
    sub = df[
        (df["eval_mode"] == eval_mode)
        & (df["method"].isin([normalize_method(m) for m in methods]))
        & (df["phi_mode"].isin([normalize_phi_mode(p) for p in phi_modes]))
    ].copy()
    if sub.empty:
        raise SystemExit(f"No prediction rows found for eval_mode={eval_mode}, methods={methods}, phi_modes={phi_modes}.")
    if "train_block_id" in sub.columns:
        max_by_group = sub.groupby(["method", "phi_mode"])["train_block_id"].max()
        final_block = int(max_by_group.min())
        sub = sub[pd.to_numeric(sub["train_block_id"], errors="coerce") == final_block]
    return sub


def select_location_from_predictions(df: pd.DataFrame, args: argparse.Namespace) -> tuple[int, dict[str, float | int | str]]:
    if args.location_index is not None:
        loc = int(args.location_index)
        sub = df[df["location_index"] == loc]
        if sub.empty:
            raise SystemExit(f"Requested location_index={loc} is absent from prediction file.")
    else:
        base = _final_block_rows(df, eval_mode="seen_history", methods=["structured_joint"], phi_modes=["base"])
        base["sq_error"] = (base["pred_mean"] - base["y_true"]) ** 2
        grouped = base.groupby("location_index", as_index=False).agg(
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
            median_rmse = grouped["rmse"].median()
            row = grouped.iloc[(grouped["rmse"] - median_rmse).abs().argsort().iloc[0]]
        loc = int(row["location_index"])
        sub = df[df["location_index"] == loc]
    first = sub.iloc[0]
    return loc, {
        "location_index": loc,
        "selection_strategy": "manual" if args.location_index is not None else args.location_strategy,
        "lat": float(first["latitude"]),
        "lon": float(first["longitude"]),
    }


def require_values(df: pd.DataFrame, column: str, values: list[str], context: str) -> None:
    present = set(df[column].astype(str))
    missing = [value for value in values if value not in present]
    if missing:
        raise SystemExit(f"{context} is missing {column} value(s): {', '.join(missing)}. Regenerate the corresponding evaluation output.")


def plot_seen_history_method_comparison_from_predictions(
    df: pd.DataFrame,
    args: argparse.Namespace,
    loc: int,
    loc_meta: dict[str, float | int | str],
) -> tuple[list[str], pd.DataFrame]:
    methods = ["no_transfer", "mean_field", "structured_joint"]
    sub = _final_block_rows(df, eval_mode="seen_history", methods=methods, phi_modes=["base"])
    sub = sub[sub["location_index"] == loc].sort_values("time_index")
    require_values(sub, "method", methods, "seen_history method comparison")
    k = interval_multiplier(args.interval)
    fig, ax = plt.subplots(figsize=(6.8, 2.8), constrained_layout=True)
    truth = sub[["time_index", "y_true"]].drop_duplicates().sort_values("time_index")
    ax.plot(truth["time_index"], truth["y_true"], color="black", linewidth=1.2, label="Ground truth")
    for method in methods:
        item = sub[sub["method"] == method].sort_values("time_index")
        color = METHOD_COLORS.get(method, "#666666")
        ax.plot(item["time_index"], item["pred_mean"], color=color, linewidth=1.25, label=METHOD_LABELS.get(method, method))
        ax.fill_between(
            item["time_index"],
            item["pred_mean"] - k * item["pred_std_y"],
            item["pred_mean"] + k * item["pred_std_y"],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    add_block_boundaries(ax, sub["time_index"], args.block_size)
    ax.set_title(f"ERA5 seen-history at location {loc} (lat {loc_meta['lat']:.2f}, lon {loc_meta['lon']:.2f})")
    ax.set_xlabel("time index")
    ax.set_ylabel("standardized target")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    paths = save_figure(fig, args.output_dir, f"single_location_seen_history_method_comparison_loc{loc:04d}", args.format)
    return paths, sub


def plot_base_vs_rich_from_predictions(
    df: pd.DataFrame,
    args: argparse.Namespace,
    loc: int,
    loc_meta: dict[str, float | int | str],
) -> tuple[list[str], pd.DataFrame]:
    phi_modes = ["base", "rich_seasonal_spatial"]
    sub = _final_block_rows(df, eval_mode="seen_history", methods=["structured_joint"], phi_modes=phi_modes)
    sub = sub[sub["location_index"] == loc].sort_values("time_index")
    require_values(sub, "phi_mode", phi_modes, "base-vs-rich Phi diagnostic")
    k = interval_multiplier(args.interval)
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.8), sharex=True, constrained_layout=True)
    for ax, phi_mode, color in zip(axes, phi_modes, ["#59A14F", "#B07AA1"]):
        item = sub[sub["phi_mode"] == phi_mode].sort_values("time_index")
        ax.plot(item["time_index"], item["y_true"], color="black", linewidth=1.1, label="Ground truth")
        ax.plot(item["time_index"], item["pred_mean"], color=color, linewidth=1.25, label="Structured joint")
        ax.fill_between(
            item["time_index"],
            item["pred_mean"] - k * item["pred_std_y"],
            item["pred_mean"] + k * item["pred_std_y"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        add_block_boundaries(ax, item["time_index"], args.block_size)
        ax.set_title(PHI_LABELS.get(phi_mode, phi_mode))
        ax.set_ylabel("target")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time index")
    fig.suptitle(f"ERA5 Phi diagnostic at location {loc} (lat {loc_meta['lat']:.2f}, lon {loc_meta['lon']:.2f})", y=1.02)
    paths = save_figure(fig, args.output_dir, f"single_location_base_vs_rich_phi_loc{loc:04d}", args.format)
    return paths, sub


def choose_future_origin(sub: pd.DataFrame, block_size: int, origin_block: int | None) -> int:
    if origin_block is not None:
        return int(origin_block)
    candidates = []
    for origin, group in sub.groupby("origin_block"):
        try:
            origin_int = int(float(origin))
        except (TypeError, ValueError):
            continue
        basis_ok = {"observed", "extended"}.issubset(set(group["basis_mode"]))
        horizons = set(pd.to_numeric(group["horizon"], errors="coerce").dropna().astype(int))
        horizon_ok = set(range(1, block_size + 1)).issubset(horizons)
        if basis_ok and horizon_ok:
            candidates.append(origin_int)
    if not candidates:
        raise SystemExit(
            "No future origin block contains both observed/extended basis predictions for all horizons. "
            "Regenerate future per-location artifacts with both --future-basis-mode observed and extended."
        )
    return sorted(candidates)[len(candidates) // 2]


def plot_future_horizon_from_predictions(
    df: pd.DataFrame,
    args: argparse.Namespace,
    loc: int,
    loc_meta: dict[str, float | int | str],
) -> tuple[list[str], pd.DataFrame]:
    sub = df[
        (df["eval_mode"] == "future")
        & (df["method"] == "structured_joint")
        & (df["phi_mode"] == "base")
        & (df["location_index"] == loc)
    ].copy()
    require_values(sub, "basis_mode", ["observed", "extended"], "future horizon diagnostic")
    if "origin_block" not in sub.columns or "horizon" not in sub.columns:
        raise SystemExit("Future prediction rows must contain origin_block and horizon columns.")
    origin = choose_future_origin(sub, args.block_size, args.origin_block)
    sub = sub[pd.to_numeric(sub["origin_block"], errors="coerce") == origin].copy()
    sub["horizon"] = pd.to_numeric(sub["horizon"], errors="coerce").astype(int)
    horizons = set(sub["horizon"])
    missing_h = sorted(set(range(1, args.block_size + 1)) - horizons)
    if missing_h:
        raise SystemExit(f"Future diagnostic origin_block={origin} is missing horizon(s): {missing_h}. Regenerate full block-ahead artifacts.")
    require_values(sub, "basis_mode", ["observed", "extended"], "future horizon diagnostic")
    k = interval_multiplier(args.interval)
    fig, ax = plt.subplots(figsize=(5.2, 2.9), constrained_layout=True)
    truth = sub[["horizon", "y_true"]].drop_duplicates().sort_values("horizon")
    ax.plot(truth["horizon"], truth["y_true"], color="black", linewidth=1.2, marker="o", markersize=3.0, label="Ground truth")
    for basis_mode in ["observed", "extended"]:
        item = sub[sub["basis_mode"] == basis_mode].sort_values("horizon")
        color = BASIS_COLORS[basis_mode]
        ax.plot(item["horizon"], item["pred_mean"], color=color, linewidth=1.25, marker="o", markersize=2.8, label=f"{basis_mode} basis")
        ax.fill_between(
            item["horizon"],
            item["pred_mean"] - k * item["pred_std_y"],
            item["pred_mean"] + k * item["pred_std_y"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    ax.set_title(f"ERA5 block-ahead future at location {loc}; origin block {origin}")
    ax.set_xlabel("horizon h inside next block")
    ax.set_ylabel("standardized target")
    ax.set_xticks(range(1, args.block_size + 1))
    ax.legend(loc="best")
    paths = save_figure(fig, args.output_dir, f"single_location_future_horizon_observed_vs_extended_loc{loc:04d}_origin{origin:03d}", args.format)
    return paths, sub


def main_from_predictions_csv(args: argparse.Namespace) -> None:
    if args.predictions_path is None:
        raise SystemExit("--predictions_path is required for strict complete diagnostics.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = _norm_prediction_columns(pd.read_csv(args.predictions_path))
    loc, loc_meta = select_location_from_predictions(df, args)
    outputs: list[str] = []
    csv_files: list[str] = []
    paths, plotted = plot_seen_history_method_comparison_from_predictions(df, args, loc, loc_meta)
    outputs.extend(paths)
    if args.save_csv:
        csv_files.append(write_csv(args.output_dir / f"single_location_seen_history_method_comparison_loc{loc:04d}.csv", plotted))
    paths, plotted = plot_base_vs_rich_from_predictions(df, args, loc, loc_meta)
    outputs.extend(paths)
    if args.save_csv:
        csv_files.append(write_csv(args.output_dir / f"single_location_base_vs_rich_phi_loc{loc:04d}.csv", plotted))
    paths, plotted = plot_future_horizon_from_predictions(df, args, loc, loc_meta)
    outputs.extend(paths)
    if args.save_csv:
        csv_files.append(write_csv(args.output_dir / f"single_location_future_horizon_observed_vs_extended_loc{loc:04d}.csv", plotted))
    metadata = {
        **loc_meta,
        "predictions_path": str(args.predictions_path),
        "output_dir": str(args.output_dir),
        "interval": args.interval,
        "block_size": args.block_size,
        "origin_block": args.origin_block,
        "output_files": outputs,
        "csv_files": csv_files,
        "available_methods": sorted(df["method"].unique().tolist()),
        "available_eval_modes": sorted(df["eval_mode"].unique().tolist()),
        "available_phi_modes": sorted(df["phi_mode"].unique().tolist()),
        "available_basis_modes": sorted(df["basis_mode"].unique().tolist()),
    }
    metadata_path = args.output_dir / "single_location_time_dependency_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions_path is not None:
        main_from_predictions_csv(args)
        return

    if args.input_dir is None:
        raise SystemExit("Either --predictions_path or --input_dir must be provided.")

    all_rows, warnings = load_all_snapshots(args.input_dir)
    selection_rows = filter_rows(
        all_rows,
        eval_mode=args.eval_mode,
        methods=args.methods,
        phi_modes=args.phi_modes,
        basis_modes=args.basis_modes,
    )
    if selection_rows.empty:
        selection_rows = all_rows
        warnings.append("Requested filters were empty for location selection; all loaded rows were used.")
    loc, loc_meta = select_location(selection_rows, args)

    output_files: list[str] = []
    csv_files: list[str] = []
    source_files = sorted(all_rows["source_file"].unique().tolist())

    paths, plotted, plot_warnings = plot_methods_over_time(all_rows, args, loc, loc_meta)
    output_files.extend(paths)
    warnings.extend(plot_warnings)
    if args.save_csv and not plotted.empty:
        csv_files.append(write_csv(args.output_dir / f"single_location_{args.eval_mode}_methods_loc{loc:04d}.csv", plotted))

    paths, plotted, plot_warnings = plot_base_vs_rich_phi(all_rows, args, loc, loc_meta)
    output_files.extend(paths)
    warnings.extend(plot_warnings)
    if args.save_csv and not plotted.empty:
        csv_files.append(write_csv(args.output_dir / f"single_location_base_vs_rich_phi_loc{loc:04d}.csv", plotted))

    paths, plotted, plot_warnings = plot_future_horizon(all_rows, args, loc, loc_meta)
    output_files.extend(paths)
    warnings.extend(plot_warnings)
    if args.save_csv and not plotted.empty:
        origin_label = "available"
        csv_files.append(write_csv(args.output_dir / f"single_location_future_horizon_loc{loc:04d}_{origin_label}.csv", plotted))

    paths, plot_warnings = maybe_plot_aggregate_horizon(args.input_dir, args.output_dir, args.format)
    output_files.extend(paths)
    warnings.extend(plot_warnings)

    metadata = {
        **loc_meta,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "methods": [normalize_method(m) for m in args.methods],
        "phi_modes": [normalize_phi_mode(p) for p in args.phi_modes],
        "eval_mode": args.eval_mode,
        "basis_modes": args.basis_modes,
        "interval": args.interval,
        "block_size": args.block_size,
        "origin_block": args.origin_block,
        "source_files_used": source_files,
        "output_files": output_files,
        "csv_files": csv_files,
        "warnings": warnings,
        "available_methods": sorted(all_rows["method"].unique().tolist()),
        "available_eval_modes": sorted(all_rows["eval_mode"].unique().tolist()),
        "available_phi_modes": sorted(all_rows["phi_mode"].unique().tolist()),
        "available_basis_modes": sorted(all_rows["basis_mode"].unique().tolist()),
    }
    metadata_path = args.output_dir / "single_location_time_dependency_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
