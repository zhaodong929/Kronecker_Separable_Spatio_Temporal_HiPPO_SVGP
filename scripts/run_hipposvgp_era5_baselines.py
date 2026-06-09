#!/usr/bin/env python
"""Run online baselines on a processed HiPPO-SVGP ERA5 subset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback.
    resource = None

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.online_baselines import PredictionResult, make_baseline, timer
from stvgp_kronecker.data.hipposvgp_era5 import (
    HippoERA5Dataset,
    build_phi_features,
    iter_online_blocks,
    load_hipposvgp_era5,
)
from scripts.run_hipposvgp_era5_routeb import augment_dataset_phi, select_hyperparams_from_calibration_fullgp_mll


Z90 = 1.6448536269514722


def gaussian_nll(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(np.asarray(var, dtype=float), 1e-10)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (np.asarray(y) - mean) ** 2 / var))


def coverage90(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    half = Z90 * np.sqrt(np.maximum(var, 1e-10))
    return float(np.mean((y >= mean - half) & (y <= mean + half)))


def ece_gaussian(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    levels = np.asarray([0.5, 0.8, 0.9, 0.95])
    z_values = np.asarray([0.67448975, 1.28155157, 1.64485363, 1.95996398])
    sigma = np.sqrt(np.maximum(var, 1e-10))
    errors = []
    for level, z in zip(levels, z_values):
        half = z * sigma
        empirical = np.mean((y >= mean - half) & (y <= mean + half))
        errors.append(abs(float(empirical) - float(level)))
    return float(np.mean(errors))


def memory_mb() -> float:
    if resource is None:
        return float("nan")
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes. The project is WSL/Linux here.
    return float(usage / 1024.0)


def phi_for_slice(dataset: HippoERA5Dataset, block: slice) -> np.ndarray:
    S = dataset.coords.shape[0]
    rows = []
    for t_idx in range(block.start or 0, block.stop or dataset.Y.shape[0]):
        rows.extend(range(t_idx * S, (t_idx + 1) * S))
    return dataset.Phi[np.asarray(rows)]


def predict_block(model, dataset: HippoERA5Dataset, block: slice) -> PredictionResult:
    times = dataset.times[block]
    phi = phi_for_slice(dataset, block)
    return model.predict(times, dataset.coords, phi)


def model_diagnostics(model) -> dict:
    if hasattr(model, "diagnostics"):
        return dict(model.diagnostics())
    return {}


def metric_row(
    method: str,
    eval_mode: str,
    block_id: int,
    y: np.ndarray,
    pred: PredictionResult,
    runtime: float,
    diagnostics: dict,
) -> dict:
    var = np.maximum(pred.variance, 1e-10)
    std = np.sqrt(var)
    return {
        "method": method,
        "eval_mode": eval_mode,
        "block_id": block_id,
        "rmse": float(np.sqrt(np.mean((y - pred.mean) ** 2))),
        "mae": float(np.mean(np.abs(y - pred.mean))),
        "nll": gaussian_nll(y, pred.mean, var),
        "coverage90": coverage90(y, pred.mean, var),
        "ece": ece_gaussian(y, pred.mean, var),
        "avg_var": float(np.mean(var)),
        "avg_std": float(np.mean(std)),
        "avg_width90": float(np.mean(2.0 * Z90 * std)),
        "avg_predictive_variance": float(np.mean(var)),
        "avg_interval_width90": float(np.mean(2.0 * Z90 * std)),
        "runtime": runtime,
        "runtime_per_block": runtime,
        "num_train": int(diagnostics.get("num_train", 0)),
        "num_test": int(np.asarray(y).size),
        "coverage_sample_count": int(np.asarray(y).size),
        "memory_mb": memory_mb(),
        **diagnostics,
    }


def eval_blocks(blocks: list[slice], block_id: int, mode: str) -> list[slice]:
    if mode == "current":
        return [blocks[block_id]]
    if mode == "seen_history":
        return [slice(0, blocks[block_id].stop)]
    if mode == "future":
        return [] if block_id + 1 >= len(blocks) else [blocks[block_id + 1]]
    raise ValueError(f"Unknown eval mode: {mode}")


def run_one_method(dataset: HippoERA5Dataset, method_name: str, args: argparse.Namespace) -> list[dict]:
    blocks = iter_online_blocks(dataset.Y.shape[0], args.block_size)
    if not blocks:
        raise ValueError("No online blocks were produced")
    model = make_baseline(
        method_name,
        ridge=args.ridge,
        training_iterations=args.gp_training_iterations,
        learning_rate=args.gp_learning_rate,
        inducing_points=args.gp_inducing_points,
        minibatch_size=args.gp_minibatch_size,
        noise_lower_bound=args.gp_noise_lower_bound,
        inducing_init=args.gp_inducing_init,
        seed=args.seed,
        kernel_type=args.gp_kernel_type,
        fixed_lengthscale=args.gp_fixed_lengthscale,
        fixed_noise=args.gp_fixed_noise,
        fixed_outputscale=args.gp_fixed_outputscale,
        freeze_kernel_hyperparams=args.gp_freeze_hyperparams,
    )
    rows: list[dict] = []
    method = model.name()
    fitted = False
    for block_id, block in enumerate(blocks):
        t0 = timer()
        if not fitted:
            model.fit_initial_task(dataset.times[block], dataset.coords, dataset.Y[block], phi_for_slice(dataset, block))
            fitted = True
        else:
            model.update_block(dataset.times[block], dataset.coords, dataset.Y[block], phi_for_slice(dataset, block))
        runtime = timer() - t0

        for mode in args.eval_modes:
            mode_blocks = eval_blocks(blocks, block_id, mode)
            if not mode_blocks:
                continue
            for eval_block in mode_blocks:
                pred = predict_block(model, dataset, eval_block)
                row = metric_row(method, mode, block_id, dataset.Y[eval_block], pred, runtime, model_diagnostics(model))
                row.update(
                    {
                        "dataset": "era5_processed_timeseries_4",
                        "tasks": ",".join(dataset.tasks),
                        "variable_index": dataset.variable_index,
                        "num_time": dataset.Y.shape[0],
                        "num_space": dataset.Y.shape[1],
                        "block_size": args.block_size,
                        "seed": args.seed,
                        "scale": "scaled" if dataset.scaled else "unscaled",
                        "phi_mode": args.phi_mode,
                        "gp_kernel_type": args.gp_kernel_type,
                        "gp_hyperparam_fit_mode": args.gp_hyperparam_fit_mode,
                        "gp_fixed_lengthscale": args.gp_fixed_lengthscale if args.gp_fixed_lengthscale is not None else "",
                        "gp_fixed_noise": args.gp_fixed_noise if args.gp_fixed_noise is not None else "",
                        "gp_fixed_outputscale": args.gp_fixed_outputscale if args.gp_fixed_outputscale is not None else "",
                        "gp_freeze_hyperparams": bool(args.gp_freeze_hyperparams),
                    }
                )
                rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["eval_mode"]), []).append(row)
    summary = []
    for (method, mode), group in sorted(groups.items()):
        out = {"method": method, "eval_mode": mode, "num_rows": len(group)}
        for key in [
            "rmse",
            "mae",
            "nll",
            "coverage90",
            "ece",
            "avg_var",
            "avg_std",
            "avg_width90",
            "avg_predictive_variance",
            "avg_interval_width90",
            "runtime",
            "runtime_per_block",
            "num_train",
            "num_test",
            "coverage_sample_count",
            "memory_mb",
        ]:
            if key not in group[0]:
                continue
            vals = np.asarray([float(row[key]) for row in group], dtype=float)
            out[key] = float(np.mean(vals))
            out[f"{key}_se"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        summary.append(out)
    return summary


def plot_summary(summary: list[dict], outdir: Path, *, prefix: str = "era5_baselines") -> None:
    if not summary:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["rmse", "nll", "coverage90", "ece"]
    methods = sorted({row["method"] for row in summary})
    modes = sorted({row["eval_mode"] for row in summary})
    for metric in metrics:
        fig, axes = plt.subplots(1, len(modes), figsize=(4.2 * len(modes), 3.2), squeeze=False)
        for ax, mode in zip(axes[0], modes):
            vals = []
            for method in methods:
                found = [row for row in summary if row["method"] == method and row["eval_mode"] == mode]
                vals.append(found[0][metric] if found else np.nan)
            ax.bar(np.arange(len(methods)), vals)
            ax.set_title(mode)
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels(methods, rotation=35, ha="right")
            ax.set_ylabel(metric)
            if metric == "coverage90":
                ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{prefix}_{metric}.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--tasks", nargs="+", default=["task_1"])
    parser.add_argument("--variable-index", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--first-n-locations", type=int, default=None)
    parser.add_argument("--random-n-locations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--lat-bounds", nargs=2, type=float, default=None)
    parser.add_argument("--lon-bounds", nargs=2, type=float, default=None)
    parser.add_argument("--max-time", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument(
        "--phi-mode",
        choices=["base", "rich_v1", "rich_v2", "rich_v3", "rich_seasonal_spatial", "lag_ar", "rich_v3_lag_ar"],
        default="base",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["persistence", "climatology", "ridge"],
        choices=[
            "persistence",
            "climatology",
            "ridge",
            "independent_temporal_gp",
            "independent_gp",
            "gpytorch_sgpr",
            "sgpr",
            "gpytorch_sgpr_phi",
            "sgpr_phi",
            "gpytorch_svgp",
            "svgp",
            "gpytorch_svgp_phi",
            "svgp_phi",
        ],
    )
    parser.add_argument("--eval-modes", nargs="+", default=["current", "seen_history", "future"])
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--gp-training-iterations", type=int, default=20)
    parser.add_argument("--gp-learning-rate", type=float, default=0.05)
    parser.add_argument("--gp-inducing-points", type=int, default=32)
    parser.add_argument("--gp-minibatch-size", type=int, default=None)
    parser.add_argument("--gp-noise-lower-bound", type=float, default=1e-5)
    parser.add_argument("--gp-inducing-init", choices=["linspace", "random"], default="linspace")
    parser.add_argument("--gp-kernel-type", choices=["rbf", "matern32"], default="rbf")
    parser.add_argument("--gp-freeze-hyperparams", action="store_true")
    parser.add_argument("--gp-fixed-lengthscale", type=float, default=None)
    parser.add_argument("--gp-fixed-noise", type=float, default=None)
    parser.add_argument("--gp-fixed-outputscale", type=float, default=None)
    parser.add_argument("--gp-hyperparam-fit-mode", choices=["none", "routeb_initial_task_fullgp_grid"], default="none")
    parser.add_argument("--calibration-tasks", nargs="+", default=["task_1"])
    parser.add_argument("--ell-t-grid", nargs="+", type=float, default=[0.0125, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20])
    parser.add_argument("--noise-grid", nargs="+", type=float, default=[0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80])
    parser.add_argument("--kernel-variance-grid", nargs="+", type=float, default=[0.10, 0.25, 0.50, 1.00, 1.50])
    parser.add_argument("--hyperparam-fit-max-time", type=int, default=30)
    parser.add_argument("--hyperparam-fit-max-locations", type=int, default=30)
    parser.add_argument("--kernel-type", choices=["rbf", "matern32", "ard_rbf"], default="matern32")
    parser.add_argument("--kernel-variance", type=float, default=0.1)
    parser.add_argument("--spatial-lengthscale", type=float, default=0.35)
    parser.add_argument("--spatial-ard-lengthscales", nargs=2, type=float, default=None)
    parser.add_argument("--model-ell-t", type=float, default=0.05)
    parser.add_argument("--ell-t-fit-mode", choices=["none"], default="none")
    parser.add_argument("--outdir", default="results/experiments_era5_baselines")
    parser.add_argument("--output-prefix", default="era5_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    rows: list[dict] = []
    errors: list[dict] = []
    dataset_shape = None
    for seed in seeds:
        args.seed = seed
        if args.gp_hyperparam_fit_mode == "routeb_initial_task_fullgp_grid":
            args.kernel_type = "matern32" if args.gp_kernel_type == "matern32" else "rbf"
            calibration_dataset = load_hipposvgp_era5(
                args.root,
                tasks=args.calibration_tasks,
                variable_index=args.variable_index,
                prefer_scaled=True,
                split=args.split,
                first_n_locations=args.first_n_locations,
                random_n_locations=args.random_n_locations,
                seed=args.seed,
                lat_bounds=tuple(args.lat_bounds) if args.lat_bounds else None,
                lon_bounds=tuple(args.lon_bounds) if args.lon_bounds else None,
                max_time=args.max_time,
            )
            calibration_dataset = augment_dataset_phi(calibration_dataset, phi_mode=args.phi_mode)
            ell_t, sigma2, kernel_variance, score, grid_scores = select_hyperparams_from_calibration_fullgp_mll(
                calibration_dataset,
                args,
            )
            args.gp_fixed_lengthscale = ell_t
            args.gp_fixed_noise = sigma2
            args.gp_fixed_outputscale = kernel_variance
            args.gp_freeze_hyperparams = True
            args.kernel_variance = kernel_variance
            args.gp_hyperparam_fit_score = score
            args.gp_hyperparam_grid_scores = grid_scores
        dataset = load_hipposvgp_era5(
            args.root,
            tasks=args.tasks,
            variable_index=args.variable_index,
            prefer_scaled=True,
            split=args.split,
            first_n_locations=args.first_n_locations,
            random_n_locations=args.random_n_locations,
            seed=args.seed,
            lat_bounds=tuple(args.lat_bounds) if args.lat_bounds else None,
            lon_bounds=tuple(args.lon_bounds) if args.lon_bounds else None,
            max_time=args.max_time,
        )
        dataset = augment_dataset_phi(dataset, phi_mode=args.phi_mode)
        dataset_shape = {"T": dataset.Y.shape[0], "S": dataset.Y.shape[1], "p": dataset.Phi.shape[1]}
        for method in args.methods:
            try:
                rows.extend(run_one_method(dataset, method, args))
            except ImportError as exc:
                errors.append({"method": method, "seed": seed, "error": str(exc)})
                print(f"Skipping {method} for seed {seed}: {exc}")
    summary = summarize(rows)
    write_csv(rows, outdir / f"{args.output_prefix}_metrics.csv")
    write_csv(summary, outdir / f"{args.output_prefix}_summary.csv")
    plot_summary(summary, outdir, prefix=args.output_prefix)
    report = {
        "args": vars(args),
        "dataset_shape": dataset_shape,
        "summary": summary,
        "errors": errors,
    }
    (outdir / f"{args.output_prefix}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
