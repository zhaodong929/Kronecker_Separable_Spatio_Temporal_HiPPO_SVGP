#!/usr/bin/env python
"""Mechanism diagnostics for the standard Route-B synthetic setting."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_joint_ssgp_kron_experiments import normalize_eval_modes, run_all_requested


METHODS = ["mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-time", type=int, default=100)
    parser.add_argument("--num-space", type=int, default=6)
    parser.add_argument("--ms", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--mt-grid", nargs="*", type=int, default=[5, 8, 12, 16])
    parser.add_argument("--block-size-grid", nargs="*", type=int, default=[5, 10])
    parser.add_argument("--gp-signal-grid", nargs="*", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--coupling-grid", nargs="*", choices=["weak", "medium", "strong"], default=["medium", "strong"])
    parser.add_argument("--hyperfit-modes", nargs="*", choices=["ell_only", "noise_kernel"], default=["ell_only", "noise_kernel"])
    parser.add_argument("--noise-fit-grid-values", nargs="*", type=float, default=[0.05, 0.08, 0.10])
    parser.add_argument("--kernel-variance-fit-grid-values", nargs="*", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--outdir", type=Path, default=Path("results/experiments_routeB_standard_diagnostic_ablation"))
    return parser.parse_args()


def experiment_args(args: argparse.Namespace, *, mt: int, block_size: int, gp_signal: float, coupling: str, hyperfit: str) -> SimpleNamespace:
    ns = SimpleNamespace(
        dataset="synthetic",
        synthetic_regime="standard",
        num_seeds=args.num_seeds,
        num_time=args.num_time,
        num_space=args.num_space,
        block_size=block_size,
        mt=mt,
        ms=args.ms,
        noise=args.noise,
        model_noise=None,
        model_kernel_variance=None,
        ell_t=None,
        model_ell_t=None,
        model_ell_t_sweep=None,
        ell_t_fit_mode="initial_task_fullgp",
        initial_task_blocks=None,
        initial_task_fraction=0.2,
        time_normalization="expected_horizon",
        time_scale=1.0,
        ell_t_grid_source="time_scale",
        ell_t_grid_values=None,
        fit_noise_from_initial_task=hyperfit == "noise_kernel",
        noise_fit_grid_values=args.noise_fit_grid_values,
        fit_kernel_variance_from_initial_task=hyperfit == "noise_kernel",
        kernel_variance_fit_grid_values=args.kernel_variance_fit_grid_values,
        methods=METHODS,
        linear_dim=None,
        linear_signal_strength=1.0,
        gp_signal_strength=gp_signal,
        beta_u_correlation_design=coupling,
        noise_sweep=None,
        beta_u_correlation_sweep=None,
        eval_mode=None,
        eval_modes=["current", "seen_history"],
        missing_rate=0.5,
        include_mean_field_ablation=False,
        include_dense_reference_small=False,
        routeB=False,
        outdir=args.outdir,
        era5_root=Path("data/era5/processed_timeseries_4"),
        era5_task_dirs=None,
        era5_variable_index=0,
        era5_split="train",
        era5_shuffle_locations=False,
    )
    normalize_eval_modes(ns)
    return ns


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_metric(rows: list[dict[str, object]], metric: str) -> float:
    vals = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else float("nan")


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    config_keys = ["mt", "block_size", "gp_signal_strength", "beta_u_correlation_design", "hyperfit_mode"]
    configs = sorted({tuple(r[k] for k in config_keys) for r in rows})
    out = []
    for config in configs:
        cfg_rows = [r for r in rows if tuple(r[k] for k in config_keys) == config and r["eval_mode"] == "seen_history"]
        metrics_by_method = {}
        for method in METHODS:
            method_rows = [r for r in cfg_rows if r["method"] == method]
            metrics_by_method[method] = {
                metric: mean_metric(method_rows, metric)
                for metric in ["rmse", "nll", "coverage90", "rmse_forgetting", "nll_forgetting"]
            }
        mf = metrics_by_method["mean_field_ssgp_transfer"]
        rb = metrics_by_method["structured_joint_ssgp_transfer"]
        out.append(
            {
                "mt": config[0],
                "block_size": config[1],
                "gp_signal_strength": config[2],
                "beta_u_correlation_design": config[3],
                "hyperfit_mode": config[4],
                "routeB_rmse": rb["rmse"],
                "mean_field_rmse": mf["rmse"],
                "routeB_nll": rb["nll"],
                "mean_field_nll": mf["nll"],
                "routeB_cov90": rb["coverage90"],
                "mean_field_cov90": mf["coverage90"],
                "routeB_rmse_forgetting": rb["rmse_forgetting"],
                "mean_field_rmse_forgetting": mf["rmse_forgetting"],
                "routeB_nll_forgetting": rb["nll_forgetting"],
                "mean_field_nll_forgetting": mf["nll_forgetting"],
                "routeB_beats_rmse": rb["rmse"] < mf["rmse"],
                "routeB_beats_nll": rb["nll"] < mf["nll"],
                "routeB_beats_rmse_forgetting": rb["rmse_forgetting"] < mf["rmse_forgetting"],
                "routeB_beats_nll_forgetting": rb["nll_forgetting"] < mf["nll_forgetting"],
                "routeB_beats_rmse_and_nll": rb["rmse"] < mf["rmse"] and rb["nll"] < mf["nll"],
            }
        )
    return out


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, object]] = []
    for mt in args.mt_grid:
        for block_size in args.block_size_grid:
            for gp_signal in args.gp_signal_grid:
                for coupling in args.coupling_grid:
                    for hyperfit in args.hyperfit_modes:
                        cfg = experiment_args(
                            args,
                            mt=mt,
                            block_size=block_size,
                            gp_signal=gp_signal,
                            coupling=coupling,
                            hyperfit=hyperfit,
                        )
                        rows = run_all_requested(cfg)
                        for row in rows:
                            row["hyperfit_mode"] = hyperfit
                        all_rows.extend(rows)
                        print(
                            "finished",
                            f"mt={mt}",
                            f"block={block_size}",
                            f"gp={gp_signal}",
                            f"coupling={coupling}",
                            f"hyperfit={hyperfit}",
                            flush=True,
                        )
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "standard_diagnostic_metrics.csv", all_rows)
    summary = summarize(all_rows)
    write_csv(args.outdir / "standard_diagnostic_summary.csv", summary)
    report = {
        "num_rows": len(all_rows),
        "num_configs": len(summary),
        "num_seeds": args.num_seeds,
        "wins_rmse": int(sum(bool(r["routeB_beats_rmse"]) for r in summary)),
        "wins_nll": int(sum(bool(r["routeB_beats_nll"]) for r in summary)),
        "wins_rmse_and_nll": int(sum(bool(r["routeB_beats_rmse_and_nll"]) for r in summary)),
        "summary_path": str(args.outdir / "standard_diagnostic_summary.csv"),
    }
    (args.outdir / "standard_diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
