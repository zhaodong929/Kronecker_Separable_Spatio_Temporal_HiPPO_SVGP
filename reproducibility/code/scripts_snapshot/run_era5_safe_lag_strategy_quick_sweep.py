#!/usr/bin/env python3
"""Quick safe-lag strategy sweep for medium-ERA5.

This diagnostic tests lightweight fixes for the teacher-forcing/free-running
lag mismatch:

- lag noise augmentation on training Phi,
- lag coefficient shrinkage via a smaller prior variance on lag beta columns,
- lag uncertainty propagation via predictive variance inflation,
- and combinations of strategies that individually improve both RMSE and NLL.

The sweep uses medium-ERA5, structured-joint Route B, safe-lag held-out split 0,
and Matern-3/2 because the safe-lag kernel diagnostic found it to be the most
stable residual kernel.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_READY = ROOT / "results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready"
OUTDIR = PAPER_READY / "safe_lag_strategy_quick_sweep"
RUNNER = ROOT / "scripts/run_hipposvgp_era5_routeb.py"
PYTHON = Path(sys.executable)

BASE_RUN_ARGS = [
    "--root",
    "data/era5/processed_timeseries_4",
    "--calibration-tasks",
    "task_1",
    "--online-tasks",
    "task_2",
    "--variable-index",
    "0",
    "--split",
    "all",
    "--block-size",
    "10",
    "--routeb-methods",
    "structured_joint",
    "--eval-modes",
    "seen_history",
    "--phi-mode",
    "medium_era5",
    "--ohsvgp-heldout-eval",
    "--heldout-split-seeds",
    "0",
    "--seeds",
    "0",
    "--prediction-mode",
    "streaming_sylvester",
    "--prediction-chunk-size",
    "8192",
    "--hyperparam-fit-mode",
    "none",
    "--ell-t-fit-mode",
    "none",
    "--model-ell-t",
    "0.2",
    "--routeb-noise",
    "0.05",
    "--kernel-type",
    "matern32",
    "--kernel-variance",
    "0.25",
    "--mt",
    "16",
    "--ms",
    "128",
]


SINGLE_STRATEGIES: list[dict[str, Any]] = [
    {"name": "baseline_matern32", "family": "baseline", "args": []},
    {"name": "lag_noise_0.05", "family": "lag_noise", "args": ["--lag-train-noise-std", "0.05"]},
    {"name": "lag_noise_0.10", "family": "lag_noise", "args": ["--lag-train-noise-std", "0.10"]},
    {"name": "lag_noise_0.20", "family": "lag_noise", "args": ["--lag-train-noise-std", "0.20"]},
    {"name": "lag_beta_prior_1.0", "family": "lag_beta_shrink", "args": ["--lag-beta-prior-variance", "1.0"]},
    {"name": "lag_beta_prior_0.3", "family": "lag_beta_shrink", "args": ["--lag-beta-prior-variance", "0.3"]},
    {"name": "lag_beta_prior_0.1", "family": "lag_beta_shrink", "args": ["--lag-beta-prior-variance", "0.1"]},
    {"name": "var_scale_1.5", "family": "lag_uncertainty", "args": ["--safe-lag-variance-scale", "1.5"]},
    {"name": "var_scale_2.0", "family": "lag_uncertainty", "args": ["--safe-lag-variance-scale", "2.0"]},
    {"name": "var_add_0.05", "family": "lag_uncertainty", "args": ["--safe-lag-variance-add", "0.05"]},
    {"name": "var_add_0.10", "family": "lag_uncertainty", "args": ["--safe-lag-variance-add", "0.10"]},
]


def read_summary(run_dir: Path) -> dict[str, float]:
    with (run_dir / "era5_ohsvgp_heldout_summary.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("method") == "structured_joint":
            return {k: float(v) if _is_float(v) else v for k, v in row.items()}
    raise ValueError(f"No structured_joint row in {run_dir}")


def _is_float(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def run_variant(variant: dict[str, Any], *, force: bool) -> dict[str, Any]:
    run_dir = OUTDIR / variant["name"]
    if force or not (run_dir / "era5_ohsvgp_heldout_summary.csv").exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(PYTHON), str(RUNNER), "--outdir", str(run_dir), *BASE_RUN_ARGS, *variant["args"]]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    summary = read_summary(run_dir)
    return {
        "name": variant["name"],
        "family": variant["family"],
        "args": " ".join(variant["args"]),
        "rmse": float(summary["rmse"]),
        "nll": float(summary["nll"]),
        "coverage90": float(summary["coverage90"]),
        "ece": float(summary["ece"]),
        "avg_predictive_variance": float(summary["avg_predictive_variance"]),
        "runtime_per_block": float(summary["runtime_per_block"]),
        "run_dir": str(run_dir.relative_to(PAPER_READY)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_combo_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(r for r in rows if r["name"] == "baseline_matern32")
    improved = [
        r
        for r in rows
        if r["name"] != "baseline_matern32"
        and float(r["rmse"]) < float(baseline["rmse"])
        and float(r["nll"]) < float(baseline["nll"])
    ]
    best_by_family: dict[str, dict[str, Any]] = {}
    for row in improved:
        old = best_by_family.get(row["family"])
        if old is None or (float(row["rmse"]) + float(row["nll"])) < (float(old["rmse"]) + float(old["nll"])):
            best_by_family[row["family"]] = row

    combos: list[dict[str, Any]] = []
    families = sorted(best_by_family)
    if len(families) >= 2:
        args: list[str] = []
        names: list[str] = []
        for fam in families:
            row = best_by_family[fam]
            args.extend(str(row["args"]).split())
            names.append(row["name"])
        combos.append({"name": "combo_all_improving", "family": "combo", "args": args, "components": "+".join(names)})
    if "lag_noise" in best_by_family and "lag_uncertainty" in best_by_family:
        args = str(best_by_family["lag_noise"]["args"]).split() + str(best_by_family["lag_uncertainty"]["args"]).split()
        combos.append({"name": "combo_noise_uncertainty", "family": "combo", "args": args, "components": f"{best_by_family['lag_noise']['name']}+{best_by_family['lag_uncertainty']['name']}"})
    if "lag_beta_shrink" in best_by_family and "lag_uncertainty" in best_by_family:
        args = str(best_by_family["lag_beta_shrink"]["args"]).split() + str(best_by_family["lag_uncertainty"]["args"]).split()
        combos.append({"name": "combo_shrink_uncertainty", "family": "combo", "args": args, "components": f"{best_by_family['lag_beta_shrink']['name']}+{best_by_family['lag_uncertainty']['name']}"})
    return combos


def plot_rows(rows: list[dict[str, Any]]) -> Path:
    plot_path = OUTDIR / "fig_safe_lag_strategy_quick_sweep.png"
    labels = [r["name"].replace("_", "\n") for r in rows]
    x = np.arange(len(rows))
    rmse = [float(r["rmse"]) for r in rows]
    nll = [float(r["nll"]) for r in rows]
    cov = [float(r["coverage90"]) for r in rows]
    colors = {
        "baseline": "#6B7C93",
        "lag_noise": "#4E79A7",
        "lag_beta_shrink": "#59A14F",
        "lag_uncertainty": "#F28E2B",
        "combo": "#B07AA1",
    }
    bar_colors = [colors.get(str(r["family"]), "#999999") for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    for ax, vals, title, ylabel in [
        (axes[0], rmse, "RMSE", "lower is better"),
        (axes[1], nll, "NLL/NLPD", "lower is better"),
        (axes[2], cov, "Coverage90", "nominal 0.90"),
    ]:
        ax.bar(x, vals, color=bar_colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.2)
    axes[2].axhline(0.9, color="black", linestyle="--", linewidth=0.8)
    fig.suptitle("Safe-lag medium-ERA5 strategy quick sweep (Matern-3/2, held-out split 0)", fontsize=12)
    fig.savefig(plot_path, dpi=240)
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    single_rows = [run_variant(v, force=args.force) for v in SINGLE_STRATEGIES]
    combos = build_combo_variants(single_rows)
    combo_rows = [run_variant(v, force=args.force) | {"components": v.get("components", "")} for v in combos]
    rows = single_rows + combo_rows
    write_csv(OUTDIR / "safe_lag_strategy_quick_sweep_summary.csv", rows)
    plot_path = plot_rows(rows)
    summary = {
        "scope": "quick safe-lag strategy sweep; split 0 diagnostic",
        "base_protocol": "medium_era5 structured_joint safe-lag heldout, Matern-3/2, Mt=16, Ms=128",
        "summary_csv": str(OUTDIR / "safe_lag_strategy_quick_sweep_summary.csv"),
        "figure": str(plot_path),
        "rows": rows,
    }
    (OUTDIR / "safe_lag_strategy_quick_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
