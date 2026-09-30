#!/usr/bin/env python3
"""Sanity-check whether Q=1, zero-mean spectral mixture can recover RBF."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stvgp_kronecker.joint_ssgp_kron.synthetic import covariance_kernel


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/xlag_lag_kernel_validation")
OUT = ROOT / "sm_q1_rbf_sanity_check"
RUNS = OUT / "runs"
PYTHON = Path(".venv/bin/python")

LAG = 10
VAL_SEED = 0
TEST_SEED = 1
ELL_T = 0.05
SPATIAL_ELL = 0.35


def write_params() -> Path:
    # In the NumPy covariance path, "scales" is sqrt(v) in
    # exp(-2*pi^2*tau^2*scale^2). Therefore RBF exp(-tau^2/(2 ell^2))
    # corresponds to scale = 1 / (2*pi*ell).
    #
    # The analytic temporal RFF path divides sampled base frequencies by
    # model_ell_t, so temporal scale=1.0 and mean=0 reproduce the RBF spectral
    # distribution up to finite-RFF Monte Carlo variation.
    params = {
        "schema": "sm_q1_zero_mean_rbf_sanity",
        "temporal_weights": [1.0],
        "temporal_means": [0.0],
        "temporal_scales": [1.0],
        "spatial_weights": [1.0],
        "spatial_means": [0.0],
        "spatial_scales": [1.0 / (2.0 * math.pi * SPATIAL_ELL)],
        "notes": {
            "temporal": "analytic RFF base frequency scale; current_frequencies divides by model_ell_t",
            "spatial": "exact covariance scale matching RBF lengthscale 0.35",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sm_q1_zero_mean_match_rbf_params.json"
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return path


def matrix_sanity(params_path: Path) -> dict[str, float]:
    params = json.loads(params_path.read_text())
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 2))
    k_rbf = covariance_kernel(x, lengthscale=SPATIAL_ELL, variance=1.0, kernel_type="rbf")
    k_sm = covariance_kernel(
        x,
        variance=1.0,
        kernel_type="spectral_mixture",
        spectral_mixture_weights=np.asarray(params["spatial_weights"], dtype=float),
        spectral_mixture_means=np.asarray(params["spatial_means"], dtype=float),
        spectral_mixture_scales=np.asarray(params["spatial_scales"], dtype=float),
    )
    diff = k_sm - k_rbf
    out = {
        "spatial_max_abs_diff": float(np.max(np.abs(diff))),
        "spatial_mean_abs_diff": float(np.mean(np.abs(diff))),
        "spatial_relative_fro_diff": float(np.linalg.norm(diff) / max(np.linalg.norm(k_rbf), 1e-12)),
        "spatial_rbf_diag_mean": float(np.mean(np.diag(k_rbf))),
        "spatial_sm_diag_mean": float(np.mean(np.diag(k_sm))),
    }
    (OUT / "matrix_sanity_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_routeb(name: str, kernel_type: str, split_seed: int, params_path: Path | None = None) -> Path:
    outdir = RUNS / name
    if (outdir / "era5_routeb_summary.csv").exists():
        return outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        "scripts/run_hipposvgp_era5_routeb.py",
        "--outdir",
        str(outdir),
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
        "--ohsvgp-heldout-eval",
        "--heldout-split-seeds",
        str(split_seed),
        "--seeds",
        "0",
        "--mt",
        "8",
        "--ms",
        "64",
        "--prediction-mode",
        "streaming_sylvester",
        "--prediction-chunk-size",
        "8192",
        "--hyperparam-fit-mode",
        "none",
        "--ell-t-fit-mode",
        "none",
        "--model-ell-t",
        str(ELL_T),
        "--routeb-noise",
        "0.1",
        "--kernel-variance",
        "1.0",
        "--spatial-lengthscale",
        str(SPATIAL_ELL),
        "--phi-mode",
        "medium_era5_xlag",
        "--xlag-length",
        str(LAG),
        "--kernel-type",
        kernel_type,
    ]
    if params_path is not None:
        cmd.extend(["--spectral-mixture-param-path", str(params_path)])
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (outdir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (outdir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (outdir / "walltime_seconds.txt").write_text(f"{time.perf_counter() - started:.6f}\n", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed; see {outdir / 'stdout.log'}")
    return outdir


def read_summary(path: Path) -> dict[str, float]:
    df = pd.read_csv(path / "era5_routeb_summary.csv")
    df = df[(df["method"] == "structured_joint") & (df["eval_mode"] == "seen_history")]
    row = df.iloc[0]
    return {
        "rmse": float(row["rmse"]),
        "nll": float(row["nll"]),
        "coverage90": float(row["coverage90"]),
        "ece": float(row["ece"]),
        "runtime": float((path / "walltime_seconds.txt").read_text().strip()),
    }


def make_report(results: pd.DataFrame, matrix_metrics: dict[str, float]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), constrained_layout=True)
    labels = results["setting"].tolist()
    colors = ["#4C8A4A" if "RBF" in x else "#8E5C87" for x in labels]
    axes[0].bar(np.arange(len(results)), results["test_rmse"], color=colors)
    axes[0].set_xticks(np.arange(len(results)))
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("test RMSE")
    axes[0].set_title("Route B sanity check")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(np.arange(len(results)), results["test_nll"], color=colors)
    axes[1].set_xticks(np.arange(len(results)))
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("test NLL/NLPD")
    axes[1].set_title("Probabilistic score")
    axes[1].grid(axis="y", alpha=0.25)
    for ax, metric in zip(axes, ["test_rmse", "test_nll"]):
        for i, value in enumerate(results[metric]):
            ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT / "sm_q1_vs_rbf_metrics.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "sm_q1_vs_rbf_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    result_lines = [
        "| setting | kernel_type | lag_length | val_rmse | val_nll | test_rmse | test_nll | runtime |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples():
        result_lines.append(
            f"| {row.setting} | {row.kernel_type} | {int(row.lag_length)} | "
            f"{row.val_rmse:.4f} | {row.val_nll:.4f} | {row.test_rmse:.4f} | "
            f"{row.test_nll:.4f} | {row.runtime:.1f} |"
        )
    lines = [
        "# SM-Q1 zero-mean RBF sanity check",
        "",
        "This check asks whether a one-component, zero-mean spectral-mixture kernel can recover the current RBF setting.",
        "",
        "## Matrix-level check",
        "",
        f"- spatial max absolute difference: {matrix_metrics['spatial_max_abs_diff']:.3e}",
        f"- spatial mean absolute difference: {matrix_metrics['spatial_mean_abs_diff']:.3e}",
        f"- spatial relative Frobenius difference: {matrix_metrics['spatial_relative_fro_diff']:.3e}",
        "",
        "The spatial covariance path matches RBF to numerical precision when scale=1/(2*pi*ell).",
        "",
        "## Route B result",
        "",
        *result_lines,
        "",
        "Interpretation: if SM-Q1 remains close to RBF, the basic SM implementation and parameter plumbing are likely sound. If full SM remains worse, the issue is more likely model/parameter suitability or finite-RFF/initialization, not a gross formula error.",
    ]
    (OUT / "sm_q1_rbf_sanity_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    params_path = write_params()
    matrix_metrics = matrix_sanity(params_path)
    rows = []
    for setting, kernel, params in [
        ("RBF", "rbf", None),
        ("SM-Q1 zero-mean", "spectral_mixture", params_path),
    ]:
        val_dir = run_routeb(f"{setting.lower().replace(' ', '_').replace('-', '_')}_val_seed0", kernel, VAL_SEED, params)
        test_dir = run_routeb(f"{setting.lower().replace(' ', '_').replace('-', '_')}_test_seed1", kernel, TEST_SEED, params)
        val = read_summary(val_dir)
        test = read_summary(test_dir)
        rows.append(
            {
                "setting": setting,
                "kernel_type": kernel,
                "lag_length": LAG,
                "val_rmse": val["rmse"],
                "val_nll": val["nll"],
                "test_rmse": test["rmse"],
                "test_nll": test["nll"],
                "runtime": val["runtime"] + test["runtime"],
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "sm_q1_rbf_sanity_check_results.csv", index=False)
    make_report(results, matrix_metrics)
    print(results.to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
