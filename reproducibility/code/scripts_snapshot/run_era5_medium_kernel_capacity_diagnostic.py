#!/usr/bin/env python3
"""Standalone medium-ERA5 kernel/capacity diagnostic.

This script runs a paper-ready diagnostic without changing the main ERA5
runner. The main protocol is inherited from the earlier kernel-family
diagnostic: task_1 calibration, task_2 online evaluation, structured-joint Route
B, seen-history evaluation, and location-99 pointwise predictions.

The only extension is a local spectral-mixture kernel patch used by this
diagnostic script. The main runner still supports its original kernel choices.
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
OUTDIR = PAPER_READY / "medium_kernel_capacity_diagnostic"
PLOT_DIR = OUTDIR / "plots"
RUNNER = ROOT / "scripts/run_hipposvgp_era5_routeb.py"
PYTHON = Path(sys.executable)

CAPACITIES = [(8, 64), (8, 128), (16, 128), (32, 256)]
KERNELS = [
    ("rbf", "RBF"),
    ("matern32", "Matern-3/2"),
    ("ard_rbf", "ARD-RBF"),
    ("spectral_mixture", "Spectral mixture"),
]

BASE_ARGS = [
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
    "--save-per-location-predictions",
    "--per-location-indices",
    "99",
    "--prediction-mode",
    "streaming_sylvester",
    "--prediction-chunk-size",
    "8192",
    "--hyperparam-fit-max-time",
    "30",
    "--hyperparam-fit-max-locations",
    "30",
    "--ell-t-grid",
    "0.0125",
    "0.025",
    "0.05",
    "0.075",
    "0.1",
    "0.15",
    "0.2",
    "--noise-grid",
    "0.05",
    "0.1",
    "0.2",
    "0.3",
    "0.5",
    "0.8",
    "--kernel-variance-grid",
    "0.25",
    "0.5",
    "1.0",
    "1.5",
]


def spectral_mixture_kernel(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    lengthscale: float | np.ndarray = 1.0,
    variance: float = 1.0,
    kernel_type: str = "spectral_mixture",
) -> np.ndarray:
    """Small fixed spectral-mixture kernel for the standalone diagnostic.

    The mixture contains a smooth component and an oscillatory component. The
    runner's full-GP grid still chooses the common temporal lengthscale,
    observation noise, and kernel variance.
    """

    del kernel_type
    x = np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    diff = x[:, None, :] - y[None, :, :]
    ls = np.maximum(np.asarray(lengthscale, dtype=float), 1e-12)
    if ls.ndim == 0:
        scaled = diff / float(ls)
    else:
        scaled = diff / ls.reshape((1, 1, -1))

    weights = np.asarray([0.65, 0.35], dtype=float)
    weights /= weights.sum()
    frequencies = np.asarray([0.0, 1.0], dtype=float)
    scales = np.asarray([1.0, 0.45], dtype=float)
    out = np.zeros(scaled.shape[:2], dtype=float)
    for weight, freq, scale in zip(weights, frequencies, scales):
        component_diff = scaled / scale
        envelope = np.exp(-0.5 * np.sum(component_diff * component_diff, axis=-1))
        carrier = np.prod(np.cos(2.0 * np.pi * freq * scaled), axis=-1)
        out += weight * envelope * carrier
    return float(variance) * out


def run_spectral_child(argv: list[str]) -> None:
    sys.path.insert(0, str(ROOT))
    import scripts.run_hipposvgp_era5_routeb as routeb
    import stvgp_kronecker.joint_ssgp_kron.synthetic as synthetic

    routeb.covariance_kernel = spectral_mixture_kernel
    synthetic.covariance_kernel = spectral_mixture_kernel
    sys.argv = [str(RUNNER)] + argv
    routeb.main()


def run_command(kernel: str, run_args: list[str]) -> None:
    if kernel == "spectral_mixture":
        cmd = [str(PYTHON), str(Path(__file__).resolve()), "--spectral-child", "--", *run_args]
    else:
        cmd = [str(PYTHON), str(RUNNER), *run_args]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def routeb_args_for_run(
    *,
    outdir: Path,
    kernel: str,
    mt: int,
    ms: int,
    fit_hyperparams: bool,
    selected: dict[str, float] | None,
) -> list[str]:
    kernel_arg = "rbf" if kernel == "spectral_mixture" else kernel
    args = [
        "--outdir",
        str(outdir),
        *BASE_ARGS,
        "--mt",
        str(mt),
        "--ms",
        str(ms),
        "--kernel-type",
        kernel_arg,
    ]
    if kernel == "ard_rbf":
        args.extend(["--spatial-ard-lengthscales", "0.25", "0.55"])
    if fit_hyperparams:
        args.extend(["--hyperparam-fit-mode", "initial_task_fullgp_grid"])
    else:
        if selected is None:
            raise ValueError("selected hyperparameters are required when fit_hyperparams=False")
        args.extend(
            [
                "--hyperparam-fit-mode",
                "none",
                "--ell-t-fit-mode",
                "none",
                "--model-ell-t",
                f"{selected['ell_t']:.12g}",
                "--routeb-noise",
                f"{np.sqrt(selected['sigma2']):.12g}",
                "--kernel-variance",
                f"{selected['kernel_variance']:.12g}",
            ]
        )
    return args


def read_summary(run_dir: Path) -> dict[str, float]:
    with (run_dir / "era5_routeb_summary.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("method") == "structured_joint" and row.get("eval_mode") == "seen_history":
            return {k: float(v) if _is_float(v) else v for k, v in row.items()}
    raise ValueError(f"No structured_joint seen_history row in {run_dir}")


def read_kernel_variance(run_dir: Path) -> float:
    report = json.loads((run_dir / "era5_routeb_report.json").read_text(encoding="utf-8"))
    return float(report["args"]["kernel_variance"])


def _is_float(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def run_all(force: bool) -> list[dict[str, Any]]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selected_by_kernel: dict[str, dict[str, float]] = {}

    for kernel, kernel_label in KERNELS:
        for idx, (mt, ms) in enumerate(CAPACITIES):
            run_name = f"{kernel}_medium_Mt{mt}_Ms{ms}"
            run_dir = OUTDIR / run_name
            needs_run = force or not (run_dir / "era5_routeb_summary.csv").exists()
            if needs_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                fit = idx == 0
                args = routeb_args_for_run(
                    outdir=run_dir,
                    kernel=kernel,
                    mt=mt,
                    ms=ms,
                    fit_hyperparams=fit,
                    selected=selected_by_kernel.get(kernel),
                )
                run_command(kernel, args)

            summary = read_summary(run_dir)
            if idx == 0:
                selected_by_kernel[kernel] = {
                    "ell_t": float(summary["selected_ell_t"]),
                    "sigma2": float(summary["avg_sigma2"]),
                    "kernel_variance": read_kernel_variance(run_dir),
                }
            selected = selected_by_kernel[kernel]
            rows.append(
                {
                    "kernel": kernel_label,
                    "kernel_id": kernel,
                    "phi_mode": "medium_era5",
                    "capacity": f"{mt}/{ms}",
                    "mt": mt,
                    "ms": ms,
                    "selected_ell_t": selected["ell_t"],
                    "selected_sigma": float(np.sqrt(selected["sigma2"])),
                    "selected_kernel_variance": selected["kernel_variance"],
                    "rmse": float(summary["rmse"]),
                    "nll": float(summary["nll"]),
                    "coverage90": float(summary["coverage90"]),
                    "ece": float(summary["ece"]),
                    "avg_nu_star": float(summary["avg_nu_star"]),
                    "avg_predictive_variance": float(summary["avg_predictive_variance"]),
                    "runtime_per_block": float(summary["runtime_per_block"]),
                    "run_dir": str(run_dir.relative_to(PAPER_READY)),
                }
            )

    write_csv(OUTDIR / "era5_medium_kernel_capacity_summary.csv", rows)
    plot_summary(rows)
    write_report_fragment(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, Any]]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    capacities = [f"{mt}/{ms}" for mt, ms in CAPACITIES]
    colors = {
        "RBF": "#4E79A7",
        "Matern-3/2": "#F28E2B",
        "ARD-RBF": "#59A14F",
        "Spectral mixture": "#B07AA1",
    }
    metrics = [("rmse", "RMSE"), ("nll", "NLL"), ("coverage90", "Cov90"), ("avg_predictive_variance", "Avg var")]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for kernel_label in [label for _, label in KERNELS]:
            vals = []
            for capacity in capacities:
                match = [r for r in rows if r["kernel"] == kernel_label and r["capacity"] == capacity]
                vals.append(float(match[0][metric]))
            ax.plot(capacities, vals, marker="o", linewidth=1.6, color=colors[kernel_label], label=kernel_label)
        if metric == "coverage90":
            ax.axhline(0.90, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(title)
        ax.grid(True, alpha=0.22)
        ax.set_xlabel(r"$M_t/M_s$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Medium-ERA5 kernel and inducing-point capacity diagnostic")
    fig.savefig(PLOT_DIR / "era5_medium_kernel_capacity_metrics.png", dpi=240)
    fig.savefig(PLOT_DIR / "era5_medium_kernel_capacity_metrics.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 3.4), constrained_layout=True)
    for kernel_label in [label for _, label in KERNELS]:
        vals = [float(r["runtime_per_block"]) for r in rows if r["kernel"] == kernel_label]
        ax.plot(capacities, vals, marker="o", linewidth=1.6, color=colors[kernel_label], label=kernel_label)
    ax.set_title("Runtime per block")
    ax.set_xlabel(r"$M_t/M_s$")
    ax.set_ylabel("seconds")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8)
    fig.savefig(PLOT_DIR / "era5_medium_kernel_capacity_runtime.png", dpi=240)
    plt.close(fig)


def write_report_fragment(rows: list[dict[str, Any]]) -> None:
    lines = [
        "Medium-ERA5 kernel/capacity diagnostic",
        "",
        "All rows use medium-ERA5 as the feature map, structured-joint Route B, task-1 calibration, task-2 online seen-history evaluation, and the same location-99 pointwise protocol as the previous kernel diagnostic. For each kernel family, the first capacity row fits ell_t, observation noise, and kernel variance by the task-1 full-GP MLL grid; the selected hyperparameters are then reused for the other inducing-point capacities.",
        "",
        "| Kernel | Capacity | RMSE | NLL | Cov90 | ECE | nu* | Avg var | Runtime |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {kernel} | {capacity} | {rmse:.4f} | {nll:.4f} | {coverage90:.4f} | {ece:.4f} | {avg_nu_star:.4f} | {avg_predictive_variance:.4f} | {runtime_per_block:.4f} |".format(
                **row
            )
        )
    (OUTDIR / "era5_medium_kernel_capacity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--spectral-child", action="store_true")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.spectral_child:
        runner_args = args.runner_args
        if runner_args and runner_args[0] == "--":
            runner_args = runner_args[1:]
        run_spectral_child(runner_args)
        return
    rows = run_all(force=args.force)
    print(json.dumps({"rows": rows, "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
