#!/usr/bin/env python3
"""Safe-lag medium-ERA5 kernel/capacity diagnostic.

All runs use the corrected held-out evaluation path:
``--ohsvgp-heldout-eval --phi-mode medium_era5``. Therefore target lags at test
locations are recursively filled from previous predictive means, not from
ground-truth test labels.

This is a diagnostic grid over kernel family and inducing capacity for held-out
split seed 0. It is intentionally separate from the main three-split medium/rich
ERA5 experiment.
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
OUTDIR = PAPER_READY / "medium_kernel_capacity_safe_lag_diagnostic"
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
    "--ohsvgp-heldout-eval",
    "--heldout-split-seeds",
    "0",
    "--seeds",
    "0",
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
    del kernel_type
    x = np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    diff = x[:, None, :] - y[None, :, :]
    ls = np.maximum(np.asarray(lengthscale, dtype=float), 1e-12)
    scaled = diff / float(ls) if ls.ndim == 0 else diff / ls.reshape((1, 1, -1))
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
    args = [
        "--outdir",
        str(outdir),
        *BASE_ARGS,
        "--mt",
        str(mt),
        "--ms",
        str(ms),
        "--kernel-type",
        kernel,
        "--temporal-backend",
        "analytic_hippo_rff",
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


def _is_float(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def read_heldout_summary(run_dir: Path) -> dict[str, float]:
    with (run_dir / "era5_ohsvgp_heldout_summary.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("method") == "structured_joint":
            return {k: float(v) if _is_float(v) else v for k, v in row.items()}
    raise ValueError(f"No structured_joint row in {run_dir}")


def read_routeb_summary(run_dir: Path) -> dict[str, float]:
    with (run_dir / "era5_routeb_summary.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("method") == "structured_joint":
            return {k: float(v) if _is_float(v) else v for k, v in row.items()}
    raise ValueError(f"No structured_joint Route B summary row in {run_dir}")


def read_kernel_variance(run_dir: Path) -> float:
    report = json.loads((run_dir / "era5_routeb_report.json").read_text(encoding="utf-8"))
    return float(report["args"]["kernel_variance"])


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PAPER_READY))
    except ValueError:
        return str(resolved)


def run_all(force: bool, outdir: Path = OUTDIR) -> list[dict[str, Any]]:
    global OUTDIR, PLOT_DIR
    OUTDIR = Path(outdir).resolve()
    PLOT_DIR = OUTDIR / "plots"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selected_by_kernel: dict[str, dict[str, float]] = {}
    for kernel, kernel_label in KERNELS:
        for idx, (mt, ms) in enumerate(CAPACITIES):
            run_name = f"{kernel}_medium_safe_lag_Mt{mt}_Ms{ms}"
            run_dir = OUTDIR / run_name
            needs_run = force or not (run_dir / "era5_ohsvgp_heldout_summary.csv").exists()
            if needs_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                args = routeb_args_for_run(
                    outdir=run_dir,
                    kernel=kernel,
                    mt=mt,
                    ms=ms,
                    fit_hyperparams=(idx == 0),
                    selected=selected_by_kernel.get(kernel),
                )
                run_command(kernel, args)
            summary = read_heldout_summary(run_dir)
            if idx == 0:
                routeb_summary = read_routeb_summary(run_dir)
                selected_by_kernel[kernel] = {
                    "ell_t": float(routeb_summary.get("selected_ell_t", 0.05)),
                    "sigma2": float(routeb_summary.get("avg_sigma2", 0.013977655531449635)),
                    "kernel_variance": read_kernel_variance(run_dir),
                }
            selected = selected_by_kernel[kernel]
            rows.append(
                {
                    "kernel": kernel_label,
                    "kernel_id": kernel,
                    "phi_mode": "medium_era5",
                    "protocol": "safe_lag_ohsvgp_heldout_split0",
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
                    "avg_predictive_variance": float(summary["avg_predictive_variance"]),
                    "runtime_per_block": float(summary["runtime_per_block"]),
                    "run_dir": _display_path(run_dir),
                }
            )
    write_csv(OUTDIR / "era5_medium_kernel_capacity_safe_lag_summary.csv", rows)
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
    fig.suptitle("Medium-ERA5 safe-lag held-out kernel/capacity diagnostic")
    fig.savefig(PLOT_DIR / "era5_medium_kernel_capacity_safe_lag_metrics.png", dpi=240)
    fig.savefig(PLOT_DIR / "era5_medium_kernel_capacity_safe_lag_metrics.pdf")
    plt.close(fig)


def write_report_fragment(rows: list[dict[str, Any]]) -> None:
    lines = [
        "Medium-ERA5 safe-lag kernel/capacity diagnostic",
        "",
        "All rows use medium-ERA5, structured-joint Route B, task-1 calibration, task-2 OHSVGP-style held-out seen-history evaluation, and held-out split seed 0. Target lags at test locations are recursively filled from predictive means, not from ground-truth test labels.",
        "",
        "| Kernel | Capacity | RMSE | NLL | Cov90 | ECE | Avg var | Runtime |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {kernel} | {capacity} | {rmse:.4f} | {nll:.4f} | {coverage90:.4f} | {ece:.4f} | {avg_predictive_variance:.4f} | {runtime_per_block:.4f} |".format(**row)
        )
    (OUTDIR / "era5_medium_kernel_capacity_safe_lag_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
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
    rows = run_all(force=args.force, outdir=args.outdir)
    print(json.dumps({"rows": rows, "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
