#!/usr/bin/env python
"""Run synthetic experiments for the new joint SSGP Kronecker implementation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.joint_ssgp_kron.kron_utils import dense_A_from_factors, inv_spd, solve_spd, symmetrize, vec_f
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    SyntheticDataset,
    design_matrix,
    iter_time_blocks,
    make_block_factors,
    make_spatial_projection,
    make_synthetic_dataset,
    temporal_inducing_for_block,
)


def _parse_lat_lon(path: Path) -> tuple[float, float]:
    match = re.search(r"lat_([-0-9.]+)_lon_([-0-9.]+)", path.stem.replace("_scaled", ""))
    if match is None:
        raise ValueError(f"Cannot parse coordinates from {path}")
    return float(match.group(1)), float(match.group(2))


def load_era5_processed_dataset(args: argparse.Namespace, seed: int) -> SyntheticDataset:
    root = Path(args.era5_root)
    if args.era5_task_dirs:
        task_dirs = [root / name for name in args.era5_task_dirs]
    else:
        task_dirs = sorted(root.glob("task_*"))[:1]
    files: list[Path] = []
    for task_dir in task_dirs:
        files.extend(sorted((task_dir / "sequences").glob("*_scaled.npz")))
    if not files:
        raise FileNotFoundError(f"No *_scaled.npz files found under {root}")
    rng = np.random.default_rng(seed)
    if args.era5_shuffle_locations:
        files = list(rng.permutation(files))
    files = files[: args.num_space]
    series = []
    coords = []
    time_sets = []
    split_names = ["data_train", "time_train"] if args.era5_split == "train" else ["data_test", "time_test"]
    for path in files:
        with np.load(path) as data:
            values = np.asarray(data[split_names[0]][args.era5_variable_index], dtype=float)
            times = np.asarray(data[split_names[1]], dtype=float)
        order = np.argsort(times)
        times = times[order]
        values = values[order]
        series.append((times, values))
        time_sets.append(set(times.tolist()))
        coords.append(_parse_lat_lon(path))
    common_times = sorted(set.intersection(*time_sets))
    if args.num_time:
        common_times = common_times[: args.num_time]
    if len(common_times) < args.block_size:
        raise ValueError("Not enough aligned ERA5 times for the requested block size")
    Y = np.zeros((len(files), len(common_times)))
    for i, (times, values) in enumerate(series):
        lookup = {float(t): float(v) for t, v in zip(times, values)}
        Y[i] = [lookup[float(t)] for t in common_times]
    coords_arr = np.asarray(coords, dtype=float)
    coords_arr = (coords_arr - coords_arr.mean(axis=0, keepdims=True)) / np.maximum(coords_arr.std(axis=0, keepdims=True), 1e-8)
    times_arr = np.asarray(common_times, dtype=float)
    times_scaled = (times_arr - times_arr.min()) / max(1e-12, times_arr.max() - times_arr.min())
    Phi = design_matrix(times_scaled, coords_arr[:, :1])
    return SyntheticDataset(
        times=times_scaled,
        spatial_coords=coords_arr,
        Y=Y,
        F=np.zeros_like(Y),
        Phi=Phi,
        beta_true=np.zeros(Phi.shape[1]),
        sigma2=args.noise**2,
    )


def gaussian_nll(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(var, 1e-10)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (y - mean) ** 2 / var))


def coverage(y: np.ndarray, mean: np.ndarray, var: np.ndarray, level: float) -> float:
    z = {0.5: 0.67448975, 0.9: 1.64485363, 0.95: 1.95996398}[level]
    half = z * np.sqrt(np.maximum(var, 1e-10))
    return float(np.mean((y >= mean - half) & (y <= mean + half)))


def block_variances(model: JointSSGPKronHiPPOSVGP, state, Phi: np.ndarray, T: np.ndarray, C: np.ndarray) -> np.ndarray:
    vals = []
    for t_idx in range(T.shape[0]):
        for s_idx in range(C.shape[0]):
            pred = model.predict(
                phi_star=Phi[t_idx * C.shape[0] + s_idx],
                t_proj_star=T[t_idx],
                c_proj_star=C[s_idx],
                state=state,
            )
            vals.append(pred.variance)
    return np.asarray(vals)


def make_dataset(args: argparse.Namespace, seed: int) -> SyntheticDataset:
    if args.dataset == "synthetic":
        return make_synthetic_dataset(num_time=args.num_time, num_space=args.num_space, noise=args.noise, seed=seed)
    if args.dataset == "era5":
        return load_era5_processed_dataset(args, seed)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def run_structured_method(args: argparse.Namespace, seed: int, method: str) -> list[dict[str, float | int | str]]:
    dataset = make_dataset(args, seed)
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, args.ms)
    model = JointSSGPKronHiPPOSVGP(
        Ks=Ks,
        C=C,
        sigma2=dataset.sigma2,
        beta_prior_mean=np.zeros(dataset.Phi.shape[1]),
        beta_prior_cov=10.0 * np.eye(dataset.Phi.shape[1]),
    )
    state = None
    old_z = None
    rows = []
    moving = method != "fixed_basis_exact"
    num_time = dataset.Y.shape[1]
    for block_id, block in enumerate(iter_time_blocks(num_time, args.block_size)):
        z_t = temporal_inducing_for_block(dataset.times, block, args.mt, moving=moving)
        factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=old_z)
        start = time.perf_counter()
        if method == "no_transfer":
            state = model.update_block_no_transfer(
                y_vec=factors.y_vec,
                Phi=factors.Phi,
                T_n=factors.T,
                Kt_new=factors.Kt,
                K_on_t=factors.K_on_t,
                state=state,
                inner_iters=2,
            )
        elif method == "projected_prior":
            state = model.update_block_projected_prior(
                y_vec=factors.y_vec,
                Phi=factors.Phi,
                T_n=factors.T,
                Kt_new=factors.Kt,
                K_on_t=factors.K_on_t,
                state=state,
                inner_iters=2,
            )
        elif method == "ssgp_transfer":
            state = model.update_block_ssgp_transfer(
                y_vec=factors.y_vec,
                Phi=factors.Phi,
                T_n=factors.T,
                Kt_new=factors.Kt,
                K_on_t=factors.K_on_t,
                state=state,
                inner_iters=2,
            )
        else:
            raise ValueError(f"Unsupported structured method: {method}")
        elapsed = time.perf_counter() - start
        mean = factors.Phi @ state.beta_mean + dense_A_from_factors(factors.T, C) @ vec_f(state.M_u)
        var = block_variances(model, state, factors.Phi, factors.T, C)
        rows.append(
            {
                "seed": seed,
                "method": method,
                "block": block_id,
                "rmse": float(np.sqrt(np.mean((factors.y_vec - mean) ** 2))),
                "mae": float(np.mean(np.abs(factors.y_vec - mean))),
                "nll": gaussian_nll(factors.y_vec, mean, var),
                "coverage50": coverage(factors.y_vec, mean, var, 0.5),
                "coverage90": coverage(factors.y_vec, mean, var, 0.9),
                "coverage95": coverage(factors.y_vec, mean, var, 0.95),
                "runtime_sec": elapsed,
            }
        )
        old_z = z_t
    return rows


def run_fixed_basis_exact(args: argparse.Namespace, seed: int) -> list[dict[str, float | int | str]]:
    dataset = make_dataset(args, seed)
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, args.ms)
    num_time = dataset.Y.shape[1]
    z_t = temporal_inducing_for_block(dataset.times, slice(0, num_time), args.mt, moving=False)
    beta_dim = dataset.Phi.shape[1]
    u_dim = args.ms * args.mt
    prior_beta_cov = 10.0 * np.eye(beta_dim)
    prior_u_cov = np.kron(
        make_block_factors(dataset, block=slice(0, min(args.block_size, num_time)), z_t=z_t, z_t_old=None).Kt,
        Ks,
    )
    Lambda = np.zeros((beta_dim + u_dim, beta_dim + u_dim))
    Lambda[:beta_dim, :beta_dim] = inv_spd(prior_beta_cov)
    Lambda[beta_dim:, beta_dim:] = inv_spd(prior_u_cov)
    h = np.zeros(beta_dim + u_dim)
    rows = []
    cov = inv_spd(Lambda)
    mean_state = np.zeros(beta_dim + u_dim)
    for block_id, block in enumerate(iter_time_blocks(num_time, args.block_size)):
        factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=z_t)
        A = dense_A_from_factors(factors.T, C)
        H = np.hstack([factors.Phi, A])
        start = time.perf_counter()
        Lambda = symmetrize(Lambda + H.T @ H / dataset.sigma2)
        h = h + H.T @ factors.y_vec / dataset.sigma2
        mean_state = solve_spd(Lambda, h)
        cov = inv_spd(Lambda)
        elapsed = time.perf_counter() - start
        mean = H @ mean_state
        var = np.maximum(dataset.sigma2 + np.sum((H @ cov) * H, axis=1), 1e-10)
        rows.append(
            {
                "seed": seed,
                "method": "fixed_basis_exact",
                "block": block_id,
                "rmse": float(np.sqrt(np.mean((factors.y_vec - mean) ** 2))),
                "mae": float(np.mean(np.abs(factors.y_vec - mean))),
                "nll": gaussian_nll(factors.y_vec, mean, var),
                "coverage50": coverage(factors.y_vec, mean, var, 0.5),
                "coverage90": coverage(factors.y_vec, mean, var, 0.9),
                "coverage95": coverage(factors.y_vec, mean, var, 0.95),
                "runtime_sec": elapsed,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(outdir: Path, rows: list[dict[str, float | int | str]]) -> None:
    for metric, filename in [
        ("rmse", "rmse_over_blocks.png"),
        ("nll", "nll_over_blocks.png"),
        ("coverage90", "coverage_plot.png"),
    ]:
        plt.figure(figsize=(7, 4))
        methods = sorted({str(row["method"]) for row in rows})
        for method in methods:
            blocks = sorted({int(row["block"]) for row in rows if row["method"] == method})
            means = []
            for block in blocks:
                vals = [float(row[metric]) for row in rows if row["method"] == method and int(row["block"]) == block]
                means.append(float(np.mean(vals)))
            plt.plot(blocks, means, marker="o", label=method)
        plt.xlabel("block")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=150)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "era5"], default="synthetic")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-time", type=int, default=40)
    parser.add_argument("--num-space", type=int, default=6)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--mt", type=int, default=5)
    parser.add_argument("--ms", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["no_transfer", "projected_prior", "ssgp_transfer"],
        choices=["no_transfer", "projected_prior", "ssgp_transfer", "fixed_basis_exact"],
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/experiments"))
    parser.add_argument("--era5-root", type=Path, default=Path("data/era5/processed_timeseries_4"))
    parser.add_argument("--era5-task-dirs", nargs="*", default=None)
    parser.add_argument("--era5-variable-index", type=int, default=0)
    parser.add_argument("--era5-split", choices=["train", "test"], default="train")
    parser.add_argument("--era5-shuffle-locations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.num_seeds):
        for method in args.methods:
            if method == "fixed_basis_exact":
                rows.extend(run_fixed_basis_exact(args, seed))
            else:
                rows.extend(run_structured_method(args, seed, method))
    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = "joint_ssgp_kron_synthetic" if args.dataset == "synthetic" else "joint_ssgp_kron_era5"
    metrics_path = args.outdir / f"{prefix}_metrics.csv"
    write_csv(metrics_path, rows)
    summary = {
        "num_rows": len(rows),
        "methods": args.methods,
        "mean_by_method": {
            method: {
                metric: float(np.mean([float(row[metric]) for row in rows if row["method"] == method]))
                for metric in ["rmse", "mae", "nll", "coverage90", "runtime_sec"]
            }
            for method in args.methods
        },
    }
    report_path = args.outdir / f"{prefix}_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(args.outdir, rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
