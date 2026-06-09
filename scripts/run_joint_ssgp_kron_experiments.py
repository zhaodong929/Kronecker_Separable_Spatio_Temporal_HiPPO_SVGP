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
from types import SimpleNamespace
from itertools import product

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
    rbf_kernel,
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
    Phi = design_matrix(times_arr, coords_arr[:, :1])
    return SyntheticDataset(
        times=times_arr,
        spatial_coords=coords_arr,
        Y=Y,
        F=np.zeros_like(Y),
        Phi=Phi,
        beta_true=np.zeros(Phi.shape[1]),
        sigma2=args.noise**2,
        gp_prior_variance=1.0,
    )


def apply_synthetic_routeB_controls(dataset: SyntheticDataset, args: argparse.Namespace, seed: int) -> SyntheticDataset:
    """Adjust synthetic signal strength and beta dimension for Route-B stress tests."""

    if args.dataset != "synthetic":
        return dataset
    linear_dim = args.linear_dim if args.linear_dim is not None else dataset.Phi.shape[1]
    base = dataset.Phi
    columns = [base[:, i % base.shape[1]] for i in range(linear_dim)]
    if args.beta_u_correlation_design in {"medium", "strong"} and linear_dim:
        # Smooth low-frequency covariates tend to overlap with the sparse GP
        # projection space, making beta-u posterior coupling visible.
        t = np.repeat(dataset.times, dataset.spatial_coords.shape[0])
        s = np.tile(dataset.spatial_coords[:, 0], dataset.times.shape[0])
        strong_cols = [
            np.sin(2.0 * np.pi * t),
            np.cos(2.0 * np.pi * t),
            s - np.mean(s),
            (s - np.mean(s)) * np.sin(2.0 * np.pi * t),
        ]
        if args.beta_u_correlation_design == "strong":
            columns = [strong_cols[i % len(strong_cols)] for i in range(linear_dim)]
        else:
            columns = [0.5 * columns[i] + 0.5 * strong_cols[i % len(strong_cols)] for i in range(linear_dim)]
    Phi = np.column_stack(columns) if columns else np.zeros((dataset.Phi.shape[0], 0))
    beta_true = np.linspace(0.7, -0.4, linear_dim) if linear_dim else np.zeros(0)
    mean = (Phi @ beta_true).reshape(dataset.Y.shape, order="F") if linear_dim else np.zeros_like(dataset.Y)
    rng = np.random.default_rng(seed + 10_000)
    noise = np.sqrt(dataset.sigma2) * rng.normal(size=dataset.Y.shape)
    Y = args.linear_signal_strength * mean + args.gp_signal_strength * dataset.F + noise
    return SyntheticDataset(
        times=dataset.times,
        spatial_coords=dataset.spatial_coords,
        Y=Y,
        F=args.gp_signal_strength * dataset.F,
        Phi=Phi,
        beta_true=args.linear_signal_strength * beta_true,
        sigma2=dataset.sigma2,
        gp_prior_variance=(args.gp_signal_strength**2) * dataset.gp_prior_variance,
        temporal_lengthscale=dataset.temporal_lengthscale,
        spatial_lengthscale=dataset.spatial_lengthscale,
    )


def gaussian_nll(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(var, 1e-10)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (y - mean) ** 2 / var))


def coverage(y: np.ndarray, mean: np.ndarray, var: np.ndarray, level: float) -> float:
    z = {0.5: 0.67448975, 0.9: 1.64485363, 0.95: 1.95996398}[level]
    half = z * np.sqrt(np.maximum(var, 1e-10))
    return float(np.mean((y >= mean - half) & (y <= mean + half)))


def block_variance_diagnostics(model: JointSSGPKronHiPPOSVGP, state, Phi: np.ndarray, T: np.ndarray, C: np.ndarray) -> dict[str, np.ndarray | float]:
    vals = []
    sigma2 = []
    nu = []
    u_terms = []
    beta_terms = []
    for t_idx in range(T.shape[0]):
        for s_idx in range(C.shape[0]):
            decomp = model.predictive_variance_decomposition(
                phi_star=Phi[t_idx * C.shape[0] + s_idx],
                t_proj_star=T[t_idx],
                c_proj_star=C[s_idx],
                state=state,
            )
            vals.append(decomp.total_variance)
            sigma2.append(decomp.sigma2)
            nu.append(decomp.nu_star)
            u_terms.append(decomp.u_posterior_term)
            beta_terms.append(decomp.beta_schur_term)
    var = np.asarray(vals)
    return {
        "variance": var,
        "avg_predictive_variance": float(np.mean(var)),
        "avg_interval_width90": float(np.mean(2.0 * 1.64485363 * np.sqrt(np.maximum(var, 1e-10)))),
        "avg_sigma2": float(np.mean(sigma2)),
        "avg_nu_star": float(np.mean(nu)),
        "avg_u_posterior_term": float(np.mean(u_terms)),
        "avg_beta_schur_term": float(np.mean(beta_terms)),
    }


def block_variances(model: JointSSGPKronHiPPOSVGP, state, Phi: np.ndarray, T: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.asarray(block_variance_diagnostics(model, state, Phi, T, C)["variance"])


def optional_float(value, default: float = float("nan")) -> float:
    return default if value is None else float(value)


def initial_task_block_count(args: argparse.Namespace, num_time: int) -> int:
    blocks = iter_time_blocks(num_time, args.block_size)
    if args.initial_task_blocks is not None:
        count = int(args.initial_task_blocks)
    else:
        if not 0.0 < float(args.initial_task_fraction) <= 1.0:
            raise ValueError("--initial-task-fraction must be in (0, 1]")
        initial_points = max(1, int(np.ceil(float(args.initial_task_fraction) * num_time)))
        count = int(np.ceil(initial_points / args.block_size))
    if args.ell_t_fit_mode == "initial_task_fullgp" and count < 1:
        raise ValueError("Initial-task full-GP fitting needs at least one initial block")
    if count > len(blocks):
        raise ValueError("Initial task exceeds the number of online blocks")
    return count


def apply_time_normalization(dataset: SyntheticDataset, args: argparse.Namespace) -> SyntheticDataset:
    raw_times = np.asarray(dataset.times, dtype=float)
    blocks = iter_time_blocks(dataset.Y.shape[1], args.block_size)
    count = initial_task_block_count(args, dataset.Y.shape[1])
    initial_stop = blocks[count - 1].stop or dataset.Y.shape[1]
    raw_span = float(raw_times[-1] - raw_times[0])
    initial_span = float(raw_times[initial_stop - 1] - raw_times[0])
    mode = args.time_normalization
    if mode == "none":
        scale = 1.0
    elif mode in {"custom", "expected_horizon"}:
        scale = float(args.time_scale)
        if scale <= 0.0:
            raise ValueError("--time-scale must be positive for the selected time normalization")
    elif mode == "initial_task":
        scale = initial_span
        if scale <= 0.0:
            raise ValueError("Initial-task time span must be positive for initial_task normalization")
    else:
        raise ValueError(f"Unsupported time normalization: {mode}")
    scaled_times = raw_times if mode == "none" else (raw_times - raw_times[0]) / scale
    args.resolved_time_scale = float(scale)
    args.initial_task_blocks_used = count
    args.initial_task_span = initial_span
    args.raw_time_span_if_available = raw_span
    return SyntheticDataset(
        times=scaled_times,
        spatial_coords=dataset.spatial_coords,
        Y=dataset.Y,
        F=dataset.F,
        Phi=dataset.Phi,
        beta_true=dataset.beta_true,
        sigma2=(float(args.model_noise) ** 2 if getattr(args, "model_noise", None) is not None else dataset.sigma2),
        gp_prior_variance=(
            float(args.model_kernel_variance)
            if getattr(args, "model_kernel_variance", None) is not None
            else dataset.gp_prior_variance
        ),
        temporal_lengthscale=dataset.temporal_lengthscale,
        spatial_lengthscale=dataset.spatial_lengthscale,
    )


def eval_blocks_for_mode(blocks: list[slice], block_id: int, mode: str) -> list[slice]:
    block = blocks[block_id]
    if mode == "current":
        return [block]
    if mode == "seen_history":
        return [slice(0, block.stop)]
    if mode == "future":
        return [] if block_id + 1 >= len(blocks) else [blocks[block_id + 1]]
    raise ValueError(f"Unsupported eval mode: {mode}")


def evaluate_state_on_factors(
    *,
    model: JointSSGPKronHiPPOSVGP,
    state,
    factors,
    C_eval: np.ndarray,
    method: str,
    seed: int,
    block_id: int,
    eval_mode: str,
    args: argparse.Namespace,
    elapsed: float,
    base_current_metrics: dict[int, dict[str, float]],
) -> dict[str, float | int | str]:
    mean = factors.Phi @ state.beta_mean + dense_A_from_factors(factors.T, C_eval) @ vec_f(state.M_u)
    diagnostics = block_variance_diagnostics(model, state, factors.Phi, factors.T, C_eval)
    var = np.asarray(diagnostics["variance"])
    rmse = float(np.sqrt(np.mean((factors.y_vec - mean) ** 2)))
    nll = gaussian_nll(factors.y_vec, mean, var)
    Lambda_cond = state.Lambda_beta_given_u
    row: dict[str, float | int | str] = {
        "seed": seed,
        "method": method,
        "block": block_id,
        "eval_mode": eval_mode,
        "regime_name": args.synthetic_regime,
        "mt": int(args.mt),
        "ms": int(args.ms),
        "block_size": int(args.block_size),
        "noise": args.noise,
        "gp_signal_strength": float(args.gp_signal_strength),
        "linear_signal_strength": float(args.linear_signal_strength),
        "ell_t": float(args.ell_t) if args.ell_t is not None else (0.8 if args.synthetic_regime == "long_memory" else 0.25),
        "model_ell_t": model_temporal_lengthscale(args),
        "data_ell_t": float(args.ell_t) if args.ell_t is not None else (0.8 if args.synthetic_regime == "long_memory" else 0.25),
        "ell_t_fit_mode": getattr(args, "ell_t_fit_mode", "fixed"),
        "initial_task_blocks": int(getattr(args, "initial_task_blocks_used", 0) or 0),
        "initial_task_fraction": float(args.initial_task_fraction),
        "time_normalization": args.time_normalization,
        "time_scale": float(getattr(args, "resolved_time_scale", args.time_scale)),
        "ell_t_grid_source": args.ell_t_grid_source,
        "model_noise": optional_float(getattr(args, "model_noise", None)),
        "model_kernel_variance": optional_float(getattr(args, "model_kernel_variance", None)),
        "fit_noise_from_initial_task": bool(getattr(args, "fit_noise_from_initial_task", False)),
        "fit_kernel_variance_from_initial_task": bool(getattr(args, "fit_kernel_variance_from_initial_task", False)),
        "fitted_ell_t": float(getattr(args, "fitted_ell_t", float("nan"))),
        "fitted_noise": float(getattr(args, "fitted_noise", float("nan"))),
        "fitted_kernel_variance": float(getattr(args, "fitted_kernel_variance", float("nan"))),
        "selected_ell_t": float(getattr(args, "selected_ell_t", model_temporal_lengthscale(args))),
        "selected_noise": float(getattr(args, "selected_noise", float("nan"))),
        "selected_kernel_variance": float(getattr(args, "selected_kernel_variance", float("nan"))),
        "candidate_ell_t": getattr(args, "candidate_ell_t", ""),
        "candidate_noise": getattr(args, "candidate_noise", ""),
        "candidate_kernel_variance": getattr(args, "candidate_kernel_variance", ""),
        "candidate_score": getattr(args, "candidate_score", ""),
        "selected_candidate_score": float(getattr(args, "selected_candidate_score", float("nan"))),
        "ell_t_grid": getattr(args, "ell_t_grid_used", ""),
        "initial_task_span": float(getattr(args, "initial_task_span", float("nan"))),
        "raw_time_span_if_available": float(getattr(args, "raw_time_span_if_available", float("nan"))),
        "beta_u_correlation_design": args.beta_u_correlation_design,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(factors.y_vec - mean))),
        "nll": nll,
        "coverage50": coverage(factors.y_vec, mean, var, 0.5),
        "coverage90": coverage(factors.y_vec, mean, var, 0.9),
        "coverage95": coverage(factors.y_vec, mean, var, 0.95),
        "avg_predictive_variance": diagnostics["avg_predictive_variance"],
        "avg_interval_width90": diagnostics["avg_interval_width90"],
        "avg_sigma2": diagnostics["avg_sigma2"],
        "avg_nu_star": diagnostics["avg_nu_star"],
        "avg_u_posterior_term": diagnostics["avg_u_posterior_term"],
        "avg_beta_schur_term": diagnostics["avg_beta_schur_term"],
        "R_uu_norm": float(np.linalg.norm(state.B_temporal) * np.linalg.norm(state.G)),
        "R_beta_u_norm": float(0.0 if state.R_beta_u is None else np.linalg.norm(state.R_beta_u)),
        "Lambda_beta_given_u_cond": float(np.nan if Lambda_cond is None or Lambda_cond.size == 0 else np.linalg.cond(Lambda_cond)),
        "runtime_per_block": elapsed,
        "runtime_sec": elapsed,
        "rmse_forgetting": float("nan"),
        "nll_forgetting": float("nan"),
    }
    if eval_mode == "current":
        base_current_metrics[block_id] = {"rmse": rmse, "nll": nll}
        row["rmse_forgetting"] = 0.0
        row["nll_forgetting"] = 0.0
    return row


def make_dataset(args: argparse.Namespace, seed: int) -> SyntheticDataset:
    if args.dataset == "synthetic":
        ell_t = 0.25
        ell_s = 0.35
        if args.synthetic_regime == "long_memory":
            ell_t = 0.8
        if args.ell_t is not None:
            ell_t = args.ell_t
        dataset = make_synthetic_dataset(
            num_time=args.num_time,
            num_space=args.num_space,
            noise=args.noise,
            seed=seed,
            ell_t=ell_t,
            ell_s=ell_s,
        )
        dataset = apply_synthetic_routeB_controls(dataset, args, seed)
        return apply_time_normalization(dataset, args)
    if args.dataset == "era5":
        return apply_time_normalization(load_era5_processed_dataset(args, seed), args)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def train_spatial_indices(args: argparse.Namespace, dataset: SyntheticDataset, seed: int, block_id: int, num_blocks: int) -> np.ndarray:
    ns = dataset.Y.shape[0]
    if args.synthetic_regime == "sparse_current":
        rng = np.random.default_rng(seed + 123)
        keep = max(1, int(round(ns * (1.0 - args.missing_rate))))
        return np.sort(rng.choice(ns, size=keep, replace=False))
    if args.synthetic_regime == "old_region_retention":
        # The structured Kronecker state assumes one fixed spatial Gram factor
        # G=C^T C throughout a run. A true left-to-right observation-mask shift
        # needs block-specific spatial factors and is therefore reported as a
        # follow-up rather than approximated by an invalid dynamic-G update.
        return np.arange(ns)
    return np.arange(ns)


def eval_spatial_indices(args: argparse.Namespace, dataset: SyntheticDataset, eval_mode: str) -> np.ndarray:
    if args.synthetic_regime == "old_region_retention" and eval_mode == "seen_history":
        coords = dataset.spatial_coords[:, 0]
        return np.flatnonzero(coords <= float(np.median(coords)))
    return np.arange(dataset.Y.shape[0])


def make_indexed_block_factors(
    dataset: SyntheticDataset,
    *,
    block: slice,
    z_t: np.ndarray,
    z_t_old: np.ndarray | None,
    spatial_idx: np.ndarray,
    lengthscale: float,
):
    full = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=z_t_old, lengthscale=lengthscale)
    spatial_idx = np.asarray(spatial_idx, dtype=int)
    Y = full.Y[spatial_idx, :]
    ns_full = dataset.Y.shape[0]
    start = block.start or 0
    stop = block.stop or dataset.Y.shape[1]
    row_idx: list[int] = []
    for t_idx in range(start, stop):
        row_idx.extend([t_idx * ns_full + int(s) for s in spatial_idx])
    return SimpleNamespace(
        y_vec=vec_f(Y),
        Phi=dataset.Phi[np.asarray(row_idx)],
        Y=Y,
        T=full.T,
        Kt=full.Kt,
        K_on_t=full.K_on_t,
        block_slice=block,
        inducing_times=z_t,
        spatial_idx=spatial_idx,
    )


def model_temporal_lengthscale(args: argparse.Namespace) -> float:
    return 0.25 if args.model_ell_t is None else float(args.model_ell_t)


def run_structured_method(args: argparse.Namespace, seed: int, method: str) -> list[dict[str, float | int | str]]:
    dataset = make_dataset(args, seed)
    _, Ks, C_full = make_spatial_projection(dataset.spatial_coords, args.ms)
    blocks = iter_time_blocks(dataset.Y.shape[1], args.block_size)
    first_idx = train_spatial_indices(args, dataset, seed, 0, len(blocks))
    C_train0 = C_full[first_idx]
    model = JointSSGPKronHiPPOSVGP(
        Ks=Ks,
        C=C_train0,
        sigma2=dataset.sigma2,
        beta_prior_mean=np.zeros(dataset.Phi.shape[1]),
        beta_prior_cov=10.0 * np.eye(dataset.Phi.shape[1]),
        prior_point_variance=dataset.gp_prior_variance,
    )
    state = None
    old_z = None
    rows = []
    base_current_metrics: dict[int, dict[str, float]] = {}
    moving = method != "fixed_basis_exact"
    num_time = dataset.Y.shape[1]
    model_ell_t = model_temporal_lengthscale(args)
    for block_id, block in enumerate(blocks):
        z_t = temporal_inducing_for_block(dataset.times, block, args.mt, moving=moving)
        train_idx = train_spatial_indices(args, dataset, seed, block_id, len(blocks))
        C_train = C_full[train_idx]
        model.C = C_train
        model.G = symmetrize(C_train.T @ C_train)
        factors = make_indexed_block_factors(dataset, block=block, z_t=z_t, z_t_old=old_z, spatial_idx=train_idx, lengthscale=model_ell_t)
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
        elif method == "mean_field_ssgp_transfer":
            state = model.update_block_mean_field_ssgp_transfer(
                y_vec=factors.y_vec,
                Phi=factors.Phi,
                T_n=factors.T,
                Kt_new=factors.Kt,
                K_on_t=factors.K_on_t,
                state=state,
                inner_iters=2,
            )
        elif method == "structured_joint_ssgp_transfer":
            state = model.update_block_structured_joint_ssgp_transfer(
                y_vec=factors.y_vec,
                Phi=factors.Phi,
                T_n=factors.T,
                Kt_new=factors.Kt,
                K_on_t=factors.K_on_t,
                state=state,
            )
        elif method == "structured_joint_no_transfer":
            state = model.update_block_structured_joint_no_transfer(
                y_vec=factors.y_vec,
                Phi=factors.Phi,
                T_n=factors.T,
                Kt_new=factors.Kt,
                K_on_t=factors.K_on_t,
                state=state,
            )
        else:
            raise ValueError(f"Unsupported structured method: {method}")
        elapsed = time.perf_counter() - start
        for eval_mode in args.eval_modes:
            aggregate_rows = []
            for eval_block in eval_blocks_for_mode(blocks, block_id, eval_mode):
                eval_idx = eval_spatial_indices(args, dataset, eval_mode)
                C_eval = C_full[eval_idx]
                eval_factors = make_indexed_block_factors(dataset, block=eval_block, z_t=z_t, z_t_old=None, spatial_idx=eval_idx, lengthscale=model_ell_t)
                aggregate_rows.append(
                    evaluate_state_on_factors(
                        model=model,
                        state=state,
                        factors=eval_factors,
                        C_eval=C_eval,
                        method=method,
                        seed=seed,
                        block_id=block_id,
                        eval_mode=eval_mode,
                        args=args,
                        elapsed=elapsed,
                        base_current_metrics=base_current_metrics,
                    )
                )
            if aggregate_rows:
                # For seen_history, metrics are computed on one concatenated slice
                # already. Compute forgetting separately over individual past blocks.
                row = aggregate_rows[0]
                if eval_mode == "seen_history" and block_id > 0:
                    rmse_deltas = []
                    nll_deltas = []
                    for past_id in range(block_id):
                        past_base = base_current_metrics.get(past_id)
                        if past_base is None:
                            continue
                        past_block = blocks[past_id]
                        eval_idx = eval_spatial_indices(args, dataset, eval_mode)
                        C_eval = C_full[eval_idx]
                        past_factors = make_indexed_block_factors(dataset, block=past_block, z_t=z_t, z_t_old=None, spatial_idx=eval_idx, lengthscale=model_ell_t)
                        past_row = evaluate_state_on_factors(
                            model=model,
                            state=state,
                            factors=past_factors,
                            C_eval=C_eval,
                            method=method,
                            seed=seed,
                            block_id=block_id,
                            eval_mode="seen_history",
                            args=args,
                            elapsed=elapsed,
                            base_current_metrics={},
                        )
                        rmse_deltas.append(float(past_row["rmse"]) - past_base["rmse"])
                        nll_deltas.append(float(past_row["nll"]) - past_base["nll"])
                    if rmse_deltas:
                        row["rmse_forgetting"] = float(np.mean(rmse_deltas))
                        row["nll_forgetting"] = float(np.mean(nll_deltas))
                    else:
                        row["forgetting_skip_reason"] = "missing_current_baseline"
                rows.append(row)
        old_z = z_t
    return rows


def run_fixed_basis_exact(args: argparse.Namespace, seed: int) -> list[dict[str, float | int | str]]:
    dataset = make_dataset(args, seed)
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, args.ms)
    num_time = dataset.Y.shape[1]
    model_ell_t = model_temporal_lengthscale(args)
    z_t = temporal_inducing_for_block(dataset.times, slice(0, num_time), args.mt, moving=False)
    beta_dim = dataset.Phi.shape[1]
    u_dim = args.ms * args.mt
    prior_beta_cov = 10.0 * np.eye(beta_dim)
    prior_u_cov = np.kron(
        make_block_factors(dataset, block=slice(0, min(args.block_size, num_time)), z_t=z_t, z_t_old=None, lengthscale=model_ell_t).Kt,
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
        factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=z_t, lengthscale=model_ell_t)
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
                "eval_mode": "current",
                "noise": args.noise,
                "ell_t": float(args.ell_t) if args.ell_t is not None else (0.8 if args.synthetic_regime == "long_memory" else 0.25),
                "model_ell_t": model_ell_t,
                "data_ell_t": float(args.ell_t) if args.ell_t is not None else (0.8 if args.synthetic_regime == "long_memory" else 0.25),
                "ell_t_fit_mode": getattr(args, "ell_t_fit_mode", "fixed"),
                "initial_task_blocks": int(getattr(args, "initial_task_blocks_used", 0) or 0),
                "initial_task_fraction": float(args.initial_task_fraction),
                "time_normalization": args.time_normalization,
                "time_scale": float(getattr(args, "resolved_time_scale", args.time_scale)),
                "ell_t_grid_source": args.ell_t_grid_source,
                "fitted_ell_t": float(getattr(args, "fitted_ell_t", float("nan"))),
                "selected_ell_t": float(getattr(args, "selected_ell_t", model_ell_t)),
                "candidate_ell_t": getattr(args, "candidate_ell_t", ""),
                "candidate_score": getattr(args, "candidate_score", ""),
                "selected_candidate_score": float(getattr(args, "selected_candidate_score", float("nan"))),
                "ell_t_grid": getattr(args, "ell_t_grid_used", ""),
                "initial_task_span": float(getattr(args, "initial_task_span", float("nan"))),
                "raw_time_span_if_available": float(getattr(args, "raw_time_span_if_available", float("nan"))),
                "beta_u_correlation_design": args.beta_u_correlation_design,
                "rmse": float(np.sqrt(np.mean((factors.y_vec - mean) ** 2))),
                "mae": float(np.mean(np.abs(factors.y_vec - mean))),
                "nll": gaussian_nll(factors.y_vec, mean, var),
                "coverage50": coverage(factors.y_vec, mean, var, 0.5),
                "coverage90": coverage(factors.y_vec, mean, var, 0.9),
                "coverage95": coverage(factors.y_vec, mean, var, 0.95),
                "avg_predictive_variance": float(np.mean(var)),
                "avg_interval_width90": float(np.mean(2.0 * 1.64485363 * np.sqrt(np.maximum(var, 1e-10)))),
                "runtime_sec": elapsed,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(outdir: Path, rows: list[dict[str, float | int | str]]) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
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
        plt.savefig(plot_dir / filename, dpi=150)
        plt.close()

    for metric, filename in [
        ("rmse_forgetting", "forgetting_rmse_over_blocks.png"),
        ("nll_forgetting", "forgetting_nll_over_blocks.png"),
    ]:
        vals_rows = [row for row in rows if row.get("eval_mode") == "seen_history" and row.get(metric, "") not in ("", "nan")]
        if not vals_rows:
            continue
        plt.figure(figsize=(7, 4))
        methods = sorted({str(row["method"]) for row in vals_rows})
        for method in methods:
            blocks = sorted({int(row["block"]) for row in vals_rows if row["method"] == method})
            means = []
            for block in blocks:
                vals = [float(row[metric]) for row in vals_rows if row["method"] == method and int(row["block"]) == block]
                means.append(float(np.mean(vals)))
            plt.plot(blocks, means, marker="o", label=method)
        plt.xlabel("online block")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=150)
        plt.close()

    for eval_mode in sorted({str(row.get("eval_mode", "current")) for row in rows}):
        mode_rows = [row for row in rows if row.get("eval_mode") == eval_mode]
        if not mode_rows or not all("noise" in row for row in mode_rows):
            continue
        for metric, filename in [
            ("rmse", f"{eval_mode}_rmse_vs_noise.png"),
            ("nll", f"{eval_mode}_nll_vs_noise.png"),
            ("coverage90", f"{eval_mode}_coverage_vs_noise.png"),
            ("avg_beta_schur_term", f"{eval_mode}_beta_schur_vs_noise.png"),
        ]:
            if metric not in mode_rows[0]:
                continue
            plt.figure(figsize=(7, 4))
            methods = sorted({str(row["method"]) for row in mode_rows})
            for method in methods:
                noises = sorted({float(row["noise"]) for row in mode_rows if row["method"] == method})
                means = []
                for noise in noises:
                    vals = [float(row[metric]) for row in mode_rows if row["method"] == method and float(row["noise"]) == noise]
                    means.append(float(np.mean(vals)))
                plt.plot(noises, means, marker="o", label=method)
            if metric == "coverage90":
                plt.axhline(0.9, color="black", linestyle="--", linewidth=1)
            plt.xlabel("noise")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / filename, dpi=150)
            plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "era5"], default="synthetic")
    parser.add_argument(
        "--synthetic-regime",
        choices=["standard", "long_memory", "sparse_current", "old_region_retention", "all"],
        default="standard",
    )
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-time", type=int, default=40)
    parser.add_argument("--num-space", type=int, default=6)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--mt", type=int, default=5)
    parser.add_argument("--ms", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--model-noise", type=float, default=None)
    parser.add_argument("--model-kernel-variance", type=float, default=None)
    parser.add_argument("--ell-t", type=float, default=None)
    parser.add_argument("--model-ell-t", type=float, default=None)
    parser.add_argument("--model-ell-t-sweep", nargs="*", type=float, default=None)
    parser.add_argument("--ell-t-fit-mode", choices=["none", "initial_task_fullgp"], default="initial_task_fullgp")
    parser.add_argument("--initial-task-blocks", type=int, default=None)
    parser.add_argument("--initial-task-fraction", type=float, default=0.2)
    parser.add_argument("--time-normalization", choices=["none", "expected_horizon", "custom", "initial_task"], default="expected_horizon")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--ell-t-grid-source", choices=["manual", "time_scale"], default="time_scale")
    parser.add_argument("--ell-t-grid-values", nargs="*", type=float, default=None)
    parser.add_argument("--fit-noise-from-initial-task", action="store_true")
    parser.add_argument("--noise-fit-grid-values", nargs="*", type=float, default=None)
    parser.add_argument("--fit-kernel-variance-from-initial-task", action="store_true")
    parser.add_argument("--kernel-variance-fit-grid-values", nargs="*", type=float, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["no_transfer", "projected_prior", "ssgp_transfer"],
        choices=[
            "no_transfer",
            "projected_prior",
            "ssgp_transfer",
            "mean_field_ssgp_transfer",
            "structured_joint_ssgp_transfer",
            "structured_joint_no_transfer",
            "fixed_basis_exact",
        ],
    )
    parser.add_argument("--linear-dim", type=int, default=None)
    parser.add_argument("--linear-signal-strength", type=float, default=1.0)
    parser.add_argument("--gp-signal-strength", type=float, default=1.0)
    parser.add_argument("--beta-u-correlation-design", choices=["weak", "medium", "strong"], default="strong")
    parser.add_argument("--noise-sweep", nargs="*", type=float, default=None)
    parser.add_argument("--beta-u-correlation-sweep", nargs="*", choices=["weak", "medium", "strong"], default=None)
    parser.add_argument("--eval-mode", choices=["current", "seen_history", "future", "all"], default=None)
    parser.add_argument("--eval-modes", nargs="+", choices=["current", "seen_history", "future"], default=None)
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--include-mean-field-ablation", action="store_true")
    parser.add_argument("--include-dense-reference-small", action="store_true")
    parser.add_argument("--routeB", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("results/experiments"))
    parser.add_argument("--era5-root", type=Path, default=Path("data/era5/processed_timeseries_4"))
    parser.add_argument("--era5-task-dirs", nargs="*", default=None)
    parser.add_argument("--era5-variable-index", type=int, default=0)
    parser.add_argument("--era5-split", choices=["train", "test"], default="train")
    parser.add_argument("--era5-shuffle-locations", action="store_true")
    return parser.parse_args()


def normalize_eval_modes(args: argparse.Namespace) -> None:
    if args.eval_mode == "all":
        args.eval_modes = ["current", "seen_history", "future"]
    elif args.eval_mode is not None:
        args.eval_modes = [args.eval_mode]
    elif args.eval_modes is None:
        args.eval_modes = ["current"]


def regime_list(args: argparse.Namespace) -> list[str]:
    if args.synthetic_regime == "all":
        return ["standard", "long_memory", "sparse_current", "old_region_retention"]
    return [args.synthetic_regime]


def ell_t_candidate_grid(args: argparse.Namespace) -> list[float]:
    if args.ell_t_grid_source == "manual":
        if not args.ell_t_grid_values:
            raise ValueError("--ell-t-grid-values is required when --ell-t-grid-source manual")
        return [float(x) for x in args.ell_t_grid_values]
    if args.time_normalization == "none":
        raise ValueError("Raw-time initial-task full-GP fitting requires --ell-t-grid-source manual")
    # Once time is normalized by its chosen scale, temporal lengthscales are in
    # normalized model units. This is the general time-scale rule for every dataset.
    return [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6]


def noise_candidate_grid(args: argparse.Namespace, dataset: SyntheticDataset) -> list[float]:
    if not getattr(args, "fit_noise_from_initial_task", False):
        return [float(args.model_noise) if getattr(args, "model_noise", None) is not None else float(np.sqrt(dataset.sigma2))]
    if args.noise_fit_grid_values:
        return [float(x) for x in args.noise_fit_grid_values]
    base = float(np.sqrt(dataset.sigma2))
    return sorted({max(1e-4, x * base) for x in [0.5, 0.75, 1.0, 1.25, 1.5]})


def kernel_variance_candidate_grid(args: argparse.Namespace, dataset: SyntheticDataset) -> list[float]:
    if not getattr(args, "fit_kernel_variance_from_initial_task", False):
        return [
            float(args.model_kernel_variance)
            if getattr(args, "model_kernel_variance", None) is not None
            else float(dataset.gp_prior_variance)
        ]
    if args.kernel_variance_fit_grid_values:
        return [float(x) for x in args.kernel_variance_fit_grid_values]
    base = float(dataset.gp_prior_variance)
    return sorted({max(1e-6, x * base) for x in [0.5, 0.75, 1.0, 1.5, 2.0]})


def full_gp_initial_task_marginal_nll(
    dataset: SyntheticDataset,
    stop_time: int,
    ell_t: float,
    *,
    noise: float | None = None,
    kernel_variance: float | None = None,
) -> float:
    """Exact GP marginal NLL on the initial task, integrating out beta.

    The score is independent of the online Route-B/mean-field update formulas:
    y ~ N(Phi m_beta, K_t(ell_t) kron K_s + Phi S_beta Phi^T + sigma2 I).
    """

    times = dataset.times[:stop_time]
    spatial = dataset.spatial_coords
    y = vec_f(dataset.Y[:, :stop_time])
    Phi = dataset.Phi[: y.shape[0]]
    beta_prior_mean = np.zeros(Phi.shape[1])
    beta_prior_cov = 10.0 * np.eye(Phi.shape[1])
    noise2 = (float(noise) ** 2) if noise is not None else dataset.sigma2
    variance = float(kernel_variance) if kernel_variance is not None else dataset.gp_prior_variance
    Kt = rbf_kernel(times, lengthscale=float(ell_t), variance=variance)
    Ks = rbf_kernel(spatial, lengthscale=dataset.spatial_lengthscale, variance=1.0)
    cov = np.kron(Kt, Ks) + Phi @ beta_prior_cov @ Phi.T + noise2 * np.eye(y.shape[0])
    cov = symmetrize(cov)
    resid = y - Phi @ beta_prior_mean
    jitter = 1e-8 * max(1.0, float(np.mean(np.diag(cov))))
    for _ in range(8):
        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        L = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, resid))
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    total_nll = 0.5 * (float(resid @ alpha) + logdet + y.shape[0] * np.log(2.0 * np.pi))
    return float(total_nll / y.shape[0])


def fit_model_hyperparameters_from_initial_task(
    args: argparse.Namespace, seed: int
) -> dict[str, object]:
    """Choose shared model hyperparameters by initial-task full-GP marginal NLL."""

    dataset = make_dataset(args, seed)
    blocks = iter_time_blocks(dataset.Y.shape[1], args.block_size)
    count = int(args.initial_task_blocks_used)
    if args.ell_t_fit_mode == "initial_task_fullgp":
        initial_stop = blocks[count - 1].stop or dataset.Y.shape[1]
        ell_grid = ell_t_candidate_grid(args)
        noise_grid = noise_candidate_grid(args, dataset)
        variance_grid = kernel_variance_candidate_grid(args, dataset)
        scored = []
        for ell_t, noise, variance in product(ell_grid, noise_grid, variance_grid):
            score = full_gp_initial_task_marginal_nll(
                dataset,
                initial_stop,
                float(ell_t),
                noise=float(noise),
                kernel_variance=float(variance),
            )
            scored.append((float(ell_t), float(noise), float(variance), float(score)))
        best_ell_t, best_noise, best_variance, best_score = min(scored, key=lambda item: item[3])
        return {
            "ell_t": best_ell_t,
            "noise": best_noise,
            "kernel_variance": best_variance,
            "score": best_score,
            "ell_grid": ell_grid,
            "noise_grid": noise_grid,
            "kernel_variance_grid": variance_grid,
            "candidate_score": scored,
        }
    raise ValueError(f"Unsupported ell_t fit mode for initial-task fitting: {args.ell_t_fit_mode}")


def run_all_requested(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    noises = args.noise_sweep if args.noise_sweep else [args.noise]
    couplings = args.beta_u_correlation_sweep if args.beta_u_correlation_sweep else [args.beta_u_correlation_design]
    model_ells = args.model_ell_t_sweep if args.model_ell_t_sweep else [args.model_ell_t]
    for regime in regime_list(args):
        for noise in noises:
            for coupling in couplings:
                for model_ell in model_ells:
                    combo_args = argparse.Namespace(**vars(args))
                    combo_args.synthetic_regime = regime
                    combo_args.noise = noise
                    combo_args.model_ell_t = model_ell
                    combo_args.beta_u_correlation_design = coupling
                    if regime == "long_memory" and args.num_time == 40:
                        combo_args.num_time = 100
                        combo_args.num_space = max(args.num_space, 10)
                    if regime in {"sparse_current", "old_region_retention"} and args.num_time == 40:
                        combo_args.num_time = 100
                        combo_args.num_space = max(args.num_space, 20)
                    for seed in range(combo_args.num_seeds):
                        combo_args.fitted_ell_t = float("nan")
                        combo_args.fitted_noise = float("nan")
                        combo_args.fitted_kernel_variance = float("nan")
                        combo_args.selected_ell_t = model_temporal_lengthscale(combo_args)
                        combo_args.selected_noise = float("nan")
                        combo_args.selected_kernel_variance = float("nan")
                        combo_args.selected_candidate_score = float("nan")
                        combo_args.candidate_ell_t = ""
                        combo_args.candidate_noise = ""
                        combo_args.candidate_kernel_variance = ""
                        combo_args.candidate_score = ""
                        combo_args.ell_t_grid_used = ""
                        # Build once to resolve time scale and initial-task metadata,
                        # including for fixed mismatch/oracle diagnostics.
                        make_dataset(combo_args, seed)
                        if combo_args.ell_t_fit_mode == "initial_task_fullgp":
                            fitted = fit_model_hyperparameters_from_initial_task(combo_args, seed)
                            combo_args.model_ell_t = float(fitted["ell_t"])
                            combo_args.model_noise = float(fitted["noise"])
                            combo_args.model_kernel_variance = float(fitted["kernel_variance"])
                            combo_args.fitted_ell_t = combo_args.model_ell_t
                            combo_args.fitted_noise = combo_args.model_noise
                            combo_args.fitted_kernel_variance = combo_args.model_kernel_variance
                            combo_args.selected_ell_t = combo_args.model_ell_t
                            combo_args.selected_noise = combo_args.model_noise
                            combo_args.selected_kernel_variance = combo_args.model_kernel_variance
                            combo_args.selected_candidate_score = float(fitted["score"])
                            combo_args.candidate_ell_t = " ".join(f"{x:g}" for x in fitted["ell_grid"])
                            combo_args.candidate_noise = " ".join(f"{x:g}" for x in fitted["noise_grid"])
                            combo_args.candidate_kernel_variance = " ".join(f"{x:g}" for x in fitted["kernel_variance_grid"])
                            combo_args.candidate_score = " ".join(
                                f"{ell:g}:{noise:g}:{variance:g}:{score:.8g}"
                                for ell, noise, variance, score in fitted["candidate_score"]
                            )
                            combo_args.ell_t_grid_used = " ".join(f"{x:g}" for x in fitted["ell_grid"])
                        for method in combo_args.methods:
                            if method == "fixed_basis_exact":
                                rows.extend(run_fixed_basis_exact(combo_args, seed))
                            else:
                                rows.extend(run_structured_method(combo_args, seed, method))
                        if combo_args.include_dense_reference_small and "fixed_basis_exact" not in combo_args.methods:
                            dense_rows = run_fixed_basis_exact(combo_args, seed)
                            for row in dense_rows:
                                row["method"] = "dense_reference_fixed_basis"
                            rows.extend(dense_rows)
    return rows


def mean_metric(rows: list[dict[str, float | int | str]], metric: str) -> float:
    vals = [float(row[metric]) for row in rows if metric in row and str(row[metric]) != "nan"]
    return float(np.mean(vals)) if vals else float("nan")


def seed_level_metric_summary(
    rows: list[dict[str, float | int | str]],
    group_keys: list[str],
    metrics: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, ...], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in group_keys)
        grouped.setdefault(key, []).append(row)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for key, group_rows in grouped.items():
        label = "|".join(f"{name}={value}" for name, value in zip(group_keys, key))
        out[label] = {}
        seeds = sorted({int(row["seed"]) for row in group_rows if "seed" in row})
        for metric in metrics:
            seed_vals = []
            for seed in seeds:
                vals = [
                    float(row[metric])
                    for row in group_rows
                    if int(row.get("seed", -1)) == seed and metric in row and str(row[metric]) != "nan"
                ]
                if vals:
                    seed_vals.append(float(np.mean(vals)))
            if seed_vals:
                arr = np.asarray(seed_vals, dtype=float)
                std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                out[label][metric] = {
                    "mean": float(np.mean(arr)),
                    "std": std,
                    "se": float(std / np.sqrt(arr.size)),
                    "num_seeds": int(arr.size),
                }
            else:
                out[label][metric] = {"mean": float("nan"), "std": float("nan"), "se": float("nan"), "num_seeds": 0}
    return out


def main() -> None:
    args = parse_args()
    normalize_eval_modes(args)
    if args.include_mean_field_ablation and "mean_field_ssgp_transfer" not in args.methods:
        args.methods = list(args.methods) + ["mean_field_ssgp_transfer"]
    rows = run_all_requested(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = "joint_ssgp_kron_synthetic" if args.dataset == "synthetic" else "joint_ssgp_kron_era5"
    metrics_path = args.outdir / f"{prefix}_metrics.csv"
    write_csv(metrics_path, rows)
    summary = {
        "num_rows": len(rows),
        "methods": args.methods,
        "routeB": bool(args.routeB),
        "dataset": args.dataset,
        "noise": args.noise,
        "model_noise": args.model_noise,
        "fit_noise_from_initial_task": args.fit_noise_from_initial_task,
        "noise_fit_grid_values": args.noise_fit_grid_values,
        "model_kernel_variance": args.model_kernel_variance,
        "fit_kernel_variance_from_initial_task": args.fit_kernel_variance_from_initial_task,
        "kernel_variance_fit_grid_values": args.kernel_variance_fit_grid_values,
        "ell_t": args.ell_t,
        "model_ell_t": args.model_ell_t,
        "model_ell_t_sweep": args.model_ell_t_sweep,
        "ell_t_fit_mode": args.ell_t_fit_mode,
        "initial_task_blocks": args.initial_task_blocks,
        "initial_task_fraction": args.initial_task_fraction,
        "time_normalization": args.time_normalization,
        "time_scale": args.time_scale,
        "ell_t_grid_source": args.ell_t_grid_source,
        "ell_t_grid_values": args.ell_t_grid_values,
        "linear_dim": args.linear_dim,
        "linear_signal_strength": args.linear_signal_strength,
        "gp_signal_strength": args.gp_signal_strength,
        "gp_prior_variance": float(make_dataset(args, 0).gp_prior_variance),
        "beta_u_correlation_design": args.beta_u_correlation_design,
        "synthetic_regime": args.synthetic_regime,
        "missing_rate": args.missing_rate,
        "noise_sweep": args.noise_sweep,
        "beta_u_correlation_sweep": args.beta_u_correlation_sweep,
        "eval_modes": args.eval_modes,
        "mean_by_method": {
            method: {
                metric: mean_metric([row for row in rows if row["method"] == method], metric)
                for metric in ["rmse", "mae", "nll", "coverage90", "avg_predictive_variance", "avg_interval_width90", "runtime_per_block"]
            }
            for method in sorted({str(row["method"]) for row in rows})
        },
        "mean_by_method_eval_mode": {
            f"{method}|{eval_mode}": {
                metric: mean_metric([row for row in rows if row["method"] == method and row.get("eval_mode") == eval_mode], metric)
                for metric in ["rmse", "mae", "nll", "coverage90", "avg_predictive_variance", "avg_interval_width90", "rmse_forgetting", "nll_forgetting"]
            }
            for method in sorted({str(row["method"]) for row in rows})
            for eval_mode in sorted({str(row.get("eval_mode", "current")) for row in rows})
        },
        "mean_by_noise_coupling_method_eval": {
            f"noise={noise}|model_ell_t={model_ell_t}|coupling={coupling}|{method}|{eval_mode}": {
                metric: mean_metric(
                    [
                        row
                        for row in rows
                        if float(row.get("noise", args.noise)) == float(noise)
                        and float(row.get("model_ell_t", model_temporal_lengthscale(args))) == float(model_ell_t)
                        and row.get("beta_u_correlation_design") == coupling
                        and row["method"] == method
                        and row.get("eval_mode") == eval_mode
                    ],
                    metric,
                )
                for metric in ["rmse", "nll", "coverage90", "avg_predictive_variance", "avg_interval_width90", "avg_sigma2", "avg_nu_star", "avg_u_posterior_term", "avg_beta_schur_term"]
            }
            for noise in sorted({float(row.get("noise", args.noise)) for row in rows})
            for model_ell_t in sorted({float(row.get("model_ell_t", model_temporal_lengthscale(args))) for row in rows})
            for coupling in sorted({str(row.get("beta_u_correlation_design", args.beta_u_correlation_design)) for row in rows})
            for method in sorted({str(row["method"]) for row in rows})
            for eval_mode in sorted({str(row.get("eval_mode", "current")) for row in rows})
        },
        "seed_level_summary_by_method_eval_mode": seed_level_metric_summary(
            rows,
            ["method", "eval_mode"],
            [
                "rmse",
                "mae",
                "nll",
                "coverage90",
                "avg_predictive_variance",
                "avg_interval_width90",
                "rmse_forgetting",
                "nll_forgetting",
                "runtime_per_block",
            ],
        ),
        "seed_level_summary_by_regime_method_eval_mode": seed_level_metric_summary(
            rows,
            ["regime_name", "method", "eval_mode"],
            [
                "rmse",
                "nll",
                "coverage90",
                "rmse_forgetting",
                "nll_forgetting",
                "avg_predictive_variance",
                "avg_interval_width90",
            ],
        ),
        "seed_level_summary_by_ablation": seed_level_metric_summary(
            rows,
            ["regime_name", "noise", "ell_t", "model_ell_t", "ell_t_fit_mode", "method", "eval_mode"],
            [
                "rmse",
                "nll",
                "coverage90",
                "rmse_forgetting",
                "nll_forgetting",
                "avg_predictive_variance",
                "avg_interval_width90",
                "avg_beta_schur_term",
            ],
        ),
    }
    report_path = args.outdir / f"{prefix}_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(args.outdir, rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
