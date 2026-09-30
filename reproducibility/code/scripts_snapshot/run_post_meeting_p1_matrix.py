#!/usr/bin/env python3
"""P1 fair batch/online matrix for matched sparse and structured-joint models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import (
    augment_dataset_phi,
    beta_prior_cov_for_dataset,
    fixed_spatial_train_test_split,
    normalise_time_dataset,
    normalise_time_dataset_with_scale,
    routeb_dataset_from_era5,
    selected_locations_from_dataset,
    vectorized_predict_with_C,
)
from scripts.run_iclr_formal_stvgp_baseline import coverage90, ece_gaussian, gaussian_nll
from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5
from stvgp_kronecker.joint_ssgp_kron.kron_utils import inv_spd, solve_sylvester_precision, unvec_f, vec_f
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.ssgp_transfer import joint_likelihood_stats
from stvgp_kronecker.joint_ssgp_kron.structured_state import StructuredKronState
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    BlockFactors,
    covariance_kernel,
    make_block_factors_analytic_hippo,
    make_spatial_projection,
)


MIN_VAR = 1e-10


def spatial_covariance(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    lengthscale: float | np.ndarray,
    kernel_type: str,
) -> np.ndarray:
    if kernel_type != "matern32_separable":
        return covariance_kernel(x1, x2, lengthscale=lengthscale, kernel_type=kernel_type)
    scales = np.asarray(lengthscale, dtype=float).reshape(-1)
    if scales.size == 1:
        scales = np.repeat(scales, x1.shape[1])
    if scales.size != x1.shape[1]:
        raise ValueError("matern32_separable requires one lengthscale per spatial dimension")
    delta = np.abs(
        np.asarray(x1, dtype=float)[:, None, :] - np.asarray(x2, dtype=float)[None, :, :]
    ) / np.maximum(scales[None, None, :], 1e-12)
    scaled = np.sqrt(3.0) * delta
    return np.prod((1.0 + scaled) * np.exp(-scaled), axis=-1)


def fixed_spatial_projection(
    spatial_coords: np.ndarray,
    inducing_coords: np.ndarray,
    *,
    lengthscale: float | np.ndarray,
    kernel_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_s = np.asarray(inducing_coords, dtype=float)
    ks = spatial_covariance(
        z_s, z_s, lengthscale=lengthscale, kernel_type=kernel_type
    )
    ks = 0.5 * (ks + ks.T) + 1e-6 * np.eye(z_s.shape[0])
    kxz = spatial_covariance(
        np.asarray(spatial_coords, dtype=float),
        z_s,
        lengthscale=lengthscale,
        kernel_type=kernel_type,
    )
    c_all = np.linalg.solve(ks, kxz.T).T
    return z_s, ks, c_all


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def phi_subset(dataset: Any, time_indices: np.ndarray, spatial_indices: np.ndarray) -> np.ndarray:
    num_space = dataset.Y.shape[1]
    rows = np.concatenate(
        [int(t) * num_space + np.asarray(spatial_indices, dtype=int) for t in time_indices]
    )
    return dataset.Phi[rows]


def temporal_factors(
    routeb_dataset: Any,
    time_indices: np.ndarray,
    *,
    representation: str,
    mt: int,
    ell_t: float,
    kernel_variance: float,
    rff_sample_size: int,
    seed: int,
    kernel_type: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    times = np.asarray(routeb_dataset.times[time_indices], dtype=float)
    if representation == "analytic_hippo_rff":
        block = slice(int(time_indices[0]), int(time_indices[-1]) + 1)
        factors = make_block_factors_analytic_hippo(
            routeb_dataset,
            block=block,
            basis_block=block,
            old_basis_block=None,
            mt=mt,
            lengthscale=ell_t,
            kernel_variance=kernel_variance,
            rff_sample_size=rff_sample_size,
            seed=seed,
            moving=True,
            kernel_type=kernel_type,
        )
        return factors.T, factors.Kt, mt
    if representation == "inducing_points":
        z_t = np.linspace(routeb_dataset.times.min(), routeb_dataset.times.max(), mt)
        kt = covariance_kernel(z_t, lengthscale=ell_t, variance=kernel_variance, kernel_type=kernel_type)
        kt = 0.5 * (kt + kt.T) + 1e-6 * np.eye(mt)
        kfu = covariance_kernel(times, z_t, lengthscale=ell_t, variance=kernel_variance, kernel_type=kernel_type)
        return np.linalg.solve(kt, kfu.T).T, kt, mt
    if representation == "ordinary_rff":
        rng = np.random.default_rng(seed)
        frequencies = rng.normal(0.0, 1.0 / max(ell_t, 1e-12), size=mt)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=mt)
        features = np.sqrt(2.0 * kernel_variance / mt) * np.cos(
            times[:, None] * frequencies[None, :] + phases[None, :]
        )
        return features, np.eye(mt), mt
    if representation == "full_temporal_kernel":
        kt = covariance_kernel(times, lengthscale=ell_t, variance=kernel_variance, kernel_type=kernel_type)
        kt = 0.5 * (kt + kt.T) + 1e-6 * np.eye(times.size)
        return np.eye(times.size), kt, int(times.size)
    raise ValueError(f"Unknown temporal representation: {representation}")


def make_factors(
    dataset: Any,
    routeb_dataset: Any,
    time_indices: np.ndarray,
    spatial_indices: np.ndarray,
    *,
    representation: str,
    mt: int,
    ell_t: float,
    kernel_variance: float,
    rff_sample_size: int,
    seed: int,
    kernel_type: str,
) -> BlockFactors:
    t_mat, kt, _ = temporal_factors(
        routeb_dataset,
        time_indices,
        representation=representation,
        mt=mt,
        ell_t=ell_t,
        kernel_variance=kernel_variance,
        rff_sample_size=rff_sample_size,
        seed=seed,
        kernel_type=kernel_type,
    )
    y_matrix = dataset.Y[np.ix_(time_indices, spatial_indices)].T
    return BlockFactors(
        y_vec=vec_f(y_matrix),
        Phi=phi_subset(dataset, time_indices, spatial_indices),
        Y=y_matrix,
        T=t_mat,
        Kt=kt,
        K_on_t=None,
        block_slice=slice(int(time_indices[0]), int(time_indices[-1]) + 1),
        inducing_times=np.asarray(routeb_dataset.times[time_indices], dtype=float),
        temporal_backend=representation,
    )


def metric_row(
    *,
    architecture: str,
    protocol: str,
    representation: str,
    split_seed: int,
    block_size: int,
    block_id: int,
    stop: int,
    mt: int,
    effective_mt: int,
    ms: int,
    y: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    update_runtime: float,
    prediction_runtime: float,
    block_runtime: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    var = np.maximum(np.asarray(var, dtype=float), MIN_VAR)
    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    return {
        "architecture": architecture,
        "protocol": protocol,
        "temporal_representation": representation,
        "heldout_split_seed": split_seed,
        "block_size": block_size,
        "block_id": block_id,
        "num_seen_time": stop,
        "is_final_block": False,
        "mt": mt,
        "effective_mt": effective_mt,
        "ms": ms,
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "nll": gaussian_nll(y, mean, var),
        "coverage90": coverage90(y, mean, var),
        "ece": ece_gaussian(y, mean, var),
        "avg_std": float(np.mean(np.sqrt(var))),
        "update_runtime_sec": update_runtime,
        "prediction_runtime_sec": prediction_runtime,
        "block_incremental_runtime_sec": block_runtime,
        "num_test": int(y.size),
        **diagnostics,
    }


def final_prediction_rows(
    dataset: Any,
    test_idx: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    method: str,
) -> list[dict[str, Any]]:
    y_mat = np.asarray(y).reshape(dataset.Y.shape[0], test_idx.size)
    mean_mat = np.asarray(mean).reshape(dataset.Y.shape[0], test_idx.size)
    var_mat = np.asarray(var).reshape(dataset.Y.shape[0], test_idx.size)
    rows: list[dict[str, Any]] = []
    for time_index in range(dataset.Y.shape[0]):
        for local_index, location_index in enumerate(test_idx):
            rows.append(
                {
                    "method": method,
                    "time_index": time_index,
                    "time": float(dataset.times[time_index]),
                    "location_index": int(location_index),
                    "lat": float(dataset.coords[location_index, 0]),
                    "lon": float(dataset.coords[location_index, 1]),
                    "y_true": float(y_mat[time_index, local_index]),
                    "pred_mean": float(mean_mat[time_index, local_index]),
                    "pred_var": float(var_mat[time_index, local_index]),
                    "error": float(y_mat[time_index, local_index] - mean_mat[time_index, local_index]),
                }
            )
    return rows


def run_matched_sparse(
    dataset: Any,
    routeb_dataset: Any,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    protocol: str,
    args: argparse.Namespace,
    ks: np.ndarray,
    c_train: np.ndarray,
    c_test: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_times = np.arange(dataset.Y.shape[0], dtype=int)
    t_all, kt, effective_mt = temporal_factors(
        routeb_dataset,
        all_times,
        representation="inducing_points",
        mt=args.mt,
        ell_t=args.ell_t,
        kernel_variance=args.kernel_variance,
        rff_sample_size=args.temporal_rff_sample_size,
        seed=args.seed,
        kernel_type=args.kernel_type,
    )
    kt_inv = inv_spd(kt)
    ks_inv = inv_spd(ks)
    g_space = c_train.T @ c_train
    model = JointSSGPKronHiPPOSVGP(
        Ks=ks,
        C=c_train,
        sigma2=args.noise**2,
        beta_prior_mean=np.zeros(0),
        beta_prior_cov=np.zeros((0, 0)),
        prior_point_variance=args.kernel_variance,
    )
    p = dataset.Phi.shape[1]
    accumulated = {
        "phi_phi": np.zeros((p, p)),
        "phi_y": np.zeros(p),
        "R_beta_u": np.zeros((p, args.mt * args.ms)),
        "B_temporal": np.zeros((args.mt, args.mt)),
        "H_info": np.zeros((args.ms, args.mt)),
    }
    rows: list[dict[str, Any]] = []
    pointwise: list[dict[str, Any]] = []
    blocks = [
        slice(start, min(dataset.Y.shape[0], start + args.block_size))
        for start in range(0, dataset.Y.shape[0], args.block_size)
    ]
    if args.final_block_only:
        blocks = [slice(0, dataset.Y.shape[0])]

    for block_id, block in enumerate(blocks):
        block_started = time.perf_counter()
        stop = int(block.stop)
        if protocol == "online":
            block_times = np.arange(int(block.start), stop, dtype=int)
            phi = phi_subset(dataset, block_times, train_idx)
            y_matrix = dataset.Y[np.ix_(block_times, train_idx)].T
            y_vec = vec_f(y_matrix)
            stats = joint_likelihood_stats(
                y_vec, phi, t_all[block_times], c_train, args.noise**2
            )
            accumulated["phi_phi"] += phi.T @ phi
            accumulated["phi_y"] += phi.T @ y_vec
            accumulated["R_beta_u"] += stats["R_beta_u"]
            accumulated["B_temporal"] += stats["B_temporal"]
            accumulated["H_info"] += stats["H_info"]
            current = accumulated
        else:
            seen_times = np.arange(stop, dtype=int)
            phi = phi_subset(dataset, seen_times, train_idx)
            y_matrix = dataset.Y[np.ix_(seen_times, train_idx)].T
            y_vec = vec_f(y_matrix)
            stats = joint_likelihood_stats(
                y_vec, phi, t_all[seen_times], c_train, args.noise**2
            )
            current = {
                "phi_phi": phi.T @ phi,
                "phi_y": phi.T @ y_vec,
                "R_beta_u": stats["R_beta_u"],
                "B_temporal": stats["B_temporal"],
                "H_info": stats["H_info"],
            }

        beta = np.linalg.solve(
            current["phi_phi"] + args.ridge * np.eye(p), current["phi_y"]
        )
        h_residual = vec_f(current["H_info"]) - current["R_beta_u"].T @ beta
        h_matrix = unvec_f(h_residual, (args.ms, args.mt))
        m_u = solve_sylvester_precision(
            kt_inv,
            ks_inv,
            current["B_temporal"],
            g_space,
            h_matrix,
        )
        state = StructuredKronState(
            beta_mean=np.zeros(0),
            beta_cov=np.zeros((0, 0)),
            M_u=m_u,
            B_temporal=current["B_temporal"].copy(),
            H_info=h_matrix,
            Kt_current=kt,
            Ks=ks,
            G=g_space,
            sigma2=args.noise**2,
            metadata={"method": f"matched_sparse_{protocol}"},
        )
        update_runtime = time.perf_counter() - block_started

        seen_times = np.arange(stop, dtype=int)
        eval_factors = make_factors(
            dataset,
            routeb_dataset,
            seen_times,
            test_idx,
            representation="inducing_points",
            mt=args.mt,
            ell_t=args.ell_t,
            kernel_variance=args.kernel_variance,
            rff_sample_size=args.temporal_rff_sample_size,
            seed=args.seed,
            kernel_type=args.kernel_type,
        )
        eval_factors = BlockFactors(
            **{**eval_factors.__dict__, "Phi": np.zeros((eval_factors.y_vec.size, 0))}
        )
        prediction_started = time.perf_counter()
        residual_mean, var, diagnostics = vectorized_predict_with_C(
            model,
            state,
            eval_factors,
            c_test,
            prediction_mode="streaming_sylvester",
            chunk_size=args.prediction_chunk_size,
        )
        phi_test = phi_subset(dataset, seen_times, test_idx)
        mean = phi_test @ beta + residual_mean
        prediction_runtime = time.perf_counter() - prediction_started
        row = metric_row(
            architecture="matched_sparse_stvgp",
            protocol=protocol,
            representation="inducing_points",
            split_seed=args.split_seed,
            block_size=args.block_size,
            block_id=block_id,
            stop=stop,
            mt=args.mt,
            effective_mt=effective_mt,
            ms=args.ms,
            y=eval_factors.y_vec,
            mean=mean,
            var=var,
            update_runtime=update_runtime,
            prediction_runtime=prediction_runtime,
            block_runtime=time.perf_counter() - block_started,
            diagnostics=diagnostics,
        )
        rows.append(row)
        if stop == dataset.Y.shape[0]:
            pointwise = final_prediction_rows(
                dataset,
                test_idx,
                eval_factors.y_vec,
                mean,
                var,
                f"matched_sparse_stvgp_{protocol}",
            )

    rows[-1]["is_final_block"] = True
    return rows, pointwise


def run_structured_batch(
    dataset: Any,
    routeb_dataset: Any,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    args: argparse.Namespace,
    ks: np.ndarray,
    c_train: np.ndarray,
    c_test: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_args = SimpleNamespace(
        beta_prior_variance=args.beta_prior_variance,
        lag_beta_prior_variance=None,
    )
    model = JointSSGPKronHiPPOSVGP(
        Ks=ks,
        C=c_train,
        sigma2=args.noise**2,
        beta_prior_mean=np.zeros(dataset.Phi.shape[1]),
        beta_prior_cov=beta_prior_cov_for_dataset(dataset, args=model_args),
        prior_point_variance=args.kernel_variance,
    )
    rows: list[dict[str, Any]] = []
    pointwise: list[dict[str, Any]] = []
    blocks = [
        slice(start, min(dataset.Y.shape[0], start + args.block_size))
        for start in range(0, dataset.Y.shape[0], args.block_size)
    ]
    if args.final_block_only:
        blocks = [slice(0, dataset.Y.shape[0])]
    for block_id, block in enumerate(blocks):
        block_started = time.perf_counter()
        stop = int(block.stop)
        seen_times = np.arange(stop, dtype=int)
        train_factors = make_factors(
            dataset,
            routeb_dataset,
            seen_times,
            train_idx,
            representation=args.temporal_representation,
            mt=args.mt,
            ell_t=args.ell_t,
            kernel_variance=args.kernel_variance,
            rff_sample_size=args.temporal_rff_sample_size,
            seed=args.seed,
            kernel_type=args.kernel_type,
        )
        state = model.update_block_structured_joint_ssgp_transfer(
            y_vec=train_factors.y_vec,
            Phi=train_factors.Phi,
            T_n=train_factors.T,
            Kt_new=train_factors.Kt,
            state=None,
            K_on_t=None,
        )
        update_runtime = time.perf_counter() - block_started
        eval_factors = make_factors(
            dataset,
            routeb_dataset,
            seen_times,
            test_idx,
            representation=args.temporal_representation,
            mt=args.mt,
            ell_t=args.ell_t,
            kernel_variance=args.kernel_variance,
            rff_sample_size=args.temporal_rff_sample_size,
            seed=args.seed,
            kernel_type=args.kernel_type,
        )
        prediction_started = time.perf_counter()
        mean, var, diagnostics = vectorized_predict_with_C(
            model,
            state,
            eval_factors,
            c_test,
            prediction_mode="streaming_sylvester",
            chunk_size=args.prediction_chunk_size,
        )
        prediction_runtime = time.perf_counter() - prediction_started
        row = metric_row(
            architecture="structured_joint_hippo_stvgp",
            protocol="batch",
            representation=args.temporal_representation,
            split_seed=args.split_seed,
            block_size=args.block_size,
            block_id=block_id,
            stop=stop,
            mt=args.mt,
            effective_mt=train_factors.Kt.shape[0],
            ms=args.ms,
            y=eval_factors.y_vec,
            mean=mean,
            var=var,
            update_runtime=update_runtime,
            prediction_runtime=prediction_runtime,
            block_runtime=time.perf_counter() - block_started,
            diagnostics=diagnostics,
        )
        rows.append(row)
        if stop == dataset.Y.shape[0]:
            pointwise = final_prediction_rows(
                dataset,
                test_idx,
                eval_factors.y_vec,
                mean,
                var,
                "structured_joint_hippo_stvgp_batch",
            )
    rows[-1]["is_final_block"] = True
    return rows, pointwise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["matched_sparse_stvgp", "structured_joint"], required=True)
    parser.add_argument("--protocol", choices=["batch", "online"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--mt", type=int, default=8)
    parser.add_argument("--ms", type=int, default=64)
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument(
        "--phi-mode",
        choices=["direct_y", "medium_era5_xlag"],
        default="medium_era5_xlag",
    )
    parser.add_argument("--ell-t", type=float, default=0.05)
    parser.add_argument("--spatial-lengthscale", type=float, default=0.35)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--kernel-type", choices=["rbf", "matern32"], default="rbf")
    parser.add_argument(
        "--spatial-kernel-type",
        choices=["rbf", "matern32", "matern32_separable"],
        default=None,
    )
    parser.add_argument("--spatial-lengthscales", nargs="+", type=float, default=None)
    parser.add_argument("--spatial-inducing-coords-npz", default=None)
    parser.add_argument("--beta-prior-variance", type=float, default=10.0)
    parser.add_argument("--spatial-inducing-selection", choices=["linspace", "farthest", "kmeans"], default="linspace")
    parser.add_argument(
        "--temporal-representation",
        choices=["analytic_hippo_rff", "inducing_points", "ordinary_rff", "full_temporal_kernel"],
        default="analytic_hippo_rff",
    )
    parser.add_argument("--temporal-rff-sample-size", type=int, default=256)
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument("--final-block-only", action="store_true")
    args = parser.parse_args()
    if args.architecture == "structured_joint" and args.protocol != "batch":
        raise ValueError("Structured-joint online must use the main Route B runner")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    calibration_raw = load_hipposvgp_era5(
        args.root, tasks=("task_1",), variable_index=0, split="all"
    )
    selected_locations = selected_locations_from_dataset(calibration_raw)
    online_raw = load_hipposvgp_era5(
        args.root,
        tasks=("task_2",),
        variable_index=0,
        split="all",
        selected_locations=selected_locations,
    )
    _, calibration_scale = normalise_time_dataset(calibration_raw)
    dataset_base = normalise_time_dataset_with_scale(
        online_raw, scale=calibration_scale, source="calibration_task_span"
    )
    dataset = augment_dataset_phi(
        dataset_base, phi_mode=args.phi_mode, xlag_length=args.xlag_length
    )
    routeb_dataset = routeb_dataset_from_era5(
        dataset, sigma2=args.noise**2, args=args
    )
    train_idx, test_idx = fixed_spatial_train_test_split(
        dataset.Y.shape[1], test_fraction=args.test_fraction, seed=args.split_seed
    )
    spatial_kernel_type = args.spatial_kernel_type or args.kernel_type
    spatial_lengthscale = (
        np.asarray(args.spatial_lengthscales, dtype=float)
        if args.spatial_lengthscales is not None
        else args.spatial_lengthscale
    )
    if args.spatial_inducing_coords_npz is not None:
        shared = np.load(args.spatial_inducing_coords_npz)
        key = f"inducing_coords_ms{args.ms}"
        if key not in shared:
            raise KeyError(f"{args.spatial_inducing_coords_npz} does not contain {key}")
        z_s, ks, c_all = fixed_spatial_projection(
            routeb_dataset.spatial_coords,
            shared[key],
            lengthscale=spatial_lengthscale,
            kernel_type=spatial_kernel_type,
        )
    else:
        if spatial_kernel_type == "matern32_separable":
            raise ValueError("matern32_separable requires --spatial-inducing-coords-npz")
        z_s, ks, c_all = make_spatial_projection(
            routeb_dataset.spatial_coords,
            args.ms,
            lengthscale=spatial_lengthscale,
            kernel_type=spatial_kernel_type,
            inducing_selection=args.spatial_inducing_selection,
        )
    c_train = c_all[train_idx]
    c_test = c_all[test_idx]

    if args.architecture == "matched_sparse_stvgp":
        rows, pointwise = run_matched_sparse(
            dataset,
            routeb_dataset,
            train_idx,
            test_idx,
            protocol=args.protocol,
            args=args,
            ks=ks,
            c_train=c_train,
            c_test=c_test,
        )
    else:
        rows, pointwise = run_structured_batch(
            dataset,
            routeb_dataset,
            train_idx,
            test_idx,
            args=args,
            ks=ks,
            c_train=c_train,
            c_test=c_test,
        )

    write_csv(rows, outdir / "block_metrics.csv")
    write_csv(pointwise, outdir / "final_pointwise_predictions.csv")
    metadata = {
        "args": vars(args),
        "num_time": int(dataset.Y.shape[0]),
        "num_space": int(dataset.Y.shape[1]),
        "num_train_space": int(train_idx.size),
        "num_test_space": int(test_idx.size),
        "num_features": int(dataset.Phi.shape[1]),
        "spatial_inducing_locations": z_s.tolist(),
        "spatial_kernel_type": spatial_kernel_type,
        "spatial_lengthscale": np.asarray(spatial_lengthscale).tolist(),
        "spatial_inducing_coordinates_source": args.spatial_inducing_coords_npz,
        "final": rows[-1],
        "block_average": {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in ["rmse", "nll", "coverage90", "block_incremental_runtime_sec"]
        },
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
