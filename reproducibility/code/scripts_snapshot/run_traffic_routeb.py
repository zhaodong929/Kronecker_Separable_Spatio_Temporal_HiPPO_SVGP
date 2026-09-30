#!/usr/bin/env python3
"""Strict-online paired-spatial traffic experiments for Kron-STGP variants.

The runner has two non-interchangeable protocols:

* ``nowcast``: at t, absorb hidden labels from t-1, condition on y_V,t, then
  predict y_H,t before revealing it.
* ``forecast``: after the same legal update at t, predict y_H,t+h using known
  future calendar/spatial features and road-context features frozen at origin
  t.  It never reads y at t+h before the prediction is complete.

It is deliberately compact and dataset-agnostic; PEMS-BAY and METR-LA differ
only in their raw HDF path and fixed spatial-split manifest.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr, spherical_jn
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.data.traffic import (
    TrafficDataset,
    build_road_context_xlag_features,
    load_spatial_split,
    load_traffic_dataset,
)
from stvgp_kronecker.joint_ssgp_kron.kron_utils import solve_spd, stable_jitter, symmetrize, vec_f
from stvgp_kronecker.joint_ssgp_kron.synthetic import BlockFactors, make_analytic_temporal_builder, matern32_kernel, select_spatial_inducing_indices, temporal_spec_for_block
from stvgp_kronecker.joint_ssgp_kron.torch_backend import TorchJointSSGPKronHiPPOSVGP
from stvgp_kronecker.routeb_empirical_bayes import DTYPE, matern32_separable, robust_cholesky
from stvgp_kronecker.traffic_spatial_kernels import SPATIAL_KERNELS, fixed_spatial_factors
from stvgp_kronecker.traffic_protocol import StrictOnlineNowcastingGuard, TrafficForecastGuard


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _trim_cpu_allocator() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def gaussian_metrics(y_true: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    mu = np.asarray(mean, dtype=float).reshape(-1)
    var = np.maximum(np.asarray(variance, dtype=float).reshape(-1), 1e-10)
    std = np.sqrt(var)
    z = (y - mu) / std
    crps = std * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi))
    levels = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    z_scores = np.asarray([0.0627067779, 0.189118426, 0.318639364, 0.45376219, 0.597760126, 0.755415026, 0.934589291, 1.15034938, 1.43953147, 1.95996398])
    coverage = np.asarray([np.mean(np.abs(y - mu) <= score * std) for score in z_scores])
    return {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "crps": float(np.mean(np.maximum(crps, 0.0))),
        "gaussian_nlpd": float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + z**2))),
        "ece": float(np.mean(np.abs(coverage - levels))),
        "coverage90": float(np.mean(np.abs(y - mu) <= 1.644853627 * std)),
        "mean_predictive_std": float(np.mean(std)),
    }


def _state_diagnostics(state, kt: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    """Compact numerical diagnostics for changing-coordinate stability audits."""

    def _norm(value) -> float:
        if value is None:
            return 0.0
        return float(torch.linalg.vector_norm(value).detach().cpu())

    beta_precision = torch.linalg.pinv(state.beta_cov)
    beta_natural = beta_precision @ state.beta_mean
    predictive_variance = np.asarray(variance, dtype=float).reshape(-1)
    return {
        "kt_condition_number": float(np.linalg.cond(np.asarray(kt, dtype=float))),
        "beta_cov_condition_number": float(torch.linalg.cond(state.beta_cov).detach().cpu()),
        "beta_mean_norm": _norm(state.beta_mean),
        "beta_natural_norm": _norm(beta_natural),
        "gp_mean_state_norm": _norm(state.M_u),
        "gp_information_norm": _norm(state.H_info),
        "temporal_precision_stat_norm": _norm(state.B_temporal),
        "predictive_variance_min": float(np.min(predictive_variance)),
        "predictive_variance_median": float(np.median(predictive_variance)),
        "predictive_variance_max": float(np.max(predictive_variance)),
    }


def _spatial_factors(dataset: TrafficDataset, visible_indices: np.ndarray, *, ms: int, lengthscale: float | Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    visible_coordinates = dataset.coordinates_standardised[visible_indices]
    local = select_spatial_inducing_indices(visible_coordinates, min(int(ms), visible_indices.size), method="farthest")
    inducing = visible_coordinates[local]
    all_coordinates = torch.as_tensor(dataset.coordinates_standardised, dtype=DTYPE)
    inducing_tensor = torch.as_tensor(inducing, dtype=DTYPE)
    lengthscales = torch.as_tensor(lengthscale, dtype=DTYPE).reshape(-1)
    if lengthscales.numel() == 1:
        lengthscales = lengthscales.repeat(2)
    if tuple(lengthscales.shape) != (2,) or torch.any(lengthscales <= 0.0):
        raise ValueError("spatial lengthscale must be one positive scalar or two positive lat/lon values")
    with torch.no_grad():
        ks = matern32_separable(inducing_tensor, inducing_tensor, lengthscales)
        ks = 0.5 * (ks + ks.T) + 1e-7 * torch.eye(ks.shape[0], dtype=DTYPE)
        kxs = matern32_separable(all_coordinates, inducing_tensor, lengthscales)
        chol = robust_cholesky(ks)
        c_all = torch.cholesky_solve(kxs.T, chol).T
    return np.asarray(ks), np.asarray(c_all), inducing, visible_indices[local]


def _ordinary_temporal_factors(times: np.ndarray, query: np.ndarray, supports: np.ndarray, *, lengthscale: float, variance: float) -> tuple[np.ndarray, np.ndarray, None, None]:
    kt = matern32_kernel(supports, lengthscale=lengthscale, variance=variance) + 1e-6 * np.eye(supports.size)
    kfu = matern32_kernel(times[np.asarray(query, dtype=int)], supports, lengthscale=lengthscale, variance=variance)
    return solve_spd(kt, kfu.T, jitter=1e-12).T, kt, None, None


def _inverse_distance_weights(
    target_coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
    *,
    neighbors: int,
    power: float,
) -> np.ndarray:
    """Return row-normalised KNN inverse-distance interpolation weights."""

    target = np.asarray(target_coordinates, dtype=float)
    reference = np.asarray(reference_coordinates, dtype=float)
    if target.ndim != 2 or reference.ndim != 2 or target.shape[1] != reference.shape[1]:
        raise ValueError("target and reference coordinates must be compatible matrices")
    if neighbors <= 0 or power <= 0.0:
        raise ValueError("neighbors and power must be positive")
    count = min(int(neighbors), reference.shape[0])
    distances = np.linalg.norm(target[:, None, :] - reference[None, :, :], axis=-1)
    nearest = np.argpartition(distances, kth=count - 1, axis=1)[:, :count]
    weights = np.zeros_like(distances)
    rows = np.arange(target.shape[0])[:, None]
    selected_distances = np.maximum(distances[rows, nearest], 1e-8)
    selected_weights = selected_distances ** (-float(power))
    selected_weights /= selected_weights.sum(axis=1, keepdims=True)
    weights[rows, nearest] = selected_weights
    return weights


def _analytic_temporal_factors(
    *,
    builder,
    times: np.ndarray,
    query: np.ndarray,
    basis: slice,
    old_basis: slice | None,
    old_temporal_basis: np.ndarray | torch.Tensor | None,
    evaluator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    spec = temporal_spec_for_block(times, basis, moving=True)
    old_spec = None if old_basis is None else temporal_spec_for_block(times, old_basis, moving=True)
    if evaluator == "scipy_frozen":
        new_basis = _scipy_analytic_temporal_basis(builder, spec)
        old_basis_array = None if old_temporal_basis is None else np.asarray(old_temporal_basis, dtype=np.float64)
        if old_basis_array is None and old_spec is not None:
            old_basis_array = _scipy_analytic_temporal_basis(builder, old_spec)
        with torch.no_grad():
            basis_tensor = torch.as_tensor(new_basis, dtype=torch.float64)
            kt = builder.add_jitter(builder.variance * (basis_tensor.T @ basis_tensor)).detach().cpu().numpy()
            query_times = torch.as_tensor(times[np.asarray(query, dtype=int)], dtype=torch.float64)
            kfu = (builder.variance * (builder.compute_feature_matrix(query_times) @ basis_tensor)).detach().cpu().numpy()
            variance = float(builder.variance.detach().cpu())
        k_on = None if old_basis_array is None else variance * (old_basis_array.T @ new_basis)
        return solve_spd(kt, kfu.T, jitter=1e-12).T, kt, k_on, new_basis
    if evaluator != "torch_autograd":
        raise ValueError(f"Unsupported temporal evaluator: {evaluator}")
    with torch.no_grad():
        kfu, kt_raw, k_on, new_basis = builder.compute_block_covariances_with_basis(
            torch.as_tensor(times[np.asarray(query, dtype=int)], dtype=torch.float64),
            spec,
            old_spec,
            old_basis=old_temporal_basis,
        )
        kt = builder.add_jitter(kt_raw).detach().cpu().numpy()
        kfu_array = kfu.detach().cpu().numpy()
        k_on_array = None if k_on is None else k_on.detach().cpu().numpy()
        new_basis_array = new_basis.detach().cpu().numpy()
    return solve_spd(kt, kfu_array.T, jitter=1e-12).T, kt, k_on_array, new_basis_array


def _scipy_analytic_temporal_basis(builder, horizon) -> np.ndarray:
    """Fast frozen-parameter evaluation of the builder's analytic HiPPO basis."""

    config = builder.config
    frequencies = builder.current_frequencies().detach().cpu().numpy()
    step_size = (float(horizon.end) - float(horizon.start)) / max(int(horizon.num_discrete_steps), 1)
    effective_frequencies = frequencies * step_size
    time_index = int(horizon.prev_discrete_steps) + int(horizon.num_discrete_steps)
    kappa = 0.5 * effective_frequencies * time_index
    if config.phase_origin_mode != "global_start" or config.globalstart_wt_mode == "none":
        phase_origin = np.zeros_like(frequencies)
    else:
        origin = float(horizon.phase_origin if horizon.phase_origin is not None else horizon.start)
        phase_origin = (effective_frequencies if config.globalstart_wt_mode == "w_eff" else frequencies) * origin
    levels = np.arange(config.inducing_size, dtype=float)[:, None]
    bessel = spherical_jn(levels, kappa)
    prefactor = np.sqrt(2.0 * levels + 1.0)
    phase = phase_origin + kappa + levels * np.pi / 2.0
    scale = (1.0 / config.rff_sample_size) ** 0.5
    return scale * np.concatenate(
        [prefactor * bessel * np.sin(phase), prefactor * bessel * np.cos(phase)],
        axis=1,
    ).T


def _analytic_temporal_projection(
    *,
    builder,
    times: np.ndarray,
    query: np.ndarray,
    temporal_basis: np.ndarray,
    kt: np.ndarray,
) -> np.ndarray:
    """Project one query row onto an already-built temporal basis.

    A delayed label and the current visible observations share the current
    basis.  Reusing that basis avoids a duplicate spherical-Bessel recurrence;
    the query feature calculation and SPD solve remain the same as the full
    analytic builder path.
    """

    with torch.no_grad():
        query_times = torch.as_tensor(times[np.asarray(query, dtype=int)], dtype=torch.float64)
        basis = torch.as_tensor(temporal_basis, dtype=torch.float64)
        kfu = (builder.variance * (builder.compute_feature_matrix(query_times) @ basis)).detach().cpu().numpy()
    return solve_spd(kt, kfu.T, jitter=1e-12).T


def _forecast_feature_matrix(
    phi: np.ndarray,
    *,
    query: np.ndarray,
    heldout: np.ndarray,
    origin: int,
    base_feature_count: int,
) -> np.ndarray:
    """Combine known target-time features with origin-frozen dynamic context."""

    query = np.asarray(query, dtype=int)
    heldout = np.asarray(heldout, dtype=int)
    selected = np.asarray(phi[query][:, heldout, :]).copy()
    if not 0 < int(base_feature_count) <= selected.shape[-1]:
        raise ValueError("base_feature_count must index a non-empty feature prefix")
    if int(base_feature_count) < selected.shape[-1]:
        origin_dynamic = np.asarray(phi[int(origin), heldout, int(base_feature_count) :])
        selected[:, :, int(base_feature_count) :] = origin_dynamic[None, :, :]
    return selected.reshape(-1, selected.shape[-1])


def _factors(dataset: TrafficDataset, query: np.ndarray, spatial_indices: np.ndarray, t_matrix: np.ndarray, kt: np.ndarray, k_on_t: np.ndarray | None, *, include_targets: bool) -> BlockFactors:
    query = np.asarray(query, dtype=int)
    indices = np.asarray(spatial_indices, dtype=int)
    phi = dataset.phi[query][:, indices, :].reshape(-1, dataset.phi.shape[-1])
    y_matrix = dataset.values_standardised[query][:, indices].T if include_targets else np.empty((indices.size, 0), dtype=float)
    return BlockFactors(
        y_vec=vec_f(y_matrix) if include_targets else np.empty(0, dtype=float),
        Phi=phi,
        Y=y_matrix,
        T=t_matrix,
        Kt=kt,
        K_on_t=k_on_t,
        block_slice=slice(int(query.min()), int(query.max()) + 1),
        inducing_times=np.empty(0, dtype=float),
        temporal_backend="traffic",
    )


def _update(
    model: TorchJointSSGPKronHiPPOSVGP,
    *,
    coupling: str,
    factors: BlockFactors,
    state,
    c_observed: np.ndarray,
    transfer: bool,
    zero_cross: bool = False,
):
    kwargs = {
        "y_vec": factors.y_vec,
        "Phi": factors.Phi,
        "T_n": factors.T,
        "Kt_new": factors.Kt,
        "state": state,
        "K_on_t": factors.K_on_t,
        "C_observed": c_observed,
        "no_transfer": not transfer,
    }
    if zero_cross:
        kwargs["zero_cross"] = True
    with torch.no_grad():
        if coupling == "structured_joint":
            return model.update_block_structured_joint_ssgp_transfer(**kwargs)
        if coupling == "mean_field":
            return model.update_block_ssgp_transfer(**kwargs)
    raise ValueError(f"Unsupported coupling: {coupling}")


def _serialisable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key != "locked_mean_feature_metadata"
    }


class TrafficRouteBRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dataset = load_traffic_dataset(args.data_root, args.dataset, task1_steps=args.task1_steps)
        self.split = load_spatial_split(args.split_manifest)
        if self.split.dataset != self.dataset.name:
            raise ValueError(f"Split dataset {self.split.dataset!r} does not match {self.dataset.name!r}")
        self.visible = np.asarray(self.split.visible_indices, dtype=int)
        self.heldout = np.asarray(self.split.heldout_indices, dtype=int)
        self.mean_feature_metadata: dict[str, object] = {
            "mode": "base",
            "total_feature_count": int(self.dataset.phi.shape[-1]),
            "uses_current_hidden_target": False,
        }
        if args.mean_feature_mode == "road_context_xlag":
            scaler = args.locked_mean_feature_metadata or {}
            self.dataset, self.mean_feature_metadata = build_road_context_xlag_features(
                self.dataset,
                context_indices=self.visible,
                scaler_fit_indices=np.asarray(self.split.visible_calibration_indices, dtype=int),
                road_distance_csv=args.road_distance_csv,
                lag_count=args.xlag_length,
                graph_diffusion=args.context_graph_diffusion,
                feature_mean=scaler.get("feature_mean"),
                feature_scale=scaler.get("feature_scale"),
            )
            if args.protocol == "forecast":
                self.mean_feature_metadata["forecast_dynamic_feature_policy"] = "origin_hold"
                self.mean_feature_metadata["forecast_uses_future_visible_targets"] = False
        self.device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float64 if args.dtype == "float64" else torch.float32
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if args.spatial_kernel == "geo_matern32":
            self.ks, self.c_all, self.inducing, self.inducing_sensor_indices = _spatial_factors(
                self.dataset, self.visible, ms=args.ms, lengthscale=args.spatial_lengthscale
            )
        else:
            spatial_theta = {
                "ell_t": args.temporal_lengthscale,
                "ell_s": args.spatial_lengthscale,
                "kernel_variance": args.kernel_variance,
                "noise_std": args.noise_std,
                "spatial_kernel": args.spatial_kernel,
                "graph_diffusion": args.graph_diffusion,
                "spatial_sm_weights": args.spatial_sm_weights,
                "spatial_sm_means": args.spatial_sm_means,
                "spatial_sm_scales": args.spatial_sm_scales,
            }
            self.ks, self.c_all, self.inducing, self.inducing_sensor_indices = fixed_spatial_factors(
                self.dataset,
                self.visible,
                ms=args.ms,
                theta=spatial_theta,
                road_distance_csv=args.road_distance_csv,
            )
        self.c_visible = self.c_all[self.visible]
        self.c_heldout = self.c_all[self.heldout]
        self.initial_hidden_mean = float(self.dataset.values_standardised[: self.dataset.task1_steps, self.visible].mean())
        self.initial_hidden_variance = float(max(self.dataset.values_standardised[: self.dataset.task1_steps, self.visible].var(), 1e-6))
        self.idw_weights = None
        if args.method == "spatial_idw":
            if args.protocol != "nowcast":
                raise ValueError("spatial_idw requires current visible sensors and is only defined for nowcasting")
            calibration = np.asarray(self.split.visible_calibration_indices, dtype=int)
            validation = np.asarray(self.split.visible_validation_indices, dtype=int)
            self.idw_weights = _inverse_distance_weights(
                self.dataset.coordinates_standardised[self.heldout],
                self.dataset.coordinates_standardised[self.visible],
                neighbors=args.knn_neighbors,
                power=args.idw_power,
            )
            validation_weights = _inverse_distance_weights(
                self.dataset.coordinates_standardised[validation],
                self.dataset.coordinates_standardised[calibration],
                neighbors=args.knn_neighbors,
                power=args.idw_power,
            )
            task1 = self.dataset.values_standardised[: self.dataset.task1_steps]
            validation_mean = task1[:, calibration] @ validation_weights.T
            residual = task1[:, validation] - validation_mean
            self.initial_hidden_variance = float(max(np.mean(residual**2), 1e-6))
        self.builder = None
        self.supports = None
        self._fixed_temporal_cache: dict[str, tuple[np.ndarray, tuple[np.ndarray, bool], np.ndarray | None]] = {}
        if args.method in {"hippo", "mean_field"}:
            self.builder = make_analytic_temporal_builder(
                mt=args.mt,
                lengthscale=args.temporal_lengthscale,
                variance=args.kernel_variance,
                rff_sample_size=args.rff,
                seed=args.model_seed,
                kernel_type="matern32",
            )
        elif args.method == "ordinary":
            self.supports = np.linspace(self.dataset.times_hours[0], self.dataset.times_hours[-1], args.mt)
        elif args.method not in {"persistence", "frozen_mean", "spatial_idw"}:
            raise ValueError(f"Unknown method: {args.method}")

    @property
    def is_gp(self) -> bool:
        return self.args.method in {"hippo", "mean_field", "ordinary"}

    @property
    def coupling(self) -> str:
        return "mean_field" if self.args.method == "mean_field" else "structured_joint"

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def peak_gpu_memory_mib(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return float(torch.cuda.max_memory_allocated(self.device) / 1024.0**2)

    def _temporal(self, query: np.ndarray, *, current_time: int, old_basis: slice | None, old_temporal_basis, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, slice | None]:
        if self.args.method == "ordinary":
            assert self.supports is not None
            cached = self._fixed_temporal_cache.get("ordinary")
            if cached is None:
                kt = matern32_kernel(
                    self.supports,
                    lengthscale=self.args.temporal_lengthscale,
                    variance=self.args.kernel_variance,
                ) + 1e-6 * np.eye(self.supports.size)
                matrix = symmetrize(np.asarray(kt, dtype=float))
                factor = cho_factor(
                    matrix + stable_jitter(matrix, 1e-12) * np.eye(matrix.shape[0]),
                    lower=True,
                    check_finite=False,
                )
                cached = (kt, factor, None)
                self._fixed_temporal_cache["ordinary"] = cached
            kt, factor, _ = cached
            kfu = matern32_kernel(
                self.dataset.times_hours[np.asarray(query, dtype=int)],
                self.supports,
                lengthscale=self.args.temporal_lengthscale,
                variance=self.args.kernel_variance,
            )
            t = cho_solve(factor, kfu.T, check_finite=False).T
            return t, kt, None, None, None
        assert self.builder is not None
        if mode == "changing":
            basis_slice = slice(0, int(current_time) + 1)
            previous = old_basis
        elif mode == "fixed":
            basis_slice = slice(0, self.dataset.task1_steps)
            previous = None
        elif mode == "fixed_global":
            basis_slice = slice(0, self.dataset.num_time)
            previous = None
        else:
            raise ValueError(mode)
        if mode in {"fixed", "fixed_global"} and self.args.temporal_evaluator == "scipy_frozen":
            cached = self._fixed_temporal_cache.get(mode)
            if cached is None:
                spec = temporal_spec_for_block(self.dataset.times_hours, basis_slice, moving=True)
                basis = _scipy_analytic_temporal_basis(self.builder, spec)
                with torch.no_grad():
                    basis_tensor = torch.as_tensor(basis, dtype=torch.float64)
                    kt = self.builder.add_jitter(
                        self.builder.variance * (basis_tensor.T @ basis_tensor)
                    ).detach().cpu().numpy()
                matrix = symmetrize(np.asarray(kt, dtype=float))
                factor = cho_factor(
                    matrix + stable_jitter(matrix, 1e-12) * np.eye(matrix.shape[0]),
                    lower=True,
                    check_finite=False,
                )
                cached = (kt, factor, basis)
                self._fixed_temporal_cache[mode] = cached
            kt, factor, basis = cached
            assert basis is not None
            with torch.no_grad():
                query_times = torch.as_tensor(
                    self.dataset.times_hours[np.asarray(query, dtype=int)],
                    dtype=torch.float64,
                )
                kfu = (
                    self.builder.variance
                    * (self.builder.compute_feature_matrix(query_times) @ torch.as_tensor(basis, dtype=torch.float64))
                ).detach().cpu().numpy()
            t = cho_solve(factor, kfu.T, check_finite=False).T
            return t, kt, None, basis, basis_slice
        t, kt, k_on, basis = _analytic_temporal_factors(
            builder=self.builder,
            times=self.dataset.times_hours,
            query=query,
            basis=basis_slice,
            old_basis=previous,
            old_temporal_basis=old_temporal_basis if mode == "changing" else None,
            evaluator=self.args.temporal_evaluator,
        )
        return t, kt, k_on, basis, basis_slice

    def _make_model(self) -> TorchJointSSGPKronHiPPOSVGP:
        return TorchJointSSGPKronHiPPOSVGP(
            Ks=self.ks,
            C=self.c_visible,
            sigma2=self.args.noise_std**2,
            beta_prior_mean=np.zeros(self.dataset.phi.shape[-1]),
            beta_prior_cov=self.args.beta_prior_variance * np.eye(self.dataset.phi.shape[-1]),
            prior_point_variance=self.args.kernel_variance,
            device=self.device,
            dtype=self.dtype,
        )

    def _initial_state(self, *, mode: str):
        if not self.is_gp:
            return None, None, None
        task_query = np.arange(self.dataset.task1_steps, dtype=int)
        t, kt, _, basis, basis_slice = self._temporal(
            task_query,
            current_time=self.dataset.task1_steps - 1,
            old_basis=None,
            old_temporal_basis=None,
            mode=mode,
        )
        model = self._make_model()
        factors = _factors(self.dataset, task_query, self.visible, t, kt, None, include_targets=True)
        state = _update(
            model,
            coupling=self.coupling,
            factors=factors,
            state=None,
            c_observed=self.c_visible,
            transfer=True,
            zero_cross=self.args.zero_cross,
        )
        return model, state, (basis_slice, basis)

    def _legal_update(self, *, model, state, current_time: int, pending_hidden: int | None, basis_state, mode: str, guard: StrictOnlineNowcastingGuard):
        if not self.is_gp:
            return state, basis_state, None, None
        old_basis, old_temporal = basis_state
        t_current, kt, k_on, new_temporal, new_basis = self._temporal(
            np.asarray([current_time]),
            current_time=current_time,
            old_basis=old_basis,
            old_temporal_basis=old_temporal,
            mode=mode,
        )
        if pending_hidden is not None:
            guard.absorb_delayed_hidden(pending_hidden)
            if self.args.method == "ordinary":
                t_delayed, _, _, _, _ = self._temporal(
                    np.asarray([pending_hidden]),
                    current_time=current_time,
                    old_basis=old_basis,
                    old_temporal_basis=old_temporal,
                    mode=mode,
                )
            else:
                assert new_temporal is not None
                t_delayed = _analytic_temporal_projection(
                    builder=self.builder,
                    times=self.dataset.times_hours,
                    query=np.asarray([pending_hidden]),
                    temporal_basis=new_temporal,
                    kt=kt,
                )
            delayed = _factors(self.dataset, np.asarray([pending_hidden]), self.heldout, t_delayed, kt, k_on, include_targets=True)
            state = _update(
                model,
                coupling=self.coupling,
                factors=delayed,
                state=state,
                c_observed=self.c_heldout,
                transfer=True,
                zero_cross=self.args.zero_cross,
            )
            k_on = None
        guard.read_current_visible(current_time)
        visible = _factors(self.dataset, np.asarray([current_time]), self.visible, t_current, kt, k_on, include_targets=True)
        state = _update(
            model,
            coupling=self.coupling,
            factors=visible,
            state=state,
            c_observed=self.c_visible,
            transfer=True,
            zero_cross=self.args.zero_cross,
        )
        return state, (new_basis, new_temporal), t_current, kt

    def _predict_gp_from_t(
        self,
        model,
        state,
        *,
        query: np.ndarray,
        t: np.ndarray,
        feature_origin: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype=int)
        if feature_origin is None:
            phi = self.dataset.phi[query][:, self.heldout, :].reshape(-1, self.dataset.phi.shape[-1])
        else:
            phi = _forecast_feature_matrix(
                self.dataset.phi,
                query=query,
                heldout=self.heldout,
                origin=int(feature_origin),
                base_feature_count=int(self.mean_feature_metadata["base_feature_count"]),
            )
        with torch.no_grad():
            mean, variance, _ = model.predict_with_C(
                state=state,
                T_eval=t,
                Phi=phi,
                C_eval=self.c_heldout,
                chunk_size=self.args.prediction_chunk_size,
                include_conditional_residual_variance=True,
                validate_conditional_residual_variance=True,
            )
        return np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)

    def _baseline_prediction(self, last_hidden: np.ndarray, *, current_time: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self.args.method == "persistence":
            return last_hidden.copy(), np.full(last_hidden.shape, self.initial_hidden_variance)
        if self.args.method == "spatial_idw":
            if current_time is None or self.idw_weights is None:
                raise ValueError("spatial_idw requires a current nowcasting time")
            current_visible = self.dataset.values_standardised[current_time, self.visible]
            return self.idw_weights @ current_visible, np.full(self.heldout.size, self.initial_hidden_variance)
        return np.full(last_hidden.shape, self.initial_hidden_mean), np.full(last_hidden.shape, self.initial_hidden_variance)

    def run_nowcast(self, *, mode: str) -> dict[str, object]:
        model, state, basis_state = self._initial_state(mode=mode)
        guard = StrictOnlineNowcastingGuard()
        stream = np.arange(self.dataset.task1_steps, self.dataset.num_time, self.args.stream_stride, dtype=int)
        if self.args.max_stream_steps:
            stream = stream[: self.args.max_stream_steps]
        if not stream.size:
            raise ValueError("No strict-online steps selected")
        pending_hidden = None
        last_hidden = np.full(self.heldout.size, self.initial_hidden_mean)
        all_y: list[np.ndarray] = []
        all_mean: list[np.ndarray] = []
        all_variance: list[np.ndarray] = []
        rows: list[dict[str, object]] = []
        state_sizes: list[int] = []
        started = time.perf_counter()
        for step, current_time in enumerate(stream):
            guard.begin(int(current_time))
            self.synchronize()
            update_started = time.perf_counter()
            if self.is_gp:
                assert basis_state is not None
                state, basis_state, t_current, kt = self._legal_update(
                    model=model,
                    state=state,
                    current_time=int(current_time),
                    pending_hidden=pending_hidden,
                    basis_state=basis_state,
                    mode=mode,
                    guard=guard,
                )
                self.synchronize()
                update_seconds = time.perf_counter() - update_started
                self.synchronize()
                prediction_started = time.perf_counter()
                assert t_current is not None
                mean, variance = self._predict_gp_from_t(
                    model,
                    state,
                    query=np.asarray([current_time]),
                    t=t_current,
                )
                self.synchronize()
                prediction_seconds = time.perf_counter() - prediction_started
                state_bytes = int(state.tensor_bytes() + model.tensor_bytes())
            else:
                if pending_hidden is not None:
                    guard.absorb_delayed_hidden(pending_hidden)
                guard.read_current_visible(int(current_time))
                update_seconds = time.perf_counter() - update_started
                prediction_started = time.perf_counter()
                mean, variance = self._baseline_prediction(last_hidden, current_time=int(current_time))
                prediction_seconds = time.perf_counter() - prediction_started
                state_bytes = int(last_hidden.nbytes + (0 if self.idw_weights is None else self.idw_weights.nbytes))
            guard.mark_prediction_complete()
            guard.reveal_current_hidden(int(current_time))
            y = self.dataset.values_standardised[current_time, self.heldout]
            all_y.append(y)
            all_mean.append(mean)
            all_variance.append(variance)
            last_hidden = y.copy()
            pending_hidden = int(current_time)
            state_sizes.append(state_bytes)
            metrics = gaussian_metrics(y, mean, variance)
            row = {
                    "step": step,
                    "time_index": int(current_time),
                    "timestamp": self.dataset.timestamps[int(current_time)].isoformat(),
                    "update_seconds": update_seconds,
                    "prediction_seconds": prediction_seconds,
                    "persistent_state_bytes": state_bytes,
                    **metrics,
                }
            if self.args.state_diagnostics and self.is_gp:
                assert kt is not None
                row.update(_state_diagnostics(state, kt, variance))
            rows.append(row)
            if (
                self.args.divergence_rmse_threshold > 0.0
                and metrics["rmse"] > self.args.divergence_rmse_threshold
            ):
                break
        y_all = np.concatenate(all_y)
        mean_all = np.concatenate(all_mean)
        variance_all = np.concatenate(all_variance)
        final = gaussian_metrics(y_all, mean_all, variance_all)
        final["rmse_speed"] = float(np.sqrt(np.mean((self.dataset.target_to_speed(y_all) - self.dataset.target_to_speed(mean_all)) ** 2)))
        self.synchronize()
        return {
            "protocol": "N",
            "mode": mode,
            "stream_indices": stream[: len(rows)],
            "y": np.vstack(all_y),
            "mean": np.vstack(all_mean),
            "variance": np.vstack(all_variance),
            "rows": rows,
            "final": final,
            "guard": guard.report(),
            "persistent_state_bytes_min": int(min(state_sizes)),
            "persistent_state_bytes_max": int(max(state_sizes)),
            "wall_clock_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": self.peak_gpu_memory_mib(),
            "stability_status": (
                "diverged"
                if self.args.divergence_rmse_threshold > 0.0
                and rows[-1]["rmse"] > self.args.divergence_rmse_threshold
                else "complete"
            ),
            "divergence_onset": (
                {
                    "step": int(rows[-1]["step"]),
                    "time_index": int(rows[-1]["time_index"]),
                    "timestamp": str(rows[-1]["timestamp"]),
                    "rmse": float(rows[-1]["rmse"]),
                    "threshold": float(self.args.divergence_rmse_threshold),
                }
                if self.args.divergence_rmse_threshold > 0.0
                and rows[-1]["rmse"] > self.args.divergence_rmse_threshold
                else None
            ),
        }

    def run_forecast(self, *, mode: str, horizons: Iterable[int]) -> dict[str, object]:
        checkpoint_path = self.args.output / "forecast_checkpoint.pt"
        stream = np.arange(self.dataset.task1_steps, self.dataset.num_time, self.args.stream_stride, dtype=int)
        if self.args.max_stream_steps:
            stream = stream[: self.args.max_stream_steps]
        horizon_values = tuple(sorted({int(value) for value in horizons if int(value) > 0}))
        if not horizon_values:
            raise ValueError("At least one positive forecast horizon is required")
        max_horizon = max(horizon_values)
        stream = stream[stream + max_horizon < self.dataset.num_time]

        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if saved["horizons"] != horizon_values or saved["mode"] != mode:
                raise ValueError("Forecast checkpoint does not match requested horizons or mechanism mode")
            model = self._make_model() if self.is_gp else None
            state = saved["state"]
            basis_state = saved["basis_state"]
            guard_nowcast = saved["guard_nowcast"]
            pending_hidden = saved["pending_hidden"]
            last_hidden = saved["last_hidden"]
            results = saved["results"]
            rows = saved["rows"]
            next_step = int(saved["next_step"])
            elapsed_before = float(saved["elapsed_seconds"])
        else:
            model, state, basis_state = self._initial_state(mode=mode)
            guard_nowcast = StrictOnlineNowcastingGuard()
            pending_hidden = None
            last_hidden = np.full(self.heldout.size, self.initial_hidden_mean)
            results = {h: {"y": [], "mean": [], "variance": []} for h in horizon_values}
            rows = []
            next_step = 0
            elapsed_before = 0.0

        chunk_steps = int(self.args.chunk_steps)
        stop_step = stream.size if chunk_steps <= 0 else min(stream.size, next_step + chunk_steps)
        started = time.perf_counter()
        for step in range(next_step, stop_step):
            current_time = int(stream[step])
            if step and step % 1000 == 0:
                _trim_cpu_allocator()
            guard_nowcast.begin(current_time)
            if self.is_gp:
                assert model is not None and basis_state is not None
                state, basis_state, _, kt_current = self._legal_update(
                    model=model,
                    state=state,
                    current_time=current_time,
                    pending_hidden=pending_hidden,
                    basis_state=basis_state,
                    mode=mode,
                    guard=guard_nowcast,
                )
            else:
                if pending_hidden is not None:
                    guard_nowcast.absorb_delayed_hidden(pending_hidden)
                guard_nowcast.read_current_visible(current_time)
                kt_current = None
            for horizon in horizon_values:
                forecast_guard = TrafficForecastGuard()
                forecast_guard.read_current_visible()
                forecast_guard.read_known_future_calendar()
                if self.is_gp:
                    assert model is not None and basis_state is not None and kt_current is not None
                    if self.args.method == "ordinary":
                        old_basis, old_temporal = basis_state
                        forecast_t, _, _, _, _ = self._temporal(
                            np.asarray([current_time + horizon]),
                            current_time=current_time,
                            old_basis=old_basis,
                            old_temporal_basis=old_temporal,
                            mode=mode,
                        )
                    else:
                        assert basis_state[1] is not None
                        forecast_t = _analytic_temporal_projection(
                            builder=self.builder,
                            times=self.dataset.times_hours,
                            query=np.asarray([current_time + horizon]),
                            temporal_basis=basis_state[1],
                            kt=kt_current,
                        )
                    mean, variance = self._predict_gp_from_t(
                        model,
                        state,
                        query=np.asarray([current_time + horizon]),
                        t=forecast_t,
                        feature_origin=current_time,
                    )
                else:
                    mean, variance = self._baseline_prediction(last_hidden)
                forecast_guard.mark_prediction_complete()
                forecast_guard.reveal_future_target()
                y = self.dataset.values_standardised[current_time + horizon, self.heldout]
                results[horizon]["y"].append(y)
                results[horizon]["mean"].append(mean)
                results[horizon]["variance"].append(variance)
                rows.append(
                    {
                        "step": step,
                        "origin_time_index": current_time,
                        "horizon_steps": horizon,
                        "target_time_index": current_time + horizon,
                        "unknown_future_exogenous_reads": forecast_guard.report()["unknown_future_exogenous_reads"],
                        **gaussian_metrics(y, mean, variance),
                    }
                )
            guard_nowcast.mark_prediction_complete()
            guard_nowcast.reveal_current_hidden(current_time)
            last_hidden = self.dataset.values_standardised[current_time, self.heldout].copy()
            pending_hidden = current_time

        elapsed = elapsed_before + time.perf_counter() - started
        if stop_step < stream.size:
            torch.save(
                {
                    "schema_version": 1,
                    "mode": mode,
                    "horizons": horizon_values,
                    "state": state,
                    "basis_state": basis_state,
                    "guard_nowcast": guard_nowcast,
                    "pending_hidden": pending_hidden,
                    "last_hidden": last_hidden,
                    "results": results,
                    "rows": rows,
                    "next_step": stop_step,
                    "elapsed_seconds": elapsed,
                },
                checkpoint_path,
            )
            return {
                "chunk_complete": False,
                "next_step": stop_step,
                "total_steps": int(stream.size),
                "checkpoint": str(checkpoint_path),
            }

        final: dict[str, dict[str, float]] = {}
        archive: dict[str, np.ndarray] = {"stream_indices": stream}
        for horizon, values in results.items():
            y = np.vstack(values["y"])
            mean = np.vstack(values["mean"])
            variance = np.vstack(values["variance"])
            summary = gaussian_metrics(y, mean, variance)
            summary["rmse_speed"] = float(np.sqrt(np.mean((self.dataset.target_to_speed(y) - self.dataset.target_to_speed(mean)) ** 2)))
            final[str(horizon)] = summary
            archive[f"y_h{horizon}"] = y
            archive[f"mean_h{horizon}"] = mean
            archive[f"variance_h{horizon}"] = variance
        checkpoint_path.unlink(missing_ok=True)
        self.synchronize()
        return {
            "protocol": "F",
            "mode": mode,
            "archive": archive,
            "rows": rows,
            "final": final,
            "nowcasting_guard": guard_nowcast.report(),
            "wall_clock_seconds": elapsed,
            "peak_gpu_memory_mib": self.peak_gpu_memory_mib(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pems_bay", "metr_la"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/traffic/raw"))
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=["nowcast", "forecast"], default="nowcast")
    parser.add_argument("--method", choices=["hippo", "ordinary", "mean_field", "persistence", "frozen_mean", "spatial_idw"], default="hippo")
    parser.add_argument("--mechanism-mode", choices=["changing", "fixed", "fixed_global"], default="changing")
    parser.add_argument(
        "--zero-cross",
        action="store_true",
        help="Zero the retained trend-residual cross statistic after each structured joint update.",
    )
    parser.add_argument("--task1-steps", type=int, default=2016)
    parser.add_argument("--stream-stride", type=int, default=1, help="Use 1 for formal runs; >1 defines a thinned diagnostic stream")
    parser.add_argument("--max-stream-steps", type=int, default=0)
    parser.add_argument("--state-diagnostics", action="store_true")
    parser.add_argument(
        "--divergence-rmse-threshold",
        type=float,
        default=0.0,
        help="Stop after the first completed prediction whose per-step RMSE exceeds this value; 0 disables.",
    )
    parser.add_argument("--chunk-steps", type=int, default=0, help="Persist and exit after this many forecast origins; rerun the same command to resume")
    parser.add_argument("--forecast-horizons", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--mt", type=int, default=32)
    parser.add_argument("--ms", type=int, default=32)
    parser.add_argument("--rff", type=int, default=256)
    parser.add_argument(
        "--temporal-evaluator",
        choices=["torch_autograd", "scipy_frozen"],
        default="torch_autograd",
        help="Use scipy_frozen only when Task-1 theta and fixed RFF frequencies are locked.",
    )
    parser.add_argument("--temporal-lengthscale", type=float, default=6.0, help="Hours")
    parser.add_argument("--spatial-lengthscale", type=float, nargs=2, default=[1.0, 1.0], metavar=("LAT", "LON"), help="Standardised latitude and longitude units")
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--spatial-kernel", choices=SPATIAL_KERNELS, default="geo_matern32")
    parser.add_argument("--road-distance-csv", type=Path)
    parser.add_argument("--graph-diffusion", type=float, default=1.0)
    parser.add_argument("--spatial-sm-weights", type=float, nargs="+", default=[0.5, 0.5])
    parser.add_argument("--spatial-sm-means", type=float, nargs="+", default=[0.0, 0.0, 0.5, 0.5])
    parser.add_argument("--spatial-sm-scales", type=float, nargs="+", default=[1.0, 1.0, 1.0, 1.0])
    parser.add_argument("--mean-feature-mode", choices=["base", "road_context_xlag"], default="base")
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument("--context-graph-diffusion", type=float, default=7.448975327393576)
    parser.add_argument("--knn-neighbors", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--beta-prior-variance", type=float, default=100.0)
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--theta-json", type=Path, help="Optional locked Task-1 empirical-Bayes hyperparameters")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()
    args.locked_mean_feature_metadata = None
    if args.stream_stride <= 0:
        raise ValueError("stream_stride must be positive")
    if args.mt <= 1 or args.ms <= 1:
        raise ValueError("mt and ms must exceed one")
    if args.zero_cross and args.method != "hippo":
        raise ValueError("--zero-cross requires --method hippo")
    if args.zero_cross and args.mechanism_mode != "changing":
        raise ValueError("--zero-cross requires --mechanism-mode changing")
    if args.theta_json is not None:
        theta_payload = json.loads(args.theta_json.read_text(encoding="utf-8"))
        theta = theta_payload.get("learned_theta", theta_payload.get("theta", theta_payload))
        theta_args = theta_payload.get("args", {})
        feature_metadata = theta_payload.get("mean_feature_metadata", {})
        recorded_feature_mode = str(feature_metadata.get("mode", theta_args.get("mean_feature_mode", "base")))
        if args.mean_feature_mode == "base" and recorded_feature_mode != "base":
            args.mean_feature_mode = recorded_feature_mode
        elif args.mean_feature_mode != recorded_feature_mode:
            raise ValueError(
                f"{args.theta_json} was calibrated with mean_feature_mode={recorded_feature_mode}, "
                f"but this online run requested {args.mean_feature_mode}"
            )
        if recorded_feature_mode == "road_context_xlag":
            recorded_lag = int(feature_metadata.get("lag_count", theta_args.get("xlag_length", 10)))
            if args.xlag_length != 10 and args.xlag_length != recorded_lag:
                raise ValueError(
                    f"{args.theta_json} was calibrated with xlag_length={recorded_lag}, "
                    f"but this online run requested {args.xlag_length}"
                )
            args.xlag_length = recorded_lag
            args.context_graph_diffusion = float(feature_metadata["graph_diffusion"])
            args.locked_mean_feature_metadata = feature_metadata
        for name, requested in (("mt", args.mt), ("ms", args.ms), ("rff", args.rff)):
            recorded = theta_args.get(name)
            if recorded is not None and int(recorded) != int(requested):
                raise ValueError(
                    f"{args.theta_json} was calibrated with {name}={recorded}, "
                    f"but this online run requested {requested}"
                )
        for key in ("ell_t", "ell_s", "kernel_variance", "noise_std"):
            if key not in theta:
                raise ValueError(f"{args.theta_json} lacks required theta key {key!r}")
        args.temporal_lengthscale = float(theta["ell_t"])
        args.spatial_lengthscale = [float(value) for value in theta["ell_s"]]
        args.kernel_variance = float(theta["kernel_variance"])
        args.noise_std = float(theta["noise_std"])
        args.spatial_kernel = str(theta.get("spatial_kernel", "geo_matern32"))
        args.graph_diffusion = float(theta.get("graph_diffusion", args.graph_diffusion))
        args.spatial_sm_weights = theta.get("spatial_sm_weights", args.spatial_sm_weights)
        args.spatial_sm_means = theta.get("spatial_sm_means", args.spatial_sm_means)
        args.spatial_sm_scales = theta.get("spatial_sm_scales", args.spatial_sm_scales)
    if args.spatial_kernel == "road_graph" and args.road_distance_csv is None:
        raise ValueError("road_graph requires --road-distance-csv")
    if args.mean_feature_mode == "road_context_xlag" and args.road_distance_csv is None:
        raise ValueError("road_context_xlag requires --road-distance-csv")
    if args.temporal_evaluator == "scipy_frozen" and args.method in {"hippo", "mean_field"} and args.theta_json is None:
        raise ValueError("scipy_frozen requires a locked Task-1 theta.json")
    run = TrafficRouteBRun(args)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = run.run_nowcast(mode=args.mechanism_mode) if args.protocol == "nowcast" else run.run_forecast(mode=args.mechanism_mode, horizons=args.forecast_horizons)
    if result.get("chunk_complete") is False:
        (args.output / "progress.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    (args.output / "progress.json").unlink(missing_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": run.dataset.name,
        "num_time": run.dataset.num_time,
        "num_sensors": run.dataset.num_sensors,
        "task1_steps": run.dataset.task1_steps,
        "heldout_sensors": int(run.heldout.size),
        "visible_sensors": int(run.visible.size),
        "visible_calibration_sensors": len(run.split.visible_calibration_indices),
        "visible_validation_sensors": len(run.split.visible_validation_indices),
        "target_standardisation": {"mean": run.dataset.target_mean, "scale": run.dataset.target_scale, "fit_prefix": "Task-1 only"},
        "coordinates": {"order": "latitude_longitude", "standardised_independently": True},
        "mean_features": run.mean_feature_metadata,
        "method": args.method,
        "device": str(run.device),
        "dtype": args.dtype,
        "split_manifest": str(args.split_manifest),
        "inducing_sensor_indices": [int(value) for value in run.inducing_sensor_indices],
        "args": _serialisable_args(args),
        "started_unix_seconds": started,
        "result": {
            key: value
            for key, value in result.items()
            if key not in {"rows", "y", "mean", "variance", "archive", "stream_indices"}
        },
    }
    write_csv(result["rows"], args.output / "per_step_metrics.csv")
    if args.protocol == "nowcast":
        np.savez_compressed(
            args.output / "predictions.npz",
            stream_indices=result["stream_indices"],
            y=result["y"],
            mean=result["mean"],
            variance=result["variance"],
        )
    else:
        np.savez_compressed(args.output / "predictions.npz", **result["archive"])
    (args.output / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = str(result.get("stability_status", "complete"))
    (args.output / "status.json").write_text(
        json.dumps({"status": status, "finite_predictions": True, "result": "result.json"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["result"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
