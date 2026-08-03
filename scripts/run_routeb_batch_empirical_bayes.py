#!/usr/bin/env python3
"""Batch empirical-Bayes Route B on the controlled ERA5 spatial holdout task."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hipposvgp_era5_routeb import (
    augment_dataset_phi,
    coverage90,
    ece_gaussian,
    gaussian_nll,
    normalise_time_dataset,
    selected_locations_from_dataset,
    vectorized_predict_with_C,
)
from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5
from stvgp_kronecker.joint_ssgp_kron.kron_utils import vec_f
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.torch_backend import (
    TorchJointSSGPKronHiPPOSVGP,
)
from stvgp_kronecker.joint_ssgp_kron.synthetic import BlockFactors
from stvgp_kronecker.routeb_empirical_bayes import (
    BatchRouteBEmpiricalBayes,
    DTYPE,
)


@dataclass
class GridData:
    times: np.ndarray
    coordinates: np.ndarray
    y: np.ndarray
    phi: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    spatial_inducing: np.ndarray


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_controlled_grid(
    *,
    root: str,
    task: str,
    controlled_npz: Path,
    ms: int,
    xlag_length: int,
) -> GridData:
    controlled = np.load(controlled_npz)
    calibration_raw = load_hipposvgp_era5(
        root, tasks=("task_1",), variable_index=0, split="all"
    )
    selected = selected_locations_from_dataset(calibration_raw)
    raw = load_hipposvgp_era5(
        root,
        tasks=(task,),
        variable_index=0,
        split="all",
        selected_locations=selected,
    )
    dataset, _ = normalise_time_dataset(raw)
    dataset = augment_dataset_phi(
        dataset, phi_mode="medium_era5_xlag", xlag_length=xlag_length
    )
    coords = np.asarray(dataset.coords, dtype=float)
    coords = (coords - coords.mean(axis=0, keepdims=True)) / np.maximum(
        coords.std(axis=0, keepdims=True), 1e-12
    )
    phi = np.asarray(dataset.Phi, dtype=float).reshape(
        dataset.Y.shape[0], dataset.Y.shape[1], -1
    )
    result = GridData(
        times=np.asarray(dataset.times, dtype=float),
        coordinates=coords,
        y=np.asarray(dataset.Y, dtype=float),
        phi=phi,
        train_indices=np.asarray(controlled["train_indices"], dtype=int),
        test_indices=np.asarray(controlled["test_indices"], dtype=int),
        spatial_inducing=np.asarray(controlled[f"inducing_coords_ms{ms}"], dtype=float),
    )
    if task == "task_2":
        np.testing.assert_allclose(result.times, controlled["times"], atol=1e-12)
        np.testing.assert_allclose(
            result.y[:, result.train_indices], controlled["y_train"], atol=1e-10
        )
        np.testing.assert_allclose(
            result.phi[:, result.train_indices], controlled["xlag_phi_train"], atol=1e-10
        )
    return result


def inner_training_validation_split(
    train_indices: np.ndarray, *, split_seed: int, validation_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1729 + int(split_seed))
    perm = rng.permutation(np.asarray(train_indices, dtype=int))
    num_val = max(1, int(round(validation_fraction * perm.size)))
    validation = np.sort(perm[:num_val])
    training = np.sort(perm[num_val:])
    return training, validation


def tensor_training_data(
    data: GridData,
    spatial_indices: np.ndarray,
    *,
    y_override: np.ndarray | None = None,
    phi_override: np.ndarray | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = DTYPE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_source = data.y if y_override is None else np.asarray(y_override, dtype=float)
    phi_source = data.phi if phi_override is None else np.asarray(phi_override, dtype=float)
    y = torch.as_tensor(
        y_source[:, spatial_indices].T, dtype=dtype, device=device
    )
    phi = torch.as_tensor(
        phi_source[:, spatial_indices].transpose(1, 0, 2),
        dtype=dtype,
        device=device,
    )
    coordinates = torch.as_tensor(
        data.coordinates[spatial_indices], dtype=dtype, device=device
    )
    return y, phi, coordinates


def numpy_factors(
    data: GridData,
    indices: np.ndarray,
    t_mat: np.ndarray,
    kt: np.ndarray,
    representation: str,
    *,
    y_override: np.ndarray | None = None,
    phi_override: np.ndarray | None = None,
) -> BlockFactors:
    y_source = data.y if y_override is None else np.asarray(y_override, dtype=float)
    phi_source = data.phi if phi_override is None else np.asarray(phi_override, dtype=float)
    y_matrix = np.asarray(y_source[:, indices].T, dtype=float)
    phi = np.asarray(phi_source[:, indices], dtype=float).reshape(
        data.times.size * indices.size, -1
    )
    return BlockFactors(
        y_vec=vec_f(y_matrix),
        Phi=phi,
        Y=y_matrix,
        T=t_mat,
        Kt=kt,
        K_on_t=None,
        block_slice=slice(0, data.times.size),
        inducing_times=np.asarray(data.times, dtype=float),
        temporal_backend=representation,
    )


def object_array_bytes(value: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(object_array_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(object_array_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return object_array_bytes(vars(value), seen)
    return 0


def evaluate(
    *,
    empirical_model: BatchRouteBEmpiricalBayes,
    data: GridData,
    posterior_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    representation: str,
    beta_prior_variance: float,
    prediction_chunk_size: int,
    posterior_y_override: np.ndarray | None = None,
    posterior_phi_override: np.ndarray | None = None,
    evaluation_phi_override: np.ndarray | None = None,
    evaluation_mean_offset: np.ndarray | None = None,
    include_conditional_residual_variance: bool = False,
    collect_pointwise: bool = True,
    solver_backend: str = "numpy",
    solver_device: torch.device | str | None = None,
    solver_dtype: torch.dtype = torch.float64,
    synchronize: Any | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], int]:
    if solver_backend not in {"numpy", "torch"}:
        raise ValueError(f"Unknown posterior solver backend: {solver_backend}")
    synchronize = synchronize or (lambda: None)
    with torch.no_grad():
        reference = next(empirical_model.parameters())
        coords_all = torch.as_tensor(
            data.coordinates, dtype=reference.dtype, device=reference.device
        )
        t_torch, c_torch, kt_torch, ks_torch = empirical_model.factor_matrices(coords_all)
        if solver_backend == "torch":
            t_mat = t_torch
            c_all = c_torch
            kt = kt_torch
            ks = ks_torch
        else:
            t_mat = np.asarray(t_torch.cpu(), dtype=float)
            c_all = np.asarray(c_torch.cpu(), dtype=float)
            kt = np.asarray(kt_torch.cpu(), dtype=float)
            ks = np.asarray(ks_torch.cpu(), dtype=float)
    train_factors = numpy_factors(
        data,
        posterior_indices,
        t_mat,
        kt,
        representation,
        y_override=posterior_y_override,
        phi_override=posterior_phi_override,
    )
    eval_factors = numpy_factors(
        data,
        evaluation_indices,
        t_mat,
        kt,
        representation,
        phi_override=evaluation_phi_override,
    )
    if solver_backend == "torch":
        device = torch.device(solver_device or reference.device)
        posterior_index_tensor = torch.as_tensor(
            posterior_indices, device=c_all.device, dtype=torch.long
        )
        evaluation_index_tensor = torch.as_tensor(
            evaluation_indices, device=c_all.device, dtype=torch.long
        )
        c_train = c_all.index_select(0, posterior_index_tensor)
        c_eval = c_all.index_select(0, evaluation_index_tensor)
    else:
        device = torch.device("cpu")
        c_train = c_all[posterior_indices]
        c_eval = c_all[evaluation_indices]
    noise_variance = float(empirical_model.noise_std.detach().square())
    kernel_variance = float(empirical_model.temporal.variance.detach())
    if solver_backend == "torch":
        model = TorchJointSSGPKronHiPPOSVGP(
            Ks=ks,
            C=c_train,
            sigma2=noise_variance,
            beta_prior_mean=np.zeros(data.phi.shape[-1]),
            beta_prior_cov=beta_prior_variance * np.eye(data.phi.shape[-1]),
            prior_point_variance=kernel_variance,
            device=device,
            dtype=solver_dtype,
        )
    else:
        model = JointSSGPKronHiPPOSVGP(
            Ks=ks,
            C=c_train,
            sigma2=noise_variance,
            beta_prior_mean=np.zeros(data.phi.shape[-1]),
            beta_prior_cov=beta_prior_variance * np.eye(data.phi.shape[-1]),
            prior_point_variance=kernel_variance,
        )
    synchronize()
    update_started = time.perf_counter()
    state = model.update_block_structured_joint_ssgp_transfer(
        y_vec=train_factors.y_vec,
        Phi=train_factors.Phi,
        T_n=train_factors.T,
        Kt_new=train_factors.Kt,
        state=None,
        K_on_t=None,
    )
    synchronize()
    update_seconds = time.perf_counter() - update_started
    synchronize()
    prediction_started = time.perf_counter()
    if solver_backend == "torch":
        mean, variance, diagnostics = model.predict_with_C(
            state=state,
            T_eval=eval_factors.T,
            Phi=eval_factors.Phi,
            C_eval=c_eval,
            chunk_size=prediction_chunk_size,
            include_conditional_residual_variance=include_conditional_residual_variance,
        )
    else:
        mean, variance, diagnostics = vectorized_predict_with_C(
            model,
            state,
            eval_factors,
            c_eval,
            prediction_mode="streaming_sylvester",
            chunk_size=prediction_chunk_size,
            include_conditional_residual_variance=include_conditional_residual_variance,
        )
    if evaluation_mean_offset is not None:
        offset = np.asarray(evaluation_mean_offset, dtype=float)
        mean = mean + vec_f(offset[:, evaluation_indices].T)
    synchronize()
    prediction_seconds = time.perf_counter() - prediction_started
    variance = np.maximum(np.asarray(variance, dtype=float), 1e-10)
    y = np.asarray(eval_factors.y_vec, dtype=float)
    metrics = {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "nll": gaussian_nll(y, mean, variance),
        "coverage90": coverage90(y, mean, variance),
        "ece": ece_gaussian(y, mean, variance),
        "mean_predictive_std": float(np.mean(np.sqrt(variance))),
        "posterior_update_seconds": update_seconds,
        "prediction_seconds": prediction_seconds,
    }
    pointwise: list[dict[str, Any]] = []
    if collect_pointwise:
        y_grid = y.reshape(data.times.size, evaluation_indices.size)
        mean_grid = mean.reshape(data.times.size, evaluation_indices.size)
        var_grid = variance.reshape(data.times.size, evaluation_indices.size)
        for time_index, time_value in enumerate(data.times):
            for local_index, global_index in enumerate(evaluation_indices):
                pointwise.append(
                    {
                        "time_index": time_index,
                        "time": float(time_value),
                        "location_index": int(global_index),
                        "y_true": float(y_grid[time_index, local_index]),
                        "pred_mean": float(mean_grid[time_index, local_index]),
                        "pred_var": float(var_grid[time_index, local_index]),
                    }
                )
    persistent_bytes = object_array_bytes(
        {
            "posterior_model": model,
            "state": state,
            "spatial_inducing": empirical_model.spatial_inducing,
            "temporal_buffers": dict(empirical_model.temporal.named_buffers()),
            "theta": dict(empirical_model.named_parameters()),
        }
    )
    metrics.update({f"diagnostic_{key}": float(value) for key, value in diagnostics.items()})
    return metrics, pointwise, persistent_bytes


def fit_ridge_mean_grid(
    data: GridData, spatial_indices: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(data.phi[:, spatial_indices], dtype=float).reshape(
        -1, data.phi.shape[-1]
    )
    target = np.asarray(data.y[:, spatial_indices], dtype=float).reshape(-1)
    precision = design.T @ design + ridge * np.eye(design.shape[1])
    beta = np.linalg.solve(precision, design.T @ target)
    mean = np.einsum("tsd,d->ts", data.phi, beta)
    return beta, mean


def serialized_training_state_bytes(
    model: BatchRouteBEmpiricalBayes, optimizer: torch.optim.Optimizer
) -> int:
    buffer = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer
    )
    return buffer.tell()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--controlled-npz", required=True)
    parser.add_argument("--fit-task", choices=["task_1", "task_2"], required=True)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--representation", choices=["analytic_hippo_rff", "inducing_points"], required=True)
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--beta-prior-variance", type=float, default=1000.0)
    parser.add_argument("--initial-ell-t", type=float, default=0.05)
    parser.add_argument("--initial-ell-s", nargs=2, type=float, default=[0.35, 0.35])
    parser.add_argument("--initial-kernel-variance", type=float, default=1.0)
    parser.add_argument("--initial-noise", type=float, default=0.1)
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--mean-mode",
        choices=["joint_xlag", "zero", "residual_xlag"],
        default="joint_xlag",
    )
    parser.add_argument("--xlag-ridge", type=float, default=1e-3)
    parser.add_argument("--test-trajectory-every", type=int, default=0)
    parser.add_argument(
        "--training-objective",
        choices=["finite_dtc", "vfe"],
        default="finite_dtc",
    )
    args = parser.parse_args()

    process_started = time.perf_counter()
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    controlled_npz = Path(args.controlled_npz)
    fit_data = load_controlled_grid(
        root=args.root,
        task=args.fit_task,
        controlled_npz=controlled_npz,
        ms=args.ms,
        xlag_length=args.xlag_length,
    )
    task2_data = (
        fit_data
        if args.fit_task == "task_2"
        else load_controlled_grid(
            root=args.root,
            task="task_2",
            controlled_npz=controlled_npz,
            ms=args.ms,
            xlag_length=args.xlag_length,
        )
    )
    inner_train, inner_validation = inner_training_validation_split(
        fit_data.train_indices,
        split_seed=args.split_seed,
        validation_fraction=args.validation_fraction,
    )
    zero_phi_fit = np.zeros((*fit_data.y.shape, 0), dtype=float)
    training_y_override = None
    training_phi_override = None
    validation_mean_offset = None
    inner_beta = None
    if args.mean_mode == "zero":
        training_phi_override = zero_phi_fit
    elif args.mean_mode == "residual_xlag":
        inner_beta, validation_mean_offset = fit_ridge_mean_grid(
            fit_data, inner_train, args.xlag_ridge
        )
        training_y_override = fit_data.y - validation_mean_offset
        training_phi_override = zero_phi_fit
    y_train, phi_train, coordinates_train = tensor_training_data(
        fit_data,
        inner_train,
        y_override=training_y_override,
        phi_override=training_phi_override,
    )
    empirical_model = BatchRouteBEmpiricalBayes(
        times=fit_data.times,
        spatial_inducing=fit_data.spatial_inducing,
        mt=args.mt,
        representation=args.representation,
        initial_ell_t=args.initial_ell_t,
        initial_ell_s=tuple(args.initial_ell_s),
        initial_kernel_variance=args.initial_kernel_variance,
        initial_noise_std=args.initial_noise,
        rff_sample_size=args.rff_sample_size,
        seed=args.model_seed,
        objective_type=args.training_objective,
    )
    optimizer = torch.optim.Adam(empirical_model.parameters(), lr=args.learning_rate)
    include_conditional_residual_variance = args.training_objective == "vfe"
    initial_base_frequencies = (
        empirical_model.temporal.builder.base_frequencies.detach().clone()
        if empirical_model.temporal.builder is not None
        else None
    )
    initial_spatial_inducing = empirical_model.spatial_inducing.detach().clone()
    trace: list[dict[str, Any]] = []
    best_validation_nll = float("inf")
    best_iteration = 0
    best_elapsed = float("nan")
    best_state: dict[str, torch.Tensor] | None = None
    training_started = time.perf_counter()
    iteration_times: list[float] = []
    validation_seconds_total = 0.0
    test_trajectory: list[dict[str, Any]] = []

    for iteration in range(1, args.iterations + 1):
        iteration_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        objective = empirical_model.objective(
            y_matrix=y_train,
            phi_tensor=phi_train,
            spatial_coordinates=coordinates_train,
            beta_prior_variance=args.beta_prior_variance,
        )
        loss = objective.nlml_per_observation
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite NLML at iteration {iteration}")
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(empirical_model.parameters(), max_norm=20.0)
        )
        optimizer.step()
        empirical_model.clamp_parameters()
        iteration_seconds = time.perf_counter() - iteration_started
        iteration_times.append(iteration_seconds)
        row: dict[str, Any] = {
            "iteration": iteration,
            "train_nlml_per_observation": float(loss.detach()),
            "finite_nlml_per_observation": float(
                objective.finite_nlml_per_observation.detach()
            ),
            "vfe_trace_correction_per_observation": float(
                objective.vfe_trace_correction_per_observation.detach()
            ),
            "vfe_trace_residual_per_observation": float(
                objective.vfe_trace_residual_per_observation.detach()
            ),
            "logdet_per_observation": float(objective.logdet_per_observation.detach()),
            "quadratic_per_observation": float(objective.quadratic_per_observation.detach()),
            "gradient_norm_before_clip": gradient_norm,
            "iteration_seconds": iteration_seconds,
            **empirical_model.theta(),
        }
        if iteration == 1 or iteration % args.validation_every == 0 or iteration == args.iterations:
            validation_started = time.perf_counter()
            validation_metrics, _, _ = evaluate(
                empirical_model=empirical_model,
                data=fit_data,
                posterior_indices=inner_train,
                evaluation_indices=inner_validation,
                representation=args.representation,
                beta_prior_variance=args.beta_prior_variance,
                prediction_chunk_size=args.prediction_chunk_size,
                posterior_y_override=training_y_override,
                posterior_phi_override=training_phi_override,
                evaluation_phi_override=training_phi_override,
                evaluation_mean_offset=validation_mean_offset,
                include_conditional_residual_variance=include_conditional_residual_variance,
            )
            validation_seconds = time.perf_counter() - validation_started
            validation_seconds_total += validation_seconds
            row.update(
                {
                    "validation_nll": validation_metrics["nll"],
                    "validation_rmse": validation_metrics["rmse"],
                    "validation_coverage90": validation_metrics["coverage90"],
                    "validation_seconds": validation_seconds,
                }
            )
            if validation_metrics["nll"] < best_validation_nll:
                best_validation_nll = validation_metrics["nll"]
                best_iteration = iteration
                best_elapsed = time.perf_counter() - training_started
                best_state = copy.deepcopy(empirical_model.state_dict())
        if (
            args.test_trajectory_every > 0
            and args.fit_task == "task_2"
            and (
                iteration == 1
                or iteration % args.test_trajectory_every == 0
                or iteration == args.iterations
            )
        ):
            diagnostic_started = time.perf_counter()
            diagnostic_metrics, _, _ = evaluate(
                empirical_model=empirical_model,
                data=task2_data,
                posterior_indices=task2_data.train_indices,
                evaluation_indices=task2_data.test_indices,
                representation=args.representation,
                beta_prior_variance=args.beta_prior_variance,
                prediction_chunk_size=args.prediction_chunk_size,
                include_conditional_residual_variance=include_conditional_residual_variance,
            )
            test_trajectory.append(
                {
                    "iteration": iteration,
                    "cumulative_update_seconds": float(np.sum(iteration_times)),
                    "diagnostic_seconds": time.perf_counter() - diagnostic_started,
                    "test_rmse": diagnostic_metrics["rmse"],
                    "test_nll": diagnostic_metrics["nll"],
                }
            )
        trace.append(row)
        if iteration == 1 or iteration % 5 == 0 or iteration == args.iterations:
            print(json.dumps(row), flush=True)

    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("No validation checkpoint was recorded")
    empirical_model.load_state_dict(best_state)
    if initial_base_frequencies is not None:
        torch.testing.assert_close(
            empirical_model.temporal.builder.base_frequencies,
            initial_base_frequencies,
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        empirical_model.spatial_inducing,
        initial_spatial_inducing,
        rtol=0.0,
        atol=0.0,
    )
    final_y_override = None
    final_phi_override = None
    final_mean_offset = None
    final_beta = None
    if args.mean_mode == "zero":
        final_phi_override = np.zeros((*task2_data.y.shape, 0), dtype=float)
    elif args.mean_mode == "residual_xlag":
        final_beta, final_mean_offset = fit_ridge_mean_grid(
            task2_data, task2_data.train_indices, args.xlag_ridge
        )
        final_y_override = task2_data.y - final_mean_offset
        final_phi_override = np.zeros((*task2_data.y.shape, 0), dtype=float)
    final_metrics, pointwise, persistent_state_bytes = evaluate(
        empirical_model=empirical_model,
        data=task2_data,
        posterior_indices=task2_data.train_indices,
        evaluation_indices=task2_data.test_indices,
        representation=args.representation,
        beta_prior_variance=args.beta_prior_variance,
        prediction_chunk_size=args.prediction_chunk_size,
        posterior_y_override=final_y_override,
        posterior_phi_override=final_phi_override,
        evaluation_phi_override=final_phi_override,
        evaluation_mean_offset=final_mean_offset,
        include_conditional_residual_variance=include_conditional_residual_variance,
    )
    optimizer_state_bytes = serialized_training_state_bytes(empirical_model, optimizer)
    process_seconds = time.perf_counter() - process_started
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "method": "batch_empirical_bayes_routeb",
        "training_objective": args.training_objective,
        "predictive_variance_protocol": (
            "vfe_conditional_residual"
            if include_conditional_residual_variance
            else "strict_finite_projected"
        ),
        "protocol": "task1_calibration_then_freeze" if args.fit_task == "task_1" else "task2_empirical_bayes",
        "fit_task": args.fit_task,
        "evaluation_task": "task_2",
        "temporal_representation": args.representation,
        "mean_mode": args.mean_mode,
        "mt": args.mt,
        "ms": args.ms,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "num_time": int(task2_data.times.size),
        "num_train_space": int(task2_data.train_indices.size),
        "num_inner_train_space": int(inner_train.size),
        "num_inner_validation_space": int(inner_validation.size),
        "num_test_space": int(task2_data.test_indices.size),
        "num_xlag_features": int(task2_data.phi.shape[-1]),
        "active_joint_mean_features": int(phi_train.shape[-1]),
        "inner_ridge_beta": None if inner_beta is None else inner_beta.tolist(),
        "final_ridge_beta": None if final_beta is None else final_beta.tolist(),
        "fixed_temporal_support": True,
        "fixed_spatial_support": True,
        "best_iteration": best_iteration,
        "best_validation_nll": best_validation_nll,
        "time_to_best_validation_seconds": best_elapsed,
        "learned_theta": empirical_model.theta(),
        "final": final_metrics,
        "diagnostic_test_trajectory": test_trajectory,
        "timing": {
            "process_total_seconds": process_seconds,
            "training_total_seconds": training_seconds,
            "mean_iteration_seconds": float(np.mean(iteration_times)),
            "median_iteration_seconds": float(np.median(iteration_times)),
            "validation_total_seconds": validation_seconds_total,
            "final_posterior_setup_seconds": final_metrics["posterior_update_seconds"],
            "final_prediction_seconds": final_metrics["prediction_seconds"],
        },
        "resources": {
            "peak_rss_mib_internal": float(peak_rss_kib / 1024.0),
            "persistent_model_state_bytes": persistent_state_bytes,
            "persistent_model_state_mib": persistent_state_bytes / (1024.0**2),
            "serialized_training_checkpoint_bytes": optimizer_state_bytes,
            "serialized_training_checkpoint_mib": optimizer_state_bytes / (1024.0**2),
            "device": "cpu",
            "dtype": "float64",
            "gpu_memory": "not applicable; CUDA unavailable with the installed driver",
        },
        "initial_theta": {
            "ell_t": args.initial_ell_t,
            "ell_s": args.initial_ell_s,
            "kernel_variance": args.initial_kernel_variance,
            "noise_std": args.initial_noise,
        },
        "args": vars(args),
    }
    write_csv(trace, outdir / "training_trace.csv")
    write_csv(pointwise, outdir / "final_pointwise_predictions.csv")
    (outdir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
