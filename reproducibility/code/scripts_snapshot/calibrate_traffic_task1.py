#!/usr/bin/env python3
"""Task-1 empirical-Bayes calibration for a fixed traffic spatial split.

Only the manifest's visible-calibration sensors fit the hyperparameters.  The
visible-validation sensors select the checkpoint using Gaussian NLPD, leaving
the held-out sensors untouched until a strict-online run.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_traffic_routeb import _factors, gaussian_metrics
from stvgp_kronecker.data.traffic import (
    build_road_context_xlag_features,
    load_spatial_split,
    load_traffic_dataset,
)
from stvgp_kronecker.joint_ssgp_kron.torch_backend import TorchJointSSGPKronHiPPOSVGP
from stvgp_kronecker.routeb_empirical_bayes import (
    BatchRouteBEmpiricalBayes,
    DTYPE,
    joint_sufficient_statistics,
)
from stvgp_kronecker.traffic_spatial_kernels import (
    SPATIAL_KERNELS,
    TrafficSpatialEmpiricalBayes,
    inducing_sensor_indices,
)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calibration_tensors(dataset, time_indices: np.ndarray, spatial_indices: np.ndarray, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = torch.as_tensor(dataset.values_standardised[time_indices][:, spatial_indices].T, dtype=DTYPE, device=device)
    phi = torch.as_tensor(dataset.phi[time_indices][:, spatial_indices, :].transpose(1, 0, 2), dtype=DTYPE, device=device)
    coordinates = torch.as_tensor(dataset.coordinates_standardised[spatial_indices], dtype=DTYPE, device=device)
    return y, phi, coordinates


def validation_metrics(
    *,
    dataset,
    time_indices: np.ndarray,
    calibration_indices: np.ndarray,
    validation_indices: np.ndarray,
    empirical: BatchRouteBEmpiricalBayes,
    beta_prior_variance: float,
    device: torch.device,
) -> dict[str, float]:
    empirical.set_temporal_query(dataset.times_hours[time_indices])
    with torch.no_grad():
        _, c_calibration, kt, ks = empirical.factor_matrices(
            torch.as_tensor(dataset.coordinates_standardised[calibration_indices], dtype=DTYPE, device=device)
        )
        t_matrix = empirical.temporal.factors()[0]
        _, c_validation, _, _ = empirical.factor_matrices(
            torch.as_tensor(dataset.coordinates_standardised[validation_indices], dtype=DTYPE, device=device)
        )
    model = TorchJointSSGPKronHiPPOSVGP(
        Ks=ks,
        C=c_calibration,
        sigma2=float(empirical.noise_std.detach().square()),
        beta_prior_mean=np.zeros(dataset.phi.shape[-1]),
        beta_prior_cov=beta_prior_variance * np.eye(dataset.phi.shape[-1]),
        prior_point_variance=float(empirical.temporal.variance.detach()),
        device=device,
        dtype=DTYPE,
    )
    train = _factors(
        dataset,
        time_indices,
        calibration_indices,
        np.asarray(t_matrix.detach().cpu()),
        np.asarray(kt.detach().cpu()),
        None,
        include_targets=True,
    )
    state = model.update_block_structured_joint_ssgp_transfer(
        y_vec=train.y_vec,
        Phi=train.Phi,
        T_n=train.T,
        Kt_new=train.Kt,
        state=None,
        C_observed=c_calibration,
    )
    prediction = _factors(
        dataset,
        time_indices,
        validation_indices,
        np.asarray(t_matrix.detach().cpu()),
        np.asarray(kt.detach().cpu()),
        None,
        include_targets=False,
    )
    mean, variance, _ = model.predict_with_C(
        state=state,
        T_eval=prediction.T,
        Phi=prediction.Phi,
        C_eval=c_validation,
        include_conditional_residual_variance=True,
        validate_conditional_residual_variance=True,
    )
    y = dataset.values_standardised[time_indices][:, validation_indices].reshape(-1)
    return gaussian_metrics(y, np.asarray(mean), np.asarray(variance))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pems_bay", "metr_la"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/traffic/raw"))
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task1-steps", type=int, default=2016)
    parser.add_argument("--calibration-stride", type=int, default=12, help="Use every nth Task-1 sample during EB calibration")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--relative-objective-tolerance", type=float, default=0.001)
    parser.add_argument("--plateau-checks", type=int, default=3)
    parser.add_argument("--mt", type=int, default=32)
    parser.add_argument("--ms", type=int, default=32)
    parser.add_argument("--rff", type=int, default=256)
    parser.add_argument("--initial-temporal-lengthscale", type=float, default=6.0)
    parser.add_argument(
        "--fixed-temporal-lengthscale",
        type=float,
        help="Fix ell_t at this value during Task-1 calibration.",
    )
    parser.add_argument("--temporal-lengthscale-min", type=float, default=0.003)
    parser.add_argument("--temporal-lengthscale-max", type=float, default=2.0)
    parser.add_argument("--initial-spatial-lengthscale", type=float, nargs=2, default=[1.0, 1.0], metavar=("LAT", "LON"))
    parser.add_argument("--initial-kernel-variance", type=float, default=1.0)
    parser.add_argument("--initial-noise-std", type=float, default=0.25)
    parser.add_argument("--spatial-kernel", choices=SPATIAL_KERNELS, default="geo_matern32")
    parser.add_argument("--road-distance-csv", type=Path)
    parser.add_argument("--graph-diffusion", type=float, default=1.0)
    parser.add_argument("--spatial-mixtures", type=int, default=2)
    parser.add_argument("--mean-feature-mode", choices=["base", "road_context_xlag"], default="base")
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument("--context-graph-diffusion", type=float, default=7.448975327393576)
    parser.add_argument("--beta-prior-variance", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.calibration_stride <= 0:
        raise ValueError("calibration_stride must be positive")
    if args.plateau_checks < 1 or args.relative_objective_tolerance <= 0.0:
        raise ValueError("plateau-checks and relative-objective-tolerance must be positive")
    if not 0.0 < args.temporal_lengthscale_min < args.temporal_lengthscale_max:
        raise ValueError("temporal lengthscale bounds must be positive and increasing")
    if args.fixed_temporal_lengthscale is not None and args.fixed_temporal_lengthscale <= 0.0:
        raise ValueError("fixed-temporal-lengthscale must be positive")
    if args.spatial_kernel == "road_graph" and args.road_distance_csv is None:
        raise ValueError("road_graph requires --road-distance-csv")
    if args.mean_feature_mode == "road_context_xlag" and args.road_distance_csv is None:
        raise ValueError("road_context_xlag requires --road-distance-csv")
    dataset = load_traffic_dataset(args.data_root, args.dataset, task1_steps=args.task1_steps)
    split = load_spatial_split(args.split_manifest)
    if split.dataset != dataset.name:
        raise ValueError("Split manifest belongs to a different dataset")
    calibration_indices = np.asarray(split.visible_calibration_indices, dtype=int)
    validation_indices = np.asarray(split.visible_validation_indices, dtype=int)
    visible_indices = np.asarray(split.visible_indices, dtype=int)
    mean_feature_metadata: dict[str, object] = {
        "mode": "base",
        "total_feature_count": int(dataset.phi.shape[-1]),
        "uses_current_hidden_target": False,
    }
    if args.mean_feature_mode == "road_context_xlag":
        dataset, mean_feature_metadata = build_road_context_xlag_features(
            dataset,
            context_indices=calibration_indices,
            scaler_fit_indices=calibration_indices,
            road_distance_csv=args.road_distance_csv,
            lag_count=args.xlag_length,
            graph_diffusion=args.context_graph_diffusion,
        )
    times = np.arange(0, dataset.task1_steps, args.calibration_stride, dtype=int)
    if times[-1] != dataset.task1_steps - 1:
        times = np.append(times, dataset.task1_steps - 1)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Sensor coordinates are known covariates. Select one fixed inducing grid
    # from all visible coordinates, while fitting with calibration labels only.
    spatial_inducing_indices = inducing_sensor_indices(dataset, visible_indices, args.ms)
    empirical = TrafficSpatialEmpiricalBayes(
        dataset=dataset,
        inducing_indices=spatial_inducing_indices,
        spatial_kernel=args.spatial_kernel,
        road_distance_csv=args.road_distance_csv,
        graph_diffusion=args.graph_diffusion,
        spatial_mixtures=args.spatial_mixtures,
        times=dataset.times_hours[times],
        mt=args.mt,
        representation="analytic_hippo_rff",
        initial_ell_t=(
            args.fixed_temporal_lengthscale
            if args.fixed_temporal_lengthscale is not None
            else args.initial_temporal_lengthscale
        ),
        initial_ell_s=tuple(args.initial_spatial_lengthscale),
        initial_kernel_variance=args.initial_kernel_variance,
        initial_noise_std=args.initial_noise_std,
        rff_sample_size=args.rff,
        seed=args.seed,
        objective_type="vfe",
        temporal_kernel="matern32",
        temporal_lengthscale_bounds=(args.temporal_lengthscale_min, args.temporal_lengthscale_max),
    ).to(device)
    if args.fixed_temporal_lengthscale is not None:
        temporal_parameter = (
            empirical.temporal.builder.log_lengthscale
            if empirical.temporal.builder is not None
            else empirical.temporal.log_lengthscale
        )
        temporal_parameter.requires_grad_(False)
    y_train, phi_train, coordinates_train = calibration_tensors(dataset, times, calibration_indices, device=device)
    sufficient_statistics = joint_sufficient_statistics(y_train, phi_train)
    optimizer = torch.optim.Adam(
        [parameter for parameter in empirical.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    trace: list[dict[str, object]] = []
    best_nlpd = float("inf")
    best_rmse = float("inf")
    best_iteration = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        objective = empirical.objective(
            y_matrix=y_train,
            phi_tensor=phi_train,
            spatial_coordinates=coordinates_train,
            beta_prior_variance=args.beta_prior_variance,
            sufficient_statistics=sufficient_statistics,
            cross_contraction="auto",
        )
        if not torch.isfinite(objective.nlml_per_observation):
            raise FloatingPointError(f"Non-finite Task-1 VFE objective at iteration {iteration}")
        objective.nlml_per_observation.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(empirical.parameters(), max_norm=20.0))
        optimizer.step()
        empirical.clamp_parameters()
        row: dict[str, object] = {
            "iteration": iteration,
            "vfe_nlml_per_observation": float(objective.nlml_per_observation.detach()),
            "finite_nlml_per_observation": float(objective.finite_nlml_per_observation.detach()),
            "vfe_trace_correction_per_observation": float(objective.vfe_trace_correction_per_observation.detach()),
            "gradient_norm": grad_norm,
            **empirical.theta(),
        }
        if iteration == 1 or iteration % args.validation_every == 0 or iteration == args.iterations:
            empirical.set_temporal_query(dataset.times_hours)
            metrics = validation_metrics(
                dataset=dataset,
                time_indices=np.arange(dataset.task1_steps, dtype=int),
                calibration_indices=calibration_indices,
                validation_indices=validation_indices,
                empirical=empirical,
                beta_prior_variance=args.beta_prior_variance,
                device=device,
            )
            empirical.set_temporal_query(dataset.times_hours[times])
            row.update({f"validation_{key}": value for key, value in metrics.items()})
            candidate = (metrics["gaussian_nlpd"], metrics["rmse"])
            if candidate < (best_nlpd, best_rmse):
                best_nlpd, best_rmse = candidate
                best_iteration = iteration
                best_state = copy.deepcopy(empirical.state_dict())
        trace.append(row)
    if best_state is None:
        raise RuntimeError("No validation checkpoint was scored")
    empirical.load_state_dict(best_state)
    objective_values = np.asarray([float(row["vfe_nlml_per_observation"]) for row in trace], dtype=float)
    relative_tail_changes = np.abs(np.diff(objective_values)) / np.maximum(np.abs(objective_values[:-1]), 1e-12)
    converged = bool(
        relative_tail_changes.size >= args.plateau_checks
        and np.all(relative_tail_changes[-args.plateau_checks :] < args.relative_objective_tolerance)
    )
    payload = {
        "schema_version": 1,
        "selection_metric": "visible-validation Gaussian NLPD, then RMSE",
        "dataset": dataset.name,
        "split_manifest": str(args.split_manifest),
        "task1_steps": dataset.task1_steps,
        "calibration_time_samples": int(times.size),
        "validation_time_samples": int(dataset.task1_steps),
        "calibration_spatial_sensors": int(calibration_indices.size),
        "validation_spatial_sensors": int(validation_indices.size),
        "inducing_grid_visible_sensors": int(visible_indices.size),
        "objective": "VFE",
        "temporal_kernel": "Matern-3/2",
        "spatial_kernel": args.spatial_kernel,
        "spatial_kernel_metadata": empirical.road_metadata,
        "mean_feature_metadata": mean_feature_metadata,
        "spatial_inducing_sensor_indices": spatial_inducing_indices.tolist(),
        "temporal_lengthscale_bounds_hours": [args.temporal_lengthscale_min, args.temporal_lengthscale_max],
        "fixed_temporal_lengthscale_hours": args.fixed_temporal_lengthscale,
        "best_iteration": best_iteration,
        "best_validation_gaussian_nlpd": best_nlpd,
        "best_validation_rmse": best_rmse,
        "calibration_status": "converged" if converged else "max_iterations_not_converged",
        "relative_objective_tolerance": args.relative_objective_tolerance,
        "tail_relative_objective_changes": relative_tail_changes[-args.plateau_checks :].tolist(),
        "learned_theta": empirical.theta(),
        "wall_clock_seconds": time.perf_counter() - started,
        "device": str(device),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(trace, args.output / "calibration_trace.csv")
    (args.output / "theta.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
