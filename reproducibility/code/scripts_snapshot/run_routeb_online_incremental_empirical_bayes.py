#!/usr/bin/env python3
"""Causal hyperparameter adaptation for online structured-joint Route B.

The new-block strategy updates theta from only the arriving block, then rebuilds
the analytic posterior from all observations seen so far. Rebuilding is required
because changing kernel/noise parameters invalidates sufficient statistics made
under the previous feature map. Consequently this is a causal adaptation
diagnostic, not a bounded-memory streaming algorithm.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_routeb_batch_empirical_bayes import (
    GridData,
    evaluate,
    inner_training_validation_split,
    load_controlled_grid,
    tensor_training_data,
    write_csv,
)
from stvgp_kronecker.joint_ssgp_kron.synthetic import temporal_spec_for_block
from stvgp_kronecker.routeb_empirical_bayes import BatchRouteBEmpiricalBayes


STRATEGIES = ("freeze", "new_block", "all_seen")
PARAMETER_POLICIES = ("all", "lengthscales_only")
NONFINITE_GRADIENT_POLICIES = ("reject_step", "zero_parameter")


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def time_slice_grid(data: GridData, time_slice: slice) -> GridData:
    """Return a time-only view while preserving the controlled spatial split."""

    start = 0 if time_slice.start is None else int(time_slice.start)
    stop = data.times.size if time_slice.stop is None else int(time_slice.stop)
    if start < 0 or stop > data.times.size or start >= stop:
        raise ValueError(f"Invalid time slice {time_slice} for {data.times.size} times")
    return GridData(
        times=np.asarray(data.times[start:stop], dtype=float),
        coordinates=data.coordinates,
        y=np.asarray(data.y[start:stop], dtype=float),
        phi=np.asarray(data.phi[start:stop], dtype=float),
        train_indices=data.train_indices,
        test_indices=data.test_indices,
        spatial_inducing=data.spatial_inducing,
    )


def theta_log_vector(theta: dict[str, Any]) -> np.ndarray:
    values = [
        float(theta["ell_t"]),
        *[float(value) for value in theta["ell_s"]],
        float(theta["kernel_variance"]),
        float(theta["noise_std"]),
    ]
    return np.log(np.asarray(values, dtype=float))


def trainable_parameter_groups(
    model: BatchRouteBEmpiricalBayes,
    *,
    parameter_policy: str,
    learning_rate: float,
    kernel_variance_lr_multiplier: float,
    noise_lr_multiplier: float,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Select log-hyperparameters and construct explicit Adam parameter groups."""

    if parameter_policy not in PARAMETER_POLICIES:
        raise ValueError(parameter_policy)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if kernel_variance_lr_multiplier < 0.0 or noise_lr_multiplier < 0.0:
        raise ValueError("Learning-rate multipliers must be non-negative")

    selected: dict[str, torch.Tensor] = {}
    groups: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        is_lengthscale = name in {
            "temporal.builder.log_lengthscale",
            "temporal.log_lengthscale",
            "log_spatial_lengthscales",
        }
        trainable = parameter_policy == "all" or is_lengthscale
        multiplier = 1.0
        if name in {
            "temporal.builder.log_variance",
            "temporal.log_variance",
        }:
            multiplier = kernel_variance_lr_multiplier
        elif name == "log_noise_std":
            multiplier = noise_lr_multiplier
        trainable = trainable and multiplier > 0.0
        parameter.requires_grad_(trainable)
        if not trainable:
            parameter.grad = None
            continue
        selected[name] = parameter
        groups.append(
            {
                "params": [parameter],
                "lr": learning_rate * multiplier,
                "parameter_name": name,
            }
        )
    if not groups:
        raise ValueError("The parameter policy selected no trainable parameters")
    return groups, selected


def proximal_log_parameter_penalty(
    parameters: dict[str, torch.Tensor],
    anchors: dict[str, torch.Tensor],
    coefficient: float,
) -> torch.Tensor:
    """Return lambda/2 times squared movement from the previous block."""

    if coefficient < 0.0:
        raise ValueError("proximal coefficient must be non-negative")
    if parameters.keys() != anchors.keys():
        raise ValueError("Proximal parameter and anchor sets differ")
    reference = next(iter(parameters.values()))
    penalty = torch.zeros((), dtype=reference.dtype, device=reference.device)
    for name, parameter in parameters.items():
        penalty = penalty + torch.sum((parameter - anchors[name]) ** 2)
    return 0.5 * coefficient * penalty


def enforce_noise_floor(
    model: BatchRouteBEmpiricalBayes, noise_floor: float
) -> None:
    if noise_floor <= 0.0:
        raise ValueError("noise_floor must be positive")
    with torch.no_grad():
        model.log_noise_std.clamp_(min=float(np.log(noise_floor)))


def tensors_are_finite(tensors: dict[str, torch.Tensor], *, gradients: bool) -> bool:
    """Return whether every selected parameter or gradient is finite."""

    for tensor in tensors.values():
        value = tensor.grad if gradients else tensor
        if value is not None and not bool(torch.isfinite(value).all()):
            return False
    return True


def nonfinite_gradient_names(tensors: dict[str, torch.Tensor]) -> list[str]:
    return [
        name
        for name, tensor in tensors.items()
        if tensor.grad is not None and not bool(torch.isfinite(tensor.grad).all())
    ]


def zero_nonfinite_gradients(
    tensors: dict[str, torch.Tensor], names: list[str]
) -> None:
    for name in names:
        gradient = tensors[name].grad
        if gradient is None:
            raise ValueError(f"Parameter {name} has no gradient to sanitize")
        gradient.zero_()


def restore_parameter_snapshot(
    parameters: dict[str, torch.Tensor], snapshot: dict[str, torch.Tensor]
) -> None:
    if parameters.keys() != snapshot.keys():
        raise ValueError("Parameter and snapshot sets differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(snapshot[name])


def calibration_path(base: Path, split_seed: int) -> Path:
    return (
        base
        / "phase_m_routeb_empirical_bayes"
        / "task1_calibration_then_freeze"
        / "analytic_hippo_rff"
        / f"seed{split_seed}"
        / "result.json"
    )


def batch_upper_bound_path(base: Path, split_seed: int) -> Path:
    return (
        base
        / "phase_m_routeb_empirical_bayes"
        / "task2_empirical_bayes"
        / "analytic_hippo_rff"
        / f"seed{split_seed}"
        / "result.json"
    )


def strict_batch_reference_path(base: Path, split_seed: int) -> Path:
    return (
        base
        / "phase_t_proximal_online_eb"
        / "task2_batch_eb_strict_dtc_reevaluation"
        / f"seed{split_seed}"
        / "result.json"
    )


def batch_reference_metrics(base: Path, split_seed: int) -> dict[str, Any]:
    strict_path = strict_batch_reference_path(base, split_seed)
    if strict_path.exists():
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
        return {
            "split_seed": split_seed,
            "rmse": float(strict["metrics"]["rmse"]),
            "nll": float(strict["metrics"]["nll"]),
            "coverage90": float(strict["metrics"]["coverage90"]),
            "best_iteration": -1,
            "variance_protocol": "strict finite/DTC projected variance",
        }
    reference = json.loads(
        batch_upper_bound_path(base, split_seed).read_text(encoding="utf-8")
    )
    return {
        "split_seed": split_seed,
        "rmse": float(reference["final"]["rmse"]),
        "nll": float(reference["final"]["nll"]),
        "coverage90": float(reference["final"]["coverage90"]),
        "best_iteration": int(reference["best_iteration"]),
        "variance_protocol": "legacy source result",
    }


def optimize_hyperparameters(
    *,
    model: BatchRouteBEmpiricalBayes,
    fit_data: GridData,
    validation_data: GridData,
    temporal_horizon: Any,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    steps: int,
    learning_rate: float,
    validation_every: int,
    beta_prior_variance: float,
    prediction_chunk_size: int,
    split_seed: int,
    block_id: int,
    strategy: str,
    parameter_policy: str,
    proximal_lambda: float,
    kernel_variance_lr_multiplier: float,
    noise_lr_multiplier: float,
    noise_floor: float,
    nonfinite_gradient_policy: str = "reject_step",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if nonfinite_gradient_policy not in NONFINITE_GRADIENT_POLICIES:
        raise ValueError(nonfinite_gradient_policy)
    y_train, phi_train, coordinates_train = tensor_training_data(
        fit_data, inner_train
    )
    parameter_groups, trainable_parameters = trainable_parameter_groups(
        model,
        parameter_policy=parameter_policy,
        learning_rate=learning_rate,
        kernel_variance_lr_multiplier=kernel_variance_lr_multiplier,
        noise_lr_multiplier=noise_lr_multiplier,
    )
    optimizer = torch.optim.Adam(parameter_groups)
    anchors = {
        name: parameter.detach().clone()
        for name, parameter in trainable_parameters.items()
    }
    best_theta = copy.deepcopy(model.theta())
    started = time.perf_counter()
    model.set_temporal_query(
        validation_data.times, temporal_horizon=temporal_horizon
    )
    initial_validation_started = time.perf_counter()
    initial_validation_metrics, _, _ = evaluate(
        empirical_model=model,
        data=validation_data,
        posterior_indices=inner_train,
        evaluation_indices=inner_validation,
        representation="analytic_hippo_rff",
        beta_prior_variance=beta_prior_variance,
        prediction_chunk_size=prediction_chunk_size,
        include_conditional_residual_variance=False,
        collect_pointwise=False,
    )
    initial_validation_seconds = time.perf_counter() - initial_validation_started
    best_validation_nll = float(initial_validation_metrics["nll"])
    best_iteration = 0
    trace: list[dict[str, Any]] = [
        {
            "split_seed": split_seed,
            "block_id": block_id,
            "strategy": strategy,
            "iteration": 0,
            "train_nlml_per_observation": float("nan"),
            "proximal_penalty": 0.0,
            "train_total_objective": float("nan"),
            "gradient_norm_before_clip": float("nan"),
            "iteration_seconds": 0.0,
            "validation_nll": initial_validation_metrics["nll"],
            "validation_rmse": initial_validation_metrics["rmse"],
            "validation_coverage90": initial_validation_metrics["coverage90"],
            "validation_seconds": initial_validation_seconds,
            **model.theta(),
        }
    ]
    iteration_seconds: list[float] = []
    validation_seconds_total = initial_validation_seconds

    for iteration in range(1, steps + 1):
        model.set_temporal_query(
            fit_data.times, temporal_horizon=temporal_horizon
        )
        iteration_started = time.perf_counter()
        parameter_snapshot = {
            name: parameter.detach().clone()
            for name, parameter in trainable_parameters.items()
        }
        optimizer.zero_grad(set_to_none=True)
        diagnostics = model.objective(
            y_matrix=y_train,
            phi_tensor=phi_train,
            spatial_coordinates=coordinates_train,
            beta_prior_variance=beta_prior_variance,
        )
        penalty = proximal_log_parameter_penalty(
            trainable_parameters, anchors, proximal_lambda
        )
        loss = diagnostics.nlml_per_observation + penalty
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite objective for seed={split_seed}, block={block_id}, "
                f"strategy={strategy}, iteration={iteration}"
            )
        loss.backward()
        bad_gradient_names = nonfinite_gradient_names(trainable_parameters)
        if bad_gradient_names and nonfinite_gradient_policy == "reject_step":
            restore_parameter_snapshot(trainable_parameters, parameter_snapshot)
            trace.append(
                {
                    "split_seed": split_seed,
                    "block_id": block_id,
                    "strategy": strategy,
                    "iteration": iteration,
                    "train_nlml_per_observation": float(
                        diagnostics.nlml_per_observation.detach()
                    ),
                    "proximal_penalty": float(penalty.detach()),
                    "train_total_objective": float(loss.detach()),
                    "gradient_norm_before_clip": float("nan"),
                    "iteration_seconds": time.perf_counter() - iteration_started,
                    "update_rejected_nonfinite": 1,
                    "partial_gradient_update": 0,
                    "nonfinite_gradient_parameters": ";".join(bad_gradient_names),
                    **model.theta(),
                }
            )
            break
        if bad_gradient_names:
            zero_nonfinite_gradients(trainable_parameters, bad_gradient_names)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(trainable_parameters.values()), max_norm=20.0
            )
        )
        optimizer.step()
        model.clamp_parameters()
        enforce_noise_floor(model, noise_floor)
        if not tensors_are_finite(trainable_parameters, gradients=False):
            restore_parameter_snapshot(trainable_parameters, parameter_snapshot)
            trace.append(
                {
                    "split_seed": split_seed,
                    "block_id": block_id,
                    "strategy": strategy,
                    "iteration": iteration,
                    "train_nlml_per_observation": float(
                        diagnostics.nlml_per_observation.detach()
                    ),
                    "proximal_penalty": float(penalty.detach()),
                    "train_total_objective": float(loss.detach()),
                    "gradient_norm_before_clip": gradient_norm,
                    "iteration_seconds": time.perf_counter() - iteration_started,
                    "update_rejected_nonfinite": 1,
                    "partial_gradient_update": 0,
                    "nonfinite_gradient_parameters": "post_optimizer_parameter",
                    **model.theta(),
                }
            )
            break
        elapsed = time.perf_counter() - iteration_started
        iteration_seconds.append(elapsed)
        row: dict[str, Any] = {
            "split_seed": split_seed,
            "block_id": block_id,
            "strategy": strategy,
            "iteration": iteration,
            "train_nlml_per_observation": float(
                diagnostics.nlml_per_observation.detach()
            ),
            "proximal_penalty": float(penalty.detach()),
            "train_total_objective": float(loss.detach()),
            "gradient_norm_before_clip": gradient_norm,
            "iteration_seconds": elapsed,
            "update_rejected_nonfinite": 0,
            "partial_gradient_update": int(bool(bad_gradient_names)),
            "nonfinite_gradient_parameters": ";".join(bad_gradient_names),
            **model.theta(),
        }
        validate = (
            iteration == 1
            or iteration == steps
            or iteration % max(validation_every, 1) == 0
        )
        if validate:
            model.set_temporal_query(
                validation_data.times, temporal_horizon=temporal_horizon
            )
            validation_started = time.perf_counter()
            validation_metrics, _, _ = evaluate(
                empirical_model=model,
                data=validation_data,
                posterior_indices=inner_train,
                evaluation_indices=inner_validation,
                representation="analytic_hippo_rff",
                beta_prior_variance=beta_prior_variance,
                prediction_chunk_size=prediction_chunk_size,
                include_conditional_residual_variance=False,
                collect_pointwise=False,
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
                best_theta = copy.deepcopy(model.theta())
        trace.append(row)

    model.set_theta(best_theta)
    return trace, {
        "best_iteration": best_iteration,
        "best_validation_nll": best_validation_nll,
        "total_seconds": time.perf_counter() - started,
        "iteration_seconds_total": float(np.sum(iteration_seconds)),
        "mean_iteration_seconds": (
            float(np.mean(iteration_seconds)) if iteration_seconds else 0.0
        ),
        "validation_seconds_total": validation_seconds_total,
    }


def run_seed_strategy(
    *,
    base: Path,
    outdir: Path,
    data_root: str,
    split_seed: int,
    strategy: str,
    mt: int,
    ms: int,
    block_size: int,
    eb_steps: int,
    learning_rate: float,
    validation_every: int,
    validation_fraction: float,
    prediction_chunk_size: int,
    checkpoint_scope: str,
    parameter_policy: str,
    proximal_lambda: float,
    kernel_variance_lr_multiplier: float,
    noise_lr_multiplier: float,
    noise_floor: float,
    nonfinite_gradient_policy: str = "reject_step",
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    if checkpoint_scope not in {"new_block", "all_seen"}:
        raise ValueError(checkpoint_scope)
    calibration = json.loads(
        calibration_path(base, split_seed).read_text(encoding="utf-8")
    )
    calibration_args = calibration["args"]
    if mt != int(calibration_args["mt"]) or ms != int(calibration_args["ms"]):
        raise ValueError(
            "This experiment warm-starts the calibrated model and therefore requires "
            f"M_t={calibration_args['mt']}, M_s={calibration_args['ms']}"
        )
    controlled_npz = (
        base
        / "phase_d_joint_xlag_controlled"
        / f"seed{split_seed}"
        / f"era5_xlag_seed{split_seed}.npz"
    )
    data = load_controlled_grid(
        root=data_root,
        task="task_2",
        controlled_npz=controlled_npz,
        ms=ms,
        xlag_length=int(calibration_args.get("xlag_length", 10)),
    )
    inner_train, inner_validation = inner_training_validation_split(
        data.train_indices,
        split_seed=split_seed,
        validation_fraction=validation_fraction,
    )
    blocks = [
        slice(start, min(data.times.size, start + block_size))
        for start in range(0, data.times.size, block_size)
    ]
    first_seen = slice(0, blocks[0].stop)
    initial_horizon = temporal_spec_for_block(data.times, first_seen, moving=True)
    torch.manual_seed(int(calibration_args.get("model_seed", 0)))
    np.random.seed(int(calibration_args.get("model_seed", 0)))
    model = BatchRouteBEmpiricalBayes(
        times=data.times[first_seen],
        spatial_inducing=data.spatial_inducing,
        mt=mt,
        representation="analytic_hippo_rff",
        initial_ell_t=float(calibration["learned_theta"]["ell_t"]),
        initial_ell_s=tuple(calibration["learned_theta"]["ell_s"]),
        initial_kernel_variance=float(
            calibration["learned_theta"]["kernel_variance"]
        ),
        initial_noise_std=float(calibration["learned_theta"]["noise_std"]),
        rff_sample_size=int(calibration_args.get("rff_sample_size", 256)),
        seed=int(calibration_args.get("model_seed", 0)),
        objective_type="finite_dtc",
        temporal_horizon=initial_horizon,
    )
    model.set_theta(calibration["learned_theta"])
    initial_frequencies = model.temporal.builder.base_frequencies.detach().clone()
    initial_spatial_inducing = model.spatial_inducing.detach().clone()
    beta_prior_variance = float(calibration_args.get("beta_prior_variance", 1000.0))
    block_rows: list[dict[str, Any]] = []
    training_trace: list[dict[str, Any]] = []
    process_started = time.perf_counter()

    for block_id, block in enumerate(blocks):
        seen = slice(0, block.stop)
        horizon = temporal_spec_for_block(data.times, seen, moving=True)
        theta_before = model.theta()
        fit_slice = block if strategy == "new_block" else seen
        fit_data = time_slice_grid(data, fit_slice)
        seen_data = time_slice_grid(data, seen)
        validation_data = (
            seen_data if checkpoint_scope == "all_seen" else fit_data
        )
        eb_summary = {
            "best_iteration": 0,
            "best_validation_nll": float("nan"),
            "total_seconds": 0.0,
            "iteration_seconds_total": 0.0,
            "mean_iteration_seconds": 0.0,
            "validation_seconds_total": 0.0,
        }
        if strategy != "freeze":
            model.set_temporal_query(
                fit_data.times, temporal_horizon=horizon
            )
            trace, eb_summary = optimize_hyperparameters(
                model=model,
                fit_data=fit_data,
                validation_data=validation_data,
                temporal_horizon=horizon,
                inner_train=inner_train,
                inner_validation=inner_validation,
                steps=eb_steps,
                learning_rate=learning_rate,
                validation_every=validation_every,
                beta_prior_variance=beta_prior_variance,
                prediction_chunk_size=prediction_chunk_size,
                split_seed=split_seed,
                block_id=block_id,
                strategy=strategy,
                parameter_policy=parameter_policy,
                proximal_lambda=proximal_lambda,
                kernel_variance_lr_multiplier=kernel_variance_lr_multiplier,
                noise_lr_multiplier=noise_lr_multiplier,
                noise_floor=noise_floor,
                nonfinite_gradient_policy=nonfinite_gradient_policy,
            )
            training_trace.extend(trace)

        theta_after = model.theta()
        model.set_temporal_query(seen_data.times, temporal_horizon=horizon)
        metrics, _, persistent_state_bytes = evaluate(
            empirical_model=model,
            data=seen_data,
            posterior_indices=data.train_indices,
            evaluation_indices=data.test_indices,
            representation="analytic_hippo_rff",
            beta_prior_variance=beta_prior_variance,
            prediction_chunk_size=prediction_chunk_size,
            include_conditional_residual_variance=False,
            collect_pointwise=False,
        )
        row = {
            "split_seed": split_seed,
            "strategy": strategy,
            "block_id": block_id,
            "block_start": int(block.start),
            "block_stop": int(block.stop),
            "num_new_times": int(block.stop - block.start),
            "num_seen_times": int(block.stop),
            "eb_fit_start": int(fit_slice.start),
            "eb_fit_stop": int(fit_slice.stop),
            "eb_num_times": int(fit_data.times.size),
            "checkpoint_validation_scope": checkpoint_scope,
            "checkpoint_validation_num_times": int(validation_data.times.size),
            "future_times_used_for_eb": 0,
            "best_eb_iteration": int(eb_summary["best_iteration"]),
            "best_validation_nll": float(eb_summary["best_validation_nll"]),
            "eb_total_seconds": float(eb_summary["total_seconds"]),
            "eb_iteration_seconds_total": float(
                eb_summary["iteration_seconds_total"]
            ),
            "eb_mean_iteration_seconds": float(
                eb_summary["mean_iteration_seconds"]
            ),
            "eb_validation_seconds_total": float(
                eb_summary["validation_seconds_total"]
            ),
            "posterior_refresh_seconds": metrics["posterior_update_seconds"],
            "prediction_seconds": metrics["prediction_seconds"],
            "rmse": metrics["rmse"],
            "nll": metrics["nll"],
            "coverage90": metrics["coverage90"],
            "ece": metrics["ece"],
            "mean_predictive_std": metrics["mean_predictive_std"],
            "persistent_state_bytes": persistent_state_bytes,
            "theta_log_step_norm": float(
                np.linalg.norm(
                    theta_log_vector(theta_after) - theta_log_vector(theta_before)
                )
            ),
            "ell_t": float(theta_after["ell_t"]),
            "ell_s_0": float(theta_after["ell_s"][0]),
            "ell_s_1": float(theta_after["ell_s"][1]),
            "kernel_variance": float(theta_after["kernel_variance"]),
            "noise_std": float(theta_after["noise_std"]),
            "parameter_policy": parameter_policy,
            "proximal_lambda": proximal_lambda,
        }
        block_rows.append(row)
        print(json.dumps(row), flush=True)

    torch.testing.assert_close(
        model.temporal.builder.base_frequencies,
        initial_frequencies,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        model.spatial_inducing,
        initial_spatial_inducing,
        rtol=0.0,
        atol=0.0,
    )
    if any(int(row["future_times_used_for_eb"]) != 0 for row in block_rows):
        raise RuntimeError("A causal EB block accessed future times")

    seed_dir = outdir / strategy / f"seed{split_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_csv(block_rows, seed_dir / "blockwise.csv")
    if training_trace:
        write_csv(training_trace, seed_dir / "training_trace.csv")
    final = block_rows[-1]
    final_validation_started = time.perf_counter()
    final_validation_metrics, _, _ = evaluate(
        empirical_model=model,
        data=time_slice_grid(data, slice(0, data.times.size)),
        posterior_indices=inner_train,
        evaluation_indices=inner_validation,
        representation="analytic_hippo_rff",
        beta_prior_variance=beta_prior_variance,
        prediction_chunk_size=prediction_chunk_size,
        include_conditional_residual_variance=False,
        collect_pointwise=False,
    )
    final_validation_seconds = time.perf_counter() - final_validation_started
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "method": "structured_joint_routeb_online_hyperparameter_adaptation",
        "strategy": strategy,
        "strategy_description": {
            "freeze": "Task-1 Route-B EB calibration, then frozen theta on Task 2",
            "new_block": (
                "theta updated from only the new block; posterior refreshed from all "
                "seen observations under the selected theta"
            ),
            "all_seen": (
                "theta re-optimized from all seen observations; posterior refreshed "
                "from all seen observations"
            ),
        }[strategy],
        "strict_bounded_memory": False,
        "causal_no_future_access": True,
        "split_seed": split_seed,
        "mt": mt,
        "ms": ms,
        "block_size": block_size,
        "num_blocks": len(blocks),
        "eb_steps_per_block": 0 if strategy == "freeze" else eb_steps,
        "learning_rate": learning_rate,
        "validation_every": validation_every,
        "validation_fraction": validation_fraction,
        "checkpoint_validation_scope": checkpoint_scope,
        "parameter_policy": parameter_policy,
        "proximal_lambda": proximal_lambda,
        "kernel_variance_lr_multiplier": kernel_variance_lr_multiplier,
        "noise_lr_multiplier": noise_lr_multiplier,
        "noise_floor": noise_floor,
        "nonfinite_gradient_policy": nonfinite_gradient_policy,
        "initial_theta": calibration["learned_theta"],
        "final_theta": model.theta(),
        "final": final,
        "final_inner_validation": final_validation_metrics,
        "timing": {
            "process_total_seconds": time.perf_counter() - process_started,
            "cumulative_eb_seconds": float(
                np.sum([row["eb_total_seconds"] for row in block_rows])
            ),
            "cumulative_posterior_refresh_seconds": float(
                np.sum([row["posterior_refresh_seconds"] for row in block_rows])
            ),
            "cumulative_prediction_seconds": float(
                np.sum([row["prediction_seconds"] for row in block_rows])
            ),
            "final_inner_validation_seconds": final_validation_seconds,
        },
        "resources": {
            "peak_rss_mib_internal": float(peak_rss_kib / 1024.0),
            "persistent_state_bytes": int(final["persistent_state_bytes"]),
            "persistent_state_mib": float(
                final["persistent_state_bytes"] / (1024.0**2)
            ),
            "device": "cpu",
            "dtype": "float64",
        },
        "protocol": {
            "task": "ERA5 task_2 variable 0",
            "spatial_split": "800 train / 200 held out",
            "inner_spatial_split": (
                f"{inner_train.size} EB train / {inner_validation.size} validation"
            ),
            "mean": "joint X-lag covariates, L=10",
            "objective": "finite/DTC marginal likelihood",
            "prediction_variance": "strict projected variance; no conditional residual",
            "basis": "cumulative HiPPO horizon [t_0,t_k]",
            "supports": "fixed RFF coordinates and fixed spatial inducing locations",
            "optimizer_state": "Adam reset at each block; theta warm-started",
            "hyperparameter_update": (
                f"policy={parameter_policy}, proximal_lambda={proximal_lambda}, "
                f"variance_lr_multiplier={kernel_variance_lr_multiplier}, "
                f"noise_lr_multiplier={noise_lr_multiplier}, noise_floor={noise_floor}"
            ),
            "checkpoint": (
                f"iteration 0 and subsequent candidates ranked by {checkpoint_scope} "
                "inner-spatial validation NLL"
            ),
            "posterior_refresh": (
                "all seen data are reprocessed after theta changes; no stale sufficient "
                "statistics are reused"
            ),
        },
    }
    (seed_dir / "result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def aggregate_results(
    *,
    base: Path,
    outdir: Path,
    split_seeds: list[int],
    strategies: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for strategy in strategies:
        for split_seed in split_seeds:
            path = outdir / strategy / f"seed{split_seed}" / "result.json"
            results.append(json.loads(path.read_text(encoding="utf-8")))

    summary: list[dict[str, Any]] = []
    for strategy in strategies:
        group = [result for result in results if result["strategy"] == strategy]
        item: dict[str, Any] = {"strategy": strategy, "n_seeds": len(group)}
        metric_paths = {
            "final_rmse": lambda result: result["final"]["rmse"],
            "final_nll": lambda result: result["final"]["nll"],
            "final_coverage90": lambda result: result["final"]["coverage90"],
            "final_validation_rmse": lambda result: result[
                "final_inner_validation"
            ]["rmse"],
            "final_validation_nll": lambda result: result[
                "final_inner_validation"
            ]["nll"],
            "final_validation_coverage90": lambda result: result[
                "final_inner_validation"
            ]["coverage90"],
            "process_total_seconds": lambda result: result["timing"][
                "process_total_seconds"
            ],
            "cumulative_eb_seconds": lambda result: result["timing"][
                "cumulative_eb_seconds"
            ],
            "cumulative_posterior_refresh_seconds": lambda result: result["timing"][
                "cumulative_posterior_refresh_seconds"
            ],
            "peak_rss_mib": lambda result: result["resources"][
                "peak_rss_mib_internal"
            ],
        }
        for name, getter in metric_paths.items():
            values = [float(getter(result)) for result in group]
            item[f"{name}_mean"] = float(np.mean(values))
            item[f"{name}_sd"] = sample_sd(values)
        summary.append(item)

    batch_upper_rows: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        batch_upper_rows.append(batch_reference_metrics(base, split_seed))
    write_csv(summary, outdir / "summary.csv")
    write_csv(batch_upper_rows, outdir / "task2_batch_eb_reference.csv")
    payload = {
        "experiment": "Route-B online hyperparameter adaptation",
        "settings": {
            "mt": args.mt,
            "ms": args.ms,
            "block_size": args.block_size,
            "eb_steps": args.eb_steps,
            "learning_rate": args.learning_rate,
            "validation_every": args.validation_every,
            "checkpoint_scope": args.checkpoint_scope,
            "parameter_policy": args.parameter_policy,
            "proximal_lambda": args.proximal_lambda,
            "kernel_variance_lr_multiplier": args.kernel_variance_lr_multiplier,
            "noise_lr_multiplier": args.noise_lr_multiplier,
            "noise_floor": args.noise_floor,
            "nonfinite_gradient_policy": args.nonfinite_gradient_policy,
            "split_seeds": split_seeds,
            "strategies": strategies,
        },
        "summary": summary,
        "task2_full_batch_eb_reference": {
            "rmse_mean": float(np.mean([row["rmse"] for row in batch_upper_rows])),
            "rmse_sd": sample_sd([row["rmse"] for row in batch_upper_rows]),
            "nll_mean": float(np.mean([row["nll"] for row in batch_upper_rows])),
            "nll_sd": sample_sd([row["nll"] for row in batch_upper_rows]),
            "note": (
                "Existing 100-step full Task-2 batch EB fit, re-evaluated with "
                "strict finite/DTC variance when available; it is not streaming."
            ),
        },
    }
    (outdir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--eb-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint-scope",
        choices=["new_block", "all_seen"],
        default="all_seen",
        help=(
            "Temporal range used for validation checkpointing. The EB gradient for "
            "strategy=new_block always uses only the arriving block."
        ),
    )
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--parameter-policy",
        choices=PARAMETER_POLICIES,
        default="all",
    )
    parser.add_argument("--proximal-lambda", type=float, default=0.0)
    parser.add_argument(
        "--kernel-variance-lr-multiplier", type=float, default=1.0
    )
    parser.add_argument("--noise-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--noise-floor", type=float, default=0.01)
    parser.add_argument(
        "--nonfinite-gradient-policy",
        choices=NONFINITE_GRADIENT_POLICIES,
        default="reject_step",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse completed per-seed result.json files before aggregation.",
    )
    args = parser.parse_args()

    if args.eb_steps <= 0:
        raise ValueError("--eb-steps must be positive")
    base = args.base.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    for strategy in args.strategies:
        for split_seed in args.split_seeds:
            existing = outdir / strategy / f"seed{split_seed}" / "result.json"
            if args.skip_existing and existing.exists():
                print(f"Reusing {existing}", flush=True)
                continue
            run_seed_strategy(
                base=base,
                outdir=outdir,
                data_root=args.root,
                split_seed=split_seed,
                strategy=strategy,
                mt=args.mt,
                ms=args.ms,
                block_size=args.block_size,
                eb_steps=args.eb_steps,
                learning_rate=args.learning_rate,
                validation_every=args.validation_every,
                validation_fraction=args.validation_fraction,
                prediction_chunk_size=args.prediction_chunk_size,
                checkpoint_scope=args.checkpoint_scope,
                parameter_policy=args.parameter_policy,
                proximal_lambda=args.proximal_lambda,
                kernel_variance_lr_multiplier=args.kernel_variance_lr_multiplier,
                noise_lr_multiplier=args.noise_lr_multiplier,
                noise_floor=args.noise_floor,
                nonfinite_gradient_policy=args.nonfinite_gradient_policy,
            )
    aggregate_results(
        base=base,
        outdir=outdir,
        split_seeds=args.split_seeds,
        strategies=args.strategies,
        args=args,
    )


if __name__ == "__main__":
    main()
