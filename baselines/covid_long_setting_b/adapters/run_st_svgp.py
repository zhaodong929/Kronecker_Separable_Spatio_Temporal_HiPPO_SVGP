#!/usr/bin/env python3
"""Causal Setting B adapter for the official AaltoML ST-SVGP implementation.

Run this file with the isolated ``baselines/.venvs/st_svgp`` interpreter.  It
uses the authors' Bayes-Newton ``MarkovVariationalGP`` unchanged.  That source
does not expose a posterior extension API for a growing irregular observation
grid, so each online step rebuilds its posterior from the legal arrived data
while reusing the Task-1 kernel, likelihood and spatial inducing locations.
This is consequently an accuracy baseline, not an online-runtime comparator.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import List, Sequence, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2

import bayesnewton
import objax
from jax.interpreters import xla as jax_xla


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.archive import PredictionArchive
from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol, KnownObservation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-weeks", type=int, default=0)
    parser.add_argument("--spatial-inducing", type=int, default=32)
    parser.add_argument("--task1-iterations", type=int, default=50000)
    parser.add_argument("--task1-check-interval", type=int, default=250)
    parser.add_argument("--task1-min-steps", type=int, default=2500)
    parser.add_argument("--task1-plateau-checks", type=int, default=10)
    parser.add_argument("--task1-plateau-relative-improvement", type=float, default=1e-3)
    parser.add_argument("--online-inference-steps", type=int, default=5)
    parser.add_argument("--task1-adam-learning-rate", type=float, default=0.01)
    parser.add_argument("--task1-newton-learning-rate", type=float, default=1.0)
    parser.add_argument(
        "--frozen-task1-state",
        type=Path,
        help="Load Task-1 kernel, likelihood and inducing state prepared by this adapter.",
    )
    parser.add_argument(
        "--write-frozen-task1-state",
        type=Path,
        help="Write the Task-1 kernel, likelihood and inducing state after fitting it.",
    )
    parser.add_argument("--segment-start", type=int, default=0)
    parser.add_argument("--segment-end", type=int)
    parser.add_argument(
        "--segment-output",
        type=Path,
        help="Write one isolated causal segment instead of a contiguous common archive.",
    )
    parser.add_argument(
        "--task1-validation-only",
        action="store_true",
        help="Fit only the 38 Task-1 fitting jurisdictions and score the four validation jurisdictions.",
    )
    return parser.parse_args()


class ArrivedObservations:
    """A chronological sparse observation list constructed solely from protocol batches."""

    def __init__(self, protocol: COVIDSettingBProtocol, locations: np.ndarray | None = None) -> None:
        self.protocol = protocol
        self._times: List[np.ndarray] = []
        self._locations: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        task1 = protocol.task1()
        locations = task1.locations if locations is None else np.asarray(locations, dtype=np.int64)
        for index, time_value in enumerate(protocol.calibration_times):
            self._times.append(np.full(locations.size, time_value, dtype=np.float64))
            self._locations.append(locations.copy())
            self._targets.append(task1.targets[index].astype(np.float64, copy=True))

    def append(self, observation: KnownObservation) -> None:
        self._times.append(np.full(observation.locations.size, observation.time, dtype=np.float64))
        self._locations.append(observation.locations.astype(np.int64, copy=True))
        self._targets.append(observation.targets.astype(np.float64, copy=True))

    def as_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.concatenate(self._times)
        locations = np.concatenate(self._locations)
        targets = np.concatenate(self._targets)
        inputs = np.column_stack([times, self.protocol.coordinates[locations]])
        return bayesnewton.utils.create_spatiotemporal_grid(inputs, targets[:, None])


def make_model(
    times: np.ndarray,
    spatial_grid: np.ndarray,
    targets: np.ndarray,
    inducing_locations: np.ndarray,
    *,
    trainable_inducing: bool,
) -> object:
    temporal_kernel = bayesnewton.kernels.Matern32(variance=1.0, lengthscale=0.2)
    spatial_kernel = bayesnewton.kernels.Separable(
        [
            bayesnewton.kernels.Matern32(variance=1.0, lengthscale=1.0),
            bayesnewton.kernels.Matern32(variance=1.0, lengthscale=1.0),
        ]
    )
    kernel = bayesnewton.kernels.SpatioTemporalKernel(
        temporal_kernel=temporal_kernel,
        spatial_kernel=spatial_kernel,
        z=inducing_locations,
        sparse=True,
        opt_z=trainable_inducing,
        conditional="Full",
    )
    return bayesnewton.models.MarkovVariationalGP(
        kernel=kernel,
        likelihood=bayesnewton.likelihoods.Gaussian(variance=0.1),
        X=times,
        R=spatial_grid,
        Y=targets,
        parallel=False,
    )


def train_task1(
    model: object,
    *,
    iterations: int,
    check_interval: int,
    min_steps: int,
    plateau_checks: int,
    plateau_relative_improvement: float,
    adam_lr: float,
    newton_lr: float,
    checkpoint_directory: Path | None,
    seed: int,
    spatial_inducing: int,
) -> dict[str, object]:
    """Fit Task 1 until the predeclared objective-plateau gate is met."""

    optimizer = objax.optimizer.Adam(model.vars())
    energy = objax.GradValues(model.energy, model.vars())

    @objax.Function.with_vars(model.vars() + optimizer.vars())
    def train_op():
        model.inference(lr=newton_lr)
        gradients, value = energy()
        optimizer(adam_lr, gradients)
        return value

    train_op = objax.Jit(train_op)
    trace: list[dict[str, object]] = []
    completed = 0
    status = "max_budget_not_converged"
    while completed < int(iterations):
        steps = min(int(check_interval), int(iterations) - completed)
        values = [float(np.asarray(train_op())) for _ in range(steps)]
        if not np.isfinite(values).all():
            raise FloatingPointError("ST-SVGP Task-1 objective became non-finite")
        completed += steps
        row: dict[str, object] = {
            "steps_completed": completed,
            "chunk_objective_median": float(np.median(values)),
            "chunk_objective_mean": float(np.mean(values)),
        }
        if checkpoint_directory is not None:
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_directory / f"checkpoint_{completed:05d}.npz"
            inducing = np.asarray(model.kernel.z.value, dtype=np.float64)
            kernel_values, likelihood_values = frozen_hyperparameters(model)
            write_frozen_task1_state(
                checkpoint_path,
                seed=seed,
                spatial_inducing=spatial_inducing,
                inducing=inducing,
                kernel_values=kernel_values,
                likelihood_values=likelihood_values,
            )
            row["checkpoint"] = str(checkpoint_path)
        window = int(plateau_checks)
        if completed >= int(min_steps) and len(trace) >= 2 * window - 1:
            combined_trace = trace + [row]
            prior = float(
                np.median([entry["chunk_objective_median"] for entry in combined_trace[-2 * window : -window]])
            )
            current = float(np.median([entry["chunk_objective_median"] for entry in combined_trace[-window:]]))
            plateau_change = abs(current - prior) / max(abs(prior), 1e-12)
            row["moving_median_relative_change"] = plateau_change
            if plateau_change < float(plateau_relative_improvement):
                status = "converged_objective_plateau"
                trace.append(row)
                break
        trace.append(row)
    return {
        "status": status,
        "steps_completed": completed,
        "max_steps": int(iterations),
        "check_interval": int(check_interval),
        "minimum_steps": int(min_steps),
        "moving_median_checks": int(plateau_checks),
        "relative_improvement_threshold": float(plateau_relative_improvement),
        "final_objective": float(trace[-1]["chunk_objective_mean"]),
        "trace": trace,
    }


def frozen_hyperparameters(model: object) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return (
        [np.asarray(value) for value in model.kernel.vars().tensors()],
        [np.asarray(value) for value in model.likelihood.vars().tensors()],
    )


def assign_frozen_hyperparameters(
    destination: object,
    kernel_values: Sequence[np.ndarray],
    likelihood_values: Sequence[np.ndarray],
) -> None:
    destination.kernel.vars().assign([np.asarray(value) for value in kernel_values])
    destination.likelihood.vars().assign([np.asarray(value) for value in likelihood_values])


def write_frozen_task1_state(
    path: Path,
    *,
    seed: int,
    spatial_inducing: int,
    inducing: np.ndarray,
    kernel_values: Sequence[np.ndarray],
    likelihood_values: Sequence[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "seed": np.asarray([seed], dtype=np.int64),
        "spatial_inducing": np.asarray([spatial_inducing], dtype=np.int64),
        "inducing": np.asarray(inducing, dtype=np.float64),
        "kernel_count": np.asarray([len(kernel_values)], dtype=np.int64),
        "likelihood_count": np.asarray([len(likelihood_values)], dtype=np.int64),
    }
    for index, value in enumerate(kernel_values):
        payload[f"kernel_{index}"] = np.asarray(value)
    for index, value in enumerate(likelihood_values):
        payload[f"likelihood_{index}"] = np.asarray(value)
    np.savez_compressed(path, **payload)


def read_frozen_task1_state(
    path: Path,
    *,
    seed: int,
    spatial_inducing: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        stored_seed = int(np.asarray(payload["seed"]).item())
        stored_spatial_inducing = int(np.asarray(payload["spatial_inducing"]).item())
        if stored_seed != seed or stored_spatial_inducing != spatial_inducing:
            raise ValueError("The frozen Task-1 state does not match the requested seed or capacity")
        kernel_count = int(np.asarray(payload["kernel_count"]).item())
        likelihood_count = int(np.asarray(payload["likelihood_count"]).item())
        inducing = np.asarray(payload["inducing"], dtype=np.float64)
        kernel_values = [np.asarray(payload[f"kernel_{index}"]) for index in range(kernel_count)]
        likelihood_values = [
            np.asarray(payload[f"likelihood_{index}"]) for index in range(likelihood_count)
        ]
    return inducing, kernel_values, likelihood_values


def predict_locations(
    model: object,
    protocol: COVIDSettingBProtocol,
    time_value: float,
    locations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # Bayes-Newton v1.1 squeezes a singleton prediction-time axis internally.
    # Two identical query times preserve the requested posterior and avoid that
    # upstream shape limitation without adding any observed label.
    query_times = np.asarray([[time_value], [time_value]], dtype=np.float64)
    query_grid = np.repeat(protocol.coordinates[None, :, :], 2, axis=0)
    mean, variance = model.predict_y(X=query_times, R=query_grid)
    return (
        np.asarray(mean, dtype=np.float64)[-1, locations],
        np.asarray(variance, dtype=np.float64)[-1, locations],
    )


def gaussian_metrics(target: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-10)
    error = np.asarray(target, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    half = 1.6448536269514722 * np.sqrt(variance)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "gaussian_nlpd": float(np.mean(0.5 * (np.log(2.0 * np.pi * variance) + error**2 / variance))),
        "coverage90": float(np.mean(np.abs(error) <= half)),
    }


def release_finished_week_model() -> None:
    """Release legacy JAX shape-specialised executables after a causal refit."""

    gc.collect()
    jax_xla._xla_callable.cache_clear()
    jax_xla.xla_primitive_callable.cache_clear()
    jax_xla.primitive_computation.cache_clear()
    jax_xla._lazy_force_computation.cache_clear()


def run_causal_segment(
    protocol: COVIDSettingBProtocol,
    frozen_inducing: np.ndarray,
    frozen_kernel_values: Sequence[np.ndarray],
    frozen_likelihood_values: Sequence[np.ndarray],
    arrived: ArrivedObservations,
    start: int,
    end: int,
    inference_steps: int,
    newton_lr: float,
) -> tuple[list[object], np.ndarray, np.ndarray, np.ndarray]:
    """Run a contiguous legal segment in the current isolated process."""

    information_rows = []
    means, variances, seconds = [], [], []
    for week in range(start, end):
        information = protocol.week(week)
        if information.delayed_hidden is not None:
            arrived.append(information.delayed_hidden)
        arrived.append(information.current_visible)
        times, spatial_grid, targets = arrived.as_grid()
        started = time.perf_counter()
        model = make_model(
            times,
            spatial_grid,
            targets,
            frozen_inducing,
            trainable_inducing=False,
        )
        assign_frozen_hyperparameters(model, frozen_kernel_values, frozen_likelihood_values)
        for _ in range(int(inference_steps)):
            model.inference(lr=newton_lr)
        mean, variance = predict_locations(
            model,
            protocol,
            information.hidden_query.time,
            protocol.hidden_locations,
        )
        seconds.append(time.perf_counter() - started)
        information_rows.append(information)
        means.append(mean)
        variances.append(variance)
        del model
        release_finished_week_model()
    return information_rows, np.stack(means), np.stack(variances), np.asarray(seconds)


def main() -> None:
    args = parse_args()
    protocol = COVIDSettingBProtocol(args.protocol_npz, args.protocol_json)
    requested_weeks = protocol.online_weeks if args.max_weeks <= 0 else int(args.max_weeks)
    if not 1 <= requested_weeks <= protocol.online_weeks:
        raise ValueError("--max-weeks must be between 1 and the full online horizon")
    if not 1 <= args.spatial_inducing <= protocol.locations:
        raise ValueError("--spatial-inducing must be between 1 and 52")
    segment_mode = args.segment_output is not None or args.segment_start != 0 or args.segment_end is not None
    segment_start = int(args.segment_start)
    segment_end = requested_weeks if args.segment_end is None else int(args.segment_end)
    if not segment_mode or args.task1_validation_only:
        segment_start, segment_end = 0, requested_weeks
    if not 0 <= segment_start < segment_end <= protocol.online_weeks:
        raise ValueError("segment bounds must satisfy 0 <= start < end <= online horizon")
    if segment_mode and args.segment_output is None:
        raise ValueError("--segment-output is required when segment bounds are supplied")
    if args.frozen_task1_state is not None and args.write_frozen_task1_state is not None:
        raise ValueError("Use either --frozen-task1-state or --write-frozen-task1-state, not both")
    if args.task1_validation_only and args.frozen_task1_state is not None:
        raise ValueError("Task-1 validation requires fitting the Task-1 posterior in this process")

    np.random.seed(args.seed)
    inducing_locations = kmeans2(
        protocol.coordinates,
        args.spatial_inducing,
        minit="points",
    )[0]
    task1_locations = protocol.fit_locations if args.task1_validation_only else None
    arrived = ArrivedObservations(protocol, task1_locations)
    task1_model = None
    if args.frozen_task1_state is not None:
        frozen_inducing, frozen_kernel_values, frozen_likelihood_values = read_frozen_task1_state(
            args.frozen_task1_state,
            seed=args.seed,
            spatial_inducing=args.spatial_inducing,
        )
        task1_seconds = 0.0
        task1_state_source = "loaded_frozen_task1_state"
    else:
        task1_times, task1_grid, task1_targets = arrived.as_grid()
        task1_started = time.perf_counter()
        task1_model = make_model(
            task1_times,
            task1_grid,
            task1_targets,
            inducing_locations,
            trainable_inducing=True,
        )
        convergence = train_task1(
            task1_model,
            iterations=args.task1_iterations,
            check_interval=args.task1_check_interval,
            min_steps=args.task1_min_steps,
            plateau_checks=args.task1_plateau_checks,
            plateau_relative_improvement=args.task1_plateau_relative_improvement,
            adam_lr=args.task1_adam_learning_rate,
            newton_lr=args.task1_newton_learning_rate,
            checkpoint_directory=args.output_dir / "task1_checkpoints",
            seed=args.seed,
            spatial_inducing=args.spatial_inducing,
        )
        task1_seconds = time.perf_counter() - task1_started
        frozen_inducing = np.asarray(task1_model.kernel.z.value, dtype=np.float64)
        frozen_kernel_values, frozen_likelihood_values = frozen_hyperparameters(task1_model)
        task1_state_source = "fitted_in_process"
        if args.write_frozen_task1_state is not None:
            write_frozen_task1_state(
                args.write_frozen_task1_state,
                seed=args.seed,
                spatial_inducing=args.spatial_inducing,
                inducing=frozen_inducing,
                kernel_values=frozen_kernel_values,
                likelihood_values=frozen_likelihood_values,
            )
    if args.frozen_task1_state is not None:
        convergence = {
            "status": "loaded_frozen_task1_state",
            "source": str(args.frozen_task1_state),
        }

    if args.task1_validation_only:
        assert task1_model is not None
        validation_locations = protocol.validation_locations
        means, variances = [], []
        for time_value in protocol.calibration_times:
            mean, variance = predict_locations(
                task1_model, protocol, float(time_value), validation_locations
            )
            means.append(mean)
            variances.append(variance)
        metrics = gaussian_metrics(
            protocol.calibration_targets(validation_locations),
            np.stack(means),
            np.stack(variances),
        )
        result = {
            "status": "task1_validation_complete",
            "method": "ST-SVGP",
            "source": "AaltoML/spatio-temporal-GPs",
            "source_commit": "c5b929e1fc07b14ff9671dd1d66b3b8041e2a2ce",
            "protocol": "Task-1-only 38-fit/4-validation spatial split",
            "seed": args.seed,
            "capacity": {"spatial_inducing": int(args.spatial_inducing)},
            "task1_iterations": int(args.task1_iterations),
            "task1_convergence": convergence,
            "task1_seconds": task1_seconds,
            "metrics": metrics,
            "validation_labels_used_only_for_scoring": int(protocol.calibration_weeks * validation_locations.size),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "task1_validation.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return

    # Rebuild the legal history before the requested segment. Current hidden
    # labels never enter this list: protocol.week() exposes only delayed hidden
    # labels and current visible labels.
    for history_week in range(segment_start):
        history = protocol.week(history_week)
        if history.delayed_hidden is not None:
            arrived.append(history.delayed_hidden)
        arrived.append(history.current_visible)

    information_rows, means, variances, online_seconds = run_causal_segment(
        protocol,
        frozen_inducing,
        frozen_kernel_values,
        frozen_likelihood_values,
        arrived,
        segment_start,
        segment_end,
        args.online_inference_steps,
        args.task1_newton_learning_rate,
    )

    if segment_mode:
        assert args.segment_output is not None
        args.segment_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.segment_output,
            week_indices=np.arange(segment_start, segment_end, dtype=np.int64),
            y_true=protocol.evaluation_targets()[segment_start:segment_end],
            pred_mean=means,
            pred_var=variances,
            times=protocol.stream_times[segment_start:segment_end],
            test_indices=protocol.hidden_locations,
        )
        segment_status = {
            "status": "segment_complete",
            "method": "ST-SVGP causal refit",
            "source_commit": "c5b929e1fc07b14ff9671dd1d66b3b8041e2a2ce",
            "seed": int(args.seed),
            "segment_start": segment_start,
            "segment_end": segment_end,
            "task1_seconds": task1_seconds,
            "online_seconds_total": float(np.sum(online_seconds)),
            "online_seconds_per_week": float(np.mean(online_seconds)),
            "audit": {
                "online_steps_completed": segment_end - segment_start,
                "delayed_hidden_labels": max(0, segment_end - max(1, segment_start)) * 10,
                "current_visible_labels": (segment_end - segment_start) * 42,
                "current_hidden_labels_read": 0,
                "hidden_predictions": (segment_end - segment_start) * 10,
            },
        }
        (args.segment_output.parent / "segment_status.json").write_text(
            json.dumps(segment_status, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(segment_status, indent=2))
        return

    archive = PredictionArchive(protocol, method="st_svgp_causal_refit", seed=args.seed)
    for information, mean, variance in zip(information_rows, means, variances):
        archive.append(information, mean, variance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = archive.write(
        args.output_dir / "predictions.npz",
        require_complete=requested_weeks == protocol.online_weeks,
        extra_metadata={
            "adapter": "official_aaltoml_st_svgp_causal_refit",
            "source_commit": "c5b929e1fc07b14ff9671dd1d66b3b8041e2a2ce",
            "task1_iterations": int(args.task1_iterations),
            "online_inference_steps": int(args.online_inference_steps),
            "spatial_inducing": int(args.spatial_inducing),
            "task1_state_source": task1_state_source,
            "task1_convergence": convergence,
        },
    )
    status = {
        "status": "complete",
        "method": "ST-SVGP causal refit",
        "source": "AaltoML/spatio-temporal-GPs",
        "source_commit": "c5b929e1fc07b14ff9671dd1d66b3b8041e2a2ce",
        "protocol": "covid_long_setting_b",
        "seed": args.seed,
        "weeks": segment_end,
        "task1_seconds": task1_seconds,
        "task1_state_source": task1_state_source,
        "task1_convergence": convergence,
        "online_seconds_total": float(np.sum(online_seconds)),
        "online_seconds_per_week": float(np.mean(online_seconds)),
        "online_update_prediction_seconds": [float(value) for value in online_seconds],
        "audit": audit,
        "note": (
            "The official API has no posterior extension method for a growing irregular grid. "
            "Each online posterior is therefore reconstructed from legal arrived observations with "
            "Task-1 kernel, likelihood and inducing locations frozen."
        ),
    }
    (args.output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
