#!/usr/bin/env python3
"""Causal Setting B adapter for the official FactorialSDE LMC/IMC SVGPs.

The authors' GPflow trainers consume complete output vectors.  This wrapper
therefore fits only on complete labels that have already arrived, then absorbs
the 42 current visible labels through analytic conditioning of the official
52-output Gaussian predictive distribution.  It does not modify the upstream
kernel, ELBO, optimiser, or likelihood implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = ROOT / "baselines/external/SeyoungKimLab_FactorialSDE"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL_ROOT))

from gpflow.utilities import parameter_dict, set_trainable
import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

from fsde.baselines.gpflow_imc_svgp import (
    get_imc_svgp,
    run_adam as run_imc_adam,
    run_adam_natgrad as run_imc_adam_natgrad,
)
from fsde.baselines.gpflow_lmc_svgp import (
    get_lmc_svgp,
    run_adam as run_lmc_adam,
    run_adam_natgrad as run_lmc_adam_natgrad,
)

from baselines.covid_long_setting_b.archive import PredictionArchive
from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("lmc", "imc"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temporal-inducing", type=int, default=16)
    parser.add_argument("--latent-rank", type=int, default=4)
    parser.add_argument("--task1-iterations", type=int, default=50000)
    parser.add_argument("--task1-check-interval", type=int, default=250)
    parser.add_argument("--task1-min-steps", type=int, default=2500)
    parser.add_argument("--task1-plateau-checks", type=int, default=10)
    parser.add_argument("--task1-plateau-relative-improvement", type=float, default=1e-3)
    parser.add_argument("--online-inference-steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-natgrad", action="store_true")
    parser.add_argument("--symmetric-lmc-init", action="store_true")
    parser.add_argument("--max-weeks", type=int, default=0)
    parser.add_argument("--task1-validation-only", action="store_true")
    parser.add_argument("--validation-weeks", type=int, default=4)
    return parser.parse_args()


def make_inducing_grid(protocol: COVIDSettingBProtocol, count: int) -> np.ndarray:
    all_times = np.concatenate([protocol.calibration_times, protocol.chronological_stream_times])
    return np.linspace(all_times.min(), all_times.max(), int(count), dtype=np.float64)[:, None]


def train_params(protocol: COVIDSettingBProtocol, args: argparse.Namespace, inducing: np.ndarray) -> dict[str, Any]:
    return {
        "M": int(args.temporal_inducing),
        "P": protocol.locations,
        "L": int(args.latent_rank),
        "kernel_type": "Matern32",
        "batch_size": int(args.batch_size),
        "n_steps": int(args.task1_iterations),
        "lr": 1e-3,
        "gamma": 1e-2,
        "fix_ind": True,
        "ind_init_mode": "equal",
        "ind_times": inducing,
    }


def constructor(method: str) -> Callable[..., object]:
    return get_lmc_svgp if method == "lmc" else get_imc_svgp


def optimizers(method: str) -> tuple[Callable[..., list[float]], Callable[..., list[float]]]:
    if method == "lmc":
        return run_lmc_adam, run_lmc_adam_natgrad
    return run_imc_adam, run_imc_adam_natgrad


def initialise_lmc_latent_kernels(model: object, args: argparse.Namespace) -> dict[str, Any]:
    """Break only the avoidable LMC/IMC initialisation symmetry."""

    kernels = tuple(getattr(model.kernel, "kernels", ()))
    if len(kernels) != int(args.latent_rank):
        raise RuntimeError("Official LMC model does not expose one kernel per latent process")
    values = [float(np.asarray(kernel.lengthscales.numpy()).reshape(())) for kernel in kernels]
    if not args.symmetric_lmc_init:
        values = np.geomspace(0.05, 0.5, int(args.latent_rank)).tolist()
        for kernel, lengthscale in zip(kernels, values):
            kernel.lengthscales.assign(lengthscale)
    parameter_ids = [id(kernel.lengthscales.unconstrained_variable) for kernel in kernels]
    if len(set(parameter_ids)) != len(parameter_ids):
        raise RuntimeError("LMC latent kernels unexpectedly share one lengthscale parameter")
    return {
        "kernel_count": len(kernels),
        "initial_lengthscales": values,
        "lengthscale_parameter_ids_distinct": True,
        "mixing_matrix_trainable": bool(model.kernel.W.trainable),
        "initialisation": "symmetric" if args.symmetric_lmc_init else "deterministic_log_spaced_symmetry_break",
    }


def variational_snapshot(model: object) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(model.q_mu.numpy(), dtype=np.float64).copy(),
        np.asarray(model.q_sqrt.numpy(), dtype=np.float64).copy(),
    )


def write_checkpoint(model: object, directory: Path, step: int) -> Path:
    """Persist a restorable numerical snapshot at a convergence check."""

    directory.mkdir(parents=True, exist_ok=True)
    parameters = parameter_dict(model)
    payload: dict[str, np.ndarray] = {}
    mapping: dict[str, str] = {}
    for index, (name, value) in enumerate(parameters.items()):
        key = f"parameter_{index:04d}"
        payload[key] = np.asarray(value.numpy(), dtype=np.float64)
        mapping[name] = key
    path = directory / f"checkpoint_{int(step):05d}.npz"
    np.savez_compressed(path, **payload)
    path.with_suffix(".json").write_text(
        json.dumps({"step": int(step), "parameters": mapping}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def run_variational_steps(
    *,
    method: str,
    model: object,
    times: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    steps: int,
    natgrad: bool,
    learning_rate: float,
    gamma: float,
) -> list[float]:
    """Call the pinned official optimiser on the current posterior state."""

    if steps <= 0:
        return []
    if times.ndim != 2 or times.shape[1] != 1:
        raise ValueError("Factorial SVGP expects times with shape [time, 1]")
    if targets.shape != (times.shape[0], model.kernel.W.shape[0]):
        raise ValueError(
            "Factorial SVGP expects targets with shape [time, output]; "
            f"got {targets.shape} for {times.shape[0]} times and {model.kernel.W.shape[0]} outputs"
        )
    dataset = tf.data.Dataset.from_tensor_slices((times, targets)).shuffle(
        times.shape[0], seed=0, reshuffle_each_iteration=False
    ).repeat()
    adam, adam_natgrad = optimizers(method)
    runner = adam_natgrad if natgrad else adam
    kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": dataset,
        "n": int(times.shape[0]),
        "batch_size": min(int(batch_size), int(times.shape[0])),
        "n_steps": int(steps),
        "lr": float(learning_rate),
    }
    if natgrad:
        kwargs["gamma"] = float(gamma)
    return [float(value) for value in runner(**kwargs)]


def fit_task1(
    protocol: COVIDSettingBProtocol,
    args: argparse.Namespace,
    times: np.ndarray,
    targets: np.ndarray,
    inducing: np.ndarray,
    checkpoint_directory: Path | None = None,
) -> tuple[object, dict[str, Any]]:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    model = constructor(args.method)(
        train_params(protocol, args, inducing),
        times,
        fix_ind=True,
        random_init=False,
        seed=args.seed,
        ind_times=inducing,
    )
    lmc_audit = None
    if args.method == "lmc":
        lmc_audit = initialise_lmc_latent_kernels(model, args)
    objectives: list[float] = []
    trace: list[dict[str, Any]] = []
    stable_checks = 0
    completed = 0
    status = "max_budget_not_converged"
    while completed < int(args.task1_iterations):
        steps = min(int(args.task1_check_interval), int(args.task1_iterations) - completed)
        before_mu, before_sqrt = variational_snapshot(model)
        chunk = run_variational_steps(
            method=args.method,
            model=model,
            times=times,
            targets=targets,
            batch_size=args.batch_size,
            steps=steps,
            natgrad=not args.no_natgrad,
            learning_rate=1e-3,
            gamma=1e-2,
        )
        if not chunk or not np.isfinite(chunk).all():
            raise FloatingPointError("Official Factorial optimiser returned a non-finite ELBO trace")
        completed += len(chunk)
        objectives.extend(chunk)
        previous_median = None
        relative_improvement = None
        if len(trace) >= 1:
            previous_median = float(trace[-1]["chunk_elbo_median"])
            relative_improvement = abs(float(np.median(chunk)) - previous_median) / max(
                abs(previous_median), 1e-12
            )
        trace.append(
            {
                "steps_completed": completed,
                "chunk_elbo_median": float(np.median(chunk)),
                "chunk_elbo_mean": float(np.mean(chunk)),
                "previous_chunk_relative_change": relative_improvement,
                "q_mu_max_abs_update": float(np.max(np.abs(variational_snapshot(model)[0] - before_mu))),
                "q_sqrt_max_abs_update": float(np.max(np.abs(variational_snapshot(model)[1] - before_sqrt))),
            }
        )
        if checkpoint_directory is not None:
            trace[-1]["checkpoint"] = str(write_checkpoint(model, checkpoint_directory, completed))
        window = int(args.task1_plateau_checks)
        if completed >= int(args.task1_min_steps) and len(trace) >= 2 * window:
            previous = float(np.median([row["chunk_elbo_median"] for row in trace[-2 * window : -window]]))
            current = float(np.median([row["chunk_elbo_median"] for row in trace[-window:]]))
            plateau_change = abs(current - previous) / max(abs(previous), 1e-12)
            trace[-1]["moving_median_relative_change"] = plateau_change
            if plateau_change < float(args.task1_plateau_relative_improvement):
                stable_checks += 1
            else:
                stable_checks = 0
            if stable_checks >= 1:
                status = "converged_elbo_plateau"
                break
    summary: dict[str, Any] = {
        "status": status,
        "steps_completed": completed,
        "max_steps": int(args.task1_iterations),
        "check_interval": int(args.task1_check_interval),
        "minimum_steps": int(args.task1_min_steps),
        "moving_median_checks": int(args.task1_plateau_checks),
        "relative_improvement_threshold": float(args.task1_plateau_relative_improvement),
        "natural_gradient": not args.no_natgrad,
        "final_elbo": float(objectives[-1]),
        "trace": trace,
        "lmc_symmetry_audit": lmc_audit,
    }
    return model, summary


def update_variational_posterior(
    protocol: COVIDSettingBProtocol,
    args: argparse.Namespace,
    model: object,
    times: np.ndarray,
    targets: np.ndarray,
    inducing: np.ndarray,
) -> dict[str, float]:
    del protocol, inducing
    set_trainable(model.kernel, False)
    set_trainable(model.likelihood, False)
    set_trainable(model.inducing_variable, False)
    model.num_data = int(times.shape[0])
    before_mu, before_sqrt = variational_snapshot(model)
    objectives = run_variational_steps(
        method=args.method,
        model=model,
        times=times,
        targets=targets,
        batch_size=args.batch_size,
        steps=args.online_inference_steps,
        natgrad=not args.no_natgrad,
        learning_rate=1e-3,
        gamma=1e-2,
    )
    after_mu, after_sqrt = variational_snapshot(model)
    if not objectives or not np.isfinite(objectives).all():
        raise FloatingPointError("Online Factorial posterior update did not produce finite ELBO values")
    return {
        "online_elbo_final": float(objectives[-1]),
        "q_mu_max_abs_update": float(np.max(np.abs(after_mu - before_mu))),
        "q_sqrt_max_abs_update": float(np.max(np.abs(after_sqrt - before_sqrt))),
    }


def joint_predictive_distribution(model: object, time_value: float) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray([[time_value]], dtype=np.float64)
    mean, covariance = model.predict_f(query, full_cov=False, full_output_cov=True)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    covariance = np.asarray(covariance, dtype=np.float64).reshape(mean.size, mean.size)
    noise = float(np.asarray(model.likelihood.variance.numpy(), dtype=np.float64))
    covariance = 0.5 * (covariance + covariance.T) + noise * np.eye(mean.size)
    return mean, covariance


def condition_current_visible(
    mean: np.ndarray,
    covariance: np.ndarray,
    visible: np.ndarray,
    visible_targets: np.ndarray,
    hidden: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    visible_covariance = covariance[np.ix_(visible, visible)]
    cross_covariance = covariance[np.ix_(hidden, visible)]
    jitter = 1e-8 * max(1.0, float(np.max(np.diag(visible_covariance))))
    factor = np.linalg.cholesky(visible_covariance + jitter * np.eye(visible.size))
    residual = np.asarray(visible_targets, dtype=np.float64) - mean[visible]
    solve_residual = np.linalg.solve(factor.T, np.linalg.solve(factor, residual))
    solve_cross = np.linalg.solve(factor.T, np.linalg.solve(factor, cross_covariance.T))
    conditional_mean = mean[hidden] + cross_covariance @ solve_residual
    conditional_covariance = covariance[np.ix_(hidden, hidden)] - cross_covariance @ solve_cross
    conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
    return conditional_mean, np.maximum(np.diag(conditional_covariance), 1e-10)


def gaussian_metrics(target: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-10)
    error = np.asarray(target, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    half_width = 1.6448536269514722 * np.sqrt(variance)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "gaussian_nlpd": float(np.mean(0.5 * (np.log(2.0 * np.pi * variance) + error**2 / variance))),
        "coverage90": float(np.mean(np.abs(error) <= half_width)),
    }


def run_online(
    protocol: COVIDSettingBProtocol,
    args: argparse.Namespace,
    task1_model: object,
    inducing: np.ndarray,
    calibration_times: np.ndarray,
    calibration_targets: np.ndarray,
    evaluation_weeks: int,
    stream_targets: np.ndarray | None = None,
    stream_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, PredictionArchive | None, list[float], list[dict[str, float]]]:
    history_times = [float(value) for value in calibration_times]
    history_targets = [row.copy() for row in calibration_targets]
    previous_visible: np.ndarray | None = None
    means, variances, seconds, posterior_updates = [], [], [], []
    archive = None if stream_targets is not None else PredictionArchive(protocol, method=f"{args.method.upper()}-SVGP", seed=args.seed)
    for week in range(evaluation_weeks):
        information = protocol.week(week) if stream_targets is None else None
        if stream_targets is None:
            assert information is not None
            current_time = information.hidden_query.time
            visible_targets = information.current_visible.targets
            hidden_targets = None
            if previous_visible is not None:
                assert information.delayed_hidden is not None
                completed = np.empty(protocol.locations, dtype=np.float64)
                completed[protocol.visible_locations] = previous_visible
                completed[protocol.hidden_locations] = information.delayed_hidden.targets
                history_times.append(information.delayed_hidden.time)
                history_targets.append(completed)
            visible = protocol.visible_locations
            hidden = protocol.hidden_locations
        else:
            assert stream_times is not None
            current_time = float(stream_times[week])
            visible = protocol.visible_locations
            hidden = protocol.hidden_locations
            visible_targets = stream_targets[week, visible]
            hidden_targets = stream_targets[week, hidden]
            if previous_visible is not None:
                completed = np.empty(protocol.locations, dtype=np.float64)
                completed[visible] = previous_visible
                completed[hidden] = stream_targets[week - 1, hidden]
                history_times.append(float(stream_times[week - 1]))
                history_targets.append(completed)
        started = time.perf_counter()
        if previous_visible is not None:
            posterior_updates.append(update_variational_posterior(
                protocol,
                args,
                task1_model,
                np.asarray(history_times, dtype=np.float64)[:, None],
                np.asarray(history_targets, dtype=np.float64),
                inducing,
            ))
        else:
            posterior_updates.append(
                {
                    "online_elbo_final": None,
                    "q_mu_max_abs_update": 0.0,
                    "q_sqrt_max_abs_update": 0.0,
                    "reason": "Task-1 posterior is the initial online state; no new complete week has arrived",
                }
            )
        mean, covariance = joint_predictive_distribution(task1_model, current_time)
        prediction, variance = condition_current_visible(mean, covariance, visible, visible_targets, hidden)
        seconds.append(time.perf_counter() - started)
        means.append(prediction)
        variances.append(variance)
        if archive is not None:
            archive.append(information, prediction, variance)
        previous_visible = np.asarray(visible_targets, dtype=np.float64).copy()
    return np.stack(means), np.stack(variances), archive, seconds, posterior_updates


def main() -> None:
    args = parse_args()
    protocol = COVIDSettingBProtocol(args.protocol_npz, args.protocol_json)
    if not 1 <= args.temporal_inducing <= protocol.calibration_weeks + protocol.online_weeks:
        raise ValueError("--temporal-inducing is outside the legal study horizon")
    if not 1 <= args.latent_rank <= protocol.locations:
        raise ValueError("--latent-rank must be between 1 and 52")
    if not 1 <= args.validation_weeks < protocol.calibration_weeks:
        raise ValueError("--validation-weeks must be between 1 and 51")
    inducing = make_inducing_grid(protocol, args.temporal_inducing)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.task1_validation_only:
        train_weeks = protocol.calibration_weeks - args.validation_weeks
        task1 = protocol.task1()
        started = time.perf_counter()
        model, convergence = fit_task1(
            protocol,
            args,
            protocol.calibration_times[:train_weeks, None],
            task1.targets[:train_weeks],
            inducing,
            args.output_dir / "task1_checkpoints",
        )
        means, variances, _, seconds, posterior_updates = run_online(
            protocol,
            args,
            model,
            inducing,
            protocol.calibration_times[:train_weeks],
            task1.targets[:train_weeks],
            args.validation_weeks,
            stream_targets=task1.targets[train_weeks:],
            stream_times=protocol.calibration_times[train_weeks:],
        )
        metrics = gaussian_metrics(task1.targets[train_weeks:, protocol.hidden_locations], means, variances)
        result = {
            "status": "task1_chronological_validation_complete",
            "method": f"{args.method.upper()}-SVGP",
            "source": "SeyoungKimLab/FactorialSDE",
            "protocol": "Task-1-only chronological 48-week history plus four held-out weekly Setting B updates",
            "seed": args.seed,
            "capacity": {"temporal_inducing": args.temporal_inducing, "latent_rank": args.latent_rank},
            "task1_convergence": convergence,
            "online_posterior_updates": posterior_updates,
            "task1_seconds": time.perf_counter() - started - float(np.sum(seconds)),
            "online_seconds_total": float(np.sum(seconds)),
            "metrics": metrics,
        }
        (args.output_dir / "task1_chronological_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    task1 = protocol.task1()
    task1_started = time.perf_counter()
    model, convergence = fit_task1(
        protocol,
        args,
        protocol.calibration_times[:, None],
        task1.targets,
        inducing,
        args.output_dir / "task1_checkpoints",
    )
    task1_seconds = time.perf_counter() - task1_started
    weeks = protocol.online_weeks if args.max_weeks <= 0 else int(args.max_weeks)
    if not 1 <= weeks <= protocol.online_weeks:
        raise ValueError("--max-weeks must be between 1 and 143")
    means, variances, archive, seconds, posterior_updates = run_online(
        protocol,
        args,
        model,
        inducing,
        protocol.calibration_times,
        task1.targets,
        weeks,
    )
    assert archive is not None
    audit = archive.write(
        args.output_dir / "predictions.npz",
        require_complete=weeks == protocol.online_weeks,
        extra_metadata={
            "source": "SeyoungKimLab/FactorialSDE",
            "method_variant": args.method,
            "temporal_inducing": args.temporal_inducing,
            "latent_rank": args.latent_rank,
            "task1_iterations": args.task1_iterations,
            "online_inference_steps": args.online_inference_steps,
            "task1_convergence": convergence,
            "online_posterior_updates": posterior_updates,
            "posterior_transfer": "single q_t carried across all legal online updates",
            "natural_gradient": not args.no_natgrad,
            "current_visible_update": "analytic Gaussian conditioning of the official full-output predictive posterior",
            "history_update": "causal refit on complete labels that have arrived by the current week",
        },
    )
    result = {
        "status": "complete" if weeks == protocol.online_weeks else "smoke_complete",
        "method": f"{args.method.upper()}-SVGP",
        "source": "SeyoungKimLab/FactorialSDE",
        "seed": args.seed,
        "capacity": {"temporal_inducing": args.temporal_inducing, "latent_rank": args.latent_rank},
        "task1_convergence": convergence,
        "task1_seconds": task1_seconds,
        "online_seconds_total": float(np.sum(seconds)),
        "online_seconds_per_week": float(np.mean(seconds)),
        "online_update_prediction_seconds": [float(value) for value in seconds],
        "audit": audit,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
