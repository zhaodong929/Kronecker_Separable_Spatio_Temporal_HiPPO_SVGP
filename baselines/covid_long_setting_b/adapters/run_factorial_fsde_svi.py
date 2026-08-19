#!/usr/bin/env python3
"""Causal Setting B adapter for the official FactorialSDE FSDE-SVI model.

The official model trains on complete historical output vectors.  This is
compatible with the delayed protocol because every vector added to its history
has fully arrived.  At the current week, the adapter conditions the official
52-output Gaussian predictive distribution on the legal 42 visible labels and
scores only the remaining ten locations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = ROOT / "baselines/external/SeyoungKimLab_FactorialSDE"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL_ROOT))

from fsde.core.model_utils import Dataset, init_params
from fsde.models import FSDE_SVI

from baselines.covid_long_setting_b.archive import PredictionArchive
from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol


LEARNING_RATES = {
    "model_lr": 1e-3,
    "var_adam_lr": 1e-3,
    "var_lr_init": 1e-5,
    "var_lr_end": 1e-4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--max-weeks", type=int, default=0)
    parser.add_argument("--task1-validation-only", action="store_true")
    parser.add_argument("--validation-weeks", type=int, default=4)
    return parser.parse_args()


def clone_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), tree)


def tree_max_abs_delta(before: Any, after: Any) -> float:
    """Return a small, implementation-agnostic posterior state-change audit."""

    before_leaves, before_definition = jax.tree_util.tree_flatten(before)
    after_leaves, after_definition = jax.tree_util.tree_flatten(after)
    if before_definition != after_definition:
        raise ValueError("FSDE parameter tree structure changed unexpectedly")
    if not before_leaves:
        return 0.0
    return float(
        max(
            np.max(np.abs(np.asarray(after_leaf) - np.asarray(before_leaf)))
            for before_leaf, after_leaf in zip(before_leaves, after_leaves)
        )
    )


def write_checkpoint(
    directory: Path,
    *,
    step: int,
    model_params: Any,
    variational_params: Any,
) -> Path:
    """Store every convergence-check state without relying on pickle."""

    directory.mkdir(parents=True, exist_ok=True)
    model_leaves, model_definition = jax.tree_util.tree_flatten(model_params)
    variational_leaves, variational_definition = jax.tree_util.tree_flatten(variational_params)
    payload: dict[str, np.ndarray] = {}
    for index, value in enumerate(model_leaves):
        payload[f"model_{index:04d}"] = np.asarray(value)
    for index, value in enumerate(variational_leaves):
        payload[f"variational_{index:04d}"] = np.asarray(value)
    path = directory / f"checkpoint_{int(step):05d}.npz"
    np.savez_compressed(path, **payload)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "step": int(step),
                "model_tree": str(model_definition),
                "variational_tree": str(variational_definition),
                "model_leaf_count": len(model_leaves),
                "variational_leaf_count": len(variational_leaves),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def make_inducing_grid(protocol: COVIDSettingBProtocol, count: int) -> jnp.ndarray:
    times = np.concatenate([protocol.calibration_times, protocol.chronological_stream_times])
    return jnp.linspace(float(times.min()), float(times.max()), int(count))


def make_model(
    *,
    protocol: COVIDSettingBProtocol,
    args: argparse.Namespace,
    inducing: jnp.ndarray,
    num_times: int,
    model_params: Any | None = None,
    variational_params: Any | None = None,
) -> FSDE_SVI:
    if model_params is None or variational_params is None:
        model_params, variational_params, cov_fn, transition_fn = init_params(
            kernel="Matern32",
            L=int(args.latent_rank),
            P=protocol.locations,
            M=int(args.temporal_inducing),
            var=0.1,
            lengthscale=1.0,
            key=jr.PRNGKey(args.seed),
        )
    else:
        _, _, cov_fn, transition_fn = init_params(
            kernel="Matern32",
            L=int(args.latent_rank),
            P=protocol.locations,
            M=int(args.temporal_inducing),
            var=0.1,
            lengthscale=1.0,
            key=jr.PRNGKey(args.seed),
        )
    return FSDE_SVI(
        model_params=clone_tree(model_params),
        v_params=clone_tree(variational_params),
        kernel="Matern32",
        jitter=1e-8,
        ind_times=inducing,
        compute_cov_infty=cov_fn,
        compute_F=transition_fn,
        num_times=int(num_times),
    )


def fit_task1(
    protocol: COVIDSettingBProtocol,
    args: argparse.Namespace,
    inducing: jnp.ndarray,
    times: np.ndarray,
    targets: np.ndarray,
    checkpoint_directory: Path | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    model = make_model(
        protocol=protocol,
        args=args,
        inducing=inducing,
        num_times=times.size,
    )
    objectives: list[float] = []
    trace: list[dict[str, Any]] = []
    completed = 0
    status = "max_budget_not_converged"
    while completed < int(args.task1_iterations):
        steps = min(int(args.task1_check_interval), int(args.task1_iterations) - completed)
        before_variational = clone_tree(model.v_params)
        metrics = model.fit(
            train_dataset=Dataset(times=jnp.asarray(times), Y=jnp.asarray(targets.T)),
            n_steps=steps,
            batch_size=min(int(args.batch_size), int(times.size)),
            key=jr.PRNGKey(int(args.seed) + completed),
            lr=LEARNING_RATES,
            lr_steps=max(1, int(args.task1_iterations)),
            use_natgrad=True,
        )
        chunk = np.asarray(metrics[0], dtype=np.float64).reshape(-1)
        if chunk.size != steps or not np.isfinite(chunk).all():
            raise FloatingPointError("Official FSDE-SVI fit returned an invalid ELBO trace")
        completed += steps
        objectives.extend(float(value) for value in chunk)
        row: dict[str, Any] = {
            "steps_completed": completed,
            "chunk_elbo_median": float(np.median(chunk)),
            "chunk_elbo_mean": float(np.mean(chunk)),
            "variational_max_abs_update": tree_max_abs_delta(before_variational, model.v_params),
        }
        if checkpoint_directory is not None:
            row["checkpoint"] = str(
                write_checkpoint(
                    checkpoint_directory,
                    step=completed,
                    model_params=model.model_params,
                    variational_params=model.v_params,
                )
            )
        window = int(args.task1_plateau_checks)
        if completed >= int(args.task1_min_steps) and len(trace) >= 2 * window - 1:
            combined_trace = trace + [row]
            prior = float(np.median([entry["chunk_elbo_median"] for entry in combined_trace[-2 * window : -window]]))
            current = float(np.median([entry["chunk_elbo_median"] for entry in combined_trace[-window:]]))
            plateau_change = abs(current - prior) / max(abs(prior), 1e-12)
            row["moving_median_relative_change"] = plateau_change
            if plateau_change < float(args.task1_plateau_relative_improvement):
                status = "converged_elbo_plateau"
                trace.append(row)
                break
        trace.append(row)
    convergence = {
        "status": status,
        "steps_completed": completed,
        "max_steps": int(args.task1_iterations),
        "check_interval": int(args.task1_check_interval),
        "minimum_steps": int(args.task1_min_steps),
        "moving_median_checks": int(args.task1_plateau_checks),
        "relative_improvement_threshold": float(args.task1_plateau_relative_improvement),
        "natural_gradient": True,
        "final_elbo": float(objectives[-1]),
        "trace": trace,
    }
    return clone_tree(model.model_params), clone_tree(model.v_params), convergence


def update_variational_posterior(
    protocol: COVIDSettingBProtocol,
    args: argparse.Namespace,
    inducing: jnp.ndarray,
    frozen_model_params: Any,
    variational_params: Any,
    times: np.ndarray,
    targets: np.ndarray,
    update_key: int,
) -> tuple[FSDE_SVI, Any]:
    """Update only q while retaining the Task-1 parameters.

    The upstream fit call jointly proposes a model-parameter and variational
    update.  For a one-step call both gradients are evaluated at the frozen
    Task-1 parameters; the proposed model update is discarded, while the
    official variational update is retained.  Reconstructing between steps
    preserves the frozen-hyperparameter protocol.
    """

    current = clone_tree(variational_params)
    model = make_model(
        protocol=protocol,
        args=args,
        inducing=inducing,
        num_times=times.size,
        model_params=frozen_model_params,
        variational_params=current,
    )
    before_model = clone_tree(model.model_params)
    metrics = model.fit(
        train_dataset=Dataset(times=jnp.asarray(times), Y=jnp.asarray(targets.T)),
        n_steps=int(args.online_inference_steps),
        batch_size=min(int(args.batch_size), int(times.size)),
        key=jr.PRNGKey(int(update_key)),
        lr=LEARNING_RATES,
        lr_steps=max(1, int(args.online_inference_steps)),
        use_natgrad=True,
    )
    elbo = np.asarray(metrics[0], dtype=np.float64).reshape(-1)
    if elbo.size != int(args.online_inference_steps) or not np.isfinite(elbo).all():
        raise FloatingPointError("Official FSDE-SVI online update returned an invalid ELBO trace")
    updated_variational = clone_tree(model.v_params)
    model_parameter_proposal_delta = tree_max_abs_delta(before_model, model.model_params)
    final_model = make_model(
        protocol=protocol,
        args=args,
        inducing=inducing,
        num_times=times.size,
        model_params=frozen_model_params,
        variational_params=updated_variational,
    )
    return final_model, updated_variational, {
        "online_elbo_final": float(elbo[-1]),
        "variational_max_abs_update": tree_max_abs_delta(current, updated_variational),
        "discarded_model_parameter_proposal_max_abs_update": model_parameter_proposal_delta,
        "frozen_model_parameters_retained": bool(
            tree_max_abs_delta(frozen_model_params, final_model.model_params) == 0.0
        ),
    }


def joint_predictive_distribution(model: FSDE_SVI, time_value: float) -> tuple[np.ndarray, np.ndarray]:
    # The upstream predict method squeezes a singleton time input internally.
    mean, covariance = model.predict(jnp.asarray([time_value, time_value]), *model.precompute_pred_args())
    mean = np.asarray(mean, dtype=np.float64)[:, -1]
    covariance = np.asarray(covariance, dtype=np.float64)[-1]
    covariance = 0.5 * (covariance + covariance.T)
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
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "gaussian_nlpd": float(
            np.mean(0.5 * (np.log(2.0 * np.pi * variance) + error**2 / variance))
        ),
        "coverage90": float(np.mean(np.abs(error) <= 1.6448536269514722 * np.sqrt(variance))),
    }


def run_online(
    protocol: COVIDSettingBProtocol,
    args: argparse.Namespace,
    inducing: jnp.ndarray,
    frozen_model_params: Any,
    task1_variational_params: Any,
    calibration_times: np.ndarray,
    calibration_targets: np.ndarray,
    evaluation_weeks: int,
    stream_targets: np.ndarray | None = None,
    stream_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, PredictionArchive | None, list[float], list[dict[str, Any]]]:
    history_times = [float(value) for value in calibration_times]
    history_targets = [row.copy() for row in calibration_targets]
    current_variational_params = clone_tree(task1_variational_params)
    previous_visible: np.ndarray | None = None
    means, variances, seconds, posterior_updates = [], [], [], []
    archive = None if stream_targets is not None else PredictionArchive(protocol, method="FSDE-SVI", seed=args.seed)

    for week in range(evaluation_weeks):
        information = protocol.week(week) if stream_targets is None else None
        if stream_targets is None:
            assert information is not None
            current_time = information.hidden_query.time
            visible_targets = information.current_visible.targets
            visible = protocol.visible_locations
            hidden = protocol.hidden_locations
            if previous_visible is not None:
                assert information.delayed_hidden is not None
                completed = np.empty(protocol.locations, dtype=np.float64)
                completed[visible] = previous_visible
                completed[hidden] = information.delayed_hidden.targets
                history_times.append(information.delayed_hidden.time)
                history_targets.append(completed)
        else:
            assert stream_times is not None
            current_time = float(stream_times[week])
            visible = protocol.visible_locations
            hidden = protocol.hidden_locations
            visible_targets = stream_targets[week, visible]
            if previous_visible is not None:
                completed = np.empty(protocol.locations, dtype=np.float64)
                completed[visible] = previous_visible
                completed[hidden] = stream_targets[week - 1, hidden]
                history_times.append(float(stream_times[week - 1]))
                history_targets.append(completed)

        started = time.perf_counter()
        if previous_visible is None:
            model = make_model(
                protocol=protocol,
                args=args,
                inducing=inducing,
                num_times=len(history_times),
                model_params=frozen_model_params,
                variational_params=current_variational_params,
            )
        else:
            model, current_variational_params, posterior_update = update_variational_posterior(
                protocol,
                args,
                inducing,
                frozen_model_params,
                current_variational_params,
                np.asarray(history_times, dtype=np.float64),
                np.asarray(history_targets, dtype=np.float64),
                update_key=args.seed * 1_000_000 + week * 100,
            )
            posterior_updates.append(posterior_update)
        if previous_visible is None:
            posterior_updates.append(
                {
                    "online_elbo_final": None,
                    "variational_max_abs_update": 0.0,
                    "reason": "Task-1 posterior is the initial online state; no new complete week has arrived",
                }
            )
        mean, covariance = joint_predictive_distribution(model, current_time)
        prediction, variance = condition_current_visible(
            mean, covariance, visible, visible_targets, hidden
        )
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
        raise ValueError("--temporal-inducing is outside the study horizon")
    if not 1 <= args.latent_rank <= protocol.locations:
        raise ValueError("--latent-rank must be between 1 and 52")
    if not 1 <= args.validation_weeks < protocol.calibration_weeks:
        raise ValueError("--validation-weeks must be between 1 and 51")
    inducing = make_inducing_grid(protocol, args.temporal_inducing)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.task1_validation_only:
        train_weeks = protocol.calibration_weeks - args.validation_weeks
        task1 = protocol.task1()
        task1_started = time.perf_counter()
        frozen_params, variational_params, convergence = fit_task1(
            protocol,
            args,
            inducing,
            protocol.calibration_times[:train_weeks],
            task1.targets[:train_weeks],
            args.output_dir / "task1_checkpoints",
        )
        means, variances, _, seconds, posterior_updates = run_online(
            protocol,
            args,
            inducing,
            frozen_params,
            variational_params,
            protocol.calibration_times[:train_weeks],
            task1.targets[:train_weeks],
            args.validation_weeks,
            stream_targets=task1.targets[train_weeks:],
            stream_times=protocol.calibration_times[train_weeks:],
        )
        record = {
            "status": "task1_chronological_validation_complete",
            "method": "FSDE-SVI",
            "source": "SeyoungKimLab/FactorialSDE",
            "protocol": "Task-1-only chronological 48-week history plus four held-out weekly Setting B updates",
            "seed": args.seed,
            "capacity": {"temporal_inducing": args.temporal_inducing, "latent_rank": args.latent_rank},
            "task1_convergence": convergence,
            "online_posterior_updates": posterior_updates,
            "task1_seconds": time.perf_counter() - task1_started - float(np.sum(seconds)),
            "online_seconds_total": float(np.sum(seconds)),
            "metrics": gaussian_metrics(task1.targets[train_weeks:, protocol.hidden_locations], means, variances),
        }
        (args.output_dir / "task1_chronological_validation.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(record, indent=2))
        return

    task1 = protocol.task1()
    task1_started = time.perf_counter()
    frozen_params, variational_params, convergence = fit_task1(
        protocol,
        args,
        inducing,
        protocol.calibration_times,
        task1.targets,
        args.output_dir / "task1_checkpoints",
    )
    task1_seconds = time.perf_counter() - task1_started
    weeks = protocol.online_weeks if args.max_weeks <= 0 else int(args.max_weeks)
    if not 1 <= weeks <= protocol.online_weeks:
        raise ValueError("--max-weeks must be between 1 and 143")
    means, variances, archive, seconds, posterior_updates = run_online(
        protocol,
        args,
        inducing,
        frozen_params,
        variational_params,
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
            "method_variant": "fsde_svi",
            "temporal_inducing": args.temporal_inducing,
            "latent_rank": args.latent_rank,
            "task1_iterations": args.task1_iterations,
            "online_inference_steps": args.online_inference_steps,
            "task1_convergence": convergence,
            "online_posterior_updates": posterior_updates,
            "current_visible_update": "analytic Gaussian conditioning of the official full-output predictive posterior",
            "history_update": "causal official FSDE-SVI variational update on complete arrived labels",
            "hyperparameters": "Task-1 parameters are frozen; each multi-step online FSDE update retains only the causal variational posterior",
        },
    )
    result = {
        "status": "complete" if weeks == protocol.online_weeks else "smoke_complete",
        "method": "FSDE-SVI",
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
