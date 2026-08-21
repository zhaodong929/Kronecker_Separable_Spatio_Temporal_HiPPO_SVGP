#!/usr/bin/env python3
"""Causal Setting B adapter for the official Online Vargp OVC-SVGP source.

Run this file with a compatible OVC environment.  Task 1 trains the authors'
``SingleTaskVariationalGP`` once.  New labels are incorporated with the
official ``get_fantasy_model`` API in the prescribed delayed-then-visible
order.  OVC represents the conditioned SVGP as an exact fantasy GP, so its
state grows with the stream; it is reported separately from fixed-state online
methods in runtime comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Tuple

import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import VariationalELBO


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.archive import PredictionArchive
from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol, KnownObservation
from volatilitygp.models import SingleTaskVariationalGP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-weeks", type=int, default=0)
    parser.add_argument("--inducing-points", type=int, default=256)
    parser.add_argument("--temporal-inducing", type=int)
    parser.add_argument("--spatial-inducing", type=int)
    parser.add_argument("--task1-iterations", type=int, default=50000)
    parser.add_argument("--task1-check-interval", type=int, default=250)
    parser.add_argument("--task1-min-steps", type=int, default=2500)
    parser.add_argument("--task1-plateau-checks", type=int, default=10)
    parser.add_argument("--task1-plateau-relative-improvement", type=float, default=1e-3)
    parser.add_argument("--task1-learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Numerical precision for the unchanged official GPyTorch model.",
    )
    parser.add_argument(
        "--task1-validation-only",
        action="store_true",
        help="Fit only the 38 Task-1 fitting jurisdictions and score the four validation jurisdictions.",
    )
    return parser.parse_args()


def flatten_task1(
    protocol: COVIDSettingBProtocol,
    locations: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    locations = np.arange(protocol.locations, dtype=np.int64) if locations is None else np.asarray(
        locations, dtype=np.int64
    )
    times = np.repeat(protocol.calibration_times, protocol.locations)
    times = np.repeat(protocol.calibration_times, locations.size)
    tiled_locations = np.tile(locations, protocol.calibration_weeks)
    targets = protocol.calibration_targets(locations).reshape(-1)
    return np.column_stack([times, protocol.coordinates[tiled_locations]]), targets


def observation_inputs(
    protocol: COVIDSettingBProtocol,
    observation: KnownObservation,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = np.column_stack(
        [
            np.full(observation.locations.size, observation.time, dtype=np.float64),
            protocol.coordinates[observation.locations],
        ]
    )
    return (
        torch.as_tensor(inputs, dtype=dtype, device=device),
        torch.as_tensor(observation.targets, dtype=dtype, device=device),
    )


def select_inducing_points(
    protocol: COVIDSettingBProtocol,
    *,
    count: int,
    temporal_inducing: int | None,
    spatial_inducing: int | None,
    seed: int,
) -> np.ndarray:
    if (temporal_inducing is None) != (spatial_inducing is None):
        raise ValueError("--temporal-inducing and --spatial-inducing must be supplied together")
    if temporal_inducing is not None and spatial_inducing is not None:
        temporal_count = int(temporal_inducing)
        spatial_count = int(spatial_inducing)
        if temporal_count < 1 or spatial_count < 1:
            raise ValueError("Temporal and spatial inducing counts must be positive")
        spatial = protocol.spatial_inducing_locations(spatial_count)
        timeline = np.concatenate([protocol.calibration_times, protocol.chronological_stream_times])
        temporal = np.linspace(float(timeline[0]), float(timeline[-1]), temporal_count)
        return np.column_stack(
            [np.repeat(temporal, spatial_count), np.tile(spatial, (temporal_count, 1))]
        )
    if not 1 <= count <= protocol.calibration_weeks * protocol.locations:
        raise ValueError("--inducing-points must be positive and no larger than the Task-1 observation count")
    temporal_count = max(1, int(np.ceil(float(count) / 32.0)))
    spatial_count = min(32, count)
    del seed
    spatial = protocol.spatial_inducing_locations(spatial_count)
    temporal = np.linspace(
        protocol.calibration_times[0], protocol.calibration_times[-1], temporal_count
    )
    grid = np.column_stack(
        [
            np.repeat(temporal, spatial_count),
            np.tile(spatial, (temporal_count, 1)),
        ]
    )
    return grid[:count]


def train_task1(
    model: object,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    iterations: int,
    check_interval: int,
    min_steps: int,
    plateau_checks: int,
    plateau_relative_improvement: float,
    lr: float,
    checkpoint_directory: Path | None,
) -> dict[str, object]:
    """Optimise Task 1 to a predeclared objective plateau and save checks."""

    model.train()
    model.likelihood.train()
    optimizer_kwargs: dict[str, object] = {"lr": lr}
    if train_x.is_cuda:
        # The legacy GPyTorch stack is incompatible with grouped CUDA Adam here.
        optimizer_kwargs["foreach"] = False
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    objective = VariationalELBO(model.likelihood, model, num_data=train_y.numel())
    trace: list[dict[str, object]] = []
    completed = 0
    status = "max_budget_not_converged"
    while completed < int(iterations):
        steps = min(int(check_interval), int(iterations) - completed)
        losses: list[float] = []
        for _ in range(steps):
            optimizer.zero_grad()
            loss = -objective(model(train_x), train_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("OVC-SVGP Task-1 objective became non-finite")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        completed += steps
        row: dict[str, object] = {
            "steps_completed": completed,
            "chunk_objective_median": float(np.median(losses)),
            "chunk_objective_mean": float(np.mean(losses)),
        }
        if checkpoint_directory is not None:
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_directory / f"checkpoint_{completed:05d}.pt"
            torch.save(
                {
                    "step": completed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
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


def condition(model: object, x: torch.Tensor, y: torch.Tensor) -> object:
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        if hasattr(model, "get_fantasy_model"):
            return model.get_fantasy_model(x, y)
        return model.condition_on_observations(x, y)


def predict_locations(
    model: object,
    protocol: COVIDSettingBProtocol,
    time_value: float,
    locations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    query = np.column_stack(
        [np.full(locations.size, time_value, dtype=np.float64), protocol.coordinates[locations]]
    )
    parameter = next(model.parameters())
    x = torch.as_tensor(query, dtype=parameter.dtype, device=parameter.device)
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        predictive = model.likelihood(model(x))
    return (
        predictive.mean.detach().cpu().numpy(),
        predictive.variance.detach().cpu().numpy(),
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


def main() -> None:
    args = parse_args()
    protocol = COVIDSettingBProtocol(args.protocol_npz, args.protocol_json)
    requested_weeks = protocol.online_weeks if args.max_weeks <= 0 else int(args.max_weeks)
    if not 1 <= requested_weeks <= protocol.online_weeks:
        raise ValueError("--max-weeks must be between 1 and the full online horizon")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = torch.device(args.device)
    torch.set_default_dtype(dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    task1_locations = protocol.fit_locations if args.task1_validation_only else None
    train_x_np, train_y_np = flatten_task1(protocol, task1_locations)
    train_x = torch.as_tensor(train_x_np, dtype=dtype, device=device)
    train_y = torch.as_tensor(train_y_np, dtype=dtype, device=device)
    inducing = torch.as_tensor(
        select_inducing_points(
            protocol,
            count=args.inducing_points,
            temporal_inducing=args.temporal_inducing,
            spatial_inducing=args.spatial_inducing,
            seed=args.seed,
        ),
        dtype=dtype,
        device=device,
    ).to(device)
    covariance = ScaleKernel(RBFKernel(ard_num_dims=3))
    task1_started = time.perf_counter()
    model = SingleTaskVariationalGP(
        init_points=inducing,
        train_inputs=train_x,
        train_targets=train_y,
        covar_module=covariance,
        use_piv_chol_init=False,
        use_whitened_var_strat=True,
    )
    convergence = train_task1(
        model,
        train_x,
        train_y,
        iterations=args.task1_iterations,
        check_interval=args.task1_check_interval,
        min_steps=args.task1_min_steps,
        plateau_checks=args.task1_plateau_checks,
        plateau_relative_improvement=args.task1_plateau_relative_improvement,
        lr=args.task1_learning_rate,
        checkpoint_directory=args.output_dir / "task1_checkpoints",
    )
    task1_seconds = time.perf_counter() - task1_started

    grid_capacity = {
        "inducing_points": int(inducing.shape[0]),
        "temporal_inducing": None if args.temporal_inducing is None else int(args.temporal_inducing),
        "spatial_inducing": None if args.spatial_inducing is None else int(args.spatial_inducing),
    }
    if args.task1_validation_only:
        validation_locations = protocol.validation_locations
        means, variances = [], []
        for time_value in protocol.calibration_times:
            mean, variance = predict_locations(
                model, protocol, float(time_value), validation_locations
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
            "method": "OVC-SVGP",
            "source": "wjmaddox/online_vargp",
            "source_commit": "7bd3da50eac32d70ca323309e3f3d80a2ae7c419",
            "protocol": "Task-1-only 38-fit/4-validation spatial split",
            "seed": args.seed,
            "capacity": grid_capacity,
            "dtype": args.dtype,
            "execution_device": args.device,
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

    archive = PredictionArchive(protocol, method="ovc_svgp_exact_fantasy", seed=args.seed)
    condition_seconds = []
    for week in range(requested_weeks):
        information = protocol.week(week)
        started = time.perf_counter()
        if information.delayed_hidden is not None:
            x_delayed, y_delayed = observation_inputs(protocol, information.delayed_hidden, dtype, device)
            model = condition(model, x_delayed, y_delayed).to(device)
        x_visible, y_visible = observation_inputs(protocol, information.current_visible, dtype, device)
        model = condition(model, x_visible, y_visible).to(device)
        mean, variance = predict_locations(
            model,
            protocol,
            information.hidden_query.time,
            protocol.hidden_locations,
        )
        condition_seconds.append(time.perf_counter() - started)
        archive.append(information, mean, variance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = archive.write(
        args.output_dir / "predictions.npz",
        require_complete=requested_weeks == protocol.online_weeks,
        extra_metadata={
            "adapter": "official_wjmaddox_ovc_svgp_exact_fantasy",
            "source_commit": "7bd3da50eac32d70ca323309e3f3d80a2ae7c419",
            "task1_iterations": int(args.task1_iterations),
            "task1_convergence": convergence,
            "dtype": args.dtype,
            "execution_device": args.device,
            **grid_capacity,
        },
    )
    status = {
        "status": "complete",
        "method": "OVC-SVGP exact fantasy continuation",
        "source": "wjmaddox/online_vargp",
        "source_commit": "7bd3da50eac32d70ca323309e3f3d80a2ae7c419",
        "protocol": "covid_long_setting_b",
        "seed": args.seed,
        "weeks": requested_weeks,
        "dtype": args.dtype,
        "execution_device": args.device,
        "task1_seconds": task1_seconds,
        "task1_convergence": convergence,
        "conditioning_seconds_total": float(np.sum(condition_seconds)),
        "conditioning_seconds_per_week": float(np.mean(condition_seconds)),
        "online_update_prediction_seconds": [float(value) for value in condition_seconds],
        "audit": audit,
        "note": (
            "The official OVC API returns an exact fantasy continuation around the Task-1 "
            "SVGP posterior. Its state therefore grows with cumulative observations."
        ),
    }
    (args.output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
