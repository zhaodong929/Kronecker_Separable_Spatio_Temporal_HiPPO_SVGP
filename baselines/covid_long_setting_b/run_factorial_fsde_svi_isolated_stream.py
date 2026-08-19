#!/usr/bin/env python3
"""Run the official FSDE-SVI adapter with one JAX process per online week.

FSDE-SVI is updated exactly as in ``run_factorial_fsde_svi.py``.  The only
change is operational: Task-1 parameters and the variational PyTree are stored
between weeks so the JAX executable cache is released when each worker exits.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.experimental.compilation_cache import compilation_cache


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = ROOT / "baselines/external/SeyoungKimLab_FactorialSDE"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL_ROOT))
COMPILATION_CACHE = Path(
    os.environ.get(
        "FSDE_SVI_COMPILATION_CACHE",
        str(ROOT / "baselines/covid_long_setting_b/results/fsde_svi_jax_compilation_cache"),
    )
)
compilation_cache.initialize_cache(str(COMPILATION_CACHE))

from fsde.core.model_utils import init_params

from baselines.covid_long_setting_b.adapters.run_factorial_fsde_svi import (
    LEARNING_RATES,
    condition_current_visible,
    fit_task1,
    joint_predictive_distribution,
    make_inducing_grid,
    make_model,
    update_variational_posterior,
)
from baselines.covid_long_setting_b.archive import PredictionArchive
from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temporal-inducing", type=int, default=4)
    parser.add_argument("--latent-rank", type=int, default=2)
    parser.add_argument("--task1-iterations", type=int, default=50)
    parser.add_argument("--online-inference-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-weeks", type=int, default=0)
    parser.add_argument("--prepare-task1-state", type=Path)
    parser.add_argument("--task1-state", type=Path)
    parser.add_argument("--previous-state", type=Path)
    parser.add_argument("--week", type=int)
    parser.add_argument("--week-output", type=Path)
    return parser.parse_args()


def leaf_payload(prefix: str, tree: Any) -> dict[str, np.ndarray]:
    leaves, _ = jax.tree_util.tree_flatten(tree)
    payload = {f"{prefix}_count": np.asarray([len(leaves)], dtype=np.int64)}
    payload.update({f"{prefix}_{index}": np.asarray(value) for index, value in enumerate(leaves)})
    return payload


def restore_tree(payload: Any, prefix: str, template: Any) -> Any:
    _, treedef = jax.tree_util.tree_flatten(template)
    count = int(np.asarray(payload[f"{prefix}_count"]).item())
    leaves = [jnp.asarray(payload[f"{prefix}_{index}"]) for index in range(count)]
    if count != treedef.num_leaves:
        raise ValueError(f"{prefix} state leaf count does not match the official FSDE parameter tree")
    return jax.tree_util.tree_unflatten(treedef, leaves)


def templates(protocol: COVIDSettingBProtocol, args: argparse.Namespace) -> tuple[Any, Any]:
    model_params, variational_params, _, _ = init_params(
        kernel="Matern32",
        L=int(args.latent_rank),
        P=protocol.locations,
        M=int(args.temporal_inducing),
        var=0.1,
        lengthscale=1.0,
        key=jr.PRNGKey(args.seed),
    )
    return model_params, variational_params


def write_task1_state(path: Path, model_params: Any, variational_params: Any, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "seed": np.asarray([args.seed], dtype=np.int64),
        "temporal_inducing": np.asarray([args.temporal_inducing], dtype=np.int64),
        "latent_rank": np.asarray([args.latent_rank], dtype=np.int64),
    }
    payload.update(leaf_payload("model", model_params))
    payload.update(leaf_payload("variational", variational_params))
    np.savez_compressed(path, **payload)


def read_task1_state(path: Path, protocol: COVIDSettingBProtocol, args: argparse.Namespace) -> tuple[Any, Any]:
    model_template, variational_template = templates(protocol, args)
    with np.load(path, allow_pickle=False) as payload:
        if (
            int(np.asarray(payload["seed"]).item()) != args.seed
            or int(np.asarray(payload["temporal_inducing"]).item()) != args.temporal_inducing
            or int(np.asarray(payload["latent_rank"]).item()) != args.latent_rank
        ):
            raise ValueError("The Task-1 FSDE state does not match the requested seed or capacity")
        return (
            restore_tree(payload, "model", model_template),
            restore_tree(payload, "variational", variational_template),
        )


def write_week_state(
    path: Path,
    *,
    week: int,
    target: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    variational_params: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "week": np.asarray([week], dtype=np.int64),
        "y_true": np.asarray(target, dtype=np.float64)[None, :],
        "pred_mean": np.asarray(mean, dtype=np.float64)[None, :],
        "pred_var": np.asarray(variance, dtype=np.float64)[None, :],
    }
    payload.update(leaf_payload("variational", variational_params))
    np.savez_compressed(path, **payload)


def read_week_variational(path: Path, week: int, template: Any) -> Any:
    with np.load(path, allow_pickle=False) as payload:
        if int(np.asarray(payload["week"]).item()) != week:
            raise ValueError(f"The prior FSDE state is not from expected week {week}")
        return restore_tree(payload, "variational", template)


def completed_history(protocol: COVIDSettingBProtocol, week: int) -> tuple[np.ndarray, np.ndarray]:
    task1 = protocol.task1()
    times = [float(value) for value in protocol.calibration_times]
    targets = [row.copy() for row in task1.targets]
    for previous_week in range(week):
        previous = protocol.week(previous_week)
        arrival = protocol.week(previous_week + 1).delayed_hidden
        if arrival is None:
            raise RuntimeError("A delayed hidden batch is missing from an arrived historical week")
        completed = np.empty(protocol.locations, dtype=np.float64)
        completed[protocol.visible_locations] = previous.current_visible.targets
        completed[protocol.hidden_locations] = arrival.targets
        times.append(float(previous.hidden_query.time))
        targets.append(completed)
    return np.asarray(times, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def prepare_task1(protocol: COVIDSettingBProtocol, args: argparse.Namespace) -> None:
    if args.prepare_task1_state is None:
        return
    inducing = make_inducing_grid(protocol, args.temporal_inducing)
    task1 = protocol.task1()
    model_params, variational_params = fit_task1(
        protocol, args, inducing, protocol.calibration_times, task1.targets
    )
    write_task1_state(args.prepare_task1_state, model_params, variational_params, args)


def run_one_week(protocol: COVIDSettingBProtocol, args: argparse.Namespace) -> None:
    if args.task1_state is None or args.week is None or args.week_output is None:
        raise ValueError("--task1-state, --week and --week-output are required for a weekly worker")
    if not 0 <= args.week < protocol.online_weeks:
        raise ValueError("--week is outside the online horizon")
    frozen_model_params, initial_variational_params = read_task1_state(args.task1_state, protocol, args)
    if args.week == 0:
        if args.previous_state is not None:
            raise ValueError("Week zero must start from the Task-1 variational posterior")
        variational_params = initial_variational_params
    else:
        if args.previous_state is None:
            raise ValueError("An online week after zero requires the previous variational state")
        variational_params = read_week_variational(
            args.previous_state, args.week - 1, initial_variational_params
        )
    times, targets = completed_history(protocol, args.week)
    inducing = make_inducing_grid(protocol, args.temporal_inducing)
    information = protocol.week(args.week)
    if args.week == 0:
        model = make_model(
            protocol=protocol,
            args=args,
            inducing=inducing,
            num_times=times.size,
            model_params=frozen_model_params,
            variational_params=variational_params,
        )
    else:
        model, variational_params = update_variational_posterior(
            protocol,
            args,
            inducing,
            frozen_model_params,
            variational_params,
            times,
            targets,
            update_key=args.seed * 1_000_000 + args.week * 100,
        )
    mean, covariance = joint_predictive_distribution(model, information.hidden_query.time)
    prediction, variance = condition_current_visible(
        mean,
        covariance,
        protocol.visible_locations,
        information.current_visible.targets,
        protocol.hidden_locations,
    )
    write_week_state(
        args.week_output,
        week=args.week,
        target=protocol.evaluation_targets()[args.week],
        mean=prediction,
        variance=variance,
        variational_params=variational_params,
    )


def worker_command(args: argparse.Namespace, state: Path, week: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--protocol-npz", str(args.protocol_npz), "--seed", str(args.seed),
        "--temporal-inducing", str(args.temporal_inducing), "--latent-rank", str(args.latent_rank),
        "--task1-iterations", str(args.task1_iterations),
        "--online-inference-steps", str(args.online_inference_steps), "--batch-size", str(args.batch_size),
        "--task1-state", str(state), "--week", str(week), "--week-output", str(output),
    ]
    if week:
        command.extend(["--previous-state", str(output.parent / f"week_{week - 1:03d}.npz")])
    return command


def valid_week(path: Path, week: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                int(np.asarray(payload["week"]).item()) == week
                and np.asarray(payload["y_true"]).shape == (1, 10)
                and np.asarray(payload["pred_mean"]).shape == (1, 10)
                and np.asarray(payload["pred_var"]).shape == (1, 10)
                and np.isfinite(np.asarray(payload["pred_mean"])).all()
                and np.isfinite(np.asarray(payload["pred_var"])).all()
                and (np.asarray(payload["pred_var"]) >= 0.0).all()
            )
    except (KeyError, OSError, ValueError):
        return False


def orchestrate(protocol: COVIDSettingBProtocol, args: argparse.Namespace) -> None:
    if args.output_dir is None:
        raise ValueError("--output-dir is required for stream orchestration")
    weeks = protocol.online_weeks if args.max_weeks <= 0 else int(args.max_weeks)
    if not 1 <= weeks <= protocol.online_weeks:
        raise ValueError("--max-weeks must be between 1 and the full online horizon")
    output = args.output_dir
    task1_state = output / "task1_state.npz"
    week_root = output / "weekly_states"
    started = time.perf_counter()
    if not task1_state.is_file():
        completed = subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve()), "--protocol-npz", str(args.protocol_npz),
                "--seed", str(args.seed), "--temporal-inducing", str(args.temporal_inducing),
                "--latent-rank", str(args.latent_rank), "--task1-iterations", str(args.task1_iterations),
                "--online-inference-steps", str(args.online_inference_steps), "--batch-size", str(args.batch_size),
                "--prepare-task1-state", str(task1_state),
            ],
            cwd=ROOT,
        )
        if completed.returncode or not task1_state.is_file():
            raise RuntimeError("The official FSDE Task-1 state preparation failed")
    worker_seconds: list[float] = []
    for week in range(weeks):
        week_output = week_root / f"week_{week:03d}.npz"
        if valid_week(week_output, week):
            continue
        worker_started = time.perf_counter()
        completed = subprocess.run(worker_command(args, task1_state, week, week_output), cwd=ROOT)
        if completed.returncode or not valid_week(week_output, week):
            raise RuntimeError(f"The isolated FSDE worker failed at week {week}")
        worker_seconds.append(time.perf_counter() - worker_started)

    archive = PredictionArchive(protocol, method="fsde_svi_isolated", seed=args.seed)
    expected_targets = protocol.evaluation_targets()
    for week in range(weeks):
        with np.load(week_root / f"week_{week:03d}.npz", allow_pickle=False) as payload:
            target = np.asarray(payload["y_true"], dtype=np.float64)[0]
            mean = np.asarray(payload["pred_mean"], dtype=np.float64)[0]
            variance = np.asarray(payload["pred_var"], dtype=np.float64)[0]
        if not np.array_equal(target, expected_targets[week]):
            raise RuntimeError(f"FSDE worker target disagrees with the audited protocol at week {week}")
        archive.append(protocol.week(week), mean, variance)
    audit = archive.write(
        output / "predictions.npz",
        require_complete=weeks == protocol.online_weeks,
        extra_metadata={
            "adapter": "official_factorial_sde_fsde_svi_isolated_state_handoff",
            "source_commit": "6e9ad8fb904e8d94e0325b21429111f26d8b6e69",
            "temporal_inducing": int(args.temporal_inducing),
            "latent_rank": int(args.latent_rank),
            "online_inference_steps": int(args.online_inference_steps),
            "task1_state": "official Task-1 parameters and posterior; worker handoff stores only q leaves",
            "worker_processes": weeks,
            "jax_compilation_cache": str(COMPILATION_CACHE),
        },
    )
    result = {
        "status": "complete" if weeks == protocol.online_weeks else "smoke_complete",
        "method": "FSDE-SVI isolated state handoff",
        "seed": args.seed,
        "weeks": weeks,
        "capacity": {"temporal_inducing": int(args.temporal_inducing), "latent_rank": int(args.latent_rank)},
        "total_seconds": time.perf_counter() - started,
        "worker_seconds_total": float(np.sum(worker_seconds)),
        "worker_seconds_per_completed_week": float(np.mean(worker_seconds)) if worker_seconds else 0.0,
        "audit": audit,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    args = parse_args()
    protocol = COVIDSettingBProtocol(args.protocol_npz, args.protocol_json)
    modes = sum(value is not None for value in (args.prepare_task1_state, args.week))
    if modes > 1:
        raise ValueError("Choose one worker mode at a time")
    if args.prepare_task1_state is not None:
        prepare_task1(protocol, args)
    elif args.week is not None:
        run_one_week(protocol, args)
    else:
        orchestrate(protocol, args)


if __name__ == "__main__":
    main()
