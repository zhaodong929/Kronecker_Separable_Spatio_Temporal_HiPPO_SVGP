#!/usr/bin/env python3
"""Measure whether OVC exact-fantasy memory growth is intentional state growth.

Run this program twice in separate fresh processes before accepting an OVC
cloud formal configuration.  It deliberately stops after a 32-week seed-0
prefix and records the exact-fantasy observation count alongside RSS and all
reachable PyTorch tensor storage after forced garbage collection.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.adapters.run_ovc_svgp import (
    condition,
    flatten_task1,
    observation_inputs,
    select_inducing_points,
    train_task1,
)
from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol
from volatilitygp.models import SingleTaskVariationalGP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-npz",
        type=Path,
        default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed0/protocol.npz"),
    )
    parser.add_argument("--protocol-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicate-id", type=int, choices=(1, 2), required=True)
    parser.add_argument("--temporal-inducing", type=int, required=True)
    parser.add_argument("--spatial-inducing", type=int, default=32)
    parser.add_argument("--task1-iterations", type=int, default=50000)
    parser.add_argument("--task1-check-interval", type=int, default=250)
    parser.add_argument("--task1-min-steps", type=int, default=2500)
    parser.add_argument("--task1-plateau-checks", type=int, default=10)
    parser.add_argument("--task1-plateau-relative-improvement", type=float, default=1e-3)
    parser.add_argument("--task1-learning-rate", type=float, default=0.01)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--weeks", type=int, default=32)
    return parser.parse_args()


def absolute(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else ROOT / path


def rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")


def reachable_unique_tensor_storage_bytes() -> int:
    """Count storage reachable from Python, including accidental retained tensors."""

    seen: set[tuple[str, int]] = set()
    total = 0
    for value in gc.get_objects():
        try:
            if not torch.is_tensor(value):
                continue
            storage = value.untyped_storage()
            pointer = int(storage.data_ptr())
            key = (str(value.device), pointer)
            if pointer and key not in seen:
                seen.add(key)
                total += int(storage.nbytes())
        except (ReferenceError, RuntimeError, TypeError):
            continue
    return total


def reachable_model_instances() -> int:
    """Detect accidental retention of past fantasy model objects after GC."""

    return sum(isinstance(value, SingleTaskVariationalGP) for value in gc.get_objects())


def fantasy_observation_count(model: object) -> int | None:
    try:
        inputs = model.train_inputs[0]
        return int(inputs.shape[-2])
    except (AttributeError, IndexError, TypeError):
        return None


def main() -> None:
    args = parse_args()
    protocol = COVIDSettingBProtocol(absolute(args.protocol_npz), absolute(args.protocol_json))
    if not 1 <= args.weeks <= min(32, protocol.online_weeks):
        raise ValueError("The OVC memory audit is fixed to a non-empty prefix of at most 32 weeks")
    output = absolute(args.output_dir) / f"replicate_{args.replicate_id}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite memory-audit evidence: {output}")
    output.mkdir(parents=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)
    np.random.seed(0)
    train_x_np, train_y_np = flatten_task1(protocol)
    train_x = torch.as_tensor(train_x_np, dtype=dtype)
    train_y = torch.as_tensor(train_y_np, dtype=dtype)
    inducing = torch.as_tensor(
        select_inducing_points(
            protocol,
            count=args.temporal_inducing * args.spatial_inducing,
            temporal_inducing=args.temporal_inducing,
            spatial_inducing=args.spatial_inducing,
            seed=0,
        ),
        dtype=dtype,
    )
    model = SingleTaskVariationalGP(
        init_points=inducing,
        train_inputs=train_x,
        train_targets=train_y,
        covar_module=ScaleKernel(RBFKernel(ard_num_dims=3)),
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
        checkpoint_directory=output / "task1_checkpoints",
    )
    rows: list[dict[str, Any]] = []
    for week in range(args.weeks):
        information = protocol.week(week)
        if information.delayed_hidden is not None:
            delayed_x, delayed_y = observation_inputs(protocol, information.delayed_hidden, dtype)
            model = condition(model, delayed_x, delayed_y)
        visible_x, visible_y = observation_inputs(protocol, information.current_visible, dtype)
        model = condition(model, visible_x, visible_y)
        if (week + 1) % 8 == 0:
            gc.collect()
            expected = protocol.calibration_weeks * protocol.locations + 42 * (week + 1) + 10 * week
            rows.append(
                {
                    "week": week + 1,
                    "rss_bytes": rss_bytes(),
                    "reachable_unique_tensor_storage_bytes": reachable_unique_tensor_storage_bytes(),
                    "reachable_model_instances": reachable_model_instances(),
                    "fantasy_state_observation_count": fantasy_observation_count(model),
                    "expected_arrived_observation_count": expected,
                }
            )
    slope = float(np.polyfit([row["week"] for row in rows], [row["rss_bytes"] for row in rows], 1)[0]) if len(rows) > 1 else 0.0
    payload = {
        "status": "complete",
        "purpose": "clean-process OVC exact-fantasy retention audit; repeat once in a separate process",
        "replicate_id": args.replicate_id,
        "protocol": str(protocol.npz_path),
        "capacity": {"temporal_inducing": args.temporal_inducing, "spatial_inducing": args.spatial_inducing, "joint_inducing": int(inducing.shape[0])},
        "task1_convergence": convergence,
        "forced_gc_every_weeks": 8,
        "rows": rows,
        "rss_slope_bytes_per_week": slope,
        "interpretation_rule": "Continue only when the two clean-process traces show state growth consistent with exact fantasy conditioning and no unexplained retained-model or autograd-cache growth.",
    }
    (output / "memory_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "rss_slope_bytes_per_week": slope}, indent=2))


if __name__ == "__main__":
    main()
