#!/usr/bin/env python3
"""Measure stage-level wall time for one empirical-Bayes Route B update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_routeb_batch_empirical_bayes import (
    inner_training_validation_split,
    load_controlled_grid,
    tensor_training_data,
)
from stvgp_kronecker.routeb_empirical_bayes import (
    BatchRouteBEmpiricalBayes,
    finite_joint_nlml_from_factors,
)


def measure_case(
    *,
    data: object,
    inner_train: np.ndarray,
    representation: str,
    mean_dimension: int,
    repeats: int,
    warmup: int,
) -> dict[str, object]:
    y, phi_full, coordinates = tensor_training_data(data, inner_train)
    phi = phi_full if mean_dimension else torch.empty(
        y.shape[0], y.shape[1], 0, dtype=y.dtype
    )
    model = BatchRouteBEmpiricalBayes(
        times=data.times,
        spatial_inducing=data.spatial_inducing,
        mt=128,
        representation=representation,
        initial_ell_t=0.025,
        initial_ell_s=(0.55, 0.61),
        initial_kernel_variance=0.30,
        initial_noise_std=0.065,
        rff_sample_size=256,
        seed=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    records = []
    total_iterations = warmup + repeats
    for iteration in range(total_iterations):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        t_mat, c_mat, kt, ks = model.factor_matrices(coordinates)
        factor_seconds = time.perf_counter() - started

        started = time.perf_counter()
        objective = finite_joint_nlml_from_factors(
            y_matrix=y,
            phi_tensor=phi,
            temporal_projection=t_mat,
            spatial_projection=c_mat,
            temporal_prior=kt,
            spatial_prior=ks,
            noise_variance=model.noise_std.square(),
            beta_prior_variance=1000.0,
        )
        solve_seconds = time.perf_counter() - started

        started = time.perf_counter()
        objective.nlml_per_observation.backward()
        backward_seconds = time.perf_counter() - started

        started = time.perf_counter()
        optimizer.step()
        optimizer_seconds = time.perf_counter() - started
        if iteration >= warmup:
            records.append(
                [factor_seconds, solve_seconds, backward_seconds, optimizer_seconds]
            )
    values = np.asarray(records, dtype=float)
    names = ["factor", "structured_objective", "backward", "adam"]
    result: dict[str, object] = {
        "representation": representation,
        "mean_dimension": int(phi.shape[-1]),
        "num_time": int(y.shape[1]),
        "num_inner_train_space": int(y.shape[0]),
        "mt": 128,
        "ms": 128,
        "repeats": repeats,
        "warmup": warmup,
    }
    for index, name in enumerate(names):
        result[f"{name}_seconds_mean"] = float(values[:, index].mean())
        result[f"{name}_seconds_sd"] = float(values[:, index].std(ddof=1))
    totals = values.sum(axis=1)
    result["total_seconds_mean"] = float(totals.mean())
    result["total_seconds_sd"] = float(totals.std(ddof=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    data = load_controlled_grid(
        root="data/era5/processed_timeseries_4",
        task="task_2",
        controlled_npz=Path(args.controlled_npz),
        ms=128,
        xlag_length=10,
    )
    inner_train, _ = inner_training_validation_split(
        data.train_indices, split_seed=0, validation_fraction=0.1
    )
    rows = [
        measure_case(
            data=data,
            inner_train=inner_train,
            representation="analytic_hippo_rff",
            mean_dimension=133,
            repeats=args.repeats,
            warmup=args.warmup,
        ),
        measure_case(
            data=data,
            inner_train=inner_train,
            representation="inducing_points",
            mean_dimension=133,
            repeats=args.repeats,
            warmup=args.warmup,
        ),
        measure_case(
            data=data,
            inner_train=inner_train,
            representation="inducing_points",
            mean_dimension=0,
            repeats=args.repeats,
            warmup=args.warmup,
        ),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
