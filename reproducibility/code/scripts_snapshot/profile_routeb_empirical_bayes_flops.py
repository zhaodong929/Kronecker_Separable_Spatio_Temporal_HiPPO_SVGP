#!/usr/bin/env python3
"""Profile one Route B empirical-Bayes optimization step with PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_routeb_batch_empirical_bayes import (
    inner_training_validation_split,
    load_controlled_grid,
    tensor_training_data,
)
from stvgp_kronecker.routeb_empirical_bayes import BatchRouteBEmpiricalBayes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--mt", type=int, default=128)
    parser.add_argument("--ms", type=int, default=128)
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    args = parser.parse_args()

    data = load_controlled_grid(
        root=args.root,
        task="task_2",
        controlled_npz=Path(args.controlled_npz),
        ms=args.ms,
        xlag_length=args.xlag_length,
    )
    inner_train, _ = inner_training_validation_split(
        data.train_indices, split_seed=args.split_seed, validation_fraction=0.1
    )
    y, phi, coordinates = tensor_training_data(data, inner_train)
    rows = []
    for representation in ("analytic_hippo_rff", "inducing_points"):
        model = BatchRouteBEmpiricalBayes(
            times=data.times,
            spatial_inducing=data.spatial_inducing,
            mt=args.mt,
            representation=representation,
            initial_ell_t=0.05,
            initial_ell_s=(0.35, 0.35),
            initial_kernel_variance=1.0,
            initial_noise_std=0.1,
            rff_sample_size=args.rff_sample_size,
            seed=0,
        )
        with FlopCounterMode(display=False) as counter:
            loss = model.objective(
                y_matrix=y,
                phi_tensor=phi,
                spatial_coordinates=coordinates,
                beta_prior_variance=1000.0,
            ).nlml_per_observation
            loss.backward()
        rows.append(
            {
                "method": "Route B " + representation,
                "scope": "one_forward_and_backward",
                "flops": int(counter.get_total_flops()),
                "gflops": float(counter.get_total_flops() / 1e9),
                "num_time_steps": int(data.times.size),
                "num_inner_train_space": int(inner_train.size),
                "num_xlag_features": int(phi.shape[-1]),
                "mt": args.mt,
                "ms": args.ms,
                "note": (
                    "PyTorch supported-ATen-op count; optimizer bookkeeping, validation, "
                    "NumPy posterior refit, prediction, and unsupported operators are excluded."
                ),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
