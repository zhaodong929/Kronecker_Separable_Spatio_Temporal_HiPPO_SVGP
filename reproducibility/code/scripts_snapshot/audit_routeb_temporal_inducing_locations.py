#!/usr/bin/env python3
"""Trace fixed temporal inducing coordinates around the Route B batch update."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_post_meeting_p1_matrix as runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-metadata", required=True)
    parser.add_argument("--outdir", required=True)
    args_cli = parser.parse_args()

    metadata_path = Path(args_cli.run_metadata)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    args = Namespace(**metadata["args"])

    calibration_raw = runner.load_hipposvgp_era5(
        args.root, tasks=("task_1",), variable_index=0, split="all"
    )
    selected_locations = runner.selected_locations_from_dataset(calibration_raw)
    online_raw = runner.load_hipposvgp_era5(
        args.root,
        tasks=("task_2",),
        variable_index=0,
        split="all",
        selected_locations=selected_locations,
    )
    _, calibration_scale = runner.normalise_time_dataset(calibration_raw)
    dataset_base = runner.normalise_time_dataset_with_scale(
        online_raw, scale=calibration_scale, source="calibration_task_span"
    )
    dataset = runner.augment_dataset_phi(
        dataset_base, phi_mode=args.phi_mode, xlag_length=args.xlag_length
    )
    routeb_dataset = runner.routeb_dataset_from_era5(
        dataset, sigma2=args.noise**2, args=args
    )
    train_idx, test_idx = runner.fixed_spatial_train_test_split(
        dataset.Y.shape[1], test_fraction=args.test_fraction, seed=args.split_seed
    )

    spatial_kernel_type = args.spatial_kernel_type or args.kernel_type
    spatial_lengthscale = np.asarray(args.spatial_lengthscales, dtype=float)
    shared = np.load(args.spatial_inducing_coords_npz)
    _, ks, c_all = runner.fixed_spatial_projection(
        routeb_dataset.spatial_coords,
        shared[f"inducing_coords_ms{args.ms}"],
        lengthscale=spatial_lengthscale,
        kernel_type=spatial_kernel_type,
    )

    original_linspace = runner.np.linspace
    captures: list[np.ndarray] = []
    t_min = float(routeb_dataset.times.min())
    t_max = float(routeb_dataset.times.max())

    def traced_linspace(start: float, stop: float, num: int = 50, *extra: object, **kwargs: object) -> np.ndarray:
        values = original_linspace(start, stop, num, *extra, **kwargs)
        if (
            int(num) == int(args.mt)
            and np.isclose(float(start), t_min)
            and np.isclose(float(stop), t_max)
        ):
            captures.append(np.asarray(values, dtype=float).copy())
        return values

    runner.np.linspace = traced_linspace
    try:
        runner.run_structured_batch(
            dataset,
            routeb_dataset,
            train_idx,
            test_idx,
            args=args,
            ks=ks,
            c_train=c_all[train_idx],
            c_test=c_all[test_idx],
        )
    finally:
        runner.np.linspace = original_linspace

    if len(captures) != 2:
        raise RuntimeError(f"Expected train/evaluation captures, found {len(captures)}")
    before_update, after_update = captures
    delta = after_update - before_update

    outdir = Path(args_cli.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "ordinary_temporal_inducing_locations_before_after_seed0.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "before_posterior_update", "after_posterior_update", "difference"])
        for index, (before, after, change) in enumerate(zip(before_update, after_update, delta)):
            writer.writerow([index, f"{before:.17g}", f"{after:.17g}", f"{change:.17g}"])

    audit = {
        "run_metadata": str(metadata_path),
        "temporal_representation": args.temporal_representation,
        "mt": int(args.mt),
        "num_observation_times": int(dataset.Y.shape[0]),
        "time_min": t_min,
        "time_max": t_max,
        "capture_count": len(captures),
        "capture_semantics": [
            "training factors before structured posterior update",
            "evaluation factors after structured posterior update",
        ],
        "array_type": f"{type(before_update).__module__}.{type(before_update).__name__}",
        "array_equal": bool(np.array_equal(before_update, after_update)),
        "max_abs_change": float(np.max(np.abs(delta))),
        "coordinates_csv": str(csv_path),
        "before": before_update.tolist(),
        "after": after_update.tolist(),
    }
    json_path = outdir / "ordinary_temporal_inducing_locations_audit_seed0.json"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
