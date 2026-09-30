#!/usr/bin/env python3
"""Plot the final short-stream Task-2 spatial snapshot from formal archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stvgp_kronecker.data.hipposvgp_era5 import (
    discover_sequence_files,
    parse_lat_lon_from_filename,
    select_location_files,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--plot-seed", type=int, default=0)
    args = parser.parse_args()

    run_root = args.formal_root / "runs" / "short" / "online" / "kronhippo_stgp"
    archive_paths = [run_root / f"seed{seed}" / "predictions.npz" for seed in range(5)]
    missing = [str(path) for path in archive_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing formal prediction archives: " + ", ".join(missing))

    protocol_path = args.protocol / "protocol.npz"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)

    with np.load(protocol_path) as protocol:
        coordinates = np.asarray(protocol["coordinates"], dtype=float)
        train_indices = np.asarray(protocol["train_indices"], dtype=int)
        protocol_test_indices = np.asarray(protocol["test_indices"], dtype=int)
    raw_files = select_location_files(
        discover_sequence_files(args.data_root, tasks=("task_1",), prefer_scaled=True)
    )
    raw_coordinates = np.asarray(
        [parse_lat_lon_from_filename(path) for path in raw_files], dtype=float
    )
    if raw_coordinates.shape != coordinates.shape:
        raise ValueError(
            f"Raw coordinate shape {raw_coordinates.shape} does not match protocol {coordinates.shape}"
        )
    raw_standardized = (raw_coordinates - raw_coordinates.mean(axis=0, keepdims=True)) / np.maximum(
        raw_coordinates.std(axis=0, keepdims=True), 1e-12
    )
    if not np.allclose(raw_standardized, coordinates, rtol=0.0, atol=1e-12):
        raise ValueError("Task-1 file coordinates do not reproduce the formal standardized coordinates")
    with np.load(archive_paths[args.plot_seed]) as prediction:
        truth = np.asarray(prediction["y_true"], dtype=float)
        mean = np.asarray(prediction["pred_mean"], dtype=float)
        variance = np.asarray(prediction["pred_var"], dtype=float)
        test_indices = np.asarray(prediction["test_indices"], dtype=int)
        times = np.asarray(prediction["times"], dtype=float)

    if truth.shape != (186, 200) or mean.shape != truth.shape or variance.shape != truth.shape:
        raise ValueError(f"Expected Task-2 archive shape (186, 200), got {truth.shape}, {mean.shape}, {variance.shape}")
    if test_indices.shape != (200,):
        raise ValueError(f"Expected 200 held-out indices, got {test_indices.shape}")
    if not np.array_equal(test_indices, protocol_test_indices):
        raise ValueError("Prediction archive test_indices differ from the formal protocol")
    if train_indices.shape != (800,):
        raise ValueError(f"Expected 800 training indices, got {train_indices.shape}")
    if not (np.isfinite(truth).all() and np.isfinite(mean).all() and np.isfinite(variance).all()):
        raise ValueError("Prediction archive contains non-finite values")
    if np.any(variance <= 0):
        raise ValueError("Prediction archive contains non-positive variances")

    coords = raw_coordinates[test_indices]
    background_coords = raw_coordinates[train_indices]
    truth_final = truth[-1]
    mean_final = mean[-1]
    error_final = mean_final - truth_final
    value_limit = max(float(np.max(np.abs(truth_final))), float(np.max(np.abs(mean_final))), 1e-9)
    error_limit = max(float(np.max(np.abs(error_final))), 1e-9)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.55), constrained_layout=True)
    panels = (
        (truth_final, "Ground truth", value_limit, "RdBu_r"),
        (mean_final, "KronHiPPO-STGP prediction", value_limit, "RdBu_r"),
        (error_final, "Prediction error", error_limit, "coolwarm"),
    )
    for axis, (values, title, limit, cmap) in zip(axes, panels):
        axis.scatter(
            background_coords[:, 1], background_coords[:, 0],
            s=10, color="#d9dde2", edgecolors="none", alpha=0.78,
            label="Training locations",
        )
        artist = axis.scatter(
            coords[:, 1], coords[:, 0], c=values, s=27, cmap=cmap,
            vmin=-limit, vmax=limit, edgecolors="white", linewidths=0.16,
            label="Held-out locations",
        )
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#e1e1e1", linewidth=0.45, alpha=0.65)
        axis.set_axisbelow(True)
        fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("Latitude")
    axes[0].legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.92)
    fig.suptitle(
        "Short-stream Task-2 held-out UK spatial field | final online block",
        fontsize=13,
    )
    save_figure(fig, args.output)

    metadata = {
        "figure": "Figure 4.10",
        "scope": "short-stream Task-2",
        "method": "KronHiPPO-STGP",
        "likelihood": "Gaussian",
        "objective": "VFE",
        "temporal_kernel": "Spectral Mixture",
        "representation": "analytic_hippo_rff",
        "mt": 128,
        "ms": 128,
        "rff_sample_size": 256,
        "formal_seed_archives": [
            {"seed": seed, "path": str(path), "sha256": sha256(path)}
            for seed, path in enumerate(archive_paths)
        ],
        "plot_seed": args.plot_seed,
        "plot_endpoint": "last of 186 Task-2 stream times / 19th online block",
        "prediction_shape": list(truth.shape),
        "coordinate_order": "raw latitude, longitude from Task-1 filenames",
        "plot_axes": {"x": "longitude", "y": "latitude"},
        "training_background_locations": int(train_indices.size),
        "held_out_locations": int(test_indices.size),
        "spatial_split_note": (
            "The five formal seeds use different held-out spatial indices; the map uses "
            "seed-0's 200 held-out UK locations rather than averaging misaligned points. "
            "The other 800 locations are shown in light gray only as spatial context."
        ),
        "source_protocol": str(protocol_path),
        "source_data_root": str(args.data_root),
        "outputs": [str(args.output.with_suffix(".pdf")), str(args.output.with_suffix(".png"))],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
