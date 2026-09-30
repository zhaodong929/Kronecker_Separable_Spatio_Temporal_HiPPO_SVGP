#!/usr/bin/env python3
"""Render a formal ERA5 Tasks 1--10 endpoint from archived predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAT_LON_RE = re.compile(r"lat_([-0-9.]+)_lon_([-0-9.]+)")


def parse_lat_lon_from_filename(path: Path) -> tuple[float, float]:
    match = LAT_LON_RE.search(path.stem.replace("_scaled", ""))
    if match is None:
        raise ValueError(f"Cannot parse lat/lon from {path.name}")
    return float(match.group(1)), float(match.group(2))


def discover_scaled_task1_files(root: Path) -> list[Path]:
    sequence_dir = root / "task_1" / "sequences"
    files = sorted(p for p in sequence_dir.iterdir() if p.is_file() and p.name.endswith("_scaled.npz"))
    if not files:
        raise FileNotFoundError(f"No scaled Task-1 sequences under {sequence_dir}")
    seen: set[tuple[float, float]] = set()
    selected: list[Path] = []
    for path in files:
        coord = parse_lat_lon_from_filename(path)
        if coord not in seen:
            selected.append(path)
            seen.add(coord)
    return sorted(selected, key=parse_lat_lon_from_filename)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--plot-seed", type=int, default=0)
    args = parser.parse_args()

    archive = args.archive_root / f"seed{args.plot_seed}" / "predictions.npz"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    protocol_path = args.protocol / "protocol.npz"
    with np.load(protocol_path) as protocol:
        coordinates = np.asarray(protocol["coordinates"], dtype=float)
        train_indices = np.asarray(protocol["train_indices"], dtype=int)
        protocol_test_indices = np.asarray(protocol["test_indices"], dtype=int)
    raw_files = discover_scaled_task1_files(args.data_root)
    raw_coordinates = np.asarray(
        [parse_lat_lon_from_filename(path) for path in raw_files], dtype=float
    )
    raw_standardized = (raw_coordinates - raw_coordinates.mean(axis=0, keepdims=True)) / np.maximum(
        raw_coordinates.std(axis=0, keepdims=True), 1e-12
    )
    if not np.allclose(raw_standardized, coordinates, rtol=0.0, atol=1e-12):
        raise ValueError("Task-1 file coordinates do not reproduce the formal protocol")

    with np.load(archive) as prediction:
        truth = np.asarray(prediction["y_true"], dtype=float)
        mean = np.asarray(prediction["pred_mean"], dtype=float)
        variance = np.asarray(prediction["pred_var"], dtype=float)
        test_indices = np.asarray(prediction["test_indices"], dtype=int)
    if truth.shape != (1674, 200) or mean.shape != truth.shape or variance.shape != truth.shape:
        raise ValueError(f"Expected (1674, 200) long-stream archive, got {truth.shape}")
    if not np.array_equal(test_indices, protocol_test_indices):
        raise ValueError("Prediction archive test_indices differ from formal protocol")
    if np.any(variance <= 0):
        raise ValueError("Prediction archive contains non-positive variances")

    coords = raw_coordinates[test_indices]
    background = raw_coordinates[train_indices]
    truth_final, mean_final = truth[-1], mean[-1]
    error_final = mean_final - truth_final
    value_limit = max(float(np.max(np.abs(truth_final))), float(np.max(np.abs(mean_final))), 1e-9)
    error_limit = max(float(np.max(np.abs(error_final))), 1e-9)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    # A full-width horizontal layout keeps all three spatial diagnostics readable.
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.45), sharex=True, sharey=True,
                             constrained_layout=True)
    panels = (
        (truth_final, "(a) Ground truth", value_limit, "RdBu_r"),
        (mean_final, "(b) KronHiPPO-STGP", value_limit, "RdBu_r"),
        (error_final, "(c) Signed error", error_limit, "coolwarm"),
    )
    for axis, (values, title, limit, cmap) in zip(axes, panels):
        axis.scatter(background[:, 1], background[:, 0], s=7, color="#d9dde2", edgecolors="none", alpha=0.7)
        artist = axis.scatter(
            coords[:, 1], coords[:, 0], c=values, s=17, cmap=cmap,
            vmin=-limit, vmax=limit, edgecolors="white", linewidths=0.08,
        )
        axis.set_title(title, pad=3)
        axis.set_xlabel("Longitude")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#e1e1e1", linewidth=0.3, alpha=0.65)
        axis.set_axisbelow(True)
        fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.025)
    axes[0].set_ylabel("Latitude")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps({
        "scope": "ERA5-Land Tasks 1--10, final online block",
        "method": "KronHiPPO-STGP",
        "objective": "VFE",
        "plot_seed": args.plot_seed,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "prediction_shape": list(truth.shape),
        "train_locations": int(train_indices.size),
        "held_out_locations": int(test_indices.size),
        "endpoint": "last of 1674 stream times",
        "coordinate_order": "latitude, longitude",
        "source_protocol": str(protocol_path),
        "source_data_root": str(args.data_root),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
