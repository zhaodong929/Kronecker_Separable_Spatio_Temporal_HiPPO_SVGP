#!/usr/bin/env python3
"""Build causal COVID Route B feature variants from one fixed protocol split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def lagged(values: np.ndarray, lag: int) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    output[lag:] = values[:-lag]
    return output


def helmert_contrasts(size: int) -> np.ndarray:
    """Return an orthonormal, sum-to-zero contrast for state deviations."""
    contrasts = np.zeros((size, size - 1), dtype=np.float64)
    for column in range(size - 1):
        scale = np.sqrt((column + 1) * (column + 2))
        contrasts[: column + 1, column] = 1.0 / scale
        contrasts[column + 1, column] = -(column + 1) / scale
    return contrasts


def adjacency_matrix(path: Path, codes: list[str]) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"left_code", "right_code"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain {sorted(required)}")
    position = {code: index for index, code in enumerate(codes)}
    adjacency = np.zeros((len(codes), len(codes)), dtype=np.float64)
    for row in rows:
        left = str(row["left_code"]).zfill(2)
        right = str(row["right_code"]).zfill(2)
        if left not in position or right not in position:
            continue
        adjacency[position[left], position[right]] = 1.0
        adjacency[position[right], position[left]] = 1.0
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("Adjacency matrix is not symmetric")
    return adjacency


def neighbour_mean(values: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    denominator = adjacency.sum(axis=1)
    numerator = values @ adjacency.T
    return np.divide(
        numerator,
        denominator[None, :],
        out=np.zeros_like(numerator),
        where=denominator[None, :] > 0.0,
    )


def visible_neighbour_current(
    y: np.ndarray, adjacency: np.ndarray, train_indices: np.ndarray
) -> np.ndarray:
    visible = np.zeros_like(y, dtype=np.float64)
    visible[:, train_indices] = y[:, train_indices]
    visible_adjacency = adjacency[:, train_indices]
    denominator = visible_adjacency.sum(axis=1)
    numerator = visible[:, train_indices] @ visible_adjacency.T
    return np.divide(
        numerator,
        denominator[None, :],
        out=np.zeros_like(numerator),
        where=denominator[None, :] > 0.0,
    )


def scale_features(
    phi: np.ndarray,
    calibration_steps: int,
    train: np.ndarray,
    preserve_columns: tuple[int, ...] = (),
) -> np.ndarray:
    reference = phi[:calibration_steps, train, 1:].reshape(-1, phi.shape[-1] - 1)
    mean = reference.mean(axis=0)
    scale = np.maximum(reference.std(axis=0), 1e-12)
    for column in preserve_columns:
        if column == 0:
            continue
        mean[column - 1] = 0.0
        scale[column - 1] = 1.0
    result = phi.copy()
    result[..., 1:] = (result[..., 1:] - mean) / scale
    return result


def ridge(phi: np.ndarray, y: np.ndarray, train: np.ndarray) -> np.ndarray:
    design = phi[:, train].reshape(-1, phi.shape[-1])
    target = y[:, train].reshape(-1)
    return np.linalg.solve(design.T @ design + 1e-3 * np.eye(design.shape[1]), design.T @ target)


def build_phi(
    *,
    y: np.ndarray,
    coordinates: np.ndarray,
    train: np.ndarray,
    calibration_steps: int,
    lag_order: int,
    state_effects: str,
    adjacency: np.ndarray | None,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    time_steps, locations = y.shape
    time = np.arange(time_steps, dtype=np.float64)
    phase = 2.0 * np.pi * time / 52.1775
    lags = [lagged(y, lag) for lag in range(1, lag_order + 1)]
    visible_mean_lag1 = lagged(np.mean(y[:, train], axis=1, keepdims=True), 1)
    parts = [
        np.ones((time_steps, locations, 1), dtype=np.float64),
        np.broadcast_to((time / max(time[-1], 1.0))[:, None, None], (time_steps, locations, 1)),
        np.broadcast_to(np.sin(phase)[:, None, None], (time_steps, locations, 1)),
        np.broadcast_to(np.cos(phase)[:, None, None], (time_steps, locations, 1)),
        *[values[..., None] for values in lags],
        np.broadcast_to(visible_mean_lag1[:, None, :], (time_steps, locations, 1)),
        np.broadcast_to(coordinates[None, :, :], (time_steps, locations, coordinates.shape[1])),
    ]
    columns = [
        "intercept",
        "time_trend",
        "season_sin",
        "season_cos",
        *[f"state_lag{lag}" for lag in range(1, lag_order + 1)],
        "visible_mean_lag1",
        "latitude",
        "longitude",
    ]
    beta_prior_variance = [1000.0] * len(columns)
    added_columns: list[str] = []
    state_effect_start = len(columns)
    contrasts = helmert_contrasts(locations)
    if state_effects == "intercept":
        parts.append(np.broadcast_to(contrasts[None, :, :], (time_steps, locations, locations - 1)))
        added_columns.extend([f"state_intercept_{index}" for index in range(locations - 1)])
    elif state_effects == "lag1":
        parts.append(lags[0][..., None] * contrasts[None, :, :])
        added_columns.extend([f"state_lag1_deviation_{index}" for index in range(locations - 1)])
    elif state_effects == "lag1_growth":
        growth = lags[0] - lags[1]
        parts.extend(
            [
                lags[0][..., None] * contrasts[None, :, :],
                growth[..., None] * contrasts[None, :, :],
            ]
        )
        added_columns.extend([f"state_lag1_deviation_{index}" for index in range(locations - 1)])
        added_columns.extend([f"state_growth_deviation_{index}" for index in range(locations - 1)])
    elif state_effects != "none":
        raise ValueError(f"Unsupported state effect: {state_effects}")
    if added_columns:
        columns.extend(added_columns)
        beta_prior_variance.extend([0.25] * len(added_columns))
    if adjacency is not None:
        parts.extend(
            [
                visible_neighbour_current(y, adjacency, train)[..., None],
                neighbour_mean(lags[0], adjacency)[..., None],
                neighbour_mean(lags[1], adjacency)[..., None],
            ]
        )
        columns.extend(["neighbour_current_visible", "neighbour_lag1", "neighbour_lag2"])
        beta_prior_variance.extend([1000.0, 1000.0, 1000.0])
    preserve_columns = tuple(
        range(state_effect_start, state_effect_start + len(added_columns))
    )
    phi = scale_features(
        np.concatenate(parts, axis=-1),
        calibration_steps,
        train,
        preserve_columns=preserve_columns,
    )
    return phi, columns, np.asarray(beta_prior_variance), added_columns


def exact_feature_basis(
    calibration_phi: np.ndarray,
    stream_phi: np.ndarray,
    relative_tolerance: float,
) -> tuple[np.ndarray, int, float]:
    design = calibration_phi.reshape(-1, calibration_phi.shape[-1])
    _, singular_values, vh = np.linalg.svd(design, full_matrices=False)
    relative = singular_values / max(float(singular_values[0]), 1e-12)
    rank = int(np.count_nonzero(relative > relative_tolerance))
    basis = vh[:rank].T
    all_phi = np.concatenate([calibration_phi, stream_phi], axis=0).reshape(-1, calibration_phi.shape[-1])
    residual = all_phi - (all_phi @ basis) @ basis.T
    error = float(np.linalg.norm(residual) / max(np.linalg.norm(all_phi), 1e-12))
    return basis, rank, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--lag-order", type=int, choices=(4, 8), default=4)
    parser.add_argument("--state-effects", choices=("none", "intercept", "lag1", "lag1_growth"), default="none")
    parser.add_argument("--adjacency-edges", type=Path)
    parser.add_argument("--use-source-phi", action="store_true")
    parser.add_argument(
        "--add-all-location-inducing",
        action="store_true",
        help="Store all known location coordinates as the Ms=number-of-locations inducing set.",
    )
    parser.add_argument("--projection-output", type=Path)
    parser.add_argument("--projection-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    with np.load(args.source_npz) as source:
        payload = {name: source[name] for name in source.files}
    metadata = json.loads(args.source_json.read_text(encoding="utf-8"))
    calibration_y = np.asarray(payload["calibration_y"], dtype=np.float64)
    stream_y = np.asarray(payload["stream_y"], dtype=np.float64)
    full_y = np.concatenate([calibration_y, stream_y], axis=0)
    train = np.asarray(payload["train_indices"], dtype=int)
    if args.use_source_phi:
        phi = np.concatenate(
            [
                np.asarray(payload["calibration_phi"], dtype=np.float64),
                np.asarray(payload["stream_phi"], dtype=np.float64),
            ],
            axis=0,
        )
        columns = metadata.get("xlag", {}).get("columns", [f"feature_{index}" for index in range(phi.shape[-1])])
        beta_prior_variance = np.full(phi.shape[-1], 1000.0)
        adjacency_mode = "not_used"
        state_effects = "none"
    else:
        adjacency = None
        adjacency_mode = "not_used"
        if args.adjacency_edges is not None:
            adjacency = adjacency_matrix(args.adjacency_edges, [str(code).zfill(2) for code in metadata["location_codes"]])
            adjacency_mode = str(args.adjacency_edges.resolve())
        phi, columns, beta_prior_variance, _ = build_phi(
            y=full_y,
            coordinates=np.asarray(payload["coordinates"], dtype=np.float64),
            train=train,
            calibration_steps=calibration_y.shape[0],
            lag_order=args.lag_order,
            state_effects=args.state_effects,
            adjacency=adjacency,
        )
        state_effects = args.state_effects
    payload["calibration_phi"] = phi[: calibration_y.shape[0]].astype(np.float32)
    payload["stream_phi"] = phi[calibration_y.shape[0] :].astype(np.float32)
    payload["beta_prior_variance"] = beta_prior_variance.astype(np.float64)
    beta = ridge(payload["calibration_phi"], calibration_y, train)
    batch_beta = ridge(payload["stream_phi"], stream_y, train)
    payload["task1_ridge_beta"] = beta
    payload["batch_ridge_beta"] = batch_beta
    payload["task1_calibration_mean"] = np.einsum("tsp,p->ts", payload["calibration_phi"], beta).astype(np.float32)
    payload["task1_stream_mean"] = np.einsum("tsp,p->ts", payload["stream_phi"], beta).astype(np.float32)
    payload["batch_stream_mean"] = np.einsum("tsp,p->ts", payload["stream_phi"], batch_beta).astype(np.float32)
    if args.add_all_location_inducing:
        payload[f"inducing_coords_ms{payload['coordinates'].shape[0]}"] = np.asarray(
            payload["coordinates"], dtype=np.float64
        )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **payload)
    output_json = args.output_npz.with_suffix(".json")
    metadata["npz"] = str(args.output_npz.resolve())
    metadata["feature_source"] = "COVID Route B causal dynamics variant"
    metadata["xlag"] = {
        "mode": "state history with optional partial-pooling and neighbour exposure",
        "columns": columns,
        "lag_order": args.lag_order,
        "state_effects": state_effects,
        "neighbour_current_uses": "current-week visible locations only",
        "neighbour_past_uses": "past labels only",
        "neighbour_growth": "omitted because neighbour_lag1 - neighbour_lag2 is an exact linear combination",
        "adjacency_edges": adjacency_mode,
        "all_location_inducing": bool(args.add_all_location_inducing),
    }
    if args.projection_output is not None:
        basis, rank, error = exact_feature_basis(
            payload["calibration_phi"],
            payload["stream_phi"],
            args.projection_tolerance,
        )
        args.projection_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.projection_output,
            basis=basis,
            numerical_rank=np.asarray(rank),
            relative_projection_error=np.asarray(error),
        )
        metadata["exact_feature_projection"] = {
            "path": str(args.projection_output.resolve()),
            "rank": rank,
            "relative_projection_error": error,
        }
    output_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_npz.resolve()), "features": len(columns)}, indent=2))


if __name__ == "__main__":
    main()
