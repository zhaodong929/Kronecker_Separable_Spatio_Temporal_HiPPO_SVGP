"""Blocked, strict-online seed-0 development protocols for COVID Setting B.

The formal 52/143 archives are intentionally immutable.  This module derives
short chronological folds from their audited target, re-standardising from
each fold's training prefix before any baseline sees a label.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans


BLOCKED_WINDOWS: tuple[tuple[int, int], ...] = ((28, 36), (36, 44), (44, 52))
SELECTION_METRIC = "gaussian_nlpd_log1p_per_100k"


@dataclass(frozen=True)
class BlockedWindow:
    """One prefix-train / chronological-validation split of Task 1."""

    train_stop: int
    validation_stop: int

    @property
    def validation_start(self) -> int:
        return self.train_stop

    @property
    def validation_weeks(self) -> int:
        return self.validation_stop - self.train_stop

    @property
    def name(self) -> str:
        return f"weeks_1_{self.train_stop}_to_{self.train_stop + 1}_{self.validation_stop}"


def blocked_windows() -> tuple[BlockedWindow, ...]:
    return tuple(BlockedWindow(*window) for window in BLOCKED_WINDOWS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _availability_aware_phi(
    *,
    target: np.ndarray,
    coordinates: np.ndarray,
    visible: np.ndarray,
    hidden: np.ndarray,
    calibration_weeks: int,
) -> tuple[np.ndarray, list[str]]:
    """Rebuild the lag-4 design with no hidden label before its one-week delay."""

    times, locations = target.shape
    index = np.arange(times, dtype=np.float64)
    phase = 2.0 * np.pi * index / 52.1775
    visible_mask = np.zeros(locations, dtype=bool)
    visible_mask[np.asarray(visible, dtype=np.int64)] = True
    hidden_mask = np.zeros(locations, dtype=bool)
    hidden_mask[np.asarray(hidden, dtype=np.int64)] = True
    lags = np.full((4, times, locations), np.nan, dtype=np.float64)
    for lag_index, lag in enumerate((1, 2, 3, 4)):
        for current in range(lag, times):
            source = current - lag
            known = np.ones(locations, dtype=bool)
            if current >= calibration_weeks:
                online_week = current - calibration_weeks
                known = visible_mask.copy()
                known[hidden_mask] = source < calibration_weeks or source <= current - 1
                if online_week == 0:
                    known[hidden_mask] = source < calibration_weeks
            lags[lag_index, current, known] = target[source, known]
    visible_mean = np.mean(target[:, visible], axis=1)
    visible_lag1 = np.full(times, np.nan, dtype=np.float64)
    visible_lag1[1:] = visible_mean[:-1]
    lag_count = np.sum(np.isfinite(lags), axis=0)
    rolling4 = np.nansum(lags, axis=0) / np.maximum(lag_count, 1)
    rolling4[lag_count == 0] = np.nan
    dynamic = np.stack(
        [
            np.broadcast_to(index[:, None] / 51.0, (times, locations)),
            np.broadcast_to(np.sin(phase)[:, None], (times, locations)),
            np.broadcast_to(np.cos(phase)[:, None], (times, locations)),
            *lags,
            rolling4,
            lags[0] - lags[1],
            np.broadcast_to(visible_lag1[:, None], (times, locations)),
            np.broadcast_to(coordinates[None, :, 0], (times, locations)),
            np.broadcast_to(coordinates[None, :, 1], (times, locations)),
        ],
        axis=-1,
    )
    reference = dynamic[:calibration_weeks, visible].reshape(-1, dynamic.shape[-1])
    mean = np.nanmean(reference, axis=0)
    scale = np.maximum(np.nanstd(reference, axis=0), 1e-12)
    dynamic = np.where(np.isfinite(dynamic), dynamic, mean[None, None, :])
    phi = np.concatenate(
        [np.ones((times, locations, 1), dtype=np.float64), (dynamic - mean) / scale], axis=-1
    )
    return phi, [
        "intercept",
        "time_trend",
        "season_sin",
        "season_cos",
        "state_lag1",
        "state_lag2",
        "state_lag3",
        "state_lag4",
        "state_rolling4",
        "state_growth1",
        "visible_mean_lag1",
        "latitude",
        "longitude",
    ]


def _ridge(phi: np.ndarray, target: np.ndarray, locations: np.ndarray) -> np.ndarray:
    design = phi[:, locations].reshape(-1, phi.shape[-1])
    values = target[:, locations].reshape(-1)
    return np.linalg.solve(design.T @ design + 1e-3 * np.eye(phi.shape[-1]), design.T @ values)


def _spatial_inducing(coordinates: np.ndarray, visible: np.ndarray, sizes: Iterable[int]) -> dict[str, np.ndarray]:
    candidates = coordinates[np.asarray(visible, dtype=np.int64)]
    output: dict[str, np.ndarray] = {}
    for requested in sizes:
        count = min(int(requested), candidates.shape[0])
        if count == candidates.shape[0]:
            values = candidates.copy()
        else:
            values = KMeans(n_clusters=count, random_state=0, n_init=10).fit(candidates).cluster_centers_
        output[f"inducing_coords_ms{int(requested)}"] = np.asarray(values, dtype=np.float64)
    return output


def build_development_protocols(
    *,
    formal_npz: Path,
    formal_json: Path,
    output_root: Path,
    inducing_sizes: Iterable[int] = (16, 32, 52),
) -> list[Path]:
    """Write the three development folds without changing the formal archive."""

    formal_npz = Path(formal_npz)
    formal_json = Path(formal_json)
    output_root = Path(output_root)
    metadata = json.loads(formal_json.read_text(encoding="utf-8"))
    standardization = metadata["target_standardization"]
    original_mean = float(standardization["mean"])
    original_scale = float(standardization["scale"])
    with np.load(formal_npz, allow_pickle=False) as arrays:
        calibration_standardized = np.asarray(arrays["calibration_y"], dtype=np.float64)
        visible = np.asarray(arrays["train_indices"], dtype=np.int64)
        fit = np.asarray(arrays["fit_indices"], dtype=np.int64)
        validation = np.asarray(arrays["validation_indices"], dtype=np.int64)
        hidden = np.asarray(arrays["test_indices"], dtype=np.int64)
        coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
        dates = np.asarray(arrays["calibration_week_dates"], dtype="U10")
    if calibration_standardized.shape != (52, 52):
        raise ValueError("Blocked development folds require the audited 52-week seed-0 Task-1 archive")
    raw_target = calibration_standardized * original_scale + original_mean
    output_paths: list[Path] = []
    for fold_id, window in enumerate(blocked_windows(), start=1):
        reference = raw_target[: window.train_stop, visible]
        fold_mean = float(np.mean(reference))
        fold_scale = float(max(np.std(reference), 1e-12))
        fold_target = (raw_target[: window.validation_stop] - fold_mean) / fold_scale
        phi, columns = _availability_aware_phi(
            target=fold_target,
            coordinates=coordinates,
            visible=visible,
            hidden=hidden,
            calibration_weeks=window.train_stop,
        )
        beta = _ridge(phi[: window.train_stop], fold_target[: window.train_stop], visible)
        fold_dir = output_root / f"fold_{fold_id}_{window.name}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        npz_path = fold_dir / "protocol.npz"
        payload = {
            "train_indices": visible,
            "fit_indices": fit,
            "validation_indices": validation,
            "test_indices": hidden,
            "calibration_y": fold_target[: window.train_stop],
            "stream_y": fold_target[window.train_stop : window.validation_stop],
            "calibration_phi": phi[: window.train_stop],
            "stream_phi": phi[window.train_stop : window.validation_stop],
            "task1_ridge_beta": beta,
            "task1_calibration_mean": np.einsum("tsp,p->ts", phi[: window.train_stop], beta),
            "task1_stream_mean": np.einsum("tsp,p->ts", phi[window.train_stop : window.validation_stop], beta),
            "coordinates": coordinates,
            "calibration_times": np.arange(window.train_stop, dtype=np.float64) / 51.0,
            "stream_times": np.arange(window.validation_weeks, dtype=np.float64) / 51.0,
            "calibration_week_dates": dates[: window.train_stop],
            "stream_week_dates": dates[window.train_stop : window.validation_stop],
            "block_start": np.arange(window.validation_weeks, dtype=np.int64),
            "block_stop": np.arange(1, window.validation_weeks + 1, dtype=np.int64),
        }
        payload.update(_spatial_inducing(coordinates, visible, inducing_sizes))
        np.savez_compressed(npz_path, **payload)
        fold_metadata = {
            "schema_version": 1,
            "development_protocol": True,
            "dataset": metadata["dataset"],
            "formal_protocol_npz": str(formal_npz.resolve()),
            "formal_protocol_sha256": sha256_file(formal_npz),
            "split_seed": 0,
            "fold": {
                "id": fold_id,
                "train_weeks": [1, window.train_stop],
                "validation_weeks": [window.train_stop + 1, window.validation_stop],
                "validation_length": window.validation_weeks,
            },
            "selection_metric": SELECTION_METRIC,
            "target": metadata["target"],
            "target_standardization": {
                "mean": fold_mean,
                "scale": fold_scale,
                "fit_scope": "development training prefix visible locations only",
                "metric_scale": "original log1p(per-100k)",
            },
            "xlag": {
                "mode": "availability-aware state lag4 and visible aggregate lag",
                "delay_weeks": 1,
                "features": len(columns),
                "columns": columns,
                "test_target_lag_information": "Only delayed hidden labels or current visible labels available at the prediction week.",
            },
            "num_locations": 52,
            "num_train_locations": 42,
            "num_test_locations": 10,
            "num_calibration_times": window.train_stop,
            "num_stream_times": window.validation_weeks,
            "npz_sha256": sha256_file(npz_path),
        }
        npz_path.with_suffix(".json").write_text(json.dumps(fold_metadata, indent=2) + "\n", encoding="utf-8")
        output_paths.append(npz_path)
    manifest = {
        "status": "complete",
        "purpose": "seed-0 blocked chronological development only",
        "formal_protocol": str(formal_npz.resolve()),
        "selection_metric": SELECTION_METRIC,
        "folds": [str(path.resolve()) for path in output_paths],
    }
    (output_root / "development_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_paths


def gaussian_metrics_original_scale(
    *,
    target: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    standardization: dict[str, object],
) -> dict[str, float]:
    """Score a standardized archive on its fold's original log1p target scale."""

    scale = float(standardization["scale"])
    offset = float(standardization["mean"])
    y = np.asarray(target, dtype=np.float64) * scale + offset
    mu = np.asarray(mean, dtype=np.float64) * scale + offset
    var = np.maximum(np.asarray(variance, dtype=np.float64) * scale**2, 1e-12)
    error = y - mu
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "gaussian_nlpd": float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + error**2 / var))),
        "coverage90": float(np.mean(np.abs(error) <= 1.6448536269514722 * np.sqrt(var))),
    }
