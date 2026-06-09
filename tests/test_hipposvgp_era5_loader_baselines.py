from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baselines.online_baselines import (
    ClimatologyBaseline,
    GPyTorchSGPRBaseline,
    GPyTorchSVGPBaseline,
    IndependentTemporalGPBaseline,
    PersistenceBaseline,
    RidgeBaseline,
)
from stvgp_kronecker.data.hipposvgp_era5 import (
    build_phi_features,
    iter_online_blocks,
    load_hipposvgp_era5,
    make_routeb_block_factors,
    parse_lat_lon_from_filename,
)


def _write_tiny_task(root: Path, task: str, offset: float = 0.0) -> None:
    seq = root / task / "sequences"
    seq.mkdir(parents=True)
    coords = [(49.0, -0.4), (50.0, 0.2), (51.0, 0.7)]
    for loc_idx, (lat, lon) in enumerate(coords):
        times_train = np.asarray([0.0, 1.0, 2.0]) + offset
        times_val = np.asarray([3.0]) + offset
        times_test = np.asarray([4.0, 5.0]) + offset
        base = lat * 0.01 + lon * 0.02 + loc_idx
        data_train = np.vstack(
            [
                base + 0.1 * times_train,
                2.0 + base + 0.05 * times_train,
            ]
        )
        data_val = np.vstack(
            [
                base + 0.1 * times_val,
                2.0 + base + 0.05 * times_val,
            ]
        )
        data_test = np.vstack(
            [
                base + 0.1 * times_test,
                2.0 + base + 0.05 * times_test,
            ]
        )
        stem = f"lat_{lat:.4f}_lon_{lon:.4f}"
        for suffix, scale in [("", 1.0), ("_scaled", 0.1)]:
            np.savez(
                seq / f"{stem}{suffix}.npz",
                data_train=data_train * scale,
                time_train=times_train,
                data_val=data_val * scale,
                time_val=times_val,
                data_test=data_test * scale,
                time_test=times_test,
            )
    (root / task / "scaler.pkl").write_bytes(b"not-a-real-pickle-for-loader-tests")


@pytest.fixture()
def tiny_era5_root(tmp_path: Path) -> Path:
    root = tmp_path / "processed_timeseries_4"
    _write_tiny_task(root, "task_1", offset=0.0)
    _write_tiny_task(root, "task_2", offset=0.0)
    (root / "global_scaler.pkl").write_bytes(b"not-a-real-pickle-for-loader-tests")
    return root


def test_loader_shapes_and_blocks(tiny_era5_root: Path) -> None:
    dataset = load_hipposvgp_era5(
        tiny_era5_root,
        tasks=("task_1",),
        variable_index=0,
        first_n_locations=2,
        split="all",
    )
    assert dataset.Y.shape == (6, 2)
    assert dataset.coords.shape == (2, 2)
    assert dataset.Phi.shape[0] == 12
    assert dataset.Phi.shape[1] >= 6
    blocks = iter_online_blocks(dataset.Y.shape[0], block_size=2)
    assert [(block.start, block.stop) for block in blocks] == [(0, 2), (2, 4), (4, 6)]


def test_loader_reuses_selected_locations_across_tasks(tiny_era5_root: Path) -> None:
    calibration = load_hipposvgp_era5(
        tiny_era5_root,
        tasks=("task_1",),
        variable_index=0,
        random_n_locations=2,
        seed=1,
        split="all",
    )
    selected = [parse_lat_lon_from_filename(path) for path in calibration.selected_files]
    online = load_hipposvgp_era5(
        tiny_era5_root,
        tasks=("task_2",),
        variable_index=0,
        selected_locations=selected,
        split="all",
    )
    assert online.tasks == ("task_2",)
    assert online.coords.shape == calibration.coords.shape
    assert np.allclose(online.coords, calibration.coords)
    assert online.Y.shape == calibration.Y.shape


def test_loader_converts_to_routeb_factors(tiny_era5_root: Path) -> None:
    dataset = load_hipposvgp_era5(tiny_era5_root, tasks=("task_1",), first_n_locations=2, max_time=4)
    factors = make_routeb_block_factors(
        dataset,
        block=slice(0, 2),
        z_t=np.linspace(dataset.times[0], dataset.times[1], 2),
        z_t_old=None,
        lengthscale=0.5,
        sigma2=0.01,
    )
    assert factors.Y.shape == (2, 2)
    assert factors.Phi.shape[0] == 4
    assert np.isfinite(factors.y_vec).all()


@pytest.mark.parametrize("baseline_cls", [PersistenceBaseline, ClimatologyBaseline, RidgeBaseline])
def test_deterministic_baselines_shapes_no_leakage_and_finite_variance(baseline_cls) -> None:
    times = np.arange(6, dtype=float)
    coords = np.asarray([[49.0, -0.4], [50.0, 0.2]], dtype=float)
    Y = np.column_stack([0.1 * times, 1.0 + 0.2 * times])
    Phi = build_phi_features(times, coords)
    baseline = baseline_cls()
    baseline.fit_initial_task(times[:3], coords, Y[:3], build_phi_features(times[:3], coords))

    future_times = times[3:5]
    future_phi = build_phi_features(future_times, coords)
    pred_before = baseline.predict(future_times, coords, future_phi)
    Y_modified_future = Y.copy()
    Y_modified_future[3:5] += 1000.0
    pred_after_unseen_change = baseline.predict(future_times, coords, future_phi)
    assert np.allclose(pred_before.mean, pred_after_unseen_change.mean)
    assert pred_before.mean.shape == (2, 2)
    assert pred_before.variance.shape == (2, 2)
    assert np.isfinite(pred_before.variance).all()
    assert np.all(pred_before.variance > 0.0)


def test_gpytorch_baselines_smoke_if_available() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    times = np.arange(4, dtype=float)
    coords = np.asarray([[49.0, -0.4]], dtype=float)
    Y = np.sin(times)[:, None]
    Phi = build_phi_features(times, coords)
    for baseline in [
        IndependentTemporalGPBaseline(training_iterations=1),
        GPyTorchSGPRBaseline(training_iterations=1, inducing_points=2),
        GPyTorchSVGPBaseline(training_iterations=1, inducing_points=2),
    ]:
        baseline.fit_initial_task(times, coords, Y, Phi)
        pred = baseline.predict(times[:2], coords, build_phi_features(times[:2], coords))
        assert pred.mean.shape == (2, 1)
        assert pred.variance.shape == (2, 1)
        assert np.isfinite(pred.variance).all()
        assert np.all(pred.variance > 0.0)
