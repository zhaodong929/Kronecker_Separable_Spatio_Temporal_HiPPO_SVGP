import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name, forbid_backend=False):
    original_import = builtins.__import__

    def guarded_import(module_name, *args, **kwargs):
        if forbid_backend and module_name.split(".")[0] in {"bayesnewton", "jax", "jaxlib", "objax"}:
            raise AssertionError("protocol helpers must not import %s" % module_name)
        return original_import(module_name, *args, **kwargs)

    if forbid_backend:
        builtins.__import__ = guarded_import
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if forbid_backend:
            builtins.__import__ = original_import


def test_wrapper_pure_protocol_helpers_do_not_require_legacy_backend():
    module = load_module(
        ROOT / "scripts" / "run_official_stvgp_legacy.py",
        "official_stvgp_validation_refit_test_module",
        forbid_backend=True,
    )
    times = np.arange(4, dtype=float)
    coordinates = np.column_stack([np.arange(1000), -np.arange(1000)]).astype(float)
    y = np.arange(times.size * 1000, dtype=float).reshape(times.size, 1000)
    mean = 0.25 * y
    fit = np.arange(720)
    validation = np.arange(720, 800)
    train = np.arange(800)
    test = np.arange(800, 1000)
    auxiliary = {
        "fit_coords": coordinates[fit],
        "validation_coords": coordinates[validation],
        "y_fit": y[:, fit],
        "y_validation": y[:, validation],
        "xlag_mean_fit": mean[:, fit],
        "xlag_mean_validation": mean[:, validation],
        "xlag_mean_train": mean[:, train],
        "xlag_mean_test": mean[:, test],
        "train_indices": train,
        "test_indices": test,
    }
    values = (times, coordinates[train], coordinates[test], y[:, train], y[:, test])
    splits, learned = module.prepare_splits(values, auxiliary, use_xlag_mean=True)

    assert learned is None
    assert splits["has_validation"]
    assert splits["fit"]["coords"].shape == (720, 2)
    assert splits["validation"]["coords"].shape == (80, 2)
    assert splits["train"]["coords"].shape == (800, 2)
    assert splits["test"]["coords"].shape == (200, 2)
    np.testing.assert_allclose(splits["validation"]["model_y"], 0.75 * y[:, validation])

    selection = module.make_phase_data(
        "st_vgp",
        times,
        splits["fit"],
        [("validation", splits["validation"]), ("test", splits["test"])],
    )
    assert selection["model_coords"].shape == (1000, 2)
    assert selection["y_train_grid"].shape == (4, 1000, 1)
    assert np.isfinite(selection["y_train_grid"][:, :720]).all()
    assert np.isnan(selection["y_train_grid"][:, 720:]).all()
    assert (selection["slices"]["validation"].start, selection["slices"]["validation"].stop) == (720, 800)
    assert (selection["slices"]["test"].start, selection["slices"]["test"].stop) == (800, 1000)

    refit = module.make_phase_data(
        "st_vgp", times, splits["train"], [("test", splits["test"]) ]
    )
    assert refit["model_coords"].shape == (1000, 2)
    assert np.isfinite(refit["y_train_grid"][:, :800]).all()
    assert np.isnan(refit["y_train_grid"][:, 800:]).all()
    assert (refit["slices"]["test"].start, refit["slices"]["test"].stop) == (800, 1000)

    sparse_selection = module.make_phase_data(
        "st_svgp",
        times,
        splits["fit"],
        [("validation", splits["validation"]), ("test", splits["test"])],
    )
    sparse_refit = module.make_phase_data(
        "st_svgp", times, splits["train"], [("test", splits["test"]) ]
    )
    assert sparse_selection["model_coords"].shape == (720, 2)
    assert sparse_refit["model_coords"].shape == (800, 2)
    assert np.isfinite(sparse_selection["y_train_grid"]).all()
    assert np.isfinite(sparse_refit["y_train_grid"]).all()


def test_exporter_exports_all_splits_and_preserves_protocol_inducing_arrays(tmp_path):
    module = load_module(
        ROOT / "scripts" / "export_iclr_protocol_for_official_stvgp.py",
        "official_stvgp_protocol_export_test_module",
        forbid_backend=True,
    )
    times = np.arange(3, dtype=float)
    coordinates = np.column_stack([np.arange(1000), np.arange(1000) + 0.5]).astype(float)
    y = np.arange(times.size * 1000, dtype=float).reshape(times.size, 1000)
    mean = y + 10.0
    fit = np.arange(720)
    validation = np.arange(720, 800)
    train = np.arange(800)
    test = np.arange(800, 1000)
    inducing64 = np.full((64, 2), 64.25)
    inducing128 = np.full((128, 2), 128.25)
    source = tmp_path / "protocol.npz"
    output = tmp_path / "official.npz"
    np.savez_compressed(
        source,
        stream_times=times,
        coordinates=coordinates,
        stream_y=y,
        batch_stream_mean=mean,
        fit_indices=fit,
        validation_indices=validation,
        train_indices=train,
        test_indices=test,
        inducing_coords_ms64=inducing64,
        inducing_coords_ms128=inducing128,
    )

    module.export_protocol(source, output)
    with np.load(output) as arrays:
        for key, size in {
            "fit_coords": 720,
            "validation_coords": 80,
            "train_coords": 800,
            "test_coords": 200,
        }.items():
            assert arrays[key].shape[0] == size
        for key, size in {
            "y_fit": 720,
            "y_validation": 80,
            "y_train": 800,
            "y_test": 200,
            "xlag_mean_fit": 720,
            "xlag_mean_validation": 80,
            "xlag_mean_train": 800,
            "xlag_mean_test": 200,
        }.items():
            assert arrays[key].shape[1] == size
        np.testing.assert_array_equal(arrays["fit_coords"], coordinates[fit])
        np.testing.assert_array_equal(arrays["validation_coords"], coordinates[validation])
        np.testing.assert_array_equal(arrays["train_coords"], coordinates[train])
        np.testing.assert_array_equal(arrays["test_coords"], coordinates[test])
        np.testing.assert_array_equal(arrays["inducing_coords_ms64"], inducing64)
        np.testing.assert_array_equal(arrays["inducing_coords_ms128"], inducing128)


def test_validation_checkpoints_and_long_taskwise_slices_are_deterministic():
    module = load_module(
        ROOT / "scripts" / "run_official_stvgp_legacy.py",
        "official_stvgp_validation_metrics_test_module",
        forbid_backend=True,
    )
    checkpoints = [
        iteration
        for iteration in range(1, 26)
        if module.validation_checkpoint(iteration, 25, 10)
    ]
    assert checkpoints == [10, 20, 25]

    y_true = np.zeros((1674, 200))
    mean = np.zeros_like(y_true)
    variance = np.ones_like(y_true)
    rows = module.taskwise_metrics(y_true, mean, variance)
    assert len(rows) == 9
    assert [(row["task"], row["start"], row["stop"]) for row in rows] == [
        (task, (task - 2) * 186, (task - 1) * 186) for task in range(2, 11)
    ]
    assert all(np.isclose(row["nll"], 0.5 * np.log(2.0 * np.pi)) for row in rows)


def test_validation_nll_selects_step_and_refit_uses_exactly_that_count():
    module = load_module(
        ROOT / "scripts" / "run_official_stvgp_legacy.py",
        "official_stvgp_selection_refit_test_module",
        forbid_backend=True,
    )

    class FakeModel:
        def __init__(self):
            self.iteration = 0

        def predict_y(self, X, R):
            error = {10: 2.0, 20: 0.5, 25: 1.0}[self.iteration]
            return np.asarray([[error]]), np.asarray([[1.0]])

    def built_phase(model):
        observed = module._record(
            np.asarray([[0.0, 0.0]]), np.asarray([[0.0]]), np.asarray([[0.0]])
        )
        phase = module.make_phase_data("st_svgp", np.asarray([0.0]), observed)

        def train_op(current_y):
            model.iteration += 1
            return [np.asarray(float(model.iteration))]

        return {"model": model, "phase": phase, "train_op": train_op}, observed

    args = SimpleNamespace(
        seed=0,
        validation_every=10,
        trajectory_every=0,
        log_every=1000,
        early_stop_relative_tol=None,
        early_stop_min_iterations=20,
        early_stop_patience=10,
    )
    selection_built, validation = built_phase(FakeModel())
    selection = module._train_phase(
        args,
        selection_built,
        np.asarray([0.0]),
        25,
        validation_record=validation,
    )
    assert [row["iteration"] for row in selection["validation_trace"]] == [10, 20, 25]
    assert selection["best_iteration"] == 20
    assert selection["best_step"] == 20
    assert np.isclose(
        selection["best_validation_nll"],
        0.5 * (np.log(2.0 * np.pi) + 0.25),
    )

    refit_built, _ = built_phase(FakeModel())
    refit = module._train_phase(
        args,
        refit_built,
        np.asarray([0.0]),
        selection["best_iteration"],
    )
    assert refit_built["model"] is not selection_built["model"]
    assert refit["iterations"] == 20
