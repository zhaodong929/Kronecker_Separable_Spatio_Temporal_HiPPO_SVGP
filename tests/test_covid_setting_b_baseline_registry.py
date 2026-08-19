from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baselines.covid_long_setting_b.archive import PredictionArchive
from baselines.covid_long_setting_b.development import build_development_protocols, sha256_file
from baselines.covid_long_setting_b.evaluate_formal_gaussian import methods_for_evaluation
from baselines.covid_long_setting_b.formalization import snapshot_archives, verify_snapshot
from baselines.covid_long_setting_b.generate_baseline_fairness_protocol import build_selected_configs
from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol
from baselines.covid_long_setting_b.run_blocked_development import (
    FACTORIAL_GRID,
    lmc_imc_cross_audits,
    practical_stability,
    summarize_candidate,
    tag,
)


def _write_protocol(tmp_path) -> tuple[object, np.ndarray]:
    calibration = np.arange(52 * 52, dtype=np.float64).reshape(52, 52)
    stream = 100000.0 + np.arange(143 * 52, dtype=np.float64).reshape(143, 52)
    visible = np.arange(42, dtype=np.int64)
    fit = np.arange(38, dtype=np.int64)
    validation = np.arange(38, 42, dtype=np.int64)
    hidden = np.arange(42, 52, dtype=np.int64)
    path = tmp_path / "protocol.npz"
    np.savez_compressed(
        path,
        calibration_y=calibration,
        stream_y=stream,
        train_indices=visible,
        fit_indices=fit,
        validation_indices=validation,
        test_indices=hidden,
        calibration_times=np.arange(52, dtype=np.float64),
        stream_times=np.arange(52, 195, dtype=np.float64),
        coordinates=np.column_stack(
            [np.linspace(25.0, 50.0, 52), np.linspace(-125.0, -65.0, 52)]
        ),
        inducing_coords_ms32=np.zeros((32, 2), dtype=np.float64),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "xlag": {"delay_weeks": 1},
                "target_standardization": {"fit_scope": "Task-1 visible locations only"},
            }
        ),
        encoding="utf-8",
    )
    return path, stream


def test_setting_b_protocol_exposes_only_arrived_labels(tmp_path) -> None:
    path, stream = _write_protocol(tmp_path)
    protocol = COVIDSettingBProtocol(path)

    first = protocol.week(0)
    assert first.delayed_hidden is None
    assert first.current_visible.targets.shape == (42,)
    assert not hasattr(first.hidden_query, "targets")
    assert first.hidden_query.time > protocol.calibration_times[-1]
    assert protocol.chronological_stream_times[0] > protocol.calibration_times[-1]

    fourth = protocol.week(4)
    assert fourth.delayed_hidden is not None
    assert fourth.delayed_hidden.stream_week == 3
    np.testing.assert_array_equal(fourth.delayed_hidden.targets, stream[3, 42:])
    np.testing.assert_array_equal(fourth.current_visible.targets, stream[4, :42])
    assert fourth.hidden_query.stream_week == 4
    assert fourth.hidden_query.locations.shape == (10,)
    np.testing.assert_array_equal(protocol.fit_locations, np.arange(38))
    np.testing.assert_array_equal(protocol.validation_locations, np.arange(38, 42))
    assert protocol.calibration_targets(protocol.validation_locations).shape == (52, 4)


def test_setting_b_prediction_archive_records_complete_leakage_audit(tmp_path) -> None:
    path, stream = _write_protocol(tmp_path)
    protocol = COVIDSettingBProtocol(path)
    archive = PredictionArchive(protocol, method="test_adapter", seed=0)

    for week in range(protocol.online_weeks):
        information = protocol.week(week)
        archive.append(
            information,
            predictive_mean=np.full(10, float(week)),
            predictive_variance=np.full(10, 0.25),
        )

    output = tmp_path / "predictions.npz"
    audit = archive.write(output)
    assert audit["passed"] is True
    assert audit["current_hidden_labels_read"] == 0
    assert audit["delayed_hidden_labels"] == 1420
    with np.load(output, allow_pickle=False) as saved:
        assert saved["pred_mean"].shape == (143, 10)
        assert saved["pred_var"].shape == (143, 10)
        np.testing.assert_array_equal(saved["y_true"], stream[:, 42:])


def test_setting_b_prediction_archive_writes_a_valid_smoke_prefix(tmp_path) -> None:
    path, stream = _write_protocol(tmp_path)
    protocol = COVIDSettingBProtocol(path)
    archive = PredictionArchive(protocol, method="test_adapter", seed=0)

    for week in range(5):
        archive.append(protocol.week(week), np.zeros(10), np.ones(10))

    output = tmp_path / "smoke_predictions.npz"
    audit = archive.write(output, require_complete=False)
    assert audit["online_steps_completed"] == 5
    assert audit["passed"] is False
    with np.load(output, allow_pickle=False) as saved:
        assert saved["y_true"].shape == (5, 10)
        np.testing.assert_array_equal(saved["y_true"], stream[:5, 42:])


def test_setting_b_catalog_keeps_repaired_methods_pending_and_out_of_main_table() -> None:
    path = Path(__file__).resolve().parents[1] / "baselines/covid_long_setting_b/catalog.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    methods = {entry["id"]: entry for entry in catalog["methods"]}

    assert methods["routeb_cumulative_hippo"]["setting_b_status"] == "formal_result_available"
    pending = ("ohsvgp_rbf", "ovc_svgp", "st_svgp", "lmc_svgp", "imc_svgp", "fsde_svi")
    assert all(methods[identifier]["setting_b_status"] == "validation_pending" for identifier in pending)
    assert all(methods[identifier]["category"] == "official_adapter_pending" for identifier in pending)
    assert "M=4, Q=2" in methods["fsde_svi"]["validation_pending_reason"]
    current_main_table = methods_for_evaluation(catalog, fairness=None)
    assert all(identifier not in current_main_table for identifier in pending)
    assert "routeb_cumulative_hippo" in current_main_table
    assert methods["earth"]["setting_b_status"] != "formal_result_available"
    assert methods["earth"]["category"] == "official_protocol_incompatible"

    assert methods["earth"]["source_commit"] == "aeff23fcec2c138a9c859f6bf8897321d9babdbf"
    assert methods["earth"]["incompatibility_evidence"].endswith("earth_setting_b_incompatibility.json")


def test_setting_b_catalog_locks_the_gaussian_metric_contract() -> None:
    path = Path(__file__).resolve().parents[1] / "baselines/covid_long_setting_b/catalog.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    protocol = catalog["protocol"]
    methods = {entry["id"]: entry for entry in catalog["methods"]}

    assert protocol["likelihood"] == "Gaussian"
    assert protocol["formal_metrics"] == [
        "RMSE",
        "CRPS",
        "Gaussian NLPD",
        "ECE",
        "Coverage90",
    ]
    assert protocol["target_scale"] == "log1p(weekly COVID admissions per 100,000)"
    assert all(entry["likelihood"].startswith("Gaussian") for entry in catalog["methods"])
    assert methods["st_svgp"]["setting_b_status"] == "validation_pending"
    assert methods["ovc_svgp"]["setting_b_status"] == "validation_pending"


def test_capacity_policy_matches_only_comparable_inducing_semantics() -> None:
    path = Path(__file__).resolve().parents[1] / "baselines/covid_long_setting_b/capacity_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))["families"]

    assert policy["routeb_ordinary_vs_cumulative_hippo"]["capacity"] == {"Mt": 32, "Ms": 32}
    ovc_grid = policy["ovc_svgp"]["candidates"]
    assert [(item["temporal_inducing"], item["spatial_inducing"]) for item in ovc_grid] == [
        (4, 32), (8, 32), (12, 32)
    ]
    assert [item["spatial_inducing"] for item in policy["st_svgp"]["candidates"]] == [16, 32, 52]
    multi_output = policy["factorial_lmc_imc_fsde"]
    assert [(item["temporal_inducing"], item["latent_rank"]) for item in multi_output["candidates"]] == [
        (16, 4),
        (32, 4),
        (50, 4),
        (32, 8),
        (32, 16),
    ]
    assert multi_output["online_posterior_steps"] == [5, 25, 100]


def test_fairness_lock_excludes_failed_gate_without_blocking_other_methods() -> None:
    factorial_candidate = {"temporal_inducing": 32, "latent_rank": 8}
    capacity = {
        "selected": {
            "ohsvgp": {"candidate": {"inducing_size": 64, "rff_sample_size": 128}},
            "ovc": {"candidate": {"temporal_inducing": 4, "spatial_inducing": 32}},
            "st_svgp": {"candidate": {"spatial_inducing": 32}},
            "factorial_lmc_imc_fsde_shared": {
                "status": "selected",
                "candidate": factorial_candidate,
                "excluded_methods": [],
            },
        }
    }
    online_steps = {
        "selected": {
            short: {"candidate": {**factorial_candidate, "online_inference_steps": 25}}
            for short in ("lmc", "imc", "fsde")
        }
    }

    configs, exclusions = build_selected_configs(
        capacity,
        online_steps,
        ohsvgp_gate_passed=False,
        ovc_memory_passed=True,
    )

    assert "ohsvgp_rbf" not in configs
    assert exclusions["ohsvgp_rbf"] == "official_ohsvgp_reproduction_gate_not_passed"
    assert {"ovc_svgp", "st_svgp", "lmc_svgp", "imc_svgp", "fsde_svi"}.issubset(configs)


def test_blocked_development_folds_restandardize_prefix_and_pass_causal_audit(tmp_path) -> None:
    formal_npz = tmp_path / "formal.npz"
    calibration = np.arange(52 * 52, dtype=np.float64).reshape(52, 52) / 100.0
    dates = np.datetime_as_string(
        np.datetime64("2020-08-01") + np.arange(52) * np.timedelta64(7, "D"), unit="D"
    )
    np.savez_compressed(
        formal_npz,
        calibration_y=calibration,
        train_indices=np.arange(42, dtype=np.int64),
        fit_indices=np.arange(38, dtype=np.int64),
        validation_indices=np.arange(38, 42, dtype=np.int64),
        test_indices=np.arange(42, 52, dtype=np.int64),
        coordinates=np.column_stack(
            [np.linspace(25.0, 50.0, 52), np.linspace(-125.0, -65.0, 52)]
        ),
        calibration_week_dates=dates,
    )
    formal_json = formal_npz.with_suffix(".json")
    formal_json.write_text(
        json.dumps(
            {
                "dataset": "synthetic audit source",
                "target": "log1p(per-100k)",
                "target_standardization": {"mean": 2.0, "scale": 3.0},
            }
        ),
        encoding="utf-8",
    )
    source_hash = sha256_file(formal_npz)
    folds = build_development_protocols(
        formal_npz=formal_npz,
        formal_json=formal_json,
        output_root=tmp_path / "folds",
    )
    assert sha256_file(formal_npz) == source_hash
    assert len(folds) == 3
    first_meta = json.loads(folds[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert first_meta["fold"]["train_weeks"] == [1, 28]
    assert first_meta["fold"]["validation_weeks"] == [29, 36]
    assert first_meta["target_standardization"]["fit_scope"] == "development training prefix visible locations only"
    protocol = COVIDSettingBProtocol(folds[0])
    audit = protocol.make_audit()
    for week in range(protocol.online_weeks):
        audit.record_step(protocol.week(week), np.zeros(10), np.full(10, 0.5))
    assert audit.summary()["passed"] is True


def test_development_selection_uses_mean_nlpd_then_rmse_and_stability_rule() -> None:
    def fold(nlpd: float, rmse: float) -> dict[str, object]:
        return {
            "status": "scored",
            "task1_convergence_status": "converged_elbo_plateau",
            "metrics": {"native_gaussian_nlpd": nlpd, "rmse": rmse, "crps": 0.1, "ece": 0.1, "coverage90": 0.9},
        }

    lower = summarize_candidate("ovc", {"temporal_inducing": 4, "spatial_inducing": 32}, [fold(0.400, 0.200)] * 3)
    upper = summarize_candidate("ovc", {"temporal_inducing": 8, "spatial_inducing": 32}, [fold(0.395, 0.201)] * 3)
    assert upper["mean_native_gaussian_nlpd"] < lower["mean_native_gaussian_nlpd"]
    stability = practical_stability([lower, upper])
    assert stability["status"] == "practical_stability_passed"


def test_frozen_archive_snapshot_detects_drift(tmp_path) -> None:
    archive = tmp_path / "legacy" / "predictions.npz"
    archive.parent.mkdir()
    archive.write_bytes(b"first version")
    snapshot = snapshot_archives([archive])
    assert verify_snapshot(snapshot) == []
    archive.write_bytes(b"changed version")
    assert verify_snapshot(snapshot) == [f"changed: {archive.resolve()}"]


def test_lmc_imc_cross_audit_marks_machine_precision_predictions_as_collapse(tmp_path) -> None:
    phase_root = tmp_path / "capacity"
    candidate = FACTORIAL_GRID[0]
    for method in ("lmc", "imc"):
        for fold in (1, 2, 3):
            output = phase_root / method / tag(candidate) / f"fold_{fold}"
            output.mkdir(parents=True)
            np.savez_compressed(
                output / "predictions.npz",
                y_true=np.zeros((8, 10)),
                pred_mean=np.zeros((8, 10)),
                pred_var=np.ones((8, 10)),
            )
    audits = lmc_imc_cross_audits(phase_root)
    assert audits[tag(candidate)]["status"] == "empirical_collapse"
