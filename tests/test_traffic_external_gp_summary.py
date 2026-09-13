from __future__ import annotations

import numpy as np

from scripts.summarize_traffic_external_gp_a100 import aggregate, metrics


def test_gaussian_archive_metrics_are_finite_and_calibrated() -> None:
    y = np.asarray([-1.0, 0.0, 1.0])
    value = metrics(y, y.copy(), np.ones_like(y))
    assert value["rmse"] == 0.0
    assert np.isfinite(value["crps"])
    assert np.isfinite(value["gaussian_nlpd"])
    assert 0.0 <= value["ece"] <= 1.0
    assert value["coverage90"] == 1.0


def test_aggregate_respects_selected_method_subset() -> None:
    rows = [
        {
            "method": "ohsvgp",
            "rmse": 1.0,
            "rmse_mph": 2.0,
            "crps": 0.5,
            "gaussian_nlpd": 1.2,
            "ece": 0.1,
            "coverage90": 0.9,
        },
        {
            "method": "ohsvgp",
            "rmse": 1.2,
            "rmse_mph": 2.4,
            "crps": 0.6,
            "gaussian_nlpd": 1.3,
            "ece": 0.2,
            "coverage90": 0.8,
        },
    ]
    result = aggregate(rows, ("ohsvgp",))
    assert len(result) == 1
    assert result[0]["n"] == 2
    assert np.isclose(result[0]["rmse_mean"], 1.1)
