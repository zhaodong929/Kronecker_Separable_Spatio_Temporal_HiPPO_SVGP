from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from stvgp_kronecker.data import traffic
from scripts.compute_covid_long_final_metric_system import gaussian_metrics_on_common_scale
from scripts.compute_covid_long_target_likelihood_ece import exact_normal_intervals


def test_scaling_cannot_read_heldout_or_future_targets(tmp_path, monkeypatch):
    raw = np.random.default_rng(12).normal(size=(8, 325))
    frame = pd.DataFrame(raw, index=pd.date_range('2017-01-01', periods=8, freq='5min'))
    folder = tmp_path / 'pems_bay'
    folder.mkdir()
    for filename in ('PEMS-BAY.h5', 'graph_sensor_locations_bay.csv'):
        (folder / filename).touch()
    monkeypatch.setattr(traffic, '_read_traffic_hdf', lambda _: frame)
    monkeypatch.setattr(traffic, '_read_coordinate_table', lambda _: (
        tuple(map(str, range(325))), np.column_stack((np.arange(325), np.arange(325)))))
    def load():
        return traffic.load_traffic_dataset(tmp_path, 'pems_bay', task1_steps=4,
                                           scaler_fit_indices=range(234))
    original = load()
    frame.iloc[:4, 234:] += 10000  # includes validation and test, neither may fit scaling
    frame.iloc[4:, :] -= 10000
    changed = load()
    assert changed.target_mean == original.target_mean
    assert changed.target_scale == original.target_scale
    np.testing.assert_array_equal(changed.values_standardised[:4, :234],
                                  original.values_standardised[:4, :234])
    for invalid in ([], [1, 1], [-1], [325], [0.5]):
        with pytest.raises(ValueError):
            traffic.load_traffic_dataset(tmp_path, 'pems_bay', task1_steps=4,
                                         scaler_fit_indices=invalid)


def test_gaussian_ece_is_seed_invariant_and_scale_equivariant():
    rng = np.random.default_rng(2)
    a = dict(y_true=rng.normal(size=(15, 10)), pred_mean=np.zeros((15, 10)),
             pred_var=np.ones((15, 10)))
    base = gaussian_metrics_on_common_scale(a, dict(mean=0, scale=1), ece_seed=1)
    assert base == gaussian_metrics_on_common_scale(a, dict(mean=0, scale=1), ece_seed=2)
    scaled = gaussian_metrics_on_common_scale(a, dict(mean=10, scale=3), ece_seed=3)
    assert scaled['ece'] == base['ece']
    assert scaled['rmse'] == pytest.approx(3 * base['rmse'])
    assert scaled['crps'] == pytest.approx(3 * base['crps'])
    assert scaled['native_gaussian_nlpd'] == pytest.approx(base['native_gaussian_nlpd'] + np.log(3))
    with pytest.raises(ValueError):
        exact_normal_intervals(np.zeros((2, 2)), np.zeros((2, 2)))
