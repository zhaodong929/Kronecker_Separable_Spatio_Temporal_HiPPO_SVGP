import numpy as np
from scripts.prepare_fair_covid import features


def test_feature_fit_cannot_read_validation_or_future_and_hidden_lag_releases_once():
    rng = np.random.default_rng(41)
    y, coordinates = rng.normal(size=(195,52)), rng.normal(size=(52,2))
    fit = np.arange(38)
    base, stats = features(y,coordinates,fit)
    modified = y.copy()
    modified[:52,38:] += 200
    changed, changed_stats = features(modified,coordinates,fit)
    np.testing.assert_array_equal(base[:52],changed[:52])
    assert stats == changed_stats
    modified = y.copy()
    modified[52,51] += 200
    changed, changed_stats = features(modified,coordinates,fit)
    np.testing.assert_array_equal(base[:53],changed[:53])
    assert abs(base[53,51,4]-changed[53,51,4]) > 1
    assert stats == changed_stats
