import numpy as np
import pytest
from benchmarks.task_stream.data import PreparedData, LegalFeatureProvider, normalize_features, prepare, road_features
from benchmarks.task_stream.protocol import Observations


def test_only_released_lags_enter_features():
    provider = LegalFeatureProvider(np.arange(8), np.zeros((3, 2)), context_sites=[0, 1])
    provider.observe(Observations(np.arange(2), [0, 1], np.ones((2, 2))))
    provider.observe(Observations(np.arange(2, 5), [0, 1], np.full((3, 2), 2.)))
    before = provider(np.arange(2, 5), [2])
    assert np.isnan(before[..., 3]).all()
    provider.observe(Observations(np.arange(2, 5), [2], np.full((3, 1), 99.)))
    after = provider(np.arange(5, 6), [2])
    assert after[0, 0, 3] == 99.
    assert np.isnan(before[..., 3]).all()  # Already materialized historical regressors remain frozen.


def test_normalization_ignores_future_and_query_values():
    x = np.arange(60.).reshape(5, 3, 4)
    _, stats = normalize_features(x, 2, [0, 1])
    changed = x.copy(); changed[2:] = 1e9; changed[:, 2] = -1e9
    _, changed_stats = normalize_features(changed, 2, [0, 1])
    assert stats == changed_stats


def test_road_context_has_no_self_or_hidden_label():
    y = np.arange(18.).reshape(6, 3)
    heat = np.ones((3, 3))
    calendar = np.ones((6, 3, 7))
    a = road_features(y, np.zeros((3, 2)), np.array([0, 1]), heat, calendar)
    changed = y.copy(); changed[:, 2] = 1e9
    np.testing.assert_array_equal(a, road_features(changed, np.zeros((3, 2)), np.array([0, 1]), heat, calendar))
    changed = y.copy(); changed[:, 0] = 1e9
    b = road_features(changed, np.zeros((3, 2)), np.array([0, 1]), heat, calendar)
    np.testing.assert_array_equal(a[:, 0], b[:, 0])


def test_covid_preparation_selection_and_delayed_leakage(tmp_path):
    rng = np.random.default_rng(1)
    y, coords = rng.normal(size=(195, 5)), rng.normal(size=(5, 2))
    args = (coords, [0, 1, 2, 3], [4], [0, 1, 2], [3])
    a = prepare('covid', y, *args, seed=5)
    changed = y.copy(); changed[52, 4] += 100; changed[53:] += 10
    b = prepare('covid', changed, *args, seed=5)
    np.testing.assert_array_equal(a.features[:53], b.features[:53])
    np.testing.assert_array_equal(a.selection_features, b.selection_features)
    assert a.features.shape == (195, 5, 13)
    assert len(a.stream.bounds) == 143 and len(a.selection_stream.bounds) == 12
    assert len(a.selection_stream.initial_sites) == 4
    assert a.metadata['target_standardization']['fit_steps'] == 40
    a.save(tmp_path/'prepared')
    assert (tmp_path/'prepared'/'manifest.json').is_file()
    loaded = PreparedData.load(tmp_path/'prepared')
    assert loaded.stream.identity() == a.stream.identity()
    np.testing.assert_array_equal(loaded.features, a.features)
    with pytest.raises(FileExistsError):
        a.save(tmp_path/'prepared')


@pytest.mark.parametrize('dataset,total,initial,tasks,width', [('era5',1860,336,64,133), ('pems',12096,4032,672,28)])
def test_long_settings_preserve_tasks_and_fit_only_scaling(dataset,total,initial,tasks,width):
    rng = np.random.default_rng(15)
    y, coords = rng.normal(size=(total,5)), rng.normal(size=(5,2))
    kwargs = dict(weather=rng.normal(size=(total,5,6))) if dataset == 'era5' else dict(
        calendar=rng.normal(size=(total,5,7)), road_heat=np.ones((5,5)))
    seed = 0 if dataset == 'era5' else 1
    a = prepare(dataset,y,coords,[0,1,2,3],[4],[0,1,2],[3],seed=seed,**kwargs)
    changed = y.copy();changed[:,4] += 1000;changed[initial:] += 100
    b = prepare(dataset,changed,coords,[0,1,2,3],[4],[0,1,2],[3],seed=seed,**kwargs)
    assert a.metadata['target_standardization'] == b.metadata['target_standardization']
    np.testing.assert_array_equal(a.selection_features,b.selection_features)
    np.testing.assert_array_equal(a.features[:initial],b.features[:initial])
    assert a.features.shape == (total,5,width)
    assert len(a.stream.bounds) == tasks
    assert a.stream.bounds[-1][1] == total
    assert a.stream.initial_sites.tolist() == [0,1,2,3]
    assert a.selection_stream.coordinates.shape == (4,2)
    if dataset == 'era5':
        np.testing.assert_allclose(a.features[:-24, :, 1:3], a.features[24:, :, 1:3], atol=1e-12)
        np.testing.assert_allclose(a.features[:-168, :, 3:5], a.features[168:, :, 3:5], atol=1e-12)
        assert a.metadata['feature_phase']['periods'] == [24,168]
        assert a.metadata['time_divisor'] == 335
