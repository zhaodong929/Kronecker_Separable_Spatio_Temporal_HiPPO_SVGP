from types import SimpleNamespace
import numpy as np
import pytest
from baselines.causal_mean import get_mean, select_initial_targets


def test_initial_subset_is_indexed_by_public_site_ids():
    observed=SimpleNamespace(locations=np.array([1,4,8]),targets=np.array([[10.,40.,80.],[11.,41.,81.]]))
    np.testing.assert_array_equal(select_initial_targets(observed,[8,1]),[[80,10],[81,11]])
    with pytest.raises(ValueError,match='unreleased'):
        select_initial_targets(observed,[0])


def test_causal_mean_uses_historical_time_and_is_cached(tmp_path):
    p=tmp_path/'protocol.npz'
    np.savez(p,task1_calibration_mean=np.array([[1.,2.],[3.,4.]]),
        task1_stream_mean=np.array([[5.,6.],[7.,8.]]))
    protocol=SimpleNamespace(npz_path=p,locations=2,calibration_times=np.array([0.,1.]),
        chronological_stream_times=np.array([2.,3.]))
    mean=get_mean(protocol)
    np.testing.assert_array_equal(mean.at(1.,[1]),[4.])
    np.testing.assert_array_equal(mean.at(2.,[0]),[5.])
    assert get_mean(protocol) is mean
    with pytest.raises(ValueError,match='outside'):
        mean.at(4.,[0])
