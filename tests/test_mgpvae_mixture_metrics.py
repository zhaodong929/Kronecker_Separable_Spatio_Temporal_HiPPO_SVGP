import numpy as np
import pytest
from benchmarks.three_domain.metrics import gaussian_mixture_metrics


def test_single_component_matches_gaussian_closed_form():
    d = gaussian_mixture_metrics(np.zeros(3), np.zeros((1,3)), 1.)
    assert d['rmse'] == 0
    assert d['nlpd'] == pytest.approx(.5*np.log(2*np.pi))
    assert d['crps'] == pytest.approx((np.sqrt(2)-1)/np.sqrt(np.pi))
    assert d == pytest.approx(gaussian_mixture_metrics(np.zeros(3), np.zeros((4,3)), 1.))


def test_mixture_calibration_uses_mixture_cdf_not_moment_match():
    from benchmarks.three_domain.metrics import gaussian_mixture_calibration
    from scipy.special import ndtri
    y = np.array([-2., 0., 2.])
    levels = np.array([.5, .8, .95])
    actual = gaussian_mixture_calibration(y, np.zeros((1,3)), 1., levels)
    expected = (np.abs(y)[None] <= ndtri((1+levels[:,None])/2)).mean(axis=1)
    np.testing.assert_array_equal(actual['coverage'], expected)
    # With modes at +/-10 and tiny noise, y=5 lies near PIT .5, hence inside
    # the central 50% interval. Moment-matching incorrectly excludes it only
    # for a narrower interval, which also remains around PIT .5 here.
    mixture = gaussian_mixture_calibration([5.], [[-10.], [10.]], .01, [.1])
    assert mixture['coverage'] == [1.]
    assert mixture['ece'] == pytest.approx(.9)
    with pytest.raises(ValueError):
        gaussian_mixture_calibration([], np.empty((1,0)), 1.)
