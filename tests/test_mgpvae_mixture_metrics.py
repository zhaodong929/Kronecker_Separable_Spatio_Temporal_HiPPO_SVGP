import numpy as np
import pytest
from benchmarks.three_domain.metrics import gaussian_mixture_metrics


def test_single_component_matches_gaussian_closed_form():
    d = gaussian_mixture_metrics(np.zeros(3), np.zeros((1,3)), 1.)
    assert d['rmse'] == 0
    assert d['nlpd'] == pytest.approx(.5*np.log(2*np.pi))
    assert d['crps'] == pytest.approx((np.sqrt(2)-1)/np.sqrt(np.pi))
    assert d == pytest.approx(gaussian_mixture_metrics(np.zeros(3), np.zeros((4,3)), 1.))
