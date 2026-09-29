from types import SimpleNamespace
import numpy as np


def test_causal_refit_uses_hidden_targets_only_after_release():
    from baselines.covid_long_setting_b.adapters.run_st_svgp import (
        ArrivedObservations, make_model, frozen_hyperparameters, run_causal_segment)
    class Protocol:
        coordinates = np.array([[0., 0.], [.2, .3]])
        calibration_times = np.array([0., .1, .2])
        hidden_locations = np.array([1])
        locations = 2
        _cached_covariate_mean = SimpleNamespace(at=lambda t, sites: np.zeros(len(sites)))
        def __init__(self, y): self.y = y
        def task1(self):
            return SimpleNamespace(locations=np.array([0,1]), targets=np.array([[.1,.3],[.4,.2],[.5,.6]]))
        def week(self, i):
            t = .3 + .1*i
            delayed = None if i == 0 else SimpleNamespace(time=.3+.1*(i-1), locations=np.array([1]), targets=self.y[i-1,1:])
            return SimpleNamespace(delayed_hidden=delayed,
                current_visible=SimpleNamespace(time=t, locations=np.array([0]), targets=self.y[i,:1]),
                hidden_query=SimpleNamespace(time=t, stream_week=i))
    def run(y):
        protocol = Protocol(y)
        arrived = ArrivedObservations(protocol)
        times, grid, targets = arrived.as_grid()
        model = make_model(times, grid, targets, protocol.coordinates, trainable_inducing=False)
        kernel, likelihood = frozen_hyperparameters(model)
        result = run_causal_segment(protocol, protocol.coordinates, kernel, likelihood, arrived, 0, 3, 1, 1.)
        return result[1], result[2]
    y = np.array([[.4,.6],[.2,.3],[.1,.2]])
    baseline = run(y)
    changed = y.copy(); changed[0,1] += 10
    after_release = run(changed)
    np.testing.assert_allclose(baseline[0][0], after_release[0][0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(baseline[1][0], after_release[1][0], atol=1e-12, rtol=0)
    assert np.max(np.abs(baseline[0][1]-after_release[0][1])) > 1e-4
    changed = y.copy(); changed[2] += 10
    future = run(changed)
    np.testing.assert_allclose(baseline[0][:2], future[0][:2], atol=1e-12, rtol=0)
