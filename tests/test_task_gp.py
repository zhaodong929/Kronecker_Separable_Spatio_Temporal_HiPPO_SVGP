"""Small task-level checks; these do not qualify full-size device throughput."""
import numpy as np
import pytest
import torch
from benchmarks.task_stream.protocol import TaskStream
from benchmarks.task_stream.gp import KronTaskAdapter


def stream(release=False, mutation=False):
    y = np.random.default_rng(91).normal(size=(7, 4))
    if mutation:
        y[2:, 2:] += 1000.
    return TaskStream(times=np.arange(7.) / 10, targets=y,
        coordinates=np.array([[0., 0.], [.2, .4], [.6, .3], [.9, .8]]),
        visible=[0, 1], hidden=[2, 3], initial_sites=[0, 1],
        initial_steps=2, task_steps=2, release_previous=release)


def kron(s, multiple):
    return KronTaskAdapter(s.coordinates, np.array([0, 1]), s.coordinates[[0, 2]],
        dict(ell_s=[.7, .8], ell_t=.4, noise_std=.3, kernel_variance=1.),
        initial_step=.1, features=lambda t, sites: np.ones((len(t)*len(sites), 1)),
        feature_dimension=1, mt=3, rff=32, seed=11, multiple_geometry=multiple)


def test_kron_single_geometry_matches_general_solver_over_tasks():
    s = stream()
    first, second = kron(s, False), kron(s, True)
    first.initialize(s.initial()); second.initialize(s.initial())
    for i in range(3):
        a, b = first.predict_task(s.task(i)), second.predict_task(s.task(i))
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=2e-5, atol=2e-7)
            assert np.isfinite(np.asarray(x)).all()
        assert np.all(np.asarray(a[1]) > 0)


@pytest.mark.parametrize('release', [False, True])
def test_kron_unreleased_labels_do_not_affect_prediction(release):
    original, changed = stream(release), stream(release, True)
    a, b = kron(original, release), kron(changed, release)
    a.initialize(original.initial()); b.initialize(changed.initial())
    for i in range(1 if release else 3):
        for x, y in zip(a.predict_task(original.task(i)), b.predict_task(changed.task(i))):
            np.testing.assert_allclose(x, y, rtol=0, atol=0)
    if release:
        p, q = a.predict_task(original.task(1)), b.predict_task(changed.task(1))
        assert not np.allclose(p[0], q[0])


def test_oh_task_recurrence_matches_pinned_official_transition():
    from benchmarks.task_stream.gp import OHSVGPTaskAdapter
    from scripts import run_covid_ohsvgp_own_theta  # installs pinned official source path
    from hipposvgp.multidim import SE_kernel
    from hipposvgp.likelihood import GaussianLikelihood
    from hipposvgp.hippo import HiPPO_LegS
    s = stream(True)
    def create():
        return OHSVGPTaskAdapter(s.coordinates, SE_kernel(3).to(dtype=torch.float64),
            GaussianLikelihood(.2).to(dtype=torch.float64), inducing_size=3, rff=16,
            initial_steps=2, update_steps=2, grid_rows=3, batch_rows=3, seed=18, train_initial_kernel=False)
    # Construction of OHSVGPTaskAdapter imports the official tree.
    a, b = create(), create()
    b.hippo = HiPPO_LegS(3, 'cpu', max_length=100)
    torch.manual_seed(19); a.initialize(s.initial())
    torch.manual_seed(19); b.initialize(s.initial())
    assert a.previous_steps == b.previous_steps == 3
    for i in range(3):
        torch.manual_seed(20+i); actual = a.predict_task(s.task(i))
        torch.manual_seed(20+i); expected = b.predict_task(s.task(i))
        for x, y in zip(actual, expected):
            np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-10)
        for key in a.state:
            torch.testing.assert_close(a.state[key], b.state[key], rtol=1e-9, atol=1e-10)
        assert np.all(actual[1] > 0)


def test_kron_seed_is_independent_of_global_torch_rng():
    s = stream()
    torch.manual_seed(13); a = kron(s, False)
    torch.manual_seed(913); b = kron(s, False)
    a.initialize(s.initial()); b.initialize(s.initial())
    for x, y in zip(a.predict_task(s.task(0)), b.predict_task(s.task(0))):
        np.testing.assert_array_equal(x, y)


def test_kron_task_adapter_delayed_geometries_match_dense_joint_information():
    from scipy.linalg import block_diag
    s = stream(True); adapter = kron(s, True)
    model = adapter.model
    model.jitter = 0.  # isolate algebra from deliberate numerical regularization
    original_update = model.update_block_structured_joint_ssgp_transfer
    r = np.zeros((7, 7)); h = np.zeros(7)
    covariance = mean = None
    def update(**kw):
        nonlocal r, h, covariance, mean
        t = np.asarray(kw['T_n']); kt = np.asarray(kw['Kt_new'])
        c, phi, y = np.asarray(kw['C_observed']), np.asarray(kw['Phi']), np.asarray(kw['y_vec'])
        if kw['state'] is not None and kw['K_on_t'] is not None:
            # Transfer the historical information in an independently assembled
            # full matrix, before adding exactly the newly released likelihood.
            l = np.linalg.solve(kt, np.asarray(kw['K_on_t']).T).T
            transfer = block_diag(np.eye(1), np.kron(l, np.eye(2)))
            r = transfer.T @ r @ transfer; h = transfer.T @ h
        design = np.column_stack([phi, np.kron(t, c)])
        r += design.T @ design / model.sigma2
        h += design.T @ y / model.sigma2
        prior = block_diag(np.eye(1), np.kron(np.linalg.inv(kt), np.asarray(model.Ks_inv)))
        covariance = np.linalg.inv(prior+r); mean = covariance @ h
        state = original_update(**kw)
        actual = np.r_[np.asarray(state.beta_mean), np.asarray(state.M_u).reshape(-1, order='F')]
        np.testing.assert_allclose(actual, mean, rtol=3e-5, atol=2e-6)
        return state
    model.update_block_structured_joint_ssgp_transfer = update
    adapter.initialize(s.initial())
    for i in range(3):
        task = s.task(i)
        pred, var = adapter.predict_task(task)
        t, kt, _ = adapter._factors(task.times, adapter.spec)
        c = adapter.c[task.query_sites]
        gp = np.kron(np.asarray(t), c)
        x = np.column_stack([np.ones(len(gp)), gp])
        residual = 1. - np.einsum('ni,ij,nj->n', gp, np.kron(np.asarray(kt), np.asarray(model.Ks)), gp)
        expected_var = model.sigma2 + residual + np.einsum('ni,ij,nj->n', x, covariance, x)
        np.testing.assert_allclose(pred.reshape(-1), x @ mean, rtol=3e-5, atol=2e-6)
        np.testing.assert_allclose(var.reshape(-1), expected_var, rtol=3e-5, atol=2e-6)


def test_oh_initial_kernel_learns_then_is_frozen_across_tasks():
    from scripts import run_covid_ohsvgp_own_theta
    from benchmarks.task_stream.gp import OHSVGPTaskAdapter
    from hipposvgp.multidim import SE_kernel
    from hipposvgp.likelihood import GaussianLikelihood
    s = stream(True)
    kernel = SE_kernel(3).to(dtype=torch.float64)
    initial_lengthscales = kernel.log_ls.detach().clone()
    adapter = OHSVGPTaskAdapter(s.coordinates, kernel, GaussianLikelihood(.2).to(dtype=torch.float64),
        inducing_size=3, rff=16, initial_steps=3, update_steps=2, grid_rows=3, batch_rows=3, seed=18)
    adapter.initialize(s.initial())
    assert not torch.equal(adapter.kernel.log_ls, initial_lengthscales)
    selected = [x.detach().clone() for x in adapter.kernel.parameters()]
    selected_noise = adapter.likelihood.variance.detach().clone()
    for i in range(3):
        mean, variance = adapter.predict_task(s.task(i))
        assert np.isfinite(mean).all() and np.all(variance > 0)
        for value, expected in zip(adapter.kernel.parameters(), selected):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        torch.testing.assert_close(adapter.likelihood.variance, selected_noise, rtol=0, atol=0)
