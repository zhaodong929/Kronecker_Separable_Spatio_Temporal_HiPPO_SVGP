from __future__ import annotations

import math

import numpy as np
import torch

from stvgp_kronecker.routeb_empirical_bayes import (
    BatchRouteBEmpiricalBayes,
    TemporalFactorModel,
    finite_joint_nlml_from_factors,
    matern32_1d,
    matern32_separable,
    separable_vfe_trace_residual,
    vfe_corrected_joint_nlml_from_factors,
)
from stvgp_kronecker.joint_ssgp_kron.synthetic import temporal_spec_for_block


def test_joint_nlml_matches_dense_covariance() -> None:
    torch.manual_seed(4)
    dtype = torch.float64
    ns, nt, ms, mt, p = 4, 3, 2, 2, 3
    y = torch.randn(ns, nt, dtype=dtype)
    phi = torch.randn(ns, nt, p, dtype=dtype)
    c = torch.randn(ns, ms, dtype=dtype)
    t = torch.randn(nt, mt, dtype=dtype)
    raw_s = torch.randn(ms, ms, dtype=dtype)
    raw_t = torch.randn(mt, mt, dtype=dtype)
    ks = raw_s @ raw_s.T + 0.5 * torch.eye(ms, dtype=dtype)
    kt = raw_t @ raw_t.T + 0.5 * torch.eye(mt, dtype=dtype)
    noise_var = torch.tensor(0.17, dtype=dtype)
    beta_var = 7.0

    actual = finite_joint_nlml_from_factors(
        y_matrix=y,
        phi_tensor=phi,
        temporal_projection=t,
        spatial_projection=c,
        temporal_prior=kt,
        spatial_prior=ks,
        noise_variance=noise_var,
        beta_prior_variance=beta_var,
    ).nlml_per_observation

    a_u = torch.kron(t, c)
    phi_flat = phi.permute(1, 0, 2).reshape(ns * nt, p)
    y_flat = y.T.reshape(-1)
    covariance = (
        noise_var * torch.eye(ns * nt, dtype=dtype)
        + beta_var * phi_flat @ phi_flat.T
        + a_u @ torch.kron(kt, ks) @ a_u.T
    )
    chol = torch.linalg.cholesky(covariance)
    alpha = torch.cholesky_solve(y_flat[:, None], chol).squeeze(1)
    expected = 0.5 * (
        torch.dot(y_flat, alpha)
        + 2.0 * torch.log(torch.diagonal(chol)).sum()
        + y_flat.numel() * math.log(2.0 * math.pi)
    ) / y_flat.numel()
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-7)


def test_zero_mean_nlml_matches_dense_covariance() -> None:
    torch.manual_seed(5)
    dtype = torch.float64
    ns, nt, ms, mt = 4, 3, 2, 2
    y = torch.randn(ns, nt, dtype=dtype)
    phi = torch.empty(ns, nt, 0, dtype=dtype)
    c = torch.randn(ns, ms, dtype=dtype)
    t = torch.randn(nt, mt, dtype=dtype)
    raw_s = torch.randn(ms, ms, dtype=dtype)
    raw_t = torch.randn(mt, mt, dtype=dtype)
    ks = raw_s @ raw_s.T + 0.5 * torch.eye(ms, dtype=dtype)
    kt = raw_t @ raw_t.T + 0.5 * torch.eye(mt, dtype=dtype)
    noise_var = torch.tensor(0.17, dtype=dtype)

    actual = finite_joint_nlml_from_factors(
        y_matrix=y,
        phi_tensor=phi,
        temporal_projection=t,
        spatial_projection=c,
        temporal_prior=kt,
        spatial_prior=ks,
        noise_variance=noise_var,
        beta_prior_variance=7.0,
    ).nlml_per_observation

    a_u = torch.kron(t, c)
    y_flat = y.T.reshape(-1)
    covariance = (
        noise_var * torch.eye(ns * nt, dtype=dtype)
        + a_u @ torch.kron(kt, ks) @ a_u.T
    )
    chol = torch.linalg.cholesky(covariance)
    alpha = torch.cholesky_solve(y_flat[:, None], chol).squeeze(1)
    expected = 0.5 * (
        torch.dot(y_flat, alpha)
        + 2.0 * torch.log(torch.diagonal(chol)).sum()
        + y_flat.numel() * math.log(2.0 * math.pi)
    ) / y_flat.numel()
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-7)


def test_rank_deficient_temporal_objective_matches_dense_and_has_gradients() -> None:
    torch.manual_seed(9)
    dtype = torch.float64
    ns, nt, ms, mt, p = 5, 2, 3, 6, 2
    y = torch.randn(ns, nt, dtype=dtype)
    phi = torch.randn(ns, nt, p, dtype=dtype)
    c = torch.randn(ns, ms, dtype=dtype)
    t = torch.randn(nt, mt, dtype=dtype, requires_grad=True)
    raw_s = torch.randn(ms, ms, dtype=dtype)
    raw_t = torch.randn(mt, mt, dtype=dtype)
    ks = raw_s @ raw_s.T + 0.7 * torch.eye(ms, dtype=dtype)
    kt = raw_t @ raw_t.T + 0.7 * torch.eye(mt, dtype=dtype)
    noise_var = torch.tensor(0.21, dtype=dtype, requires_grad=True)
    beta_var = 4.0

    actual = finite_joint_nlml_from_factors(
        y_matrix=y,
        phi_tensor=phi,
        temporal_projection=t,
        spatial_projection=c,
        temporal_prior=kt,
        spatial_prior=ks,
        noise_variance=noise_var,
        beta_prior_variance=beta_var,
    ).nlml_per_observation

    a_u = torch.kron(t, c)
    phi_flat = phi.permute(1, 0, 2).reshape(ns * nt, p)
    y_flat = y.T.reshape(-1)
    covariance = (
        noise_var * torch.eye(ns * nt, dtype=dtype)
        + beta_var * phi_flat @ phi_flat.T
        + a_u @ torch.kron(kt, ks) @ a_u.T
    )
    chol = torch.linalg.cholesky(covariance)
    alpha = torch.cholesky_solve(y_flat[:, None], chol).squeeze(1)
    expected = 0.5 * (
        torch.dot(y_flat, alpha)
        + 2.0 * torch.log(torch.diagonal(chol)).sum()
        + y_flat.numel() * math.log(2.0 * math.pi)
    ) / y_flat.numel()
    torch.testing.assert_close(actual, expected, rtol=3e-7, atol=3e-7)
    actual.backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()
    assert noise_var.grad is not None and torch.isfinite(noise_var.grad)


def test_all_five_hyperparameters_have_finite_gradients_and_fixed_supports() -> None:
    torch.manual_seed(2)
    np.random.seed(2)
    times = np.linspace(0.0, 1.0, 6)
    z_s = np.asarray([[-1.0, -0.5], [0.2, 0.4], [1.0, 0.8]])
    model = BatchRouteBEmpiricalBayes(
        times=times,
        spatial_inducing=z_s,
        mt=4,
        representation="analytic_hippo_rff",
        initial_ell_t=0.1,
        initial_ell_s=(0.4, 0.5),
        initial_kernel_variance=0.8,
        initial_noise_std=0.2,
        rff_sample_size=32,
        seed=3,
    )
    coords = torch.as_tensor(
        [[-1.0, -0.4], [-0.1, 0.3], [0.9, 0.7], [1.2, -0.2]],
        dtype=torch.float64,
    )
    y = torch.randn(4, 6, dtype=torch.float64)
    phi = torch.randn(4, 6, 3, dtype=torch.float64)
    base_before = model.temporal.builder.base_frequencies.detach().clone()
    z_before = model.spatial_inducing.detach().clone()
    loss = model.objective(
        y_matrix=y,
        phi_tensor=phi,
        spatial_coordinates=coords,
        beta_prior_variance=10.0,
    ).nlml_per_observation
    loss.backward()
    named = dict(model.named_parameters())
    expected = {
        "temporal.builder.log_lengthscale",
        "temporal.builder.log_variance",
        "log_spatial_lengthscales",
        "log_noise_std",
    }
    assert set(named) == expected
    assert sum(parameter.numel() for parameter in named.values()) == 5
    assert all(parameter.grad is not None for parameter in named.values())
    assert all(torch.isfinite(parameter.grad).all() for parameter in named.values())
    torch.testing.assert_close(model.temporal.builder.base_frequencies, base_before)
    torch.testing.assert_close(model.spatial_inducing, z_before)


def test_temporal_query_can_use_a_separate_cumulative_hippo_horizon() -> None:
    full_times = np.linspace(0.0, 1.0, 12)
    query_times = full_times[8:12]
    horizon = temporal_spec_for_block(full_times, slice(0, 12), moving=True)
    temporal = TemporalFactorModel(
        times=full_times[:4],
        mt=6,
        representation="analytic_hippo_rff",
        initial_lengthscale=0.2,
        initial_variance=0.8,
        rff_sample_size=32,
        seed=7,
    )
    frequencies_before = temporal.builder.base_frequencies.detach().clone()
    temporal.set_query_times(query_times, temporal_horizon=horizon)
    projection, prior = temporal.factors()

    expected_kuu = temporal.builder.compute_kuu_t(horizon)
    expected_kfu = temporal.builder.compute_kfu_t(query_times, horizon)
    expected_projection = torch.cholesky_solve(
        expected_kfu.T,
        torch.linalg.cholesky(
            expected_kuu
            + 1e-7 * torch.eye(expected_kuu.shape[0], dtype=expected_kuu.dtype)
        ),
    ).T
    assert projection.shape == (query_times.size, 6)
    assert prior.shape == (6, 6)
    torch.testing.assert_close(projection, expected_projection, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        temporal.builder.base_frequencies,
        frequencies_before,
        rtol=0.0,
        atol=0.0,
    )


def test_set_theta_changes_only_hyperparameters() -> None:
    model = BatchRouteBEmpiricalBayes(
        times=np.linspace(0.0, 1.0, 8),
        spatial_inducing=np.asarray([[-1.0, 0.0], [0.0, 0.5], [1.0, -0.5]]),
        mt=5,
        representation="analytic_hippo_rff",
        initial_ell_t=0.1,
        initial_ell_s=(0.4, 0.5),
        initial_kernel_variance=1.0,
        initial_noise_std=0.1,
        rff_sample_size=24,
        seed=2,
    )
    frequencies_before = model.temporal.builder.base_frequencies.detach().clone()
    spatial_before = model.spatial_inducing.detach().clone()
    theta = {
        "ell_t": 0.07,
        "ell_s": [0.8, 0.9],
        "kernel_variance": 0.6,
        "noise_std": 0.12,
    }
    model.set_theta(theta)

    actual = model.theta()
    np.testing.assert_allclose(actual["ell_t"], theta["ell_t"])
    np.testing.assert_allclose(actual["ell_s"], theta["ell_s"])
    np.testing.assert_allclose(actual["kernel_variance"], theta["kernel_variance"])
    np.testing.assert_allclose(actual["noise_std"], theta["noise_std"])
    torch.testing.assert_close(model.temporal.builder.base_frequencies, frequencies_before)
    torch.testing.assert_close(model.spatial_inducing, spatial_before)


def test_separable_vfe_trace_matches_dense_covariance() -> None:
    dtype = torch.float64
    times = torch.tensor([0.0, 0.2, 0.55, 1.0], dtype=dtype)
    z_t = torch.tensor([0.0, 0.5, 1.0], dtype=dtype)
    coords = torch.tensor(
        [[-0.8, -0.4], [-0.2, 0.1], [0.4, 0.7], [0.9, -0.1]], dtype=dtype
    )
    z_s = coords[[0, 2, 3]]
    ell_t = torch.tensor(0.31, dtype=dtype)
    variance = torch.tensor(0.73, dtype=dtype)
    ell_s = torch.tensor([0.6, 0.8], dtype=dtype)

    ktt = matern32_1d(times, times, ell_t, variance)
    kt = matern32_1d(z_t, z_t, ell_t, variance)
    kft = matern32_1d(times, z_t, ell_t, variance)
    t = torch.linalg.solve(kt, kft.T).T
    kss = matern32_separable(coords, coords, ell_s)
    ks = matern32_separable(z_s, z_s, ell_s)
    kfs = matern32_separable(coords, z_s, ell_s)
    c = torch.linalg.solve(ks, kfs.T).T

    actual = separable_vfe_trace_residual(
        temporal_projection=t,
        spatial_projection=c,
        temporal_prior=kt,
        spatial_prior=ks,
        prior_point_variance=variance,
    )
    qff = torch.kron(t @ kt @ t.T, c @ ks @ c.T)
    kff = torch.kron(ktt, kss)
    expected = torch.trace(kff - qff)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)


def test_vfe_objective_is_finite_nlml_plus_trace_correction() -> None:
    dtype = torch.float64
    times = torch.tensor([0.0, 0.4, 1.0], dtype=dtype)
    z_t = torch.tensor([0.0, 1.0], dtype=dtype)
    coords = torch.tensor([[-0.5, 0.0], [0.1, 0.3], [0.8, -0.2]], dtype=dtype)
    z_s = coords[[0, 2]]
    ell_t = torch.tensor(0.35, dtype=dtype)
    variance = torch.tensor(0.8, dtype=dtype)
    ell_s = torch.tensor([0.7, 0.9], dtype=dtype)
    kt = matern32_1d(z_t, z_t, ell_t, variance)
    t = torch.linalg.solve(kt, matern32_1d(times, z_t, ell_t, variance).T).T
    ks = matern32_separable(z_s, z_s, ell_s)
    c = torch.linalg.solve(ks, matern32_separable(coords, z_s, ell_s).T).T
    y = torch.arange(9, dtype=dtype).reshape(3, 3) / 7.0
    phi = torch.ones(3, 3, 1, dtype=dtype)
    noise = torch.tensor(0.12, dtype=dtype)

    actual = vfe_corrected_joint_nlml_from_factors(
        y_matrix=y,
        phi_tensor=phi,
        temporal_projection=t,
        spatial_projection=c,
        temporal_prior=kt,
        spatial_prior=ks,
        noise_variance=noise,
        beta_prior_variance=5.0,
        prior_point_variance=variance,
    )
    expected = (
        actual.finite_nlml_per_observation
        + actual.vfe_trace_residual_per_observation / (2.0 * noise)
    )
    torch.testing.assert_close(actual.nlml_per_observation, expected)
    assert float(actual.vfe_trace_correction_per_observation) > 0.0


def test_vfe_trace_is_zero_with_full_inducing_grid() -> None:
    dtype = torch.float64
    times = torch.linspace(0.0, 1.0, 4, dtype=dtype)
    coords = torch.tensor([[-0.5, 0.0], [0.2, 0.4], [0.8, -0.2]], dtype=dtype)
    ell_t = torch.tensor(0.3, dtype=dtype)
    variance = torch.tensor(0.9, dtype=dtype)
    ell_s = torch.tensor([0.7, 0.8], dtype=dtype)
    kt = matern32_1d(times, times, ell_t, variance)
    ks = matern32_separable(coords, coords, ell_s)
    identity_t = torch.eye(len(times), dtype=dtype)
    identity_s = torch.eye(len(coords), dtype=dtype)
    residual = separable_vfe_trace_residual(
        temporal_projection=identity_t,
        spatial_projection=identity_s,
        temporal_prior=kt,
        spatial_prior=ks,
        prior_point_variance=variance,
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2e-12, rtol=0.0)
