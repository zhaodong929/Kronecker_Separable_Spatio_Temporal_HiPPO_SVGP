from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from scipy.special import spherical_jn

from stvgp_kronecker.temporal_analytic import (
    AnalyticTemporalBuilder,
    TemporalAnalyticConfig,
    TemporalBlockSpec,
    spherical_bessel_j,
)


def weighted_sum(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(spherical_bessel_j(weights.shape[0] - 1, x) * weights)


def test_spherical_bessel_values_match_scipy() -> None:
    x = np.linspace(-12.0, 12.0, 97)
    actual = spherical_bessel_j(
        63,
        torch.as_tensor(x, dtype=torch.float64),
    ).numpy()
    expected = np.stack(
        [
            spherical_jn(level, np.abs(x))
            * np.where(x < 0.0, (-1.0) ** level, 1.0)
            for level in range(64)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)


def test_spherical_bessel_analytic_backward_matches_finite_difference() -> None:
    x = torch.as_tensor([[0.05, 0.3, 1.2, 4.0]], dtype=torch.float64)
    x.requires_grad_(True)
    weights = torch.linspace(
        0.2, 1.4, 7, dtype=torch.float64
    ).reshape(7, 1, 1)
    weighted_sum(x, weights).backward()
    analytic = x.grad.detach().clone()

    epsilon = 1e-6
    finite = torch.zeros_like(x)
    for index in range(x.numel()):
        plus = x.detach().clone()
        minus = x.detach().clone()
        plus.reshape(-1)[index] += epsilon
        minus.reshape(-1)[index] -= epsilon
        finite.reshape(-1)[index] = (
            weighted_sum(plus, weights) - weighted_sum(minus, weights)
        ) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, finite, rtol=2e-4, atol=2e-5)


def test_spherical_bessel_zero_derivative_limit() -> None:
    x = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    values = spherical_bessel_j(3, x)
    values[1].sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 1.0 / 3.0))


def test_fused_temporal_covariances_match_separate_calls() -> None:
    torch.manual_seed(0)
    builder = AnalyticTemporalBuilder(
        TemporalAnalyticConfig(
            inducing_size=16,
            rff_sample_size=32,
            kernel_type="matern32",
            dtype=torch.float64,
            seed=0,
        )
    )
    old = TemporalBlockSpec(-0.1, 0.4, 5)
    new = TemporalBlockSpec(-0.1, 0.9, 10)
    query = torch.linspace(0.5, 0.9, 5, dtype=torch.float64)
    with torch.no_grad():
        expected_kfu = builder.compute_kfu_t(query, new)
        expected_kuu = builder.compute_kuu_t(new)
        expected_cross = builder.compute_kuu_t_cross(old, new)
        kfu, kuu, cross = builder.compute_block_covariances(query, new, old)
    torch.testing.assert_close(kfu, expected_kfu, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(kuu, expected_kuu, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(cross, expected_cross, rtol=1e-12, atol=1e-12)


def test_cached_temporal_basis_matches_recomputed_cross_covariance() -> None:
    builder = AnalyticTemporalBuilder(
        TemporalAnalyticConfig(
            inducing_size=16,
            rff_sample_size=32,
            kernel_type="matern32",
            dtype=torch.float64,
            seed=0,
        )
    )
    old = TemporalBlockSpec(-0.1, 0.4, 5)
    new = TemporalBlockSpec(-0.1, 0.9, 10)
    query = torch.linspace(0.5, 0.9, 5, dtype=torch.float64)
    with torch.no_grad():
        _, _, _, old_basis = builder.compute_block_covariances_with_basis(
            query,
            old,
        )
        recomputed = builder.compute_block_covariances(query, new, old)
        cached = builder.compute_block_covariances_with_basis(
            query,
            new,
            old_basis=old_basis,
        )[:3]
    for expected, actual in zip(recomputed, cached):
        assert expected is not None and actual is not None
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_spherical_bessel_and_temporal_builder_cuda_parity() -> None:
    x_cpu = torch.linspace(-6.0, 6.0, 33, dtype=torch.float64, requires_grad=True)
    x_cuda = x_cpu.detach().to("cuda").requires_grad_(True)
    values_cpu = spherical_bessel_j(31, x_cpu)
    values_cuda = spherical_bessel_j(31, x_cuda)
    weights_cpu = torch.linspace(0.1, 1.0, values_cpu.numel(), dtype=torch.float64).reshape_as(values_cpu)
    weights_cuda = weights_cpu.to("cuda")
    (values_cpu * weights_cpu).sum().backward()
    (values_cuda * weights_cuda).sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(values_cuda.cpu(), values_cpu, rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(x_cuda.grad.cpu(), x_cpu.grad, rtol=1e-7, atol=1e-8)

    torch.manual_seed(0)
    cpu_builder = AnalyticTemporalBuilder(
        TemporalAnalyticConfig(
            inducing_size=32,
            rff_sample_size=64,
            kernel_type="matern32",
            dtype=torch.float64,
            seed=0,
        )
    )
    cuda_builder = copy.deepcopy(cpu_builder).to(device="cuda", dtype=torch.float64)
    old = TemporalBlockSpec(-0.1, 0.4, 5)
    new = TemporalBlockSpec(-0.1, 0.9, 10)
    query = torch.linspace(0.5, 0.9, 5, dtype=torch.float64)
    with torch.no_grad():
        cpu_factors = cpu_builder.compute_block_covariances(query, new, old)
        cuda_factors = cuda_builder.compute_block_covariances(query.to("cuda"), new, old)
    for cpu_factor, cuda_factor in zip(cpu_factors, cuda_factors):
        assert cpu_factor is not None and cuda_factor is not None
        torch.testing.assert_close(
            cuda_factor.cpu(), cpu_factor, rtol=1e-7, atol=1e-8
        )
