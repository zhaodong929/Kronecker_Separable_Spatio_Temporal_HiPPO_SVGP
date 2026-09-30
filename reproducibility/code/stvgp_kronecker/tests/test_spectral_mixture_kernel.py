from __future__ import annotations

import torch

from stvgp_kronecker.spectral_mixture_kernel import SpectralMixtureKernel


def test_spectral_mixture_kernel_matrix_properties() -> None:
    x = torch.linspace(0.0, 1.0, 7, dtype=torch.float64).reshape(-1, 1)
    kernel = SpectralMixtureKernel(num_mixtures=4, ard_num_dims=1)
    kxx = kernel(x)
    assert kxx.shape == (7, 7)
    assert torch.allclose(kxx, kxx.T, atol=1e-10)
    assert torch.all(torch.diagonal(kxx) > 0.0)
    assert not torch.isnan(kxx).any()


def test_spectral_mixture_kernel_multidimensional_shape() -> None:
    x1 = torch.randn(5, 3, dtype=torch.float64)
    x2 = torch.randn(4, 3, dtype=torch.float64)
    kernel = SpectralMixtureKernel(num_mixtures=2, ard_num_dims=3)
    k = kernel(x1, x2)
    assert k.shape == (5, 4)
    assert not torch.isnan(k).any()


def test_spectral_mixture_kernel_gradients_exist() -> None:
    x = torch.randn(6, 2, dtype=torch.float64)
    kernel = SpectralMixtureKernel(num_mixtures=3, ard_num_dims=2)
    loss = kernel(x).sum()
    loss.backward()
    assert kernel.raw_weights.grad is not None
    assert kernel.raw_means.grad is not None
    assert kernel.raw_variances.grad is not None


def test_spectral_mixture_kernel_short_gp_training_run() -> None:
    torch.manual_seed(0)
    x = torch.linspace(0.0, 1.0, 12, dtype=torch.float64).reshape(-1, 1)
    y = torch.sin(2.0 * torch.pi * x[:, 0])
    kernel = SpectralMixtureKernel(num_mixtures=2, ard_num_dims=1)
    raw_noise = torch.nn.Parameter(torch.tensor(-3.0, dtype=torch.float64))
    opt = torch.optim.Adam([*kernel.parameters(), raw_noise], lr=0.01)
    for _ in range(2):
        opt.zero_grad()
        k = kernel(x) + (torch.nn.functional.softplus(raw_noise) + 1e-5) * torch.eye(
            x.shape[0], dtype=x.dtype
        )
        chol = torch.linalg.cholesky(k)
        alpha = torch.cholesky_solve(y[:, None], chol)[:, 0]
        nll = 0.5 * y.dot(alpha) + torch.log(torch.diagonal(chol)).sum()
        nll = nll + 0.5 * x.shape[0] * torch.log(torch.tensor(2.0 * torch.pi, dtype=x.dtype))
        nll.backward()
        opt.step()
    assert torch.isfinite(nll)
