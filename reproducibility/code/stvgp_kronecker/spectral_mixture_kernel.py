"""Trainable ARD spectral-mixture kernel utilities."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpectralMixtureKernel(nn.Module):
    """ARD spectral-mixture covariance with positive constrained parameters.

    For ``x1`` with shape ``[N, D]`` and ``x2`` with shape ``[M, D]``:

    ``k(x1, x2) = sum_q w_q prod_d exp(-2*pi^2*tau_d^2*v_qd) cos(2*pi*tau_d*mu_qd)``.
    """

    def __init__(
        self,
        *,
        num_mixtures: int = 4,
        ard_num_dims: int = 1,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be positive")
        if ard_num_dims <= 0:
            raise ValueError("ard_num_dims must be positive")
        self.num_mixtures = int(num_mixtures)
        self.ard_num_dims = int(ard_num_dims)
        self.eps = float(eps)
        self.raw_weights = nn.Parameter(torch.zeros(self.num_mixtures, dtype=dtype))
        self.raw_means = nn.Parameter(torch.zeros(self.num_mixtures, self.ard_num_dims, dtype=dtype))
        self.raw_variances = nn.Parameter(torch.zeros(self.num_mixtures, self.ard_num_dims, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            weights = torch.full_like(self.raw_weights, 1.0 / self.num_mixtures)
            means = torch.linspace(0.0, 1.5, self.num_mixtures, dtype=self.raw_means.dtype).reshape(-1, 1)
            means = means.repeat(1, self.ard_num_dims)
            variances = torch.linspace(0.2, 1.0, self.num_mixtures, dtype=self.raw_variances.dtype).reshape(-1, 1)
            variances = variances.repeat(1, self.ard_num_dims)
            self.raw_weights.copy_(self._inv_softplus(weights))
            self.raw_means.copy_(self._inv_softplus(means + self.eps))
            self.raw_variances.copy_(self._inv_softplus(variances + self.eps))

    @staticmethod
    def _inv_softplus(value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(value, min=torch.finfo(value.dtype).eps)
        return value + torch.log(-torch.expm1(-value))

    @property
    def weights(self) -> torch.Tensor:
        return F.softplus(self.raw_weights) + self.eps

    @property
    def means(self) -> torch.Tensor:
        return F.softplus(self.raw_means) + self.eps

    @property
    def variances(self) -> torch.Tensor:
        return F.softplus(self.raw_variances) + self.eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor | None = None) -> torch.Tensor:
        if x2 is None:
            x2 = x1
        if x1.ndim == 1:
            x1 = x1[:, None]
        if x2.ndim == 1:
            x2 = x2[:, None]
        if x1.shape[-1] != self.ard_num_dims or x2.shape[-1] != self.ard_num_dims:
            raise ValueError(f"Expected inputs with last dimension {self.ard_num_dims}")
        tau = x1[:, None, :] - x2[None, :, :]
        weights = self.weights
        means = self.means
        variances = self.variances
        out = torch.zeros(tau.shape[:2], dtype=tau.dtype, device=tau.device)
        for q in range(self.num_mixtures):
            envelope = torch.exp(-2.0 * torch.pi**2 * tau.square() * variances[q].reshape(1, 1, -1))
            carrier = torch.cos(2.0 * torch.pi * tau * means[q].reshape(1, 1, -1))
            out = out + weights[q] * torch.prod(envelope * carrier, dim=-1)
        return out
