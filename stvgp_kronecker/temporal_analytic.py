"""Temporal analytic wrapper for the Kronecker HiPPO-SVGP prototype."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _miller_rescale(
    f_nm1: torch.Tensor,
    f_n: torch.Tensor,
    f_np1: torch.Tensor,
    large_thr: float,
    small_thr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = torch.maximum(torch.maximum(f_nm1.abs(), f_n.abs()), f_np1.abs())
    scale = torch.where(
        scale > large_thr,
        scale,
        torch.where(scale < small_thr, torch.full_like(scale, small_thr), torch.ones_like(scale)),
    )
    return f_nm1 / scale, f_n / scale, f_np1 / scale, torch.log(scale)


def _miller_recurrence_python(x_safe: torch.Tensor, lmax: int, n_start: int) -> tuple[torch.Tensor, torch.Tensor]:
    out_shape = (lmax + 1,) + tuple(x_safe.shape)
    f = torch.zeros(out_shape, device=x_safe.device, dtype=x_safe.dtype)
    log_s = torch.zeros(out_shape, device=x_safe.device, dtype=x_safe.dtype)

    f_np1 = torch.zeros_like(x_safe)
    f_n = torch.ones_like(x_safe)
    cur_log_s = torch.zeros_like(x_safe)

    if x_safe.dtype == torch.float64:
        large_thr = 1e100
        small_thr = 1e-100
    else:
        large_thr = 1e15
        small_thr = 1e-15

    for n in range(n_start, 0, -1):
        f_nm1 = ((2 * n + 1) / x_safe) * f_n - f_np1
        f_nm1, f_n, f_np1, log_scale = _miller_rescale(
            f_nm1,
            f_n,
            f_np1,
            large_thr=large_thr,
            small_thr=small_thr,
        )
        cur_log_s = cur_log_s + log_scale
        if n - 1 <= lmax:
            f[n - 1] = f_nm1
            log_s[n - 1] = cur_log_s
        f_np1 = f_n
        f_n = f_nm1
    return f, log_s


def _spherical_bessel_j_values(lmax: int, x: torch.Tensor) -> torch.Tensor:
    """Evaluate spherical Bessel values via the stable Miller recurrence.

    This logic is adapted from the existing analytic solar prototype under
    `scripts/onedim/solar/test_ohsgp_analytic_solar.py`.
    """

    eps = 1e-8
    x_abs = x.abs()
    mask_nonzero = x_abs > eps
    x_safe = torch.where(mask_nonzero, x_abs, torch.full_like(x_abs, eps))

    max_x = float(x_abs.max().detach().cpu()) if x_abs.numel() > 0 else 0.0
    n_start = int(lmax + max_x + 50)

    f, log_s = _miller_recurrence_python(x_safe, lmax=lmax, n_start=n_start)
    true_j0 = torch.sin(x_safe) / x_safe
    tiny = torch.finfo(x_safe.dtype).tiny
    scale0 = true_j0 / (f[0] + tiny)
    max_log = 80.0 if x_safe.dtype == torch.float32 else 700.0
    corr = torch.exp(
        torch.clamp(
            log_s - log_s[0].unsqueeze(0), min=-max_log, max=max_log
        )
    )
    j = f * corr * scale0.unsqueeze(0)

    parity = torch.where(
        (torch.arange(lmax + 1, device=x.device) % 2) == 0,
        torch.ones(lmax + 1, device=x.device, dtype=x.dtype),
        -torch.ones(lmax + 1, device=x.device, dtype=x.dtype),
    ).view(-1, *([1] * x.ndim))
    j = torch.where((x >= 0).unsqueeze(0), j, j * parity)

    j_at_0 = torch.zeros_like(j)
    j_at_0[0] = 1.0
    return torch.where(mask_nonzero.unsqueeze(0), j, j_at_0)


class _SphericalBesselJ(torch.autograd.Function):
    """Miller-recursion forward with an analytic first derivative."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lmax: int) -> torch.Tensor:  # type: ignore[override]
        values = _spherical_bessel_j_values(int(lmax), x)
        ctx.save_for_backward(x, values)
        ctx.lmax = int(lmax)
        return values

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        x, values = ctx.saved_tensors
        lmax = ctx.lmax
        near_zero = x.abs() <= 1e-6
        x_safe = torch.where(near_zero, torch.ones_like(x), x)
        derivatives = torch.zeros_like(values)
        if lmax == 0:
            j1 = _spherical_bessel_j_values(1, x)[1]
            derivatives[0] = -j1
        else:
            derivatives[0] = -values[1]
            levels = torch.arange(
                1, lmax + 1, dtype=x.dtype, device=x.device
            ).view(-1, *([1] * x.ndim))
            derivatives[1:] = values[:-1] - (
                (levels + 1.0) * values[1:] / x_safe.unsqueeze(0)
            )
        zero_limit = torch.zeros_like(derivatives)
        if lmax >= 1:
            zero_limit[1] = 1.0 / 3.0
        derivatives = torch.where(
            near_zero.unsqueeze(0), zero_limit, derivatives
        )
        grad_x = torch.sum(grad_output * derivatives, dim=0)
        return grad_x, None


def spherical_bessel_j(lmax: int, x: torch.Tensor) -> torch.Tensor:
    """Pure-torch spherical Bessel values with stable first derivatives."""

    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    return _SphericalBesselJ.apply(x, int(lmax))


@dataclass
class TemporalBlockSpec:
    """Specification for one temporal block or horizon."""

    start: float
    end: float
    num_discrete_steps: int
    prev_discrete_steps: int = 0
    phase_origin: Optional[float] = None

    @classmethod
    def from_times(
        cls,
        times: torch.Tensor,
        num_discrete_steps: Optional[int] = None,
        prev_discrete_steps: int = 0,
        padding: float = 0.0,
    ) -> "TemporalBlockSpec":
        times = torch.as_tensor(times, dtype=torch.float64).reshape(-1)
        if times.numel() == 0:
            raise ValueError("Cannot build a TemporalBlockSpec from an empty time tensor.")
        sorted_times = torch.sort(times).values
        if sorted_times.numel() > 1:
            dt = float(torch.median(sorted_times[1:] - sorted_times[:-1]).item())
        else:
            dt = 1.0
        # Match the reference analytic HiPPO-SVGP convention: start/end are
        # continuous interval boundaries, and timestamps are
        # start + step * (1, ..., num_steps).  For observed points this makes the
        # first boundary one grid step before the first observation.
        start = float(sorted_times[0].item()) - dt - padding
        end = float(sorted_times[-1].item()) + padding
        if math.isclose(start, end):
            end = start + 1.0
        num_steps = int(num_discrete_steps or max(times.numel(), 1))
        return cls(
            start=start,
            end=end,
            num_discrete_steps=num_steps,
            prev_discrete_steps=int(prev_discrete_steps),
            phase_origin=start,
        )


@dataclass
class TemporalAnalyticConfig:
    """Configuration for the analytic temporal builder."""

    inducing_size: int
    rff_sample_size: int = 256
    variance: float = 1.0
    lengthscale: float = 1.0
    kernel_type: str = "rbf"
    globalstart_wt_mode: str = "w"
    phase_origin_mode: str = "global_start"
    num_discrete_steps: Optional[int] = None
    prev_discrete_steps: int = 0
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    jitter: float = 1e-6
    seed: int = 0
    spectral_mixture_means: tuple[float, ...] = (0.0, 1.5, 4.0)
    spectral_mixture_scales: tuple[float, ...] = (1.0, 0.8, 0.45)
    spectral_mixture_weights: tuple[float, ...] = (0.55, 0.30, 0.15)


class AnalyticTemporalBuilder(nn.Module):
    """Stateful temporal builder using fixed base frequencies and analytic HiPPO features."""

    def __init__(self, config: TemporalAnalyticConfig) -> None:
        super().__init__()
        self.config = config
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.seed))
        base_freq = self._sample_base_frequencies(config, generator)
        self.register_buffer("base_frequencies", base_freq)
        self.log_variance = nn.Parameter(torch.log(torch.as_tensor(config.variance, dtype=config.dtype)))
        self.log_lengthscale = nn.Parameter(torch.log(torch.as_tensor(config.lengthscale, dtype=config.dtype)))

    def _apply(self, fn):
        result = super()._apply(fn)
        # nn.Module.to() moves tensors but cannot update this dataclass itself.
        # Keeping these fields synchronized prevents newly-created basis tensors
        # from silently returning to CPU/float64 after a CUDA or float32 move.
        self.config.dtype = self.base_frequencies.dtype
        self.config.device = str(self.base_frequencies.device)
        return result

    @staticmethod
    def _sample_base_frequencies(config: TemporalAnalyticConfig, generator: torch.Generator) -> torch.Tensor:
        kernel_type = config.kernel_type.lower()
        if kernel_type in {"rbf", "ard_rbf"}:
            return torch.randn(1, config.rff_sample_size, generator=generator, dtype=config.dtype)
        if kernel_type in {"matern32", "matern_32", "matern3/2"}:
            # Matern-3/2 in 1D has Student-t spectral density with df=3.
            normal = torch.randn(1, config.rff_sample_size, generator=generator, dtype=config.dtype)
            chi2 = torch.distributions.Chi2(df=torch.as_tensor(3.0, dtype=config.dtype)).sample(
                (1, config.rff_sample_size)
            )
            return normal / torch.sqrt(chi2 / 3.0)
        if kernel_type == "spectral_mixture":
            means = torch.as_tensor(config.spectral_mixture_means, dtype=config.dtype)
            scales = torch.as_tensor(config.spectral_mixture_scales, dtype=config.dtype)
            weights = torch.as_tensor(config.spectral_mixture_weights, dtype=config.dtype)
            weights = weights / torch.clamp(weights.sum(), min=torch.finfo(config.dtype).eps)
            comp = torch.multinomial(weights, config.rff_sample_size, replacement=True, generator=generator)
            signs = torch.where(
                torch.rand(config.rff_sample_size, generator=generator, dtype=config.dtype) < 0.5,
                torch.ones(config.rff_sample_size, dtype=config.dtype),
                -torch.ones(config.rff_sample_size, dtype=config.dtype),
            )
            freq = signs * means[comp] + scales[comp] * torch.randn(
                config.rff_sample_size, generator=generator, dtype=config.dtype
            )
            return freq.reshape(1, -1)
        raise ValueError(f"Unsupported analytic temporal kernel_type: {config.kernel_type}")

    def scaled_jitter(self, matrix: torch.Tensor) -> torch.Tensor:
        diag_mean = torch.mean(torch.diagonal(matrix)).detach()
        scale = torch.clamp(diag_mean.abs(), min=torch.as_tensor(1.0, dtype=matrix.dtype, device=matrix.device))
        return self.config.jitter * scale

    def add_jitter(self, matrix: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        return matrix + self.scaled_jitter(matrix) * eye

    @property
    def variance(self) -> torch.Tensor:
        return torch.exp(self.log_variance)

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale)

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(tensor, dtype=self.config.dtype, device=self.base_frequencies.device)

    def current_frequencies(self) -> torch.Tensor:
        return self.base_frequencies / self.lengthscale

    def resolve_horizon(self, times_or_horizon: torch.Tensor | TemporalBlockSpec) -> TemporalBlockSpec:
        if isinstance(times_or_horizon, TemporalBlockSpec):
            return times_or_horizon
        return TemporalBlockSpec.from_times(
            times_or_horizon,
            num_discrete_steps=self.config.num_discrete_steps,
            prev_discrete_steps=self.config.prev_discrete_steps,
        )

    def get_time_stamps(self, horizon: TemporalBlockSpec) -> torch.Tensor:
        start = torch.as_tensor(horizon.start, dtype=self.config.dtype, device=self.base_frequencies.device)
        step = (horizon.end - horizon.start) / max(horizon.num_discrete_steps, 1)
        offsets = torch.arange(
            horizon.num_discrete_steps,
            dtype=self.config.dtype,
            device=self.base_frequencies.device,
        )
        return (start + step * (1.0 + offsets)).unsqueeze(-1)

    def _phase_origin_term(self, w: torch.Tensor, w_eff: torch.Tensor, horizon: TemporalBlockSpec) -> torch.Tensor:
        if self.config.phase_origin_mode != "global_start":
            return torch.zeros_like(w)
        phase_origin = horizon.phase_origin if horizon.phase_origin is not None else horizon.start
        phase_origin = torch.as_tensor(phase_origin, dtype=self.config.dtype, device=w.device)
        if self.config.globalstart_wt_mode == "none":
            return torch.zeros_like(w)
        if self.config.globalstart_wt_mode == "w_eff":
            return w_eff * phase_origin
        return w * phase_origin

    def compute_temporal_basis(self, horizon: TemporalBlockSpec) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.current_frequencies()
        step_size = (horizon.end - horizon.start) / max(horizon.num_discrete_steps, 1)
        w_eff = w * step_size
        t_index = int(horizon.prev_discrete_steps) + int(horizon.num_discrete_steps)
        kappa = 0.5 * w_eff * t_index
        phase_origin_term = self._phase_origin_term(w, w_eff, horizon)

        levels = torch.arange(
            self.config.inducing_size,
            dtype=self.config.dtype,
            device=w.device,
        )
        j = spherical_bessel_j(self.config.inducing_size - 1, kappa).squeeze(1)
        prefactor = torch.sqrt(2.0 * levels + 1.0)[:, None]
        phase = phase_origin_term + kappa + levels[:, None] * math.pi / 2.0
        z_sin = prefactor * j * torch.sin(phase)
        z_cos = prefactor * j * torch.cos(phase)
        scale = (1.0 / self.config.rff_sample_size) ** 0.5
        z = scale * torch.cat([z_sin, z_cos], dim=1).transpose(0, 1)
        return z, self.get_time_stamps(horizon)

    def compute_feature_matrix(self, query_times: torch.Tensor) -> torch.Tensor:
        query_times = self._to_device(query_times).reshape(-1, 1)
        w = self.current_frequencies()
        scale = (1.0 / self.config.rff_sample_size) ** 0.5
        return scale * torch.cat([torch.sin(w * query_times), torch.cos(w * query_times)], dim=1)

    def compute_kuu_t(self, times_or_horizon: torch.Tensor | TemporalBlockSpec) -> torch.Tensor:
        horizon = self.resolve_horizon(times_or_horizon)
        z, _ = self.compute_temporal_basis(horizon)
        return self.variance * (z.transpose(0, 1) @ z)

    def compute_kuu_t_cross(
        self,
        left_horizon: torch.Tensor | TemporalBlockSpec,
        right_horizon: torch.Tensor | TemporalBlockSpec,
    ) -> torch.Tensor:
        """Cross-covariance between two temporal inducing systems.

        When two blocks use different local horizons, both inducing systems are
        still linear functionals of the same underlying RFF-expanded temporal GP.
        Their cross-covariance is therefore given by the inner product of the two
        temporal basis matrices in that shared feature space.
        """

        left = self.resolve_horizon(left_horizon)
        right = self.resolve_horizon(right_horizon)
        z_left, _ = self.compute_temporal_basis(left)
        z_right, _ = self.compute_temporal_basis(right)
        return self.variance * (z_left.transpose(0, 1) @ z_right)

    def compute_kfu_t(
        self,
        query_times: torch.Tensor,
        horizon: torch.Tensor | TemporalBlockSpec,
    ) -> torch.Tensor:
        horizon = self.resolve_horizon(horizon)
        z, _ = self.compute_temporal_basis(horizon)
        z_f = self.compute_feature_matrix(query_times)
        return self.variance * (z_f @ z)

    def compute_ktt_diag(self, query_times: torch.Tensor) -> torch.Tensor:
        query_times = self._to_device(query_times).reshape(-1)
        return self.variance.expand(query_times.shape[0])

    def compute_block_covariances(
        self,
        query_times: torch.Tensor,
        horizon: torch.Tensor | TemporalBlockSpec,
        old_horizon: torch.Tensor | TemporalBlockSpec | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Build ``Kfu``, ``Kuu`` and optional ``K_on`` from one new basis.

        The separate public covariance methods remain available, but invoking
        them independently repeats the sequential spherical-Bessel recurrence.
        Online inference needs all three matrices together, so this fused entry
        point reuses ``Z_new`` without changing any covariance formula.
        """

        kfu, kuu, k_on, _ = self.compute_block_covariances_with_basis(
            query_times,
            horizon,
            old_horizon,
        )
        return kfu, kuu, k_on

    def compute_block_covariances_with_basis(
        self,
        query_times: torch.Tensor,
        horizon: torch.Tensor | TemporalBlockSpec,
        old_horizon: torch.Tensor | TemporalBlockSpec | None = None,
        *,
        old_basis: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Build one covariance bundle and return the reusable new basis.

        ``old_basis`` is the feature-space HiPPO matrix returned for the
        preceding horizon. Supplying it avoids a second spherical-Bessel
        recurrence when constructing the changing-basis cross-covariance.
        """

        resolved = self.resolve_horizon(horizon)
        z_new, _ = self.compute_temporal_basis(resolved)
        z_query = self.compute_feature_matrix(query_times)
        kfu = self.variance * (z_query @ z_new)
        kuu = self.variance * (z_new.transpose(0, 1) @ z_new)
        k_on = None
        if old_basis is not None:
            z_old = self._to_device(old_basis)
            if z_old.shape != z_new.shape:
                raise ValueError(
                    f"old_basis must have shape {tuple(z_new.shape)}, "
                    f"got {tuple(z_old.shape)}"
                )
            k_on = self.variance * (z_old.transpose(0, 1) @ z_new)
        elif old_horizon is not None:
            old_resolved = self.resolve_horizon(old_horizon)
            z_old, _ = self.compute_temporal_basis(old_resolved)
            k_on = self.variance * (z_old.transpose(0, 1) @ z_new)
        return kfu, kuu, k_on, z_new


def compute_kuu_t(
    times_or_horizon: torch.Tensor | TemporalBlockSpec,
    config: TemporalAnalyticConfig,
) -> torch.Tensor:
    """Convenience wrapper for the requested Stage 1 interface."""
    builder = AnalyticTemporalBuilder(config)
    return builder.compute_kuu_t(times_or_horizon)


def compute_kfu_t(
    query_times: torch.Tensor,
    horizon: torch.Tensor | TemporalBlockSpec,
    config: TemporalAnalyticConfig,
) -> torch.Tensor:
    """Convenience wrapper for the requested Stage 1 interface."""
    builder = AnalyticTemporalBuilder(config)
    return builder.compute_kfu_t(query_times, horizon)


def compute_kuu_t_cross(
    left_horizon: torch.Tensor | TemporalBlockSpec,
    right_horizon: torch.Tensor | TemporalBlockSpec,
    config: TemporalAnalyticConfig,
) -> torch.Tensor:
    """Convenience wrapper for temporal inducing cross-covariances."""

    builder = AnalyticTemporalBuilder(config)
    return builder.compute_kuu_t_cross(left_horizon, right_horizon)
