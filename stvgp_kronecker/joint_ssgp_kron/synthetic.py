"""Synthetic data and adapters for joint SSGP Kronecker experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kron_utils import solve_spd, symmetrize, vec_f


@dataclass(frozen=True)
class SyntheticDataset:
    times: np.ndarray
    spatial_coords: np.ndarray
    Y: np.ndarray
    F: np.ndarray
    Phi: np.ndarray
    beta_true: np.ndarray
    sigma2: float
    gp_prior_variance: float = 1.0
    temporal_lengthscale: float = 0.25
    spatial_lengthscale: float = 0.35

    @property
    def y_vec(self) -> np.ndarray:
        return vec_f(self.Y)


@dataclass(frozen=True)
class BlockFactors:
    y_vec: np.ndarray
    Phi: np.ndarray
    Y: np.ndarray
    T: np.ndarray
    Kt: np.ndarray
    K_on_t: np.ndarray | None
    block_slice: slice
    inducing_times: np.ndarray


def _prepare_kernel_inputs(x: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    return x, y


def rbf_kernel(x: np.ndarray, y: np.ndarray | None = None, *, lengthscale: float = 1.0, variance: float = 1.0) -> np.ndarray:
    x, y = _prepare_kernel_inputs(x, y)
    diff = x[:, None, :] - y[None, :, :]
    ls = np.asarray(lengthscale, dtype=float)
    if ls.ndim > 0:
        diff = diff / np.maximum(ls.reshape((1, 1, -1)), 1e-12)
        sqdist = np.sum(diff * diff, axis=-1)
        return variance * np.exp(-0.5 * sqdist)
    sqdist = np.sum(diff * diff, axis=-1)
    return variance * np.exp(-0.5 * sqdist / (lengthscale**2))


def matern32_kernel(x: np.ndarray, y: np.ndarray | None = None, *, lengthscale: float = 1.0, variance: float = 1.0) -> np.ndarray:
    x, y = _prepare_kernel_inputs(x, y)
    diff = x[:, None, :] - y[None, :, :]
    dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
    scaled = np.sqrt(3.0) * dist / max(float(lengthscale), 1e-12)
    return variance * (1.0 + scaled) * np.exp(-scaled)


def covariance_kernel(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    lengthscale: float = 1.0,
    variance: float = 1.0,
    kernel_type: str = "rbf",
) -> np.ndarray:
    if kernel_type in {"rbf", "ard_rbf"}:
        return rbf_kernel(x, y, lengthscale=lengthscale, variance=variance)
    if kernel_type in {"matern32", "matern_32", "matern3/2"}:
        return matern32_kernel(x, y, lengthscale=lengthscale, variance=variance)
    raise ValueError(f"Unknown kernel_type: {kernel_type}")


def design_matrix(times: np.ndarray, spatial_coords: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    t_scaled = (times - times.min()) / max(1e-12, times.max() - times.min())
    s_centered = spatial_coords[:, 0] - np.mean(spatial_coords[:, 0])
    for t in t_scaled:
        for s in s_centered:
            rows.append([1.0, float(t), float(s), float(np.sin(2.0 * np.pi * t))])
    return np.asarray(rows)


def make_synthetic_dataset(
    *,
    num_time: int,
    num_space: int,
    noise: float,
    seed: int,
    ell_t: float = 0.25,
    ell_s: float = 0.35,
    kernel_variance: float = 1.0,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 1.0, num_time)
    spatial_coords = np.linspace(0.0, 1.0, num_space)[:, None]
    Kt = rbf_kernel(times, lengthscale=ell_t, variance=kernel_variance) + 1e-6 * np.eye(num_time)
    Ks = rbf_kernel(spatial_coords, lengthscale=ell_s) + 1e-6 * np.eye(num_space)
    L = np.linalg.cholesky(np.kron(Kt, Ks))
    f_vec = L @ rng.normal(size=num_time * num_space)
    F = f_vec.reshape((num_space, num_time), order="F")
    Phi = design_matrix(times, spatial_coords)
    beta_true = np.array([0.4, -0.7, 0.25, 0.15])
    mean = (Phi @ beta_true).reshape((num_space, num_time), order="F")
    Y = mean + F + noise * rng.normal(size=(num_space, num_time))
    return SyntheticDataset(
        times=times,
        spatial_coords=spatial_coords,
        Y=Y,
        F=F,
        Phi=Phi,
        beta_true=beta_true,
        sigma2=noise**2,
        gp_prior_variance=kernel_variance,
        temporal_lengthscale=ell_t,
        spatial_lengthscale=ell_s,
    )


def make_spatial_projection(
    spatial_coords: np.ndarray,
    ms: int,
    *,
    lengthscale: float | np.ndarray = 0.35,
    kernel_type: str = "rbf",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ms > spatial_coords.shape[0]:
        raise ValueError("ms cannot exceed num_space")
    idx = np.linspace(0, spatial_coords.shape[0] - 1, ms).round().astype(int)
    z_s = spatial_coords[idx]
    Ks = covariance_kernel(z_s, lengthscale=lengthscale, kernel_type=kernel_type) + 1e-6 * np.eye(ms)
    Kxz = covariance_kernel(spatial_coords, z_s, lengthscale=lengthscale, kernel_type=kernel_type)
    C = solve_spd(Ks, Kxz.T).T
    return z_s, symmetrize(Ks), C


def temporal_inducing_for_block(times: np.ndarray, block: slice, mt: int, *, moving: bool = True) -> np.ndarray:
    block_times = times[block]
    if not moving:
        return np.linspace(times.min(), times.max(), mt)
    pad = 0.5 * max(1e-6, block_times[-1] - block_times[0])
    lo = max(times.min(), block_times[0] - pad)
    hi = min(times.max(), block_times[-1] + pad)
    if lo == hi:
        hi = min(times.max(), lo + 1e-3)
    return np.linspace(lo, hi, mt)


def make_block_factors(
    dataset: SyntheticDataset,
    *,
    block: slice,
    z_t: np.ndarray,
    z_t_old: np.ndarray | None,
    lengthscale: float = 0.25,
    kernel_variance: float | None = None,
    kernel_type: str = "rbf",
) -> BlockFactors:
    times_b = dataset.times[block]
    variance = dataset.gp_prior_variance if kernel_variance is None else kernel_variance
    Kt = covariance_kernel(z_t, lengthscale=lengthscale, variance=variance, kernel_type=kernel_type) + 1e-6 * np.eye(len(z_t))
    Kfu = covariance_kernel(times_b, z_t, lengthscale=lengthscale, variance=variance, kernel_type=kernel_type)
    T = solve_spd(Kt, Kfu.T).T
    K_on_t = None if z_t_old is None else covariance_kernel(z_t_old, z_t, lengthscale=lengthscale, variance=variance, kernel_type=kernel_type)
    Y_b = dataset.Y[:, block]
    start = block.start or 0
    stop = block.stop or dataset.Y.shape[1]
    ns = dataset.Y.shape[0]
    row_idx = []
    for t_idx in range(start, stop):
        row_idx.extend(range(t_idx * ns, (t_idx + 1) * ns))
    Phi_b = dataset.Phi[np.asarray(row_idx)]
    return BlockFactors(
        y_vec=vec_f(Y_b),
        Phi=Phi_b,
        Y=Y_b,
        T=T,
        Kt=Kt,
        K_on_t=K_on_t,
        block_slice=block,
        inducing_times=z_t,
    )


def iter_time_blocks(num_time: int, block_size: int) -> list[slice]:
    return [slice(start, min(num_time, start + block_size)) for start in range(0, num_time, block_size)]
