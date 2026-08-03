"""Synthetic data and adapters for joint SSGP Kronecker experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kron_utils import solve_spd, symmetrize, vec_f
from stvgp_kronecker.temporal_analytic import AnalyticTemporalBuilder, TemporalAnalyticConfig, TemporalBlockSpec


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
    temporal_backend: str = "inducing_points"
    temporal_builder: Any | None = None
    temporal_basis_spec: Any | None = None


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


def spectral_mixture_kernel(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    variance: float = 1.0,
    weights: np.ndarray | None = None,
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    """Stationary spectral-mixture covariance in the Wilson-Adams form."""

    x, y = _prepare_kernel_inputs(x, y)
    dim = x.shape[1]
    default_weights = np.asarray([0.65, 0.35], dtype=float)
    default_means = np.asarray([[0.0] * dim, [1.0] * dim], dtype=float)
    default_scales = np.asarray([[1.0] * dim, [0.45] * dim], dtype=float)
    weights_arr = default_weights if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    means_arr = default_means if means is None else np.asarray(means, dtype=float)
    scales_arr = default_scales if scales is None else np.asarray(scales, dtype=float)
    if means_arr.ndim == 1:
        means_arr = means_arr.reshape(-1, 1)
    if scales_arr.ndim == 1:
        scales_arr = scales_arr.reshape(-1, 1)
    if means_arr.shape[1] == 1 and dim > 1:
        means_arr = np.repeat(means_arr, dim, axis=1)
    if scales_arr.shape[1] == 1 and dim > 1:
        scales_arr = np.repeat(scales_arr, dim, axis=1)
    if means_arr.shape != scales_arr.shape:
        raise ValueError("spectral mixture means and scales must have the same shape")
    if means_arr.shape[0] != weights_arr.shape[0]:
        raise ValueError("spectral mixture weights must match the number of mixture components")
    if means_arr.shape[1] != dim:
        raise ValueError(f"spectral mixture parameter dimension {means_arr.shape[1]} does not match input dimension {dim}")
    weights_arr = np.maximum(weights_arr, 0.0)
    weight_sum = float(weights_arr.sum())
    if weight_sum <= 0.0:
        raise ValueError("spectral mixture weights must contain at least one positive value")
    diff = x[:, None, :] - y[None, :, :]
    out = np.zeros(diff.shape[:2], dtype=float)
    for weight, mean, scale in zip(weights_arr, means_arr, np.maximum(scales_arr, 1e-12)):
        envelope = np.exp(-2.0 * np.pi**2 * np.sum((diff * scale.reshape(1, 1, -1)) ** 2, axis=-1))
        carrier = np.prod(np.cos(2.0 * np.pi * diff * mean.reshape(1, 1, -1)), axis=-1)
        out += float(weight) * envelope * carrier
    return float(variance) * out


def covariance_kernel(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    lengthscale: float = 1.0,
    variance: float = 1.0,
    kernel_type: str = "rbf",
    spectral_mixture_weights: np.ndarray | None = None,
    spectral_mixture_means: np.ndarray | None = None,
    spectral_mixture_scales: np.ndarray | None = None,
) -> np.ndarray:
    if kernel_type in {"rbf", "ard_rbf"}:
        return rbf_kernel(x, y, lengthscale=lengthscale, variance=variance)
    if kernel_type in {"matern32", "matern_32", "matern3/2"}:
        return matern32_kernel(x, y, lengthscale=lengthscale, variance=variance)
    if kernel_type == "spectral_mixture":
        return spectral_mixture_kernel(
            x,
            y,
            variance=variance,
            weights=spectral_mixture_weights,
            means=spectral_mixture_means,
            scales=spectral_mixture_scales,
        )
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
    inducing_selection: str = "linspace",
    spectral_mixture_weights: np.ndarray | None = None,
    spectral_mixture_means: np.ndarray | None = None,
    spectral_mixture_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ms > spatial_coords.shape[0]:
        raise ValueError("ms cannot exceed num_space")
    idx = select_spatial_inducing_indices(spatial_coords, ms, method=inducing_selection)
    z_s = spatial_coords[idx]
    kernel_kwargs = {
        "spectral_mixture_weights": spectral_mixture_weights,
        "spectral_mixture_means": spectral_mixture_means,
        "spectral_mixture_scales": spectral_mixture_scales,
    }
    Ks = covariance_kernel(z_s, lengthscale=lengthscale, kernel_type=kernel_type, **kernel_kwargs) + 1e-6 * np.eye(ms)
    Kxz = covariance_kernel(spatial_coords, z_s, lengthscale=lengthscale, kernel_type=kernel_type, **kernel_kwargs)
    C = solve_spd(Ks, Kxz.T).T
    return z_s, symmetrize(Ks), C


def select_spatial_inducing_indices(spatial_coords: np.ndarray, ms: int, *, method: str = "linspace") -> np.ndarray:
    coords = np.asarray(spatial_coords, dtype=float)
    n = coords.shape[0]
    ms = min(int(ms), n)
    method = str(method).lower()
    if method == "linspace":
        return np.linspace(0, n - 1, ms).round().astype(int)
    coords_norm = (coords - coords.mean(axis=0, keepdims=True)) / np.maximum(coords.std(axis=0, keepdims=True), 1e-8)
    if method == "farthest":
        selected = [int(np.argmin(np.sum((coords_norm - coords_norm.mean(axis=0, keepdims=True)) ** 2, axis=1)))]
        min_dist2 = np.sum((coords_norm - coords_norm[selected[0]]) ** 2, axis=1)
        for _ in range(1, ms):
            nxt = int(np.argmax(min_dist2))
            selected.append(nxt)
            dist2 = np.sum((coords_norm - coords_norm[nxt]) ** 2, axis=1)
            min_dist2 = np.minimum(min_dist2, dist2)
        return np.asarray(selected, dtype=int)
    if method == "kmeans":
        rng = np.random.default_rng(0)
        centers = coords_norm[np.linspace(0, n - 1, ms).round().astype(int)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(20):
            dist2 = np.sum((coords_norm[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
            new_labels = np.argmin(dist2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(ms):
                mask = labels == j
                if np.any(mask):
                    centers[j] = coords_norm[mask].mean(axis=0)
                else:
                    centers[j] = coords_norm[int(rng.integers(0, n))]
        idx = []
        used: set[int] = set()
        for center in centers:
            order = np.argsort(np.sum((coords_norm - center[None, :]) ** 2, axis=1))
            for candidate in order:
                cand = int(candidate)
                if cand not in used:
                    idx.append(cand)
                    used.add(cand)
                    break
        return np.asarray(idx, dtype=int)
    raise ValueError(f"Unsupported spatial inducing selection: {method}")


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
    spectral_mixture_weights: np.ndarray | None = None,
    spectral_mixture_means: np.ndarray | None = None,
    spectral_mixture_scales: np.ndarray | None = None,
) -> BlockFactors:
    times_b = dataset.times[block]
    variance = dataset.gp_prior_variance if kernel_variance is None else kernel_variance
    kernel_kwargs = {
        "spectral_mixture_weights": spectral_mixture_weights,
        "spectral_mixture_means": spectral_mixture_means,
        "spectral_mixture_scales": spectral_mixture_scales,
    }
    Kt = covariance_kernel(z_t, lengthscale=lengthscale, variance=variance, kernel_type=kernel_type, **kernel_kwargs) + 1e-6 * np.eye(len(z_t))
    Kfu = covariance_kernel(times_b, z_t, lengthscale=lengthscale, variance=variance, kernel_type=kernel_type, **kernel_kwargs)
    T = solve_spd(Kt, Kfu.T).T
    K_on_t = None if z_t_old is None else covariance_kernel(z_t_old, z_t, lengthscale=lengthscale, variance=variance, kernel_type=kernel_type, **kernel_kwargs)
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
        temporal_backend="inducing_points",
    )


def temporal_spec_for_block(
    times: np.ndarray,
    block: slice,
    *,
    moving: bool = True,
    padding: float = 0.0,
) -> TemporalBlockSpec:
    """Build the analytic HiPPO temporal horizon for a query or basis block."""

    times = np.asarray(times, dtype=float)
    if not moving:
        if times.shape[0] > 1:
            dt = float(np.median(np.diff(np.sort(times))))
        else:
            dt = 1.0
        global_start = float(times.min() - dt)
        return TemporalBlockSpec(
            start=float(times.min() - dt),
            end=float(times.max()),
            num_discrete_steps=times.shape[0],
            prev_discrete_steps=0,
            phase_origin=global_start,
        )
    start = block.start or 0
    stop = block.stop or times.shape[0]
    num_steps = max(1, stop - start)
    if times.shape[0] > 1:
        dt = float(np.median(np.diff(np.sort(times))))
    else:
        dt = 1.0
    global_start = float(times[0] - dt)
    # The analytic HiPPO reference treats start/end as continuous interval
    # boundaries: timestamps are start + step * (1, ..., num_steps).  Therefore
    # a block covering observations t_start..t_stop-1 has boundary start one
    # grid step before the first observation and end at the last observation.
    interval_start = float(times[start] - dt)
    interval_end = float(times[stop - 1])
    if padding:
        interval_start -= float(padding)
        interval_end += float(padding)
    if interval_end <= interval_start:
        interval_end = interval_start + dt
    return TemporalBlockSpec(
        start=interval_start,
        end=interval_end,
        num_discrete_steps=num_steps,
        prev_discrete_steps=start,
        phase_origin=global_start,
    )


def make_analytic_temporal_builder(
    *,
    mt: int,
    lengthscale: float = 0.25,
    variance: float = 1.0,
    rff_sample_size: int = 256,
    seed: int = 0,
    jitter: float = 1e-6,
    kernel_type: str = "rbf",
    spectral_mixture_weights: np.ndarray | None = None,
    spectral_mixture_means: np.ndarray | None = None,
    spectral_mixture_scales: np.ndarray | None = None,
) -> AnalyticTemporalBuilder:
    """Create the shared analytic HiPPO-RFF temporal builder used by Route B."""

    return AnalyticTemporalBuilder(
        TemporalAnalyticConfig(
            inducing_size=int(mt),
            rff_sample_size=int(rff_sample_size),
            variance=float(variance),
            lengthscale=float(lengthscale),
            kernel_type=str(kernel_type),
            jitter=float(jitter),
            seed=int(seed),
            spectral_mixture_weights=tuple(np.asarray(spectral_mixture_weights, dtype=float).reshape(-1).tolist())
            if spectral_mixture_weights is not None
            else TemporalAnalyticConfig.spectral_mixture_weights,
            spectral_mixture_means=tuple(np.asarray(spectral_mixture_means, dtype=float).reshape(-1).tolist())
            if spectral_mixture_means is not None
            else TemporalAnalyticConfig.spectral_mixture_means,
            spectral_mixture_scales=tuple(np.asarray(spectral_mixture_scales, dtype=float).reshape(-1).tolist())
            if spectral_mixture_scales is not None
            else TemporalAnalyticConfig.spectral_mixture_scales,
        )
    )


def make_block_factors_analytic_hippo(
    dataset: SyntheticDataset,
    *,
    block: slice,
    basis_block: slice | None = None,
    old_basis_block: slice | None = None,
    mt: int,
    lengthscale: float = 0.25,
    kernel_variance: float | None = None,
    rff_sample_size: int = 256,
    seed: int = 0,
    moving: bool = True,
    padding: float = 0.0,
    jitter: float = 1e-6,
    kernel_type: str = "rbf",
    spectral_mixture_weights: np.ndarray | None = None,
    spectral_mixture_means: np.ndarray | None = None,
    spectral_mixture_scales: np.ndarray | None = None,
) -> BlockFactors:
    """Build Route-B factors with analytic HiPPO-RFF temporal interdomain features.

    ``block`` selects the query observations. ``basis_block`` selects the temporal
    HiPPO inducing system used for those queries; for training it is usually the
    same block, while seen-history/future evaluation can query another block
    under the current posterior basis. ``old_basis_block`` is the previous basis
    used to form ``K_on_t`` for changing-basis old-likelihood transfer.
    """

    basis_block = block if basis_block is None else basis_block
    variance = dataset.gp_prior_variance if kernel_variance is None else kernel_variance
    builder = make_analytic_temporal_builder(
        mt=mt,
        lengthscale=lengthscale,
        variance=variance,
        rff_sample_size=rff_sample_size,
        seed=seed,
        jitter=jitter,
        kernel_type=kernel_type,
        spectral_mixture_weights=spectral_mixture_weights,
        spectral_mixture_means=spectral_mixture_means,
        spectral_mixture_scales=spectral_mixture_scales,
    )
    basis_spec = temporal_spec_for_block(dataset.times, basis_block, moving=moving, padding=padding)
    query_times = dataset.times[block]
    Kt_torch = builder.add_jitter(builder.compute_kuu_t(basis_spec))
    Kt = Kt_torch.detach().cpu().numpy()
    Kfu = builder.compute_kfu_t(query_times, basis_spec).detach().cpu().numpy()
    T = solve_spd(Kt, Kfu.T, jitter=1e-12).T
    if old_basis_block is None:
        K_on_t = None
    else:
        old_spec = temporal_spec_for_block(dataset.times, old_basis_block, moving=moving, padding=padding)
        K_on_t = builder.compute_kuu_t_cross(old_spec, basis_spec).detach().cpu().numpy()

    Y_b = dataset.Y[:, block]
    start = block.start or 0
    stop = block.stop or dataset.Y.shape[1]
    ns = dataset.Y.shape[0]
    row_idx = []
    for t_idx in range(start, stop):
        row_idx.extend(range(t_idx * ns, (t_idx + 1) * ns))
    Phi_b = dataset.Phi[np.asarray(row_idx)]
    inducing_times = builder.get_time_stamps(basis_spec).detach().cpu().numpy().reshape(-1)
    return BlockFactors(
        y_vec=vec_f(Y_b),
        Phi=Phi_b,
        Y=Y_b,
        T=T,
        Kt=Kt,
        K_on_t=K_on_t,
        block_slice=block,
        inducing_times=inducing_times,
        temporal_backend="analytic_hippo_rff",
        temporal_builder=builder,
        temporal_basis_spec=basis_spec,
    )


def iter_time_blocks(num_time: int, block_size: int) -> list[slice]:
    return [slice(start, min(num_time, start + block_size)) for start in range(0, num_time, block_size)]
