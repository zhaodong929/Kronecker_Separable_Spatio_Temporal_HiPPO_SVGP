#!/usr/bin/env python3
"""Strict-online cumulative-HiPPO Route B with a Negative-Binomial likelihood.

The non-Gaussian posterior is an online Laplace/assumed-density approximation.
Each arriving block contributes one local quadratic likelihood site. Historical
sites are transferred when the cumulative HiPPO temporal basis changes, and a
label is incorporated only when the delayed-observation protocol releases it.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_iclr_era5_routeb_strict_online import temporal_factors_torch
from scripts.run_routeb_online_parity_ladder import spatial_projection
from stvgp_kronecker.joint_ssgp_kron.synthetic import make_analytic_temporal_builder
from stvgp_kronecker.joint_ssgp_kron.torch_backend import solve_spd
from stvgp_kronecker.temporal_kernel_config import load_spectral_mixture_config


ECE_COVERAGE_LEVELS = tuple(np.round(np.arange(0.05, 1.0, 0.1), 2))


@dataclass
class NegativeBinomialState:
    site_precision: torch.Tensor
    site_natural: torch.Tensor
    mean: torch.Tensor
    cholesky: torch.Tensor
    temporal_prior: torch.Tensor


def robust_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (matrix + matrix.transpose(0, 1))
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    for jitter in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        factor, info = torch.linalg.cholesky_ex(symmetric + jitter * eye)
        if int(info.max().detach().cpu()) == 0:
            return factor
    raise torch.linalg.LinAlgError("Negative-Binomial posterior precision is not positive definite")


def negative_binomial_log_prob(
    counts: torch.Tensor,
    log_mean: torch.Tensor,
    dispersion: float,
) -> torch.Tensor:
    """NB2 log probability with variance mu + mu^2 / dispersion."""

    r = torch.as_tensor(dispersion, dtype=log_mean.dtype, device=log_mean.device)
    eta = torch.clamp(log_mean, min=-20.0, max=20.0)
    mu = torch.exp(eta)
    return (
        torch.lgamma(counts + r)
        - torch.lgamma(r)
        - torch.lgamma(counts + 1.0)
        + r * (torch.log(r) - torch.log(r + mu))
        + counts * (eta - torch.log(r + mu))
    )


def estimate_dispersion(
    counts: np.ndarray,
    mean_counts: np.ndarray,
    minimum: float = 0.1,
    maximum: float = 10000.0,
) -> float:
    """Task-1 method-of-moments estimate for NB2 over-dispersion."""

    y = np.asarray(counts, dtype=np.float64)
    mu = np.maximum(np.asarray(mean_counts, dtype=np.float64), 1e-8)
    excess = float(np.sum((y - mu) ** 2 - mu))
    if not np.isfinite(excess) or excess <= 0.0:
        return float(maximum)
    estimate = float(np.sum(mu**2) / excess)
    return float(np.clip(estimate, minimum, maximum))


def posterior_from_sites(
    site_precision: torch.Tensor,
    site_natural: torch.Tensor,
    *,
    temporal_prior: torch.Tensor,
    spatial_prior_inverse: torch.Tensor,
    beta_dimension: int,
    beta_prior_variance: float,
) -> NegativeBinomialState:
    mt = temporal_prior.shape[0]
    ms = spatial_prior_inverse.shape[0]
    temporal_inverse = solve_spd(
        temporal_prior,
        torch.eye(mt, device=temporal_prior.device, dtype=temporal_prior.dtype),
        jitter=1e-10,
    )
    prior_precision = torch.zeros_like(site_precision)
    prior_precision[:beta_dimension, :beta_dimension] = (
        torch.eye(beta_dimension, device=site_precision.device, dtype=site_precision.dtype)
        / beta_prior_variance
    )
    prior_precision[beta_dimension:, beta_dimension:] = torch.kron(
        temporal_inverse.contiguous(),
        spatial_prior_inverse.contiguous(),
    )
    cholesky = robust_cholesky(prior_precision + site_precision)
    mean = torch.cholesky_solve(site_natural[:, None], cholesky).squeeze(1)
    if not torch.isfinite(mean).all():
        raise FloatingPointError("Negative-Binomial posterior mean is non-finite")
    return NegativeBinomialState(
        site_precision=site_precision,
        site_natural=site_natural,
        mean=mean,
        cholesky=cholesky,
        temporal_prior=temporal_prior,
    )


def empty_state(
    *,
    beta_dimension: int,
    temporal_prior: torch.Tensor,
    spatial_prior_inverse: torch.Tensor,
    beta_prior_variance: float,
) -> NegativeBinomialState:
    dimension = beta_dimension + temporal_prior.shape[0] * spatial_prior_inverse.shape[0]
    zeros = temporal_prior.new_zeros((dimension, dimension))
    return posterior_from_sites(
        zeros,
        temporal_prior.new_zeros((dimension,)),
        temporal_prior=temporal_prior,
        spatial_prior_inverse=spatial_prior_inverse,
        beta_dimension=beta_dimension,
        beta_prior_variance=beta_prior_variance,
    )


def transfer_state(
    state: NegativeBinomialState,
    temporal_map: torch.Tensor,
    *,
    temporal_prior: torch.Tensor,
    spatial_prior_inverse: torch.Tensor,
    beta_dimension: int,
    beta_prior_variance: float,
) -> NegativeBinomialState:
    """Transfer fixed historical likelihood sites into a new HiPPO basis."""

    old_mt, new_mt = temporal_map.shape
    ms = spatial_prior_inverse.shape[0]
    precision = state.site_precision
    r_bb = precision[:beta_dimension, :beta_dimension]
    r_bu = precision[:beta_dimension, beta_dimension:].reshape(beta_dimension, old_mt, ms)
    r_uu = precision[beta_dimension:, beta_dimension:].reshape(old_mt, ms, old_mt, ms)
    h_beta = state.site_natural[:beta_dimension]
    h_u = state.site_natural[beta_dimension:].reshape(old_mt, ms)

    r_bu_new = torch.einsum("dis,ia->das", r_bu, temporal_map)
    intermediate = torch.einsum("isjr,jb->isbr", r_uu, temporal_map)
    r_uu_new = torch.einsum("ia,isbr->asbr", temporal_map, intermediate)
    h_u_new = torch.einsum("is,ia->as", h_u, temporal_map)

    dimension = beta_dimension + new_mt * ms
    transferred_precision = precision.new_zeros((dimension, dimension))
    transferred_precision[:beta_dimension, :beta_dimension] = r_bb
    transferred_precision[:beta_dimension, beta_dimension:] = r_bu_new.reshape(beta_dimension, -1)
    transferred_precision[beta_dimension:, :beta_dimension] = r_bu_new.reshape(beta_dimension, -1).transpose(0, 1)
    transferred_precision[beta_dimension:, beta_dimension:] = r_uu_new.reshape(new_mt * ms, new_mt * ms)
    transferred_natural = state.site_natural.new_zeros((dimension,))
    transferred_natural[:beta_dimension] = h_beta
    transferred_natural[beta_dimension:] = h_u_new.reshape(-1)
    return posterior_from_sites(
        0.5 * (transferred_precision + transferred_precision.transpose(0, 1)),
        transferred_natural,
        temporal_prior=temporal_prior,
        spatial_prior_inverse=spatial_prior_inverse,
        beta_dimension=beta_dimension,
        beta_prior_variance=beta_prior_variance,
    )


def latent_mean(
    posterior_mean: torch.Tensor,
    phi: torch.Tensor,
    temporal_projection: torch.Tensor,
    spatial_projection_matrix: torch.Tensor,
) -> torch.Tensor:
    beta_dimension = phi.shape[-1]
    mt = temporal_projection.shape[1]
    ms = spatial_projection_matrix.shape[1]
    beta = posterior_mean[:beta_dimension]
    inducing_mean = posterior_mean[beta_dimension:].reshape(mt, ms)
    return (
        torch.einsum("tsp,p->ts", phi, beta)
        + torch.einsum("tm,sk,mk->ts", temporal_projection, spatial_projection_matrix, inducing_mean)
    )


def likelihood_site(
    *,
    phi: torch.Tensor,
    temporal_projection: torch.Tensor,
    spatial_projection_matrix: torch.Tensor,
    weight: torch.Tensor,
    pseudo_latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dense quadratic site without materialising all Kronecker rows."""

    nt, ns, beta_dimension = phi.shape
    mt = temporal_projection.shape[1]
    ms = spatial_projection_matrix.shape[1]
    phi_flat = phi.reshape(nt * ns, beta_dimension)
    weight_flat = weight.reshape(-1)
    pseudo_flat = pseudo_latent.reshape(-1)
    weighted_target = weight_flat * pseudo_flat

    r_bb = phi_flat.transpose(0, 1) @ (weight_flat[:, None] * phi_flat)
    h_beta = phi_flat.transpose(0, 1) @ weighted_target
    r_bu = torch.einsum(
        "tsp,ts,sk,tm->pmk",
        phi,
        weight,
        spatial_projection_matrix,
        temporal_projection,
    )
    r_uu = phi.new_zeros((mt, ms, mt, ms))
    h_u = phi.new_zeros((mt, ms))
    for time_index in range(nt):
        temporal_row = temporal_projection[time_index]
        weighted_spatial = weight[time_index, :, None] * spatial_projection_matrix
        spatial_gram = spatial_projection_matrix.transpose(0, 1) @ weighted_spatial
        temporal_gram = torch.outer(temporal_row, temporal_row)
        r_uu = r_uu + temporal_gram[:, None, :, None] * spatial_gram[None, :, None, :]
        spatial_natural = spatial_projection_matrix.transpose(0, 1) @ weighted_target.reshape(nt, ns)[time_index]
        h_u = h_u + temporal_row[:, None] * spatial_natural[None, :]

    dimension = beta_dimension + mt * ms
    precision = phi.new_zeros((dimension, dimension))
    precision[:beta_dimension, :beta_dimension] = r_bb
    precision[:beta_dimension, beta_dimension:] = r_bu.reshape(beta_dimension, -1)
    precision[beta_dimension:, :beta_dimension] = r_bu.reshape(beta_dimension, -1).transpose(0, 1)
    precision[beta_dimension:, beta_dimension:] = r_uu.reshape(mt * ms, mt * ms)
    natural = phi.new_zeros((dimension,))
    natural[:beta_dimension] = h_beta
    natural[beta_dimension:] = h_u.reshape(-1)
    return 0.5 * (precision + precision.transpose(0, 1)), natural


def laplace_update(
    state: NegativeBinomialState,
    *,
    counts: torch.Tensor,
    phi: torch.Tensor,
    temporal_projection: torch.Tensor,
    spatial_projection_matrix: torch.Tensor,
    log_exposure: torch.Tensor,
    dispersion: float,
    steps: int,
    spatial_prior_inverse: torch.Tensor,
    beta_prior_variance: float,
) -> NegativeBinomialState:
    base_precision = state.site_precision
    base_natural = state.site_natural
    candidate = state
    for _ in range(max(1, int(steps))):
        eta = log_exposure + latent_mean(
            candidate.mean,
            phi,
            temporal_projection,
            spatial_projection_matrix,
        )
        mu = torch.exp(torch.clamp(eta, min=-20.0, max=20.0))
        r = torch.as_tensor(dispersion, device=mu.device, dtype=mu.dtype)
        gradient = r * (counts - mu) / (r + mu)
        weight = torch.clamp(r * mu * (r + counts) / (r + mu).square(), min=1e-8, max=1e8)
        pseudo_latent = eta + gradient / weight - log_exposure
        block_precision, block_natural = likelihood_site(
            phi=phi,
            temporal_projection=temporal_projection,
            spatial_projection_matrix=spatial_projection_matrix,
            weight=weight,
            pseudo_latent=pseudo_latent,
        )
        candidate = posterior_from_sites(
            base_precision + block_precision,
            base_natural + block_natural,
            temporal_prior=state.temporal_prior,
            spatial_prior_inverse=spatial_prior_inverse,
            beta_dimension=phi.shape[-1],
            beta_prior_variance=beta_prior_variance,
        )
    return candidate


def prediction_design(
    phi: torch.Tensor,
    temporal_projection: torch.Tensor,
    spatial_projection_matrix: torch.Tensor,
) -> torch.Tensor:
    gp = torch.einsum(
        "tm,sk->tsmk",
        temporal_projection,
        spatial_projection_matrix,
    ).reshape(phi.shape[0] * phi.shape[1], -1)
    return torch.cat([phi.reshape(-1, phi.shape[-1]), gp], dim=1)


def predictive_latent(
    state: NegativeBinomialState,
    *,
    phi: torch.Tensor,
    temporal_projection: torch.Tensor,
    spatial_projection_matrix: torch.Tensor,
    log_exposure: torch.Tensor,
    spatial_prior: torch.Tensor,
    prior_point_variance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    design = prediction_design(phi, temporal_projection, spatial_projection_matrix)
    mean = log_exposure.reshape(-1) + design @ state.mean
    solved = torch.cholesky_solve(design.transpose(0, 1), state.cholesky)
    variance = torch.sum(design.transpose(0, 1) * solved, dim=0)
    projected_time = torch.einsum(
        "tm,mn,tn->t",
        temporal_projection,
        state.temporal_prior,
        temporal_projection,
    )
    projected_space = torch.einsum(
        "sm,mn,sn->s",
        spatial_projection_matrix,
        spatial_prior,
        spatial_projection_matrix,
    )
    conditional_residual = torch.clamp(
        prior_point_variance - torch.outer(projected_time, projected_space),
        min=0.0,
    ).reshape(-1)
    return mean, torch.clamp(variance + conditional_residual, min=1e-10, max=100.0)


def mixture_count_nll(
    counts: torch.Tensor,
    latent_mean_value: torch.Tensor,
    latent_variance: torch.Tensor,
    dispersion: float,
    quadrature_points: int,
) -> torch.Tensor:
    nodes, weights = np.polynomial.hermite.hermgauss(int(quadrature_points))
    nodes_tensor = torch.as_tensor(nodes, device=counts.device, dtype=counts.dtype)[:, None]
    log_weights = torch.log(
        torch.as_tensor(weights, device=counts.device, dtype=counts.dtype)
    )[:, None]
    eta = latent_mean_value[None, :] + torch.sqrt(2.0 * latent_variance)[None, :] * nodes_tensor
    log_prob = negative_binomial_log_prob(counts[None, :], eta, dispersion)
    return -torch.logsumexp(log_weights + log_prob, dim=0) + 0.5 * math.log(math.pi)


def sample_common_scale(
    *,
    latent_mean_value: torch.Tensor,
    latent_variance: torch.Tensor,
    exposure: torch.Tensor,
    dispersion: float,
    samples: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(int(seed))
    if latent_mean_value.is_cuda:
        torch.cuda.manual_seed_all(int(seed))
    eta = latent_mean_value[None, :] + torch.sqrt(latent_variance)[None, :] * torch.randn(
        (int(samples), latent_mean_value.numel()),
        device=latent_mean_value.device,
        dtype=latent_mean_value.dtype,
    )
    mu = torch.exp(torch.clamp(eta, min=-20.0, max=20.0))
    r = torch.full_like(mu, float(dispersion))
    poisson_rate = torch.distributions.Gamma(concentration=r, rate=r / mu).sample()
    count_samples = torch.poisson(poisson_rate)
    common = torch.log1p(count_samples / exposure.reshape(1, -1))
    coverage_levels = torch.as_tensor(
        ECE_COVERAGE_LEVELS,
        device=common.device,
        dtype=common.dtype,
    )
    lower_quantiles = torch.quantile(common, (1.0 - coverage_levels) / 2.0, dim=0)
    upper_quantiles = torch.quantile(common, (1.0 + coverage_levels) / 2.0, dim=0)
    return (
        common.mean(dim=0),
        torch.clamp(common.var(dim=0, unbiased=False), min=1e-10),
        torch.quantile(common, 0.05, dim=0),
        torch.quantile(common, 0.95, dim=0),
        count_samples.mean(dim=0),
        torch.clamp(count_samples.var(dim=0, unbiased=False), min=1e-10),
        lower_quantiles,
        upper_quantiles,
        common,
    )


def aggregate_metrics(
    y_log_rate: np.ndarray,
    pred_mean: np.ndarray,
    pred_variance: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    count_nll: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y_log_rate, dtype=np.float64).reshape(-1)
    mean = np.asarray(pred_mean, dtype=np.float64).reshape(-1)
    variance = np.maximum(np.asarray(pred_variance, dtype=np.float64).reshape(-1), 1e-10)
    return {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "log1p_per_100k_gaussian_moment_nll": float(
            np.mean(0.5 * (np.log(2.0 * np.pi * variance) + (y - mean) ** 2 / variance))
        ),
        "coverage90": float(np.mean((y >= np.asarray(lower).reshape(-1)) & (y <= np.asarray(upper).reshape(-1)))),
        "negative_binomial_count_nll": float(np.mean(np.asarray(count_nll, dtype=np.float64))),
        "mean_predictive_std": float(np.mean(np.sqrt(variance))),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--theta-json", type=Path, required=True)
    parser.add_argument("--spectral-mixture-json", type=Path, default=Path("configs/covid_sm_q2.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mt", type=int, default=32)
    parser.add_argument("--ms", type=int, default=32)
    parser.add_argument("--rff-sample-size", type=int, default=64)
    parser.add_argument("--beta-prior-variance", type=float, default=1000.0)
    parser.add_argument("--task1-laplace-steps", type=int, default=6)
    parser.add_argument("--online-laplace-steps", type=int, default=1)
    parser.add_argument("--dispersion", type=float)
    parser.add_argument("--quadrature-points", type=int, default=20)
    parser.add_argument("--predictive-samples", type=int, default=256)
    parser.add_argument("--save-common-predictive-samples", action="store_true")
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.mt != 32 or args.ms != 32:
        raise ValueError("The formal COVID target ablation is fixed at Mt=32, Ms=32")
    if args.task1_laplace_steps < 1 or args.online_laplace_steps < 1:
        raise ValueError("Laplace steps must be positive")
    if args.predictive_samples < 32:
        raise ValueError("At least 32 predictive samples are required")

    metadata = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    if metadata.get("target_mode") != "log1p_per_100k":
        raise ValueError("NB Route B requires the log1p_per_100k protocol for its causal lag features")
    theta_payload = json.loads(args.theta_json.read_text(encoding="utf-8"))
    theta = theta_payload.get("learned_theta", theta_payload)
    mixture = load_spectral_mixture_config(args.spectral_mixture_json)
    if mixture is None:
        raise ValueError("A Q=2 spectral-mixture configuration is required")

    device = torch.device(args.device)
    dtype = torch.float64
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with np.load(args.protocol_npz) as arrays:
        required = {
            "calibration_y", "stream_y", "calibration_phi", "stream_phi",
            "calibration_counts", "stream_counts", "population_per_100k",
            "coordinates", "train_indices", "fit_indices", "test_indices",
            "calibration_times", "block_start", "block_stop", f"inducing_coords_ms{args.ms}",
        }
        missing = sorted(required - set(arrays.files))
        if missing:
            raise KeyError(f"Target-ablation protocol is missing {missing}")
        calibration_y = np.asarray(arrays["calibration_y"], dtype=np.float64)
        stream_y = np.asarray(arrays["stream_y"], dtype=np.float64)
        calibration_phi_np = np.asarray(arrays["calibration_phi"], dtype=np.float64)
        stream_phi_np = np.asarray(arrays["stream_phi"], dtype=np.float64)
        calibration_counts_np = np.asarray(arrays["calibration_counts"], dtype=np.float64)
        stream_counts_np = np.asarray(arrays["stream_counts"], dtype=np.float64)
        exposure_np = np.asarray(arrays["population_per_100k"], dtype=np.float64)
        coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
        train = np.asarray(arrays["train_indices"], dtype=int)
        fit = np.asarray(arrays["fit_indices"], dtype=int)
        test = np.asarray(arrays["test_indices"], dtype=int)
        calibration_times = np.asarray(arrays["calibration_times"], dtype=np.float64)
        inducing = np.asarray(arrays[f"inducing_coords_ms{args.ms}"], dtype=np.float64)
        blocks = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(arrays["block_start"], arrays["block_stop"])
        )
    if args.max_blocks > 0:
        blocks = blocks[: args.max_blocks]

    target_mean = float(metadata["target_standardization"]["mean"])
    target_scale = float(metadata["target_standardization"]["scale"])
    fit_design = calibration_phi_np[:, fit].reshape(-1, calibration_phi_np.shape[-1])
    fit_target = calibration_y[:, fit].reshape(-1)
    task1_fit_beta = np.linalg.solve(
        fit_design.T @ fit_design + 1e-3 * np.eye(fit_design.shape[1]),
        fit_design.T @ fit_target,
    )
    proxy_standardized = np.einsum("tsp,p->ts", calibration_phi_np[:, fit], task1_fit_beta)
    proxy_log_rate = target_mean + target_scale * proxy_standardized
    proxy_counts = exposure_np[fit][None, :] * np.expm1(np.maximum(proxy_log_rate, 0.0))
    dispersion = (
        float(args.dispersion)
        if args.dispersion is not None
        else estimate_dispersion(calibration_counts_np[:, fit], proxy_counts)
    )
    if not np.isfinite(dispersion) or dispersion <= 0.0:
        raise ValueError("Negative-Binomial dispersion must be finite and positive")

    step = float(np.median(np.diff(calibration_times)))
    temporal_times = np.concatenate(
        [
            calibration_times,
            calibration_times[-1] + step * np.arange(1, stream_y.shape[0] + 1),
        ]
    )
    builder = make_analytic_temporal_builder(
        mt=args.mt,
        lengthscale=float(theta["ell_t"]),
        variance=float(theta["kernel_variance"]),
        rff_sample_size=args.rff_sample_size,
        seed=0,
        jitter=1e-7,
        kernel_type="spectral_mixture",
        spectral_mixture_weights=mixture["weights"],
        spectral_mixture_means=mixture["means"],
        spectral_mixture_scales=mixture["scales"],
    ).to(device=device, dtype=dtype)
    spatial_prior_np, c_all_np = spatial_projection(coordinates, inducing, theta["ell_s"])
    spatial_prior = torch.as_tensor(spatial_prior_np, device=device, dtype=dtype)
    spatial_inverse = solve_spd(
        spatial_prior,
        torch.eye(args.ms, device=device, dtype=dtype),
        jitter=1e-10,
    )
    c_all = torch.as_tensor(c_all_np, device=device, dtype=dtype)
    c_train = c_all[torch.as_tensor(train, device=device)]
    c_test = c_all[torch.as_tensor(test, device=device)]

    calibration_stop = calibration_y.shape[0]
    task1_slice = slice(0, calibration_stop)
    started = time.perf_counter()
    task1_t, task1_kt, _, previous_temporal_basis = temporal_factors_torch(
        builder=builder,
        times=temporal_times,
        query=task1_slice,
        basis=task1_slice,
        old_basis=None,
        old_temporal_basis=None,
        device=device,
        dtype=dtype,
    )
    task1_t = task1_t.to(device=device, dtype=dtype)
    task1_kt = task1_kt.to(device=device, dtype=dtype)
    beta_dimension = calibration_phi_np.shape[-1]
    state = empty_state(
        beta_dimension=beta_dimension,
        temporal_prior=task1_kt,
        spatial_prior_inverse=spatial_inverse,
        beta_prior_variance=args.beta_prior_variance,
    )
    state = laplace_update(
        state,
        counts=torch.as_tensor(calibration_counts_np[:, train], device=device, dtype=dtype),
        phi=torch.as_tensor(calibration_phi_np[:, train], device=device, dtype=dtype),
        temporal_projection=task1_t,
        spatial_projection_matrix=c_train,
        log_exposure=torch.log(torch.as_tensor(exposure_np[train], device=device, dtype=dtype))[None, :].expand(calibration_stop, -1),
        dispersion=dispersion,
        steps=args.task1_laplace_steps,
        spatial_prior_inverse=spatial_inverse,
        beta_prior_variance=args.beta_prior_variance,
    )
    task1_seconds = time.perf_counter() - started
    previous_basis = task1_slice

    pending: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    all_true_log_rate: list[np.ndarray] = []
    all_true_count: list[np.ndarray] = []
    all_mean_log_rate: list[np.ndarray] = []
    all_variance_log_rate: list[np.ndarray] = []
    all_lower: list[np.ndarray] = []
    all_upper: list[np.ndarray] = []
    all_mean_count: list[np.ndarray] = []
    all_variance_count: list[np.ndarray] = []
    all_count_nll: list[np.ndarray] = []
    all_interval_lower: list[np.ndarray] = []
    all_interval_upper: list[np.ndarray] = []
    all_common_samples: list[np.ndarray] = []
    delayed_rows = 0
    online_started = time.perf_counter()

    for block_id, block in enumerate(blocks):
        if pending:
            delayed = pending.pop(0)
            delayed_block = delayed["block"]
            delayed_slice = slice(
                calibration_stop + delayed_block.start,
                calibration_stop + delayed_block.stop,
            )
            delayed_t, _, _, _ = temporal_factors_torch(
                builder=builder,
                times=temporal_times,
                query=delayed_slice,
                basis=previous_basis,
                old_basis=None,
                old_temporal_basis=None,
                device=device,
                dtype=dtype,
            )
            state = laplace_update(
                state,
                counts=delayed["counts"],
                phi=delayed["phi"],
                temporal_projection=delayed_t.to(device=device, dtype=dtype),
                spatial_projection_matrix=c_test,
                log_exposure=torch.log(torch.as_tensor(exposure_np[test], device=device, dtype=dtype))[None, :],
                dispersion=dispersion,
                steps=args.online_laplace_steps,
                spatial_prior_inverse=spatial_inverse,
                beta_prior_variance=args.beta_prior_variance,
            )
            delayed_rows += int(delayed["counts"].numel())

        temporal_block = slice(calibration_stop + block.start, calibration_stop + block.stop)
        basis = slice(0, temporal_block.stop)
        current_t, current_kt, k_on_t, new_temporal_basis = temporal_factors_torch(
            builder=builder,
            times=temporal_times,
            query=temporal_block,
            basis=basis,
            old_basis=previous_basis,
            old_temporal_basis=previous_temporal_basis,
            device=device,
            dtype=dtype,
        )
        current_t = current_t.to(device=device, dtype=dtype)
        current_kt = current_kt.to(device=device, dtype=dtype)
        if k_on_t is None:
            temporal_map = torch.eye(args.mt, device=device, dtype=dtype)
        else:
            k_on_t = k_on_t.to(device=device, dtype=dtype)
            temporal_map = solve_spd(current_kt, k_on_t.transpose(0, 1), jitter=1e-10).transpose(0, 1)
        state = transfer_state(
            state,
            temporal_map,
            temporal_prior=current_kt,
            spatial_prior_inverse=spatial_inverse,
            beta_dimension=beta_dimension,
            beta_prior_variance=args.beta_prior_variance,
        )
        previous_basis = basis
        previous_temporal_basis = new_temporal_basis

        phi_block = torch.as_tensor(stream_phi_np[block], device=device, dtype=dtype)
        count_block = torch.as_tensor(stream_counts_np[block], device=device, dtype=dtype)
        state = laplace_update(
            state,
            counts=count_block[:, train],
            phi=phi_block[:, train],
            temporal_projection=current_t,
            spatial_projection_matrix=c_train,
            log_exposure=torch.log(torch.as_tensor(exposure_np[train], device=device, dtype=dtype))[None, :],
            dispersion=dispersion,
            steps=args.online_laplace_steps,
            spatial_prior_inverse=spatial_inverse,
            beta_prior_variance=args.beta_prior_variance,
        )

        test_exposure = torch.as_tensor(exposure_np[test], device=device, dtype=dtype)
        latent_prediction_mean, latent_prediction_variance = predictive_latent(
            state,
            phi=phi_block[:, test],
            temporal_projection=current_t,
            spatial_projection_matrix=c_test,
            log_exposure=torch.log(test_exposure)[None, :],
            spatial_prior=spatial_prior,
            prior_point_variance=float(theta["kernel_variance"]),
        )
        true_count = count_block[:, test].reshape(-1)
        count_nll = mixture_count_nll(
            true_count,
            latent_prediction_mean,
            latent_prediction_variance,
            dispersion,
            args.quadrature_points,
        )
        (
            common_mean,
            common_variance,
            lower,
            upper,
            count_mean,
            count_variance,
            interval_lower,
            interval_upper,
            common_samples,
        ) = sample_common_scale(
            latent_mean_value=latent_prediction_mean,
            latent_variance=latent_prediction_variance,
            exposure=test_exposure,
            dispersion=dispersion,
            samples=args.predictive_samples,
            seed=100000 * args.seed + block_id,
        )
        true_log_rate = torch.log1p(true_count / test_exposure)
        block_metrics = aggregate_metrics(
            true_log_rate.detach().cpu().numpy(),
            common_mean.detach().cpu().numpy(),
            common_variance.detach().cpu().numpy(),
            lower.detach().cpu().numpy(),
            upper.detach().cpu().numpy(),
            count_nll.detach().cpu().numpy(),
        )
        rows.append({"block_id": block_id, "stream_start": block.start, "stream_stop": block.stop, **block_metrics})
        all_true_log_rate.append(true_log_rate.detach().cpu().numpy())
        all_true_count.append(true_count.detach().cpu().numpy())
        all_mean_log_rate.append(common_mean.detach().cpu().numpy())
        all_variance_log_rate.append(common_variance.detach().cpu().numpy())
        all_lower.append(lower.detach().cpu().numpy())
        all_upper.append(upper.detach().cpu().numpy())
        all_mean_count.append(count_mean.detach().cpu().numpy())
        all_variance_count.append(count_variance.detach().cpu().numpy())
        all_count_nll.append(count_nll.detach().cpu().numpy())
        all_interval_lower.append(interval_lower.detach().cpu().numpy())
        all_interval_upper.append(interval_upper.detach().cpu().numpy())
        if args.save_common_predictive_samples:
            all_common_samples.append(common_samples.detach().cpu().numpy())
        pending.append({
            "block": block,
            "counts": count_block[:, test],
            "phi": phi_block[:, test],
        })

    online_seconds = time.perf_counter() - online_started
    arrays_out = {
        "y_true": np.vstack(all_true_log_rate),
        "y_true_count": np.vstack(all_true_count),
        "pred_mean": np.vstack(all_mean_log_rate),
        "pred_variance": np.vstack(all_variance_log_rate),
        "pred_lower90": np.vstack(all_lower),
        "pred_upper90": np.vstack(all_upper),
        "pred_mean_count": np.vstack(all_mean_count),
        "pred_variance_count": np.vstack(all_variance_count),
        "negative_binomial_count_nll": np.vstack(all_count_nll),
        "ece_coverage_levels": np.asarray(ECE_COVERAGE_LEVELS, dtype=np.float64),
        "pred_interval_lower": np.stack(all_interval_lower, axis=1),
        "pred_interval_upper": np.stack(all_interval_upper, axis=1),
        "predictive_sample_count": np.asarray(args.predictive_samples, dtype=np.int64),
    }
    if args.save_common_predictive_samples:
        arrays_out["common_predictive_samples"] = np.stack(all_common_samples, axis=1)
    overall = aggregate_metrics(
        arrays_out["y_true"],
        arrays_out["pred_mean"],
        arrays_out["pred_variance"],
        arrays_out["pred_lower90"],
        arrays_out["pred_upper90"],
        arrays_out["negative_binomial_count_nll"],
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "predictions.npz", **arrays_out)
    write_csv(rows, output_dir / "blocks.csv")
    result = {
        "implementation": "cumulative HiPPO Route B; online Laplace/ADF Negative-Binomial likelihood",
        "likelihood": "NB2 with log link and population-per-100k offset",
        "protocol": str(args.protocol_npz.resolve()),
        "split_seed": args.seed,
        "mt": args.mt,
        "ms": args.ms,
        "dispersion": dispersion,
        "dispersion_estimation": "fixed argument" if args.dispersion is not None else "Task-1 fit-state method of moments",
        "task1_laplace_steps": args.task1_laplace_steps,
        "online_laplace_steps": args.online_laplace_steps,
        "delayed_observation_rows": delayed_rows,
        "overall_current_block": overall,
        "evaluation_scale": "log1p admissions per 100,000; NB count NLL reported separately",
        "ece_intervals": {
            "coverage_levels": list(ECE_COVERAGE_LEVELS),
            "predictive_samples": args.predictive_samples,
            "quantile_method": "torch.quantile empirical predictive samples",
        },
        "common_predictive_samples": {
            "saved": bool(args.save_common_predictive_samples),
            "scale": "log1p admissions per 100,000",
            "shape": list(arrays_out["common_predictive_samples"].shape)
            if args.save_common_predictive_samples
            else None,
        },
        "timing": {"task1_seconds": task1_seconds, "online_seconds": online_seconds},
        "posterior_finite": bool(torch.isfinite(state.mean).all().detach().cpu()),
        "args": vars(args),
    }
    result["args"] = {key: str(value) if isinstance(value, Path) else value for key, value in result["args"].items()}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
