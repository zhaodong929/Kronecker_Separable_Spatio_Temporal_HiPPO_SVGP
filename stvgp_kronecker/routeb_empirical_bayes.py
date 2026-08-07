"""Differentiable empirical-Bayes objective for finite structured-joint Route B."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    make_analytic_temporal_builder,
    temporal_spec_for_block,
)
from stvgp_kronecker.temporal_analytic import TemporalBlockSpec


DTYPE = torch.float64


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def robust_cholesky(matrix: torch.Tensor, base_jitter: float = 1e-7) -> torch.Tensor:
    """Cholesky with detached scale selection and bounded jitter escalation."""

    matrix = _symmetrize(matrix)
    scale = torch.clamp(torch.diagonal(matrix).mean().detach().abs(), min=1.0)
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    jitter = base_jitter * scale
    for _ in range(8):
        chol, info = torch.linalg.cholesky_ex(matrix + jitter * eye)
        if int(info.max().item()) == 0:
            return chol
        jitter = jitter * 10.0
    raise RuntimeError("Cholesky failed after eight jitter escalations")


def matern32_separable(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscales: torch.Tensor,
) -> torch.Tensor:
    """Product Matérn-3/2 kernel with one lengthscale per input dimension."""

    delta = torch.abs(x1[:, None, :] - x2[None, :, :]) / lengthscales[None, None, :]
    scaled = math.sqrt(3.0) * delta
    return torch.prod((1.0 + scaled) * torch.exp(-scaled), dim=-1)


def matern32_1d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscale: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    delta = torch.abs(x1[:, None] - x2[None, :]) / lengthscale
    scaled = math.sqrt(3.0) * delta
    return variance * (1.0 + scaled) * torch.exp(-scaled)


class TemporalFactorModel(nn.Module):
    """Fixed-support temporal representation with trainable kernel parameters."""

    def __init__(
        self,
        *,
        times: np.ndarray,
        mt: int,
        representation: str,
        initial_lengthscale: float,
        initial_variance: float,
        rff_sample_size: int,
        seed: int,
        temporal_horizon: TemporalBlockSpec | None = None,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.mt = int(mt)
        self.register_buffer("times", torch.as_tensor(times, dtype=DTYPE))
        if representation == "analytic_hippo_rff":
            self.builder = make_analytic_temporal_builder(
                mt=mt,
                lengthscale=initial_lengthscale,
                variance=initial_variance,
                rff_sample_size=rff_sample_size,
                seed=seed,
                kernel_type="matern32",
            )
            self.horizon = temporal_horizon or temporal_spec_for_block(
                np.asarray(times, dtype=float), slice(0, len(times)), moving=True
            )
            self.register_buffer("z_t", torch.empty(0, dtype=DTYPE))
        elif representation == "inducing_points":
            self.builder = None
            self.horizon = None
            self.log_lengthscale = nn.Parameter(
                torch.log(torch.as_tensor(initial_lengthscale, dtype=DTYPE))
            )
            self.log_variance = nn.Parameter(
                torch.log(torch.as_tensor(initial_variance, dtype=DTYPE))
            )
            self.register_buffer(
                "z_t",
                torch.linspace(
                    float(np.min(times)), float(np.max(times)), mt, dtype=DTYPE
                ),
            )
        else:
            raise ValueError(f"Unsupported temporal representation: {representation}")

    def set_query_times(
        self,
        times: np.ndarray,
        *,
        temporal_horizon: TemporalBlockSpec | None = None,
    ) -> None:
        """Change query times while retaining parameters and fixed RFF coordinates."""

        times = np.asarray(times, dtype=float).reshape(-1)
        if times.size == 0:
            raise ValueError("Temporal query times must not be empty")
        self.times = torch.as_tensor(
            times, dtype=self.times.dtype, device=self.times.device
        )
        if self.builder is not None:
            self.horizon = temporal_horizon or temporal_spec_for_block(
                times, slice(0, times.size), moving=True
            )

    @property
    def lengthscale(self) -> torch.Tensor:
        if self.builder is not None:
            return self.builder.lengthscale
        return torch.exp(self.log_lengthscale)

    @property
    def variance(self) -> torch.Tensor:
        if self.builder is not None:
            return self.builder.variance
        return torch.exp(self.log_variance)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projection T=K_fu K_uu^-1 and inducing prior K_uu."""

        if self.builder is not None:
            kuu = self.builder.compute_kuu_t(self.horizon)
            kfu = self.builder.compute_kfu_t(self.times, self.horizon)
        else:
            kuu = matern32_1d(
                self.z_t, self.z_t, self.lengthscale, self.variance
            )
            kfu = matern32_1d(
                self.times, self.z_t, self.lengthscale, self.variance
            )
        chol = robust_cholesky(kuu)
        projection = torch.cholesky_solve(kfu.transpose(0, 1), chol).transpose(0, 1)
        return projection, _symmetrize(kuu) + 1e-7 * torch.eye(
            kuu.shape[0], dtype=kuu.dtype, device=kuu.device
        )

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            if self.builder is not None:
                self.builder.log_lengthscale.clamp_(math.log(0.003), math.log(2.0))
                self.builder.log_variance.clamp_(math.log(0.005), math.log(10.0))
            else:
                self.log_lengthscale.clamp_(math.log(0.003), math.log(2.0))
                self.log_variance.clamp_(math.log(0.005), math.log(10.0))


@dataclass
class JointObjectiveDiagnostics:
    nlml_per_observation: torch.Tensor
    finite_nlml_per_observation: torch.Tensor
    vfe_trace_correction_per_observation: torch.Tensor
    vfe_trace_residual_per_observation: torch.Tensor
    logdet_per_observation: torch.Tensor
    quadratic_per_observation: torch.Tensor
    beta_mean: torch.Tensor | None = None
    u_mean: torch.Tensor | None = None
    beta_precision: torch.Tensor | None = None


@dataclass(frozen=True)
class JointSufficientStatistics:
    """Kernel-independent statistics that may be cached across EB steps."""

    phi_phi: torch.Tensor
    phi_y: torch.Tensor
    y_y: torch.Tensor
    num_observations: int
    num_features: int


def joint_sufficient_statistics(
    y_matrix: torch.Tensor,
    phi_tensor: torch.Tensor,
) -> JointSufficientStatistics:
    """Compute detached fixed-data statistics for the collapsed joint objective."""

    num_space, num_time = y_matrix.shape
    num_features = phi_tensor.shape[-1]
    phi_flat = phi_tensor.permute(1, 0, 2).reshape(
        num_space * num_time, num_features
    )
    y_flat = y_matrix.transpose(0, 1).reshape(-1)
    return JointSufficientStatistics(
        phi_phi=(phi_flat.transpose(0, 1) @ phi_flat).detach(),
        phi_y=(phi_flat.transpose(0, 1) @ y_flat).detach(),
        y_y=torch.dot(y_flat, y_flat).detach(),
        num_observations=int(y_flat.numel()),
        num_features=int(num_features),
    )


def feature_gp_cross(
    spatial_projection: torch.Tensor,
    phi_tensor: torch.Tensor,
    temporal_projection: torch.Tensor,
    *,
    order: str = "einsum",
    feature_block_size: int | None = None,
) -> torch.Tensor:
    """Contract the feature/GP cross tensor with an explicit contraction order."""

    if order not in {"auto", "einsum", "spatial_first", "temporal_first"}:
        raise ValueError(f"Unsupported cross contraction order: {order}")
    num_features = phi_tensor.shape[-1]
    if num_features == 0:
        return phi_tensor.new_zeros(
            (0, spatial_projection.shape[1], temporal_projection.shape[1])
        )
    block_size = num_features if not feature_block_size else feature_block_size
    if block_size <= 0:
        raise ValueError("Feature block size must be positive")
    if order == "auto":
        num_space, num_time = phi_tensor.shape[:2]
        num_spatial_inducing = spatial_projection.shape[1]
        num_temporal_inducing = temporal_projection.shape[1]
        spatial_first_cost = (
            num_space
            * num_spatial_inducing
            * num_time
            * num_features
            + num_spatial_inducing
            * num_time
            * num_features
            * num_temporal_inducing
        )
        temporal_first_cost = (
            num_space
            * num_time
            * num_features
            * num_temporal_inducing
            + num_space
            * num_spatial_inducing
            * num_features
            * num_temporal_inducing
        )
        order = (
            "spatial_first"
            if spatial_first_cost <= temporal_first_cost
            else "temporal_first"
        )
    outputs = []
    for start in range(0, num_features, block_size):
        block = phi_tensor[..., start : start + block_size]
        if order == "einsum":
            output = torch.einsum(
                "sa,stp,tb->pab",
                spatial_projection,
                block,
                temporal_projection,
            )
        elif order == "spatial_first":
            intermediate = torch.einsum(
                "sa,stp->atp", spatial_projection, block
            )
            output = torch.einsum(
                "atp,tb->pab", intermediate, temporal_projection
            )
        else:
            intermediate = torch.einsum(
                "stp,tb->spb", block, temporal_projection
            )
            output = torch.einsum(
                "sa,spb->pab", spatial_projection, intermediate
            )
        outputs.append(output)
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


def separable_vfe_trace_residual(
    *,
    temporal_projection: torch.Tensor,
    spatial_projection: torch.Tensor,
    temporal_prior: torch.Tensor,
    spatial_prior: torch.Tensor,
    prior_point_variance: torch.Tensor,
) -> torch.Tensor:
    """Return Tr(K_ff - Q_ff) without materializing observation covariances.

    The observations form a rectangular space-by-time grid and the pointwise
    spatial kernel variance is one, so Tr(K_ff) is N * prior_point_variance.
    For Q_ff = Q_t kron Q_s, Tr(Q_ff) = Tr(Q_t) Tr(Q_s).
    """

    q_time_diag = torch.einsum(
        "ni,ij,nj->n", temporal_projection, temporal_prior, temporal_projection
    )
    q_space_diag = torch.einsum(
        "ni,ij,nj->n", spatial_projection, spatial_prior, spatial_projection
    )
    num_obs = temporal_projection.shape[0] * spatial_projection.shape[0]
    prior_trace = prior_point_variance * num_obs
    projected_trace = q_time_diag.sum() * q_space_diag.sum()
    residual = prior_trace - projected_trace

    tolerance = 1e-7 * max(float(prior_trace.detach().abs()), 1.0)
    if float(residual.detach()) < -tolerance:
        raise RuntimeError(
            "Invalid VFE covariance: projected trace exceeds the full GP trace "
            f"by {-float(residual.detach()):.6g}"
        )
    return torch.clamp(residual, min=0.0)


def _solve_du(
    rhs: torch.Tensor,
    *,
    chol_s: torch.Tensor,
    chol_t: torch.Tensor,
    eigvec_s: torch.Tensor,
    eigvec_t: torch.Tensor,
    denominator: torch.Tensor,
    combined_s: torch.Tensor | None = None,
    combined_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the inverse structured u precision to [M_s,M_t] or batched RHS."""

    if (combined_s is None) != (combined_t is None):
        raise ValueError("Combined spatial and temporal bases must be supplied together")
    if combined_s is None:
        transformed = torch.matmul(chol_s.transpose(0, 1), rhs)
        transformed = torch.matmul(transformed, chol_t)
        transformed = torch.matmul(eigvec_s.transpose(0, 1), transformed)
        transformed = torch.matmul(transformed, eigvec_t)
    else:
        transformed = torch.matmul(combined_s.transpose(0, 1), rhs)
        transformed = torch.matmul(transformed, combined_t)
    transformed = transformed / denominator
    if combined_s is None:
        solved = torch.matmul(eigvec_s, transformed)
        solved = torch.matmul(solved, eigvec_t.transpose(0, 1))
        solved = torch.matmul(chol_s, solved)
        return torch.matmul(solved, chol_t.transpose(0, 1))
    solved = torch.matmul(combined_s, transformed)
    return torch.matmul(solved, combined_t.transpose(0, 1))


def _solve_du_thin_temporal_woodbury(
    rhs: torch.Tensor,
    *,
    chol_s: torch.Tensor,
    chol_t: torch.Tensor,
    eigval_s: torch.Tensor,
    eigvec_s: torch.Tensor,
    whitened_temporal_design: torch.Tensor,
    temporal_system_cholesky: torch.Tensor,
    combined_s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the precision inverse in observation space via Woodbury.

    When a new block has fewer observations than temporal latent dimensions,
    differentiating a thin SVD is unstable at repeated or near-repeated singular
    values. For each spatial eigenvalue ``a``, this uses
    ``(I + a W'W)^-1 = I - a W'(I + a WW')^-1 W``. The observation-space
    systems are positive definite and use Cholesky solves without temporal
    eigenvectors.
    """

    if combined_s is None:
        transformed = torch.matmul(chol_s.transpose(0, 1), rhs)
        transformed = torch.matmul(transformed, chol_t)
        transformed = torch.matmul(eigvec_s.transpose(0, 1), transformed)
    else:
        transformed = torch.matmul(combined_s.transpose(0, 1), rhs)
        transformed = torch.matmul(transformed, chol_t)
    leading_shape = transformed.shape[:-2]
    num_space = transformed.shape[-2]
    num_temporal = transformed.shape[-1]
    flat = transformed.reshape(-1, num_space, num_temporal)
    projected = torch.einsum(
        "bsm,nm->bsn", flat, whitened_temporal_design
    )
    solve_rhs = projected.permute(1, 0, 2).unsqueeze(-1)
    solved = torch.cholesky_solve(
        solve_rhs, temporal_system_cholesky.unsqueeze(1)
    ).squeeze(-1).permute(1, 0, 2)
    correction = torch.einsum(
        "bsn,nm->bsm", solved, whitened_temporal_design
    )
    flat = flat - correction * eigval_s[None, :, None]
    transformed = flat.reshape(*leading_shape, num_space, num_temporal)
    solved = torch.matmul(
        eigvec_s if combined_s is None else combined_s,
        transformed,
    )
    if combined_s is None:
        solved = torch.matmul(chol_s, solved)
    return torch.matmul(solved, chol_t.transpose(0, 1))


def finite_joint_nlml_from_factors(
    *,
    y_matrix: torch.Tensor,
    phi_tensor: torch.Tensor,
    temporal_projection: torch.Tensor,
    spatial_projection: torch.Tensor,
    temporal_prior: torch.Tensor,
    spatial_prior: torch.Tensor,
    noise_variance: torch.Tensor,
    beta_prior_variance: float,
    sufficient_statistics: JointSufficientStatistics | None = None,
    combine_basis_transforms: bool = False,
    remove_redundant_solve: bool = False,
    cross_contraction: str = "einsum",
    feature_block_size: int | None = None,
) -> JointObjectiveDiagnostics:
    """Exact finite-model NLML after jointly integrating beta and u.

    ``y_matrix`` and ``phi_tensor`` use [space,time] and [space,time,feature]
    layouts. The likelihood design is ``A_u = T kron C`` under Fortran
    vectorisation. The u block is handled by a Kronecker eigensystem and beta
    by a Schur complement, so no observation-sized covariance is materialised.
    """

    y = y_matrix
    phi = phi_tensor
    t_mat = temporal_projection
    c_mat = spatial_projection
    sigma2 = noise_variance
    num_space, num_time = y.shape
    num_features = phi.shape[-1]
    num_obs = y.numel()

    if sufficient_statistics is None:
        statistics = joint_sufficient_statistics(y, phi)
    else:
        statistics = sufficient_statistics
        if statistics.num_observations != num_obs or statistics.num_features != num_features:
            raise ValueError("Cached sufficient-statistic dimensions do not match y/Phi")
        tensors = (statistics.phi_phi, statistics.phi_y, statistics.y_y)
        if any(tensor.device != y.device or tensor.dtype != y.dtype for tensor in tensors):
            raise ValueError("Cached sufficient statistics must match y device and dtype")
    phi_phi = statistics.phi_phi
    phi_y = statistics.phi_y
    y_y = statistics.y_y

    g_space = c_mat.transpose(0, 1) @ c_mat
    chol_s = robust_cholesky(spatial_prior)
    chol_t = robust_cholesky(temporal_prior)
    whitened_s = _symmetrize(chol_s.transpose(0, 1) @ g_space @ chol_s)
    eig_s, vec_s = torch.linalg.eigh(whitened_s)
    eig_s = torch.clamp(eig_s, min=0.0)
    if num_time < temporal_prior.shape[0]:
        whitened_temporal_design = (t_mat @ chol_t) / torch.sqrt(sigma2)
        observation_gram = _symmetrize(
            whitened_temporal_design @ whitened_temporal_design.transpose(0, 1)
        )
        observation_eye = torch.eye(
            num_time, dtype=y.dtype, device=y.device
        )
        temporal_systems = (
            observation_eye.unsqueeze(0)
            + eig_s[:, None, None] * observation_gram.unsqueeze(0)
        )
        temporal_system_cholesky = torch.linalg.cholesky(
            _symmetrize(temporal_systems)
        )
        logdet_latent = 2.0 * torch.log(
            torch.diagonal(temporal_system_cholesky, dim1=-2, dim2=-1)
        ).sum()
        combined_s = chol_s @ vec_s if combine_basis_transforms else None

        def solve_du(rhs: torch.Tensor) -> torch.Tensor:
            return _solve_du_thin_temporal_woodbury(
                rhs,
                chol_s=chol_s,
                chol_t=chol_t,
                eigval_s=eig_s,
                eigvec_s=vec_s,
                whitened_temporal_design=whitened_temporal_design,
                temporal_system_cholesky=temporal_system_cholesky,
                combined_s=combined_s,
            )

    else:
        b_time = (t_mat.transpose(0, 1) @ t_mat) / sigma2
        whitened_t = _symmetrize(chol_t.transpose(0, 1) @ b_time @ chol_t)
        eig_t, vec_t = torch.linalg.eigh(whitened_t)
        eig_t = torch.clamp(eig_t, min=0.0)
        denominator = 1.0 + eig_s[:, None] * eig_t[None, :]
        logdet_latent = torch.log(denominator).sum()
        combined_s = chol_s @ vec_s if combine_basis_transforms else None
        combined_t = chol_t @ vec_t if combine_basis_transforms else None

        def solve_du(rhs: torch.Tensor) -> torch.Tensor:
            return _solve_du(
                rhs,
                chol_s=chol_s,
                chol_t=chol_t,
                eigvec_s=vec_s,
                eigvec_t=vec_t,
                denominator=denominator,
                combined_s=combined_s,
                combined_t=combined_t,
            )

    h_u = (c_mat.transpose(0, 1) @ y @ t_mat) / sigma2
    d_inv_h = solve_du(h_u)

    if num_features == 0:
        u_mean = d_inv_h
        h_dot_mean = torch.sum(h_u * u_mean)
        quadratic = y_y / sigma2 - h_dot_mean
        logdet = num_obs * torch.log(sigma2) + logdet_latent
        nlml = 0.5 * (quadratic + logdet + num_obs * math.log(2.0 * math.pi))
        return JointObjectiveDiagnostics(
            nlml_per_observation=nlml / num_obs,
            finite_nlml_per_observation=nlml / num_obs,
            vfe_trace_correction_per_observation=torch.zeros_like(nlml),
            vfe_trace_residual_per_observation=torch.zeros_like(nlml),
            logdet_per_observation=logdet / num_obs,
            quadratic_per_observation=quadratic / num_obs,
            beta_mean=phi.new_zeros((0,)),
            u_mean=u_mean,
            beta_precision=phi.new_zeros((0, 0)),
        )

    cross = feature_gp_cross(
        c_mat,
        phi,
        t_mat,
        order=cross_contraction,
        feature_block_size=feature_block_size,
    ) / sigma2
    d_inv_cross = solve_du(cross)

    beta_precision = (
        torch.eye(num_features, dtype=y.dtype, device=y.device) / beta_prior_variance
        + phi_phi / sigma2
    )
    schur = beta_precision - torch.einsum("pab,qab->pq", cross, d_inv_cross)
    beta_rhs = phi_y / sigma2 - torch.einsum("pab,ab->p", cross, d_inv_h)
    chol_beta = robust_cholesky(schur, base_jitter=1e-9)
    beta_mean = torch.cholesky_solve(beta_rhs[:, None], chol_beta).squeeze(1)
    if remove_redundant_solve:
        u_mean = d_inv_h - torch.einsum("p,pab->ab", beta_mean, d_inv_cross)
    else:
        u_rhs = h_u - torch.einsum("p,pab->ab", beta_mean, cross)
        u_mean = solve_du(u_rhs)

    h_dot_mean = torch.dot(phi_y / sigma2, beta_mean) + torch.sum(h_u * u_mean)
    quadratic = y_y / sigma2 - h_dot_mean
    logdet_beta_schur = 2.0 * torch.log(torch.diagonal(chol_beta)).sum()
    logdet = (
        num_obs * torch.log(sigma2)
        + logdet_latent
        + num_features * math.log(beta_prior_variance)
        + logdet_beta_schur
    )
    nlml = 0.5 * (quadratic + logdet + num_obs * math.log(2.0 * math.pi))
    return JointObjectiveDiagnostics(
        nlml_per_observation=nlml / num_obs,
        finite_nlml_per_observation=nlml / num_obs,
        vfe_trace_correction_per_observation=torch.zeros_like(nlml),
        vfe_trace_residual_per_observation=torch.zeros_like(nlml),
        logdet_per_observation=logdet / num_obs,
        quadratic_per_observation=quadratic / num_obs,
        beta_mean=beta_mean,
        u_mean=u_mean,
        beta_precision=schur,
    )


def vfe_corrected_joint_nlml_from_factors(
    *,
    y_matrix: torch.Tensor,
    phi_tensor: torch.Tensor,
    temporal_projection: torch.Tensor,
    spatial_projection: torch.Tensor,
    temporal_prior: torch.Tensor,
    spatial_prior: torch.Tensor,
    noise_variance: torch.Tensor,
    beta_prior_variance: float,
    prior_point_variance: torch.Tensor,
    sufficient_statistics: JointSufficientStatistics | None = None,
    combine_basis_transforms: bool = False,
    remove_redundant_solve: bool = False,
    cross_contraction: str = "einsum",
    feature_block_size: int | None = None,
) -> JointObjectiveDiagnostics:
    """Negative collapsed Gaussian VFE bound for structured-joint Route B."""

    finite = finite_joint_nlml_from_factors(
        y_matrix=y_matrix,
        phi_tensor=phi_tensor,
        temporal_projection=temporal_projection,
        spatial_projection=spatial_projection,
        temporal_prior=temporal_prior,
        spatial_prior=spatial_prior,
        noise_variance=noise_variance,
        beta_prior_variance=beta_prior_variance,
        sufficient_statistics=sufficient_statistics,
        combine_basis_transforms=combine_basis_transforms,
        remove_redundant_solve=remove_redundant_solve,
        cross_contraction=cross_contraction,
        feature_block_size=feature_block_size,
    )
    trace_residual = separable_vfe_trace_residual(
        temporal_projection=temporal_projection,
        spatial_projection=spatial_projection,
        temporal_prior=temporal_prior,
        spatial_prior=spatial_prior,
        prior_point_variance=prior_point_variance,
    )
    num_obs = y_matrix.numel()
    correction = trace_residual / (2.0 * noise_variance * num_obs)
    return JointObjectiveDiagnostics(
        nlml_per_observation=finite.nlml_per_observation + correction,
        finite_nlml_per_observation=finite.nlml_per_observation,
        vfe_trace_correction_per_observation=correction,
        vfe_trace_residual_per_observation=trace_residual / num_obs,
        logdet_per_observation=finite.logdet_per_observation,
        quadratic_per_observation=finite.quadratic_per_observation,
        beta_mean=finite.beta_mean,
        u_mean=finite.u_mean,
        beta_precision=finite.beta_precision,
    )


class BatchRouteBEmpiricalBayes(nn.Module):
    """Five-parameter batch empirical-Bayes Route B model with fixed supports."""

    def __init__(
        self,
        *,
        times: np.ndarray,
        spatial_inducing: np.ndarray,
        mt: int,
        representation: str,
        initial_ell_t: float,
        initial_ell_s: tuple[float, float],
        initial_kernel_variance: float,
        initial_noise_std: float,
        rff_sample_size: int,
        seed: int,
        objective_type: str = "finite_dtc",
        temporal_horizon: TemporalBlockSpec | None = None,
    ) -> None:
        super().__init__()
        if objective_type not in {"finite_dtc", "vfe"}:
            raise ValueError(f"Unsupported objective type: {objective_type}")
        self.objective_type = objective_type
        self.temporal = TemporalFactorModel(
            times=times,
            mt=mt,
            representation=representation,
            initial_lengthscale=initial_ell_t,
            initial_variance=initial_kernel_variance,
            rff_sample_size=rff_sample_size,
            seed=seed,
            temporal_horizon=temporal_horizon,
        )
        self.register_buffer(
            "spatial_inducing", torch.as_tensor(spatial_inducing, dtype=DTYPE)
        )
        self.log_spatial_lengthscales = nn.Parameter(
            torch.log(torch.as_tensor(initial_ell_s, dtype=DTYPE))
        )
        self.log_noise_std = nn.Parameter(
            torch.log(torch.as_tensor(initial_noise_std, dtype=DTYPE))
        )

    @property
    def spatial_lengthscales(self) -> torch.Tensor:
        return torch.exp(self.log_spatial_lengthscales)

    @property
    def noise_std(self) -> torch.Tensor:
        return torch.exp(self.log_noise_std)

    def set_temporal_query(
        self,
        times: np.ndarray,
        *,
        temporal_horizon: TemporalBlockSpec | None = None,
    ) -> None:
        self.temporal.set_query_times(
            times, temporal_horizon=temporal_horizon
        )

    def set_theta(self, theta: dict[str, float | list[float]]) -> None:
        """Load the five positive hyperparameters without changing supports."""

        ell_s = torch.as_tensor(
            theta["ell_s"],
            dtype=self.log_spatial_lengthscales.dtype,
            device=self.log_spatial_lengthscales.device,
        )
        if ell_s.shape != self.log_spatial_lengthscales.shape:
            raise ValueError(
                "Spatial lengthscale shape mismatch: "
                f"expected {tuple(self.log_spatial_lengthscales.shape)}, "
                f"got {tuple(ell_s.shape)}"
            )
        values = {
            "ell_t": torch.as_tensor(
                theta["ell_t"],
                dtype=self.temporal.lengthscale.dtype,
                device=self.temporal.lengthscale.device,
            ),
            "kernel_variance": torch.as_tensor(
                theta["kernel_variance"],
                dtype=self.temporal.variance.dtype,
                device=self.temporal.variance.device,
            ),
            "noise_std": torch.as_tensor(
                theta["noise_std"],
                dtype=self.log_noise_std.dtype,
                device=self.log_noise_std.device,
            ),
        }
        if any(float(value) <= 0.0 for value in (*values.values(), *ell_s)):
            raise ValueError("All Route-B hyperparameters must be positive")
        with torch.no_grad():
            if self.temporal.builder is not None:
                self.temporal.builder.log_lengthscale.copy_(torch.log(values["ell_t"]))
                self.temporal.builder.log_variance.copy_(
                    torch.log(values["kernel_variance"])
                )
            else:
                self.temporal.log_lengthscale.copy_(torch.log(values["ell_t"]))
                self.temporal.log_variance.copy_(
                    torch.log(values["kernel_variance"])
                )
            self.log_spatial_lengthscales.copy_(torch.log(ell_s))
            self.log_noise_std.copy_(torch.log(values["noise_std"]))

    def factor_matrices(
        self, spatial_coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_mat, kt = self.temporal.factors()
        ks = matern32_separable(
            self.spatial_inducing,
            self.spatial_inducing,
            self.spatial_lengthscales,
        )
        kxs = matern32_separable(
            spatial_coordinates,
            self.spatial_inducing,
            self.spatial_lengthscales,
        )
        chol_s = robust_cholesky(ks)
        c_mat = torch.cholesky_solve(kxs.transpose(0, 1), chol_s).transpose(0, 1)
        ks = _symmetrize(ks) + 1e-7 * torch.eye(
            ks.shape[0], dtype=ks.dtype, device=ks.device
        )
        return t_mat, c_mat, kt, ks

    def objective(
        self,
        *,
        y_matrix: torch.Tensor,
        phi_tensor: torch.Tensor,
        spatial_coordinates: torch.Tensor,
        beta_prior_variance: float,
        sufficient_statistics: JointSufficientStatistics | None = None,
        combine_basis_transforms: bool = False,
        remove_redundant_solve: bool = False,
        cross_contraction: str = "einsum",
        feature_block_size: int | None = None,
    ) -> JointObjectiveDiagnostics:
        t_mat, c_mat, kt, ks = self.factor_matrices(spatial_coordinates)
        kwargs = {
            "y_matrix": y_matrix,
            "phi_tensor": phi_tensor,
            "temporal_projection": t_mat,
            "spatial_projection": c_mat,
            "temporal_prior": kt,
            "spatial_prior": ks,
            "noise_variance": self.noise_std.square(),
            "beta_prior_variance": beta_prior_variance,
            "sufficient_statistics": sufficient_statistics,
            "combine_basis_transforms": combine_basis_transforms,
            "remove_redundant_solve": remove_redundant_solve,
            "cross_contraction": cross_contraction,
            "feature_block_size": feature_block_size,
        }
        if self.objective_type == "vfe":
            return vfe_corrected_joint_nlml_from_factors(
                **kwargs,
                prior_point_variance=self.temporal.variance,
            )
        return finite_joint_nlml_from_factors(**kwargs)

    def clamp_parameters(self) -> None:
        self.temporal.clamp_parameters()
        with torch.no_grad():
            self.log_spatial_lengthscales.clamp_(math.log(0.02), math.log(5.0))
            self.log_noise_std.clamp_(math.log(0.01), math.log(1.0))

    def theta(self) -> dict[str, float | list[float]]:
        return {
            "ell_t": float(self.temporal.lengthscale.detach()),
            "ell_s": [float(x) for x in self.spatial_lengthscales.detach()],
            "kernel_variance": float(self.temporal.variance.detach()),
            "noise_std": float(self.noise_std.detach()),
        }
