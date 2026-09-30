"""Spatial kernels used by the Task-1-only PEMS-BAY improvement audit."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from stvgp_kronecker.joint_ssgp_kron.synthetic import select_spatial_inducing_indices
from stvgp_kronecker.routeb_empirical_bayes import (
    BatchRouteBEmpiricalBayes,
    DTYPE,
    _symmetrize,
    matern32_separable,
    robust_cholesky,
)


SPATIAL_KERNELS = ("geo_matern32", "road_graph", "spectral_mixture")


def inducing_sensor_indices(dataset, visible_indices: np.ndarray, ms: int) -> np.ndarray:
    visible = np.asarray(visible_indices, dtype=int)
    local = select_spatial_inducing_indices(
        dataset.coordinates_standardised[visible], min(int(ms), visible.size), method="farthest"
    )
    return visible[local]


def load_road_laplacian(
    path: str | Path,
    sensor_ids: tuple[str, ...],
    *,
    normalized_k: float = 0.1,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Build the symmetric normalized Laplacian from official DCRNN distances."""

    position = {str(sensor): index for index, sensor in enumerate(sensor_ids)}
    edges: list[tuple[int, int, float]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for source, target, distance, *_ in csv.reader(handle):
            if source in position and target in position:
                value = float(distance)
                if np.isfinite(value) and value > 0.0:
                    edges.append((position[source], position[target], value))
    if not edges:
        raise ValueError(f"No usable road-distance edges found in {path}")
    distances = np.asarray([edge[2] for edge in edges], dtype=np.float64)
    sigma = float(np.std(distances))
    adjacency = np.zeros((len(sensor_ids), len(sensor_ids)), dtype=np.float64)
    for source, target, distance in edges:
        weight = math.exp(-((distance / max(sigma, 1e-12)) ** 2))
        if weight >= normalized_k:
            adjacency[source, target] = max(adjacency[source, target], weight)
    adjacency = np.maximum(adjacency, adjacency.T)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0.0] = degree[degree > 0.0] ** -0.5
    laplacian = np.eye(len(sensor_ids)) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    isolated = degree == 0.0
    laplacian[isolated, isolated] = 0.0
    metadata = {
        "road_edges_in_file": len(edges),
        "road_edges_after_threshold_undirected": int(np.count_nonzero(np.triu(adjacency, 1))),
        "road_distance_std_m": sigma,
        "road_normalized_k": normalized_k,
        "isolated_sensors": int(isolated.sum()),
    }
    return laplacian, metadata


def _normalise_covariance(kxz: torch.Tensor, kzz: torch.Tensor, query_diag: torch.Tensor, inducing_diag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale_x = torch.sqrt(torch.clamp(query_diag, min=1e-12))
    scale_z = torch.sqrt(torch.clamp(inducing_diag, min=1e-12))
    return kxz / (scale_x[:, None] * scale_z[None, :]), kzz / (scale_z[:, None] * scale_z[None, :])


def spectral_mixture_covariance(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    logits: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    delta = x1[:, None, :] - x2[None, :, :]
    weights = torch.softmax(logits, dim=0)
    exponent = torch.exp(-2.0 * math.pi**2 * torch.sum(delta[:, :, None, :].square() * scales[None, None, :, :].square(), dim=-1))
    phase = torch.cos(2.0 * math.pi * torch.sum(delta[:, :, None, :] * means[None, None, :, :], dim=-1))
    return torch.sum(weights[None, None, :] * exponent * phase, dim=-1)


class TrafficSpatialEmpiricalBayes(BatchRouteBEmpiricalBayes):
    """Route-B EB with interchangeable normalized spatial covariance families."""

    def __init__(
        self,
        *,
        dataset,
        inducing_indices: np.ndarray,
        spatial_kernel: str,
        road_distance_csv: str | Path | None = None,
        graph_diffusion: float = 1.0,
        spatial_mixtures: int = 2,
        **kwargs,
    ) -> None:
        if spatial_kernel not in SPATIAL_KERNELS:
            raise ValueError(f"Unsupported spatial kernel: {spatial_kernel}")
        self.dataset_ref = dataset
        self.spatial_kernel_name = spatial_kernel
        self.inducing_indices_np = np.asarray(inducing_indices, dtype=int)
        super().__init__(
            spatial_inducing=dataset.coordinates_standardised[self.inducing_indices_np],
            **kwargs,
        )
        self.register_buffer("all_coordinates", torch.as_tensor(dataset.coordinates_standardised, dtype=DTYPE))
        self.register_buffer("inducing_indices_tensor", torch.as_tensor(self.inducing_indices_np, dtype=torch.long))
        self.road_metadata: dict[str, float | int] | None = None
        if spatial_kernel == "road_graph":
            if road_distance_csv is None:
                raise ValueError("road_graph requires road_distance_csv")
            laplacian, self.road_metadata = load_road_laplacian(road_distance_csv, dataset.sensor_ids)
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            self.register_buffer("graph_eigenvalues", torch.as_tensor(np.maximum(eigenvalues, 0.0), dtype=DTYPE))
            self.register_buffer("graph_eigenvectors", torch.as_tensor(eigenvectors, dtype=DTYPE))
            self.log_graph_diffusion = nn.Parameter(torch.log(torch.as_tensor(graph_diffusion, dtype=DTYPE)))
            self.log_spatial_lengthscales.requires_grad_(False)
        elif spatial_kernel == "spectral_mixture":
            q = int(spatial_mixtures)
            if q < 1:
                raise ValueError("spatial_mixtures must be positive")
            self.spatial_sm_logits = nn.Parameter(torch.zeros(q, dtype=DTYPE))
            means = torch.zeros((q, 2), dtype=DTYPE)
            if q > 1:
                means[1:, :] = torch.linspace(0.25, 1.0, q - 1, dtype=DTYPE)[:, None]
            self.spatial_sm_means = nn.Parameter(means)
            self.log_spatial_sm_scales = nn.Parameter(torch.zeros((q, 2), dtype=DTYPE))
            self.log_spatial_lengthscales.requires_grad_(False)
        else:
            self.register_buffer("graph_eigenvalues", torch.empty(0, dtype=DTYPE))
            self.register_buffer("graph_eigenvectors", torch.empty(0, dtype=DTYPE))

    def _coordinate_indices(self, coordinates: torch.Tensor) -> torch.Tensor:
        distances = torch.sum((coordinates[:, None, :] - self.all_coordinates[None, :, :]) ** 2, dim=-1)
        indices = torch.argmin(distances, dim=1)
        if float(torch.max(torch.gather(distances, 1, indices[:, None]))) > 1e-16:
            raise ValueError("Spatial coordinates do not map to the canonical traffic sensor order")
        return indices

    def _kernel_blocks(self, query_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.inducing_indices_tensor
        if self.spatial_kernel_name == "geo_matern32":
            query = self.all_coordinates[query_indices]
            inducing = self.all_coordinates[z]
            return (
                matern32_separable(query, inducing, self.spatial_lengthscales),
                matern32_separable(inducing, inducing, self.spatial_lengthscales),
            )
        if self.spatial_kernel_name == "road_graph":
            spectrum = torch.exp(-torch.exp(self.log_graph_diffusion) * self.graph_eigenvalues)
            uq = self.graph_eigenvectors[query_indices]
            uz = self.graph_eigenvectors[z]
            kxz = (uq * spectrum[None, :]) @ uz.T
            kzz = (uz * spectrum[None, :]) @ uz.T
            full_diag = torch.sum(self.graph_eigenvectors.square() * spectrum[None, :], dim=1)
            return _normalise_covariance(kxz, kzz, full_diag[query_indices], full_diag[z])
        scales = torch.exp(self.log_spatial_sm_scales)
        query = self.all_coordinates[query_indices]
        inducing = self.all_coordinates[z]
        return (
            spectral_mixture_covariance(query, inducing, logits=self.spatial_sm_logits, means=self.spatial_sm_means, scales=scales),
            spectral_mixture_covariance(inducing, inducing, logits=self.spatial_sm_logits, means=self.spatial_sm_means, scales=scales),
        )

    def factor_matrices(self, spatial_coordinates: torch.Tensor):
        t_mat, kt = self.temporal.factors()
        query_indices = self._coordinate_indices(spatial_coordinates)
        kxs, ks = self._kernel_blocks(query_indices)
        chol_s = robust_cholesky(ks)
        c_mat = torch.cholesky_solve(kxs.transpose(0, 1), chol_s).transpose(0, 1)
        ks = _symmetrize(ks) + 1e-7 * torch.eye(ks.shape[0], dtype=ks.dtype, device=ks.device)
        return t_mat, c_mat, kt, ks

    def clamp_parameters(self) -> None:
        super().clamp_parameters()
        with torch.no_grad():
            if self.spatial_kernel_name == "road_graph":
                self.log_graph_diffusion.clamp_(math.log(0.05), math.log(20.0))
            elif self.spatial_kernel_name == "spectral_mixture":
                self.spatial_sm_means.clamp_(-4.0, 4.0)
                self.log_spatial_sm_scales.clamp_(math.log(0.05), math.log(10.0))

    def theta(self) -> dict[str, float | list[float]]:
        values = super().theta()
        values["spatial_kernel"] = self.spatial_kernel_name
        if self.spatial_kernel_name == "road_graph":
            values["graph_diffusion"] = float(torch.exp(self.log_graph_diffusion).detach())
        elif self.spatial_kernel_name == "spectral_mixture":
            values["spatial_sm_weights"] = torch.softmax(self.spatial_sm_logits.detach(), dim=0).cpu().tolist()
            values["spatial_sm_means"] = self.spatial_sm_means.detach().cpu().tolist()
            values["spatial_sm_scales"] = torch.exp(self.log_spatial_sm_scales.detach()).cpu().tolist()
        return values


def fixed_spatial_factors(
    dataset,
    visible_indices: np.ndarray,
    *,
    ms: int,
    theta: dict[str, object],
    road_distance_csv: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct fixed online spatial factors from a locked Task-1 theta record."""

    inducing_indices = inducing_sensor_indices(dataset, visible_indices, ms)
    model = TrafficSpatialEmpiricalBayes(
        dataset=dataset,
        inducing_indices=inducing_indices,
        spatial_kernel=str(theta.get("spatial_kernel", "geo_matern32")),
        road_distance_csv=road_distance_csv,
        graph_diffusion=float(theta.get("graph_diffusion", 1.0)),
        spatial_mixtures=len(theta.get("spatial_sm_weights", [0.5, 0.5])),
        times=np.asarray([0.0, 1.0]),
        mt=2,
        representation="inducing_points",
        initial_ell_t=1.0,
        initial_ell_s=tuple(theta["ell_s"]),
        initial_kernel_variance=float(theta["kernel_variance"]),
        initial_noise_std=float(theta["noise_std"]),
        rff_sample_size=4,
        seed=0,
    )
    model.set_theta(theta)
    with torch.no_grad():
        if model.spatial_kernel_name == "road_graph":
            model.log_graph_diffusion.copy_(torch.log(torch.as_tensor(theta["graph_diffusion"], dtype=DTYPE)))
        elif model.spatial_kernel_name == "spectral_mixture":
            weights = torch.as_tensor(theta["spatial_sm_weights"], dtype=DTYPE)
            model.spatial_sm_logits.copy_(torch.log(torch.clamp(weights, min=1e-12)))
            model.spatial_sm_means.copy_(torch.as_tensor(theta["spatial_sm_means"], dtype=DTYPE))
            model.log_spatial_sm_scales.copy_(torch.log(torch.as_tensor(theta["spatial_sm_scales"], dtype=DTYPE)))
        kxs, ks = model._kernel_blocks(torch.arange(dataset.num_sensors, dtype=torch.long))
        chol = robust_cholesky(ks)
        c_all = torch.cholesky_solve(kxs.T, chol).T
        ks = _symmetrize(ks) + 1e-7 * torch.eye(ks.shape[0], dtype=DTYPE)
    return np.asarray(ks), np.asarray(c_all), dataset.coordinates_standardised[inducing_indices], inducing_indices
