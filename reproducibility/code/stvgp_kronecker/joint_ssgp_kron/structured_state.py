"""Structured posterior state for the scalable SSGP transfer path."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .kron_utils import inv_spd, symmetrize, vec_f


@dataclass(frozen=True)
class StructuredKronState:
    beta_mean: np.ndarray
    beta_cov: np.ndarray
    M_u: np.ndarray
    B_temporal: np.ndarray
    H_info: np.ndarray
    Kt_current: np.ndarray
    Ks: np.ndarray
    G: np.ndarray
    sigma2: float
    metadata: dict[str, Any] = field(default_factory=dict)
    # Route-B likelihood natural statistics.  The older mean-field/GP-only path
    # leaves these as ``None``; the structured joint path fills them explicitly.
    R_beta_beta: np.ndarray | None = None
    R_beta_u: np.ndarray | None = None
    h_beta: np.ndarray | None = None
    beta_prior_precision: np.ndarray | None = None
    beta_prior_natural: np.ndarray | None = None
    Lambda_beta_given_u: np.ndarray | None = None
    S_beta_beta: np.ndarray | None = None

    def copy_with(self, **kwargs: Any) -> "StructuredKronState":
        return replace(self, **kwargs)

    @property
    def mt(self) -> int:
        return self.Kt_current.shape[0]

    @property
    def ms(self) -> int:
        return self.Ks.shape[0]

    def dense_precision(self, jitter: float = 1e-6) -> np.ndarray:
        Kt_inv = inv_spd(self.Kt_current, jitter=jitter)
        Ks_inv = inv_spd(self.Ks, jitter=jitter)
        return symmetrize(np.kron(Kt_inv, Ks_inv) + np.kron(self.B_temporal, self.G))

    def dense_information(self) -> np.ndarray:
        return vec_f(self.H_info)

    def dense_mean(self) -> np.ndarray:
        return vec_f(self.M_u)

    def routeB_A_beta(self, jitter: float = 1e-6) -> np.ndarray:
        if self.R_beta_beta is None:
            raise ValueError("Route-B beta-beta likelihood statistic is not available")
        if self.beta_prior_precision is None:
            if self.beta_cov.size == 0:
                prior_precision = np.zeros((0, 0))
            else:
                prior_precision = inv_spd(self.beta_cov, jitter=jitter)
        else:
            prior_precision = self.beta_prior_precision
        return symmetrize(prior_precision + self.R_beta_beta)

    def routeB_h_beta_total(self, jitter: float = 1e-6) -> np.ndarray:
        if self.h_beta is None:
            raise ValueError("Route-B beta natural vector is not available")
        if self.beta_prior_natural is None:
            if self.beta_cov.size == 0:
                prior_natural = np.zeros(0)
            else:
                prior_natural = inv_spd(self.beta_cov, jitter=jitter) @ self.beta_mean
        else:
            prior_natural = self.beta_prior_natural
        return prior_natural + self.h_beta

    def routeB_dense_joint_precision(self, jitter: float = 1e-6) -> np.ndarray:
        if self.R_beta_u is None:
            raise ValueError("Route-B beta-u likelihood statistic is not available")
        A_beta = self.routeB_A_beta(jitter=jitter)
        D_u = self.dense_precision(jitter=jitter)
        return symmetrize(np.block([[A_beta, self.R_beta_u], [self.R_beta_u.T, D_u]]))

    def routeB_dense_joint_information(self, jitter: float = 1e-6) -> np.ndarray:
        return np.concatenate([self.routeB_h_beta_total(jitter=jitter), self.dense_information()])
