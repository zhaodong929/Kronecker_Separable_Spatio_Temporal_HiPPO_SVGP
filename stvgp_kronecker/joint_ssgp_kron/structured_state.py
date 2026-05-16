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
