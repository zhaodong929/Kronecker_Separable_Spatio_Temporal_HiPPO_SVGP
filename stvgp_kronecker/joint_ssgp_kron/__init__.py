"""Joint SSGP Kronecker HiPPO implementation.

This package is intentionally separate from the baseline training modules.
"""

from .kron_utils import (
    dense_A_from_factors,
    kron_mv,
    kron_t_mv,
    make_spd_matrix,
    relative_fro_error,
    solve_sylvester_precision,
)
from .model import JointSSGPKronHiPPOSVGP
from .structured_state import StructuredKronState

__all__ = [
    "JointSSGPKronHiPPOSVGP",
    "StructuredKronState",
    "dense_A_from_factors",
    "kron_mv",
    "kron_t_mv",
    "make_spd_matrix",
    "relative_fro_error",
    "solve_sylvester_precision",
]
