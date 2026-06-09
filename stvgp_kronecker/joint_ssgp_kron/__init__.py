"""Joint SSGP Kronecker HiPPO implementation.

This package is intentionally separate from the baseline training modules.
"""

from .kron_utils import (
    apply_Lon_to_beta_u_cross_block,
    dense_A_from_factors,
    dense_Du_for_tests,
    dense_Lon_for_tests,
    dense_joint_posterior_reference,
    kron_mv,
    kron_t_mv,
    make_spd_matrix,
    relative_fro_error,
    schur_recover_posterior,
    solve_Du_sylvester,
    solve_sylvester_precision,
)
from .model import JointSSGPKronHiPPOSVGP
from .structured_state import StructuredKronState

__all__ = [
    "JointSSGPKronHiPPOSVGP",
    "StructuredKronState",
    "apply_Lon_to_beta_u_cross_block",
    "dense_A_from_factors",
    "dense_Du_for_tests",
    "dense_Lon_for_tests",
    "dense_joint_posterior_reference",
    "kron_mv",
    "kron_t_mv",
    "make_spd_matrix",
    "relative_fro_error",
    "schur_recover_posterior",
    "solve_Du_sylvester",
    "solve_sylvester_precision",
]
