"""Matched interventions. Never select a different fit for a weaker arm."""
from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass(frozen=True)
class Arm:
    name: str
    zero_cross: bool = False
    identity_transfer: bool = False

    def __post_init__(self):
        allowed = {'joint_transfer': (False, False), 'zero_cross': (True, False),
                   'identity_transfer': (False, True)}
        if allowed.get(self.name) != (self.zero_cross, self.identity_transfer):
            raise ValueError('An arm must change exactly its declared intervention')


ARMS = (Arm('joint_transfer'), Arm('zero_cross', zero_cross=True),
        Arm('identity_transfer', identity_transfer=True))


def matched_manifest(*, protocol_sha256, fit_sha256, source_commit, model_config):
    identity = dict(protocol_sha256=protocol_sha256, fit_sha256=fit_sha256,
                    source_commit=source_commit, model_config=model_config)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(shared=identity, comparison_sha256=fingerprint,
        arms=[asdict(arm) for arm in ARMS], independent_tuning_per_arm=False,
        interpretation={
            'zero_cross': 'Remove accumulated historical AND current trend-residual likelihood coupling; not an isolated historical-revision test.',
            'identity_transfer': 'Reuse historical coordinates without transport at the SAME changing basis; not a fixed-global-basis experiment.',
            'solver': 'Separate matched-system dense/structured timing and residual qualification; identical precision and RHS, no predictive configuration change.'})


def compare_linear_solvers(matrix, rhs, structured_solve, *, synchronize=lambda: None, max_dimension=2048):
    """Bounded diagnostic of the SAME Gaussian system; never densify full runs."""
    import time
    import torch
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] > max_dimension:
        raise ValueError('Dense solver comparison exceeds its explicit dimension bound')
    if not torch.isfinite(matrix).all() or not torch.isfinite(rhs).all():
        raise ValueError('Finite matched system required')
    torch.linalg.cholesky(matrix)  # require the advertised SPD Gaussian target
    synchronize(); start = time.perf_counter()
    reference = torch.linalg.solve(matrix, rhs)
    synchronize(); dense_seconds = time.perf_counter() - start
    synchronize(); start = time.perf_counter()
    actual = structured_solve(rhs)
    synchronize(); structured_seconds = time.perf_counter() - start
    norm = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.dtype).tiny)
    error = torch.linalg.vector_norm(actual-reference) / torch.linalg.vector_norm(reference).clamp_min(torch.finfo(rhs.dtype).tiny)
    return dict(dimension=matrix.shape[0], dense_seconds=dense_seconds, structured_seconds=structured_seconds,
        relative_solution_error=float(error),
        structured_relative_residual=float(torch.linalg.vector_norm(matrix@actual-rhs)/norm),
        dense_relative_residual=float(torch.linalg.vector_norm(matrix@reference-rhs)/norm),
        status='diagnostic', timing_scope='solve_only; construction and setup excluded',
        first_use_may_include_compilation=True)
