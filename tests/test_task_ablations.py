import numpy as np
import pytest
import torch
from benchmarks.task_stream.ablations import Arm, matched_manifest, compare_linear_solvers
from stvgp_kronecker.joint_ssgp_kron.multi_geometry import TorchMultiGeometryHiPPOSVGP, SumKronSolver


def test_zero_cross_is_the_independent_modified_gaussian_target():
    model = TorchMultiGeometryHiPPOSVGP(Ks=[[1.]], C=[[1.]], sigma2=1.,
        beta_prior_mean=[0.], beta_prior_cov=[[1.]], jitter=0.)
    state = None
    for c in [1., 2.]:
        state = model.update_block_structured_joint_ssgp_transfer(y_vec=[1.], Phi=[[1.]], T_n=[[1.]],
                 Kt_new=[[1.]], C_observed=[[c]], state=state, zero_cross=True)
    assert state.beta_mean.item() == pytest.approx(2./3.)
    assert state.M_u.item() == pytest.approx(3./6.)
    np.testing.assert_array_equal(state.R_beta_u, [[0.]])


def test_matched_arms_do_not_change_tuning_or_multiple_components():
    result = matched_manifest(protocol_sha256='p', fit_sha256='f', source_commit='s', model_config={'mt': 4})
    assert len(result['arms']) == 3 and not result['independent_tuning_per_arm']
    with pytest.raises(ValueError): Arm('zero_cross', zero_cross=True, identity_transfer=True)


def test_same_system_solver_measurement_records_residuals():
    eye = torch.eye(2, dtype=torch.float64)
    terms = ((eye, eye * 2), (eye * 3, eye * .5))
    solver = SumKronSolver(eye, eye, terms)
    matrix = torch.kron(eye, eye) + sum(torch.kron(b, g) for g, b in terms)
    result = compare_linear_solvers(matrix, torch.arange(1., 5., dtype=torch.float64), solver.solve)
    assert result['relative_solution_error'] < 1e-10
    assert result['structured_relative_residual'] < 1e-10
    with pytest.raises(ValueError, match='bound'):
        compare_linear_solvers(matrix, torch.ones(4), solver.solve, max_dimension=2)
