from __future__ import annotations

import numpy as np

from stvgp_kronecker.joint_ssgp_kron.kron_utils import dense_A_from_factors, vec_f
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    iter_time_blocks,
    make_block_factors,
    make_spatial_projection,
    make_synthetic_dataset,
    temporal_inducing_for_block,
)


def _small_model(seed: int = 0):
    dataset = make_synthetic_dataset(num_time=12, num_space=4, noise=0.08, seed=seed)
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, ms=3)
    model = JointSSGPKronHiPPOSVGP(
        Ks=Ks,
        C=C,
        sigma2=dataset.sigma2,
        beta_prior_mean=np.zeros(dataset.Phi.shape[1]),
        beta_prior_cov=10.0 * np.eye(dataset.Phi.shape[1]),
    )
    return dataset, model, C


def test_model_one_block_no_nan() -> None:
    dataset, model, C = _small_model()
    block = slice(0, 4)
    z_t = temporal_inducing_for_block(dataset.times, block, mt=3, moving=True)
    factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=None)
    state = model.update_block_ssgp_transfer(
        y_vec=factors.y_vec,
        Phi=factors.Phi,
        T_n=factors.T,
        Kt_new=factors.Kt,
        inner_iters=2,
    )
    mean = factors.Phi @ state.beta_mean + dense_A_from_factors(factors.T, C) @ vec_f(state.M_u)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(state.M_u))
    assert np.all(np.isfinite(state.B_temporal))
    assert np.all(np.isfinite(state.H_info))


def test_model_multi_block_no_nan() -> None:
    dataset, model, C = _small_model(seed=1)
    state = None
    old_z = None
    for block in iter_time_blocks(dataset.Y.shape[1], 4):
        z_t = temporal_inducing_for_block(dataset.times, block, mt=3, moving=True)
        factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=old_z)
        state = model.update_block_ssgp_transfer(
            y_vec=factors.y_vec,
            Phi=factors.Phi,
            T_n=factors.T,
            Kt_new=factors.Kt,
            K_on_t=factors.K_on_t,
            state=state,
            inner_iters=2,
        )
        mean = factors.Phi @ state.beta_mean + dense_A_from_factors(factors.T, C) @ vec_f(state.M_u)
        assert np.all(np.isfinite(mean))
        old_z = z_t


def test_baseline_imports_still_work() -> None:
    import stvgp_kronecker.st_model_batch as st_model_batch
    import stvgp_kronecker.st_model_online as st_model_online
    import stvgp_kronecker.train_batch as train_batch
    import stvgp_kronecker.train_online as train_online
    import stvgp_kronecker.train_online_joint as train_online_joint

    assert hasattr(st_model_batch, "BatchKroneckerSTHiPPOSVGP")
    assert hasattr(st_model_online, "OnlinePosteriorSummarySTGP")
    assert hasattr(train_batch, "main")
    assert hasattr(train_online, "main")
    assert hasattr(train_online_joint, "main")
