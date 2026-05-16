#!/usr/bin/env python
"""Verify derivations for the new joint SSGP Kronecker implementation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.joint_ssgp_kron.kron_utils import (
    dense_A_from_factors,
    inv_spd,
    make_spd_matrix,
    relative_fro_error,
    solve_spd,
    solve_sylvester_precision,
    symmetrize,
    vec_f,
)
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.ssgp_transfer import (
    compute_Lt,
    projected_prior_transfer_dense,
    transfer_information_matrix,
    transfer_temporal_precision,
)
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    iter_time_blocks,
    make_block_factors,
    make_spatial_projection,
    make_synthetic_dataset,
    temporal_inducing_for_block,
)


def _pass(value: bool, metric: float | None = None) -> dict[str, object]:
    out: dict[str, object] = {"passed": bool(value)}
    if metric is not None:
        out["metric"] = float(metric)
    return out


def check_l_on_identity(seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mt_old, mt_new, ms = 3, 4, 2
    K_on_t = rng.normal(size=(mt_old, mt_new))
    K_nn_t = make_spd_matrix(mt_new, seed=seed + 1)
    Ks = make_spd_matrix(ms, seed=seed + 2)
    L_t = compute_Lt(K_on_t, K_nn_t, jitter=0.0)
    L_dense = np.kron(K_on_t, Ks) @ inv_spd(np.kron(K_nn_t, Ks), jitter=0.0)
    L_kron = np.kron(L_t, np.eye(ms))
    err = relative_fro_error(L_dense, L_kron)
    return _pass(err < 1e-8, err)


def check_old_likelihood_transfer(seed: int = 1) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mt_old, mt_new, ms, ns = 3, 4, 2, 3
    K_on_t = rng.normal(size=(mt_old, mt_new))
    K_nn_t = make_spd_matrix(mt_new, seed=seed + 1)
    Ks = make_spd_matrix(ms, seed=seed + 2)
    C = rng.normal(size=(ns, ms))
    G = C.T @ C
    B_old = make_spd_matrix(mt_old, seed=seed + 3)
    L_t = compute_Lt(K_on_t, K_nn_t, jitter=0.0)
    L_dense = np.kron(K_on_t, Ks) @ inv_spd(np.kron(K_nn_t, Ks), jitter=0.0)
    Lambda_dense = L_dense.T @ np.kron(B_old, G) @ L_dense
    Lambda_kron = np.kron(transfer_temporal_precision(B_old, L_t), G)
    err = relative_fro_error(Lambda_dense, Lambda_kron)
    return _pass(err < 1e-8, err)


def check_fixed_basis_streaming_equals_batch(seed: int = 2) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    dim, n_blocks, rows = 7, 4, 5
    m0 = rng.normal(size=dim)
    S0 = make_spd_matrix(dim, seed=seed + 1)
    z_true = rng.normal(size=dim)
    sigma2 = 0.04
    H_blocks = [rng.normal(size=(rows, dim)) for _ in range(n_blocks)]
    y_blocks = [H @ z_true + np.sqrt(sigma2) * rng.normal(size=rows) for H in H_blocks]
    Lambda_stream = inv_spd(S0)
    h_stream = Lambda_stream @ m0
    for H, y in zip(H_blocks, y_blocks):
        Lambda_stream += H.T @ H / sigma2
        h_stream += H.T @ y / sigma2
    m_stream = solve_spd(Lambda_stream, h_stream)
    H_all = np.vstack(H_blocks)
    y_all = np.concatenate(y_blocks)
    Lambda_batch = inv_spd(S0) + H_all.T @ H_all / sigma2
    h_batch = inv_spd(S0) @ m0 + H_all.T @ y_all / sigma2
    m_batch = solve_spd(Lambda_batch, h_batch)
    mean_err = np.linalg.norm(m_stream - m_batch) / max(1.0, np.linalg.norm(m_batch))
    prec_err = relative_fro_error(Lambda_stream, Lambda_batch)
    return {"passed": bool(mean_err < 1e-8 and prec_err < 1e-8), "mean_error": float(mean_err), "precision_error": float(prec_err)}


def check_no_linear_mean(seed: int = 3) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    ns, ms, nt, mt = 3, 2, 5, 4
    Ks = make_spd_matrix(ms, seed=seed)
    C = rng.normal(size=(ns, ms))
    Kt = make_spd_matrix(mt, seed=seed + 1)
    T = rng.normal(size=(nt, mt))
    y = rng.normal(size=ns * nt)
    model = JointSSGPKronHiPPOSVGP(Ks=Ks, C=C, sigma2=0.05, beta_prior_mean=np.zeros(2), beta_prior_cov=np.eye(2))
    state = model.update_block_ssgp_transfer(y_vec=y, Phi=np.zeros((ns * nt, 2)), T_n=T, Kt_new=Kt, inner_iters=1)
    Y = y.reshape((ns, nt), order="F")
    B_gp = T.T @ T / model.sigma2
    H_gp = C.T @ Y @ T / model.sigma2
    M_gp = solve_sylvester_precision(inv_spd(Kt), inv_spd(Ks), B_gp, C.T @ C, H_gp)
    return {
        "passed": bool(
            np.allclose(state.beta_mean, np.zeros(2), atol=1e-8)
            and np.allclose(state.B_temporal, B_gp, atol=1e-8)
            and np.allclose(state.H_info, H_gp, atol=1e-8)
            and np.allclose(state.M_u, M_gp, atol=1e-7)
        )
    }


def check_no_old_data(seed: int = 4) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mt_old, mt_new, ms, ns, nt = 3, 4, 2, 3, 5
    B_old = np.zeros((mt_old, mt_old))
    H_old = np.zeros((ms, mt_old))
    L_t = rng.normal(size=(mt_old, mt_new))
    T = rng.normal(size=(nt, mt_new))
    C = rng.normal(size=(ns, ms))
    residual = rng.normal(size=(ns, nt))
    sigma2 = 0.2
    B_trans = transfer_temporal_precision(B_old, L_t)
    H_trans = transfer_information_matrix(H_old, L_t)
    B_new = B_trans + T.T @ T / sigma2
    H_new = H_trans + C.T @ residual @ T / sigma2
    return {"passed": bool(np.allclose(B_trans, 0) and np.allclose(H_trans, 0)), "B_norm": float(np.linalg.norm(B_new)), "H_norm": float(np.linalg.norm(H_new))}


def check_projected_prior_and_structured_transfer(seed: int = 5) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mt_old, mt_new, ms, ns = 3, 4, 2, 3
    Koo_t = make_spd_matrix(mt_old, seed=seed)
    Knn_t = make_spd_matrix(mt_new, seed=seed + 1)
    Ks = make_spd_matrix(ms, seed=seed + 2)
    K_on_t = rng.normal(size=(mt_old, mt_new))
    K_on = np.kron(K_on_t, Ks)
    K_no = K_on.T
    Koo = np.kron(Koo_t, Ks)
    Knn = np.kron(Knn_t, Ks)
    m_old = rng.normal(size=mt_old * ms)
    S_old = make_spd_matrix(mt_old * ms, seed=seed + 3)
    m_proj, S_proj = projected_prior_transfer_dense(m_old, S_old, Koo, Knn, K_no, K_on)
    m_ref = K_no @ solve_spd(Koo, m_old)
    S_ref = Knn + K_no @ solve_spd(Koo, S_old - Koo) @ solve_spd(Koo, K_on)
    proj_err = np.linalg.norm(m_proj - m_ref) / max(1.0, np.linalg.norm(m_ref)) + relative_fro_error(S_proj, symmetrize(S_ref))

    C = rng.normal(size=(ns, ms))
    G = C.T @ C
    B_old = make_spd_matrix(mt_old, seed=seed + 4)
    H_old = rng.normal(size=(ms, mt_old))
    L_t = compute_Lt(K_on_t, Knn_t, jitter=0.0)
    L_dense = np.kron(K_on_t, Ks) @ inv_spd(Knn, jitter=0.0)
    Lambda_dense = L_dense.T @ np.kron(B_old, G) @ L_dense
    h_dense = L_dense.T @ vec_f(H_old)
    Lambda_kron = np.kron(transfer_temporal_precision(B_old, L_t), G)
    h_kron = vec_f(transfer_information_matrix(H_old, L_t))
    transfer_err = relative_fro_error(Lambda_dense, Lambda_kron) + np.linalg.norm(h_dense - h_kron) / max(1.0, np.linalg.norm(h_kron))
    return {"passed": bool(proj_err < 1e-8 and transfer_err < 1e-8), "projected_prior_error": float(proj_err), "structured_transfer_error": float(transfer_err)}


def check_synthetic_feasibility(seed: int = 6) -> dict[str, object]:
    dataset = make_synthetic_dataset(num_time=20, num_space=5, noise=0.08, seed=seed)
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, ms=3)
    model = JointSSGPKronHiPPOSVGP(Ks=Ks, C=C, sigma2=dataset.sigma2, beta_prior_mean=np.zeros(4), beta_prior_cov=10.0 * np.eye(4))
    state = None
    old_z = None
    preds = []
    truth = []
    for block in iter_time_blocks(20, 5):
        z_t = temporal_inducing_for_block(dataset.times, block, mt=4, moving=True)
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
        preds.append(mean)
        truth.append(factors.y_vec)
        old_z = z_t
    pred = np.concatenate(preds)
    y = np.concatenate(truth)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    nll = float(0.5 * np.mean(np.log(2.0 * np.pi * dataset.sigma2) + (y - pred) ** 2 / dataset.sigma2))
    return {"passed": bool(np.isfinite(rmse) and np.isfinite(nll)), "rmse": rmse, "nll": nll}


def run_all() -> dict[str, object]:
    checks = {
        "l_on_kron_identity": check_l_on_identity(),
        "old_likelihood_transfer_kron_identity": check_old_likelihood_transfer(),
        "fixed_basis_streaming_equals_batch": check_fixed_basis_streaming_equals_batch(),
        "no_linear_mean_reduces_to_gp_only": check_no_linear_mean(),
        "no_old_data_transfer_zero": check_no_old_data(),
        "projected_prior_and_structured_transfer": check_projected_prior_and_structured_transfer(),
        "synthetic_feasibility": check_synthetic_feasibility(),
    }
    return {"all_passed": all(bool(v["passed"]) for v in checks.values()), "checks": checks}


def main() -> None:
    report = run_all()
    outdir = Path("results/verification")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "joint_ssgp_kron_verification.json"
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved verification report to {outpath}")


if __name__ == "__main__":
    main()
