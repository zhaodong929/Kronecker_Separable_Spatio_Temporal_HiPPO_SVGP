from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.joint_ssgp_kron.kron_utils import dense_A_from_factors, solve_spd, vec_f
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    iter_time_blocks,
    make_block_factors,
    make_spatial_projection,
    make_synthetic_dataset,
    rbf_kernel,
    temporal_inducing_for_block,
)


METHODS = ["no_transfer", "mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]
LABELS = {
    "no_transfer": "no_transfer",
    "mean_field_ssgp_transfer": "mean-field",
    "structured_joint_ssgp_transfer": "Route B",
}


def train_method(dataset, C, Ks, args, method: str):
    model = JointSSGPKronHiPPOSVGP(
        Ks=Ks,
        C=C,
        sigma2=dataset.sigma2,
        beta_prior_mean=np.zeros(dataset.Phi.shape[1]),
        beta_prior_cov=10.0 * np.eye(dataset.Phi.shape[1]),
        prior_point_variance=dataset.gp_prior_variance,
    )
    state = None
    old_z = None
    final_z = None
    blocks = iter_time_blocks(dataset.Y.shape[1], args.block_size)
    for block in blocks:
        z_t = temporal_inducing_for_block(dataset.times, block, args.mt, moving=True)
        factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=old_z, lengthscale=args.model_ell_t)
        if method == "no_transfer":
            state = model.update_block_no_transfer(
                y_vec=factors.y_vec,
                Phi=factors.Phi,
                T_n=factors.T,
                Kt_new=factors.Kt,
                K_on_t=factors.K_on_t,
                state=state,
                inner_iters=2,
            )
        elif method == "mean_field_ssgp_transfer":
            state = model.update_block_mean_field_ssgp_transfer(
                y_vec=factors.y_vec,
                Phi=factors.Phi,
                T_n=factors.T,
                Kt_new=factors.Kt,
                K_on_t=factors.K_on_t,
                state=state,
                inner_iters=2,
            )
        elif method == "structured_joint_ssgp_transfer":
            state = model.update_block_structured_joint_ssgp_transfer(
                y_vec=factors.y_vec,
                Phi=factors.Phi,
                T_n=factors.T,
                Kt_new=factors.Kt,
                K_on_t=factors.K_on_t,
                state=state,
            )
        else:
            raise ValueError(method)
        old_z = z_t
        final_z = z_t
    assert state is not None and final_z is not None
    return model, state, final_z


def predict_location(model, state, dataset, C, z_t, loc_idx: int, model_ell_t: float):
    ns = dataset.Y.shape[0]
    row_idx = np.asarray([t * ns + loc_idx for t in range(dataset.Y.shape[1])])
    Phi_loc = dataset.Phi[row_idx]
    Kt = rbf_kernel(z_t, lengthscale=model_ell_t, variance=dataset.gp_prior_variance) + 1e-6 * np.eye(len(z_t))
    Kfu = rbf_kernel(dataset.times, z_t, lengthscale=model_ell_t, variance=dataset.gp_prior_variance)
    T_full = solve_spd(Kt, Kfu.T).T
    C_loc = C[[loc_idx]]
    mean = Phi_loc @ state.beta_mean + dense_A_from_factors(T_full, C_loc) @ vec_f(state.M_u)
    vars_ = []
    for i in range(dataset.Y.shape[1]):
        decomp = model.predictive_variance_decomposition(
            phi_star=Phi_loc[i],
            c_proj_star=C[loc_idx],
            t_proj_star=T_full[i],
            state=state,
        )
        vars_.append(decomp.total_variance)
    return mean, np.asarray(vars_)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-time", type=int, default=100)
    parser.add_argument("--num-space", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--mt", type=int, default=16)
    parser.add_argument("--ms", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--data-ell-t", type=float, default=0.8)
    parser.add_argument("--model-ell-t", type=float, default=0.25)
    parser.add_argument("--location-index", type=int, default=5)
    parser.add_argument("--outdir", type=Path, default=Path("results/routeB_experiment_report/long_memory_time_dependence"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    dataset = make_synthetic_dataset(
        num_time=args.num_time,
        num_space=args.num_space,
        noise=args.noise,
        seed=args.seed,
        ell_t=args.data_ell_t,
        ell_s=0.35,
    )
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, args.ms)
    loc_idx = min(max(args.location_index, 0), args.num_space - 1)
    times = dataset.times
    ns = dataset.Y.shape[0]
    row_idx = np.asarray([t * ns + loc_idx for t in range(dataset.Y.shape[1])])
    linear_true = (dataset.Phi[row_idx] @ dataset.beta_true)
    residual_true = dataset.F[loc_idx]
    observed = dataset.Y[loc_idx]

    predictions = {}
    for method in METHODS:
        model, state, z_t = train_method(dataset, C, Ks, args, method)
        mean, var = predict_location(model, state, dataset, C, z_t, loc_idx, args.model_ell_t)
        predictions[method] = {"mean": mean, "var": var}

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(times, observed, color="black", linewidth=1.5, label="observation y")
    axes[0].plot(times, linear_true, color="#1b9e77", linestyle="--", label="true linear component Phi beta")
    axes[0].plot(times, linear_true + residual_true, color="#7570b3", alpha=0.8, label="latent mean Phi beta + f")
    axes[0].set_ylabel("value")
    axes[0].set_title(f"Long-memory synthetic time dependence at location {loc_idx}")
    axes[0].legend(ncol=3, fontsize=8)

    basis = np.column_stack(
        [
            np.ones_like(times),
            (times - times.min()) / max(1e-12, times.max() - times.min()),
            np.full_like(times, dataset.spatial_coords[loc_idx, 0] - np.mean(dataset.spatial_coords[:, 0])),
            np.sin(2.0 * np.pi * ((times - times.min()) / max(1e-12, times.max() - times.min()))),
        ]
    )
    for i, name in enumerate(["1", "t_scaled", "s_centered", "sin(2pi t)"]):
        axes[1].plot(times, basis[:, i], label=name)
    axes[1].set_ylabel("basis value")
    axes[1].set_title("Linear regression basis used in Phi")
    axes[1].legend(ncol=4, fontsize=8)

    colors = {
        "no_transfer": "#8da0cb",
        "mean_field_ssgp_transfer": "#66c2a5",
        "structured_joint_ssgp_transfer": "#fc8d62",
    }
    axes[2].plot(times, observed, color="black", linewidth=1.2, label="observation y")
    for method in METHODS:
        mean = predictions[method]["mean"]
        sd = np.sqrt(np.maximum(predictions[method]["var"], 0.0))
        axes[2].plot(times, mean, color=colors[method], label=LABELS[method])
        axes[2].fill_between(times, mean - 1.645 * sd, mean + 1.645 * sd, color=colors[method], alpha=0.12)
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("prediction")
    axes[2].set_title("Final online posterior predictions with 90% intervals")
    axes[2].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig_path = args.outdir / "long_memory_location_time_dependence.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    summary = {
        "linear_basis": ["1", "t_scaled", "s_centered", "sin(2*pi*t_scaled)"],
        "beta_true": dataset.beta_true.tolist(),
        "seed": args.seed,
        "location_index": loc_idx,
        "data_ell_t": args.data_ell_t,
        "model_ell_t": args.model_ell_t,
        "mt": args.mt,
        "block_size": args.block_size,
        "noise": args.noise,
        "plot": str(fig_path),
    }
    (args.outdir / "long_memory_location_time_dependence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
