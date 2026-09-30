#!/usr/bin/env python
"""Single-location protocol comparison for Route B synthetic experiments.

The figure is designed to answer a visual fitting question: for the same
synthetic setting, how do the current/single-stream, continual seen-history,
and batch full-prefix protocols differ at one spatial location?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_joint_ssgp_kron_experiments import (
    fit_model_hyperparameters_from_initial_task,
    initial_task_block_count,
    make_dataset,
    model_temporal_lengthscale,
)
from stvgp_kronecker.joint_ssgp_kron.kron_utils import dense_A_from_factors, vec_f
from stvgp_kronecker.joint_ssgp_kron.model import JointSSGPKronHiPPOSVGP
from stvgp_kronecker.joint_ssgp_kron.synthetic import (
    iter_time_blocks,
    make_block_factors,
    make_spatial_projection,
    rbf_kernel,
    temporal_inducing_for_block,
)


Z90 = 1.6448536269514722


def experiment_args(regime: str, seed: int) -> SimpleNamespace:
    num_space = 10 if regime == "long_memory" else 6
    args = SimpleNamespace(
        dataset="synthetic",
        synthetic_regime=regime,
        num_seeds=1,
        num_time=100,
        num_space=num_space,
        block_size=5,
        mt=5,
        ms=4,
        noise=0.08,
        model_noise=None,
        model_kernel_variance=None,
        ell_t=None,
        model_ell_t=None,
        model_ell_t_sweep=None,
        ell_t_fit_mode="initial_task_fullgp",
        initial_task_blocks=None,
        initial_task_fraction=0.2,
        time_normalization="expected_horizon",
        time_scale=1.0,
        ell_t_grid_source="time_scale",
        ell_t_grid_values=None,
        fit_noise_from_initial_task=False,
        noise_fit_grid_values=None,
        fit_kernel_variance_from_initial_task=False,
        kernel_variance_fit_grid_values=None,
        methods=["structured_joint_ssgp_transfer"],
        linear_dim=None,
        linear_signal_strength=1.0,
        gp_signal_strength=1.0,
        beta_u_correlation_design="strong",
        noise_sweep=None,
        beta_u_correlation_sweep=None,
        eval_mode=None,
        eval_modes=["current", "seen_history", "batch"],
        missing_rate=0.5,
        include_mean_field_ablation=False,
        include_dense_reference_small=False,
        routeB=False,
        outdir=Path("results/unused"),
        era5_root=Path("data/era5/processed_timeseries_4"),
        era5_task_dirs=None,
        era5_variable_index=0,
        era5_split="train",
        era5_shuffle_locations=False,
        seed=seed,
    )
    # Resolve metadata used by initial-task fitting.
    dataset = make_dataset(args, seed)
    args.initial_task_blocks_used = initial_task_block_count(args, dataset.Y.shape[1])
    fitted = fit_model_hyperparameters_from_initial_task(args, seed)
    args.model_ell_t = float(fitted["ell_t"])
    args.model_noise = float(fitted["noise"])
    args.model_kernel_variance = float(fitted["kernel_variance"])
    args.selected_candidate_score = float(fitted["score"])
    return args


def make_model(dataset, Ks, C) -> JointSSGPKronHiPPOSVGP:
    return JointSSGPKronHiPPOSVGP(
        Ks=Ks,
        C=C,
        sigma2=dataset.sigma2,
        beta_prior_mean=np.zeros(dataset.Phi.shape[1]),
        beta_prior_cov=10.0 * np.eye(dataset.Phi.shape[1]),
        prior_point_variance=dataset.gp_prior_variance,
    )


def routeb_update(model, state, factors):
    return model.update_block_structured_joint_ssgp_transfer(
        y_vec=factors.y_vec,
        Phi=factors.Phi,
        T_n=factors.T,
        Kt_new=factors.Kt,
        K_on_t=factors.K_on_t,
        state=state,
    )


def predict_location(model, state, dataset, C, z_t: np.ndarray, loc_idx: int, model_ell_t: float) -> tuple[np.ndarray, np.ndarray]:
    ns = dataset.Y.shape[0]
    row_idx = np.asarray([t * ns + loc_idx for t in range(dataset.Y.shape[1])])
    phi_loc = dataset.Phi[row_idx]
    kt = rbf_kernel(z_t, lengthscale=model_ell_t, variance=dataset.gp_prior_variance) + 1e-6 * np.eye(len(z_t))
    kfu = rbf_kernel(dataset.times, z_t, lengthscale=model_ell_t, variance=dataset.gp_prior_variance)
    t_full = np.linalg.solve(kt, kfu.T).T
    c_loc = C[[loc_idx]]
    mean = phi_loc @ state.beta_mean + dense_A_from_factors(t_full, c_loc) @ vec_f(state.M_u)
    var = []
    for i in range(dataset.Y.shape[1]):
        decomp = model.predictive_variance_decomposition(
            phi_star=phi_loc[i],
            t_proj_star=t_full[i],
            c_proj_star=C[loc_idx],
            state=state,
        )
        var.append(decomp.total_variance)
    return mean, np.asarray(var)


def run_protocols(regime: str, seed: int, location_index: int | None) -> dict[str, object]:
    args = experiment_args(regime, seed)
    dataset = make_dataset(args, seed)
    _, Ks, C = make_spatial_projection(dataset.spatial_coords, args.ms)
    blocks = iter_time_blocks(dataset.Y.shape[1], args.block_size)
    model_ell_t = model_temporal_lengthscale(args)
    loc = location_index if location_index is not None else dataset.Y.shape[0] // 2
    loc = max(0, min(int(loc), dataset.Y.shape[0] - 1))

    model_online = make_model(dataset, Ks, C)
    state = None
    old_z = None
    final_z = None
    current_mean = np.full(dataset.Y.shape[1], np.nan)
    current_var = np.full(dataset.Y.shape[1], np.nan)
    for block in blocks:
        z_t = temporal_inducing_for_block(dataset.times, block, args.mt, moving=True)
        factors = make_block_factors(dataset, block=block, z_t=z_t, z_t_old=old_z, lengthscale=model_ell_t)
        state = routeb_update(model_online, state, factors)
        pred_mean, pred_var = predict_location(model_online, state, dataset, C, z_t, loc, model_ell_t)
        current_mean[block] = pred_mean[block]
        current_var[block] = pred_var[block]
        old_z = z_t
        final_z = z_t
    assert state is not None and final_z is not None
    continual_mean, continual_var = predict_location(model_online, state, dataset, C, final_z, loc, model_ell_t)

    model_batch = make_model(dataset, Ks, C)
    full_block = slice(0, dataset.Y.shape[1])
    batch_z = temporal_inducing_for_block(dataset.times, full_block, args.mt, moving=True)
    batch_factors = make_block_factors(dataset, block=full_block, z_t=batch_z, z_t_old=None, lengthscale=model_ell_t)
    batch_state = routeb_update(model_batch, None, batch_factors)
    batch_mean, batch_var = predict_location(model_batch, batch_state, dataset, C, batch_z, loc, model_ell_t)

    y = dataset.Y[loc]
    latent = dataset.F[loc] + dataset.Phi[np.asarray([t * dataset.Y.shape[0] + loc for t in range(dataset.Y.shape[1])])] @ dataset.beta_true
    protocols = {
        "single_stream_current": {"mean": current_mean, "var": current_var, "label": "single-stream/current", "color": "#4C78A8"},
        "continual_seen_history": {"mean": continual_mean, "var": continual_var, "label": "continual seen-history", "color": "#F58518"},
        "batch_full_prefix": {"mean": batch_mean, "var": batch_var, "label": "batch full-prefix", "color": "#54A24B"},
    }
    metrics = {}
    for key, value in protocols.items():
        mean = np.asarray(value["mean"], dtype=float)
        metrics[key] = {
            "rmse": float(np.sqrt(np.nanmean((mean - y) ** 2))),
            "mae": float(np.nanmean(np.abs(mean - y))),
        }
    return {
        "args": vars(args),
        "times": dataset.times,
        "y": y,
        "latent": latent,
        "protocols": protocols,
        "metrics": metrics,
        "location_index": loc,
        "spatial_coord": float(dataset.spatial_coords[loc, 0]),
        "block_size": args.block_size,
        "regime": regime,
        "seed": seed,
    }


def add_block_lines(ax: plt.Axes, times: np.ndarray, block_size: int) -> None:
    for boundary in range(block_size, len(times), block_size):
        ax.axvline(times[boundary], color="0.82", linestyle="--", linewidth=0.6, zorder=0)


def plot_regime(result: dict[str, object], outdir: Path) -> Path:
    times = np.asarray(result["times"])
    y = np.asarray(result["y"])
    latent = np.asarray(result["latent"])
    protocols = result["protocols"]
    metrics = result["metrics"]
    regime = str(result["regime"])
    loc = int(result["location_index"])

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax = axes[0]
    ax.plot(times, y, color="black", linewidth=1.35, label="observed y")
    ax.plot(times, latent, color="0.35", linewidth=1.0, linestyle=":", label="latent mean")
    for key, item in protocols.items():
        mean = np.asarray(item["mean"])
        var = np.asarray(item["var"])
        sd = np.sqrt(np.maximum(var, 0.0))
        label = f"{item['label']} (RMSE {metrics[key]['rmse']:.3f})"
        ax.plot(times, mean, color=item["color"], linewidth=1.35, label=label)
        ax.fill_between(times, mean - Z90 * sd, mean + Z90 * sd, color=item["color"], alpha=0.10, linewidth=0)
    add_block_lines(ax, times, int(result["block_size"]))
    ax.set_title(f"{regime}: single-location Route B fit, location {loc}")
    ax.set_ylabel("standardized value")
    ax.legend(ncol=2, fontsize=8)

    err_ax = axes[1]
    for key, item in protocols.items():
        mean = np.asarray(item["mean"])
        err_ax.plot(times, np.abs(mean - y), color=item["color"], linewidth=1.1, label=item["label"])
    add_block_lines(err_ax, times, int(result["block_size"]))
    err_ax.set_xlabel("normalized time")
    err_ax.set_ylabel("|error|")
    err_ax.set_title("Absolute fitting error at the same location")
    err_ax.legend(ncol=3, fontsize=8)

    fig.tight_layout()
    outpath = outdir / f"routeB_single_location_protocol_fit_{regime}.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_combined(results: list[dict[str, object]], outdir: Path) -> Path:
    fig, axes = plt.subplots(len(results), 1, figsize=(10.5, 3.3 * len(results)), sharex=False)
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        times = np.asarray(result["times"])
        y = np.asarray(result["y"])
        protocols = result["protocols"]
        metrics = result["metrics"]
        ax.plot(times, y, color="black", linewidth=1.25, label="observed y")
        for key, item in protocols.items():
            mean = np.asarray(item["mean"])
            label = f"{item['label']} ({metrics[key]['rmse']:.3f})"
            ax.plot(times, mean, color=item["color"], linewidth=1.25, label=label)
        add_block_lines(ax, times, int(result["block_size"]))
        ax.set_title(f"{result['regime']} location {result['location_index']}: line label shows pointwise RMSE")
        ax.set_ylabel("value")
        ax.legend(ncol=4, fontsize=7)
    axes[-1].set_xlabel("normalized time")
    fig.tight_layout()
    outpath = outdir / "routeB_single_location_protocol_fit_standard_vs_long_memory.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--location-index", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("results/routeB_experiment_report/single_point_protocol_fit"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    results = [run_protocols(regime, args.seed, args.location_index) for regime in ["standard", "long_memory"]]
    output_files = [str(plot_regime(result, args.outdir)) for result in results]
    output_files.append(str(plot_combined(results, args.outdir)))

    serializable_results = []
    for result in results:
        serializable_results.append(
            {
                "regime": result["regime"],
                "seed": result["seed"],
                "location_index": result["location_index"],
                "spatial_coord": result["spatial_coord"],
                "block_size": result["block_size"],
                "model_ell_t": result["args"]["model_ell_t"],
                "selected_candidate_score": result["args"]["selected_candidate_score"],
                "metrics": result["metrics"],
            }
        )
    summary = {"output_files": output_files, "results": serializable_results}
    (args.outdir / "routeB_single_location_protocol_fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
