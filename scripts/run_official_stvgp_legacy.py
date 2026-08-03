#!/usr/bin/env python3
"""Paper-faithful Bayes-Newton ST-VGP/ST-SVGP smoke runner.

This file intentionally remains Python 3.7 compatible. It must be executed in
the isolated legacy environment documented alongside the P0 results.
"""

from __future__ import print_function

import argparse
import json
import os
import platform
import time

import bayesnewton
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import objax
from scipy.cluster.vq import kmeans2


def synthetic_data(seed):
    rng = np.random.RandomState(seed)
    times = np.linspace(0.0, 1.0, 12)
    all_coords = np.asarray(
        [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0], [0.5, 0.0],
         [0.5, 0.5], [0.5, 1.0], [1.0, 0.25], [1.0, 0.75]],
        dtype=np.float64,
    )
    train_coords = all_coords[:6]
    test_coords = all_coords[6:]

    def signal(coords):
        temporal = np.sin(2.0 * np.pi * times)[:, None]
        spatial = (0.7 * coords[:, 0] - 0.4 * coords[:, 1])[None, :]
        interaction = 0.25 * np.cos(4.0 * np.pi * times)[:, None] * coords[None, :, 0]
        return temporal + spatial + interaction

    y_train = signal(train_coords) + 0.03 * rng.randn(times.size, train_coords.shape[0])
    y_test = signal(test_coords)
    return times, train_coords, test_coords, y_train, y_test


def load_data(path, seed):
    if path is None:
        return synthetic_data(seed), "synthetic", {}
    data = np.load(path)
    values = (
        np.asarray(data["times"], dtype=np.float64),
        np.asarray(data["train_coords"], dtype=np.float64),
        np.asarray(data["test_coords"], dtype=np.float64),
        np.asarray(data["y_train"], dtype=np.float64),
        np.asarray(data["y_test"], dtype=np.float64),
    )
    auxiliary = {key: np.asarray(data[key]) for key in data.files if key not in {
        "times", "train_coords", "test_coords", "y_train", "y_test"
    }}
    return values, os.path.abspath(path), auxiliary


def tiled_grid(times, coords, values):
    return (
        np.asarray(times, dtype=np.float64)[:, None],
        np.tile(np.asarray(coords, dtype=np.float64)[None, :, :], (len(times), 1, 1)),
        np.asarray(values, dtype=np.float64)[:, :, None],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["st_vgp", "st_svgp", "mf_st_svgp", "st_dsvgp"],
        required=True,
    )
    parser.add_argument("--data-npz", default=None)
    parser.add_argument("--num-spatial-inducing", type=int, default=4)
    parser.add_argument("--num-temporal-inducing", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temporal-lengthscale", type=float, default=0.2)
    parser.add_argument("--spatial-lengthscale", type=float, default=0.8)
    parser.add_argument("--likelihood-variance", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--newton-rate", type=float, default=1.0)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--early-stop-relative-tol", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-iterations", type=int, default=20)
    parser.add_argument("--predictions-output", default=None)
    parser.add_argument("--use-xlag-mean", action="store_true")
    parser.add_argument("--learn-xlag-mean", action="store_true")
    parser.add_argument("--xlag-ridge", type=float, default=1e-3)
    parser.add_argument("--xlag-update-every", type=int, default=5)
    parser.add_argument("--xlag-update-spatial-count", type=int, default=200)
    parser.add_argument("--xlag-update-time-chunk-size", type=int, default=20)
    parser.add_argument("--xlag-update-damping", type=float, default=1.0)
    parser.add_argument("--fixed-spatial-inducing", action="store_true")
    parser.add_argument("--trajectory-every", type=int, default=0)
    parser.add_argument("--trajectory-output", default=None)
    parser.add_argument(
        "--hlo-cost-output",
        default=None,
        help="Write XLA HLO cost analysis for one train_op call and exit.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    (times, train_coords, test_coords, y_train, y_test), data_source, auxiliary = load_data(
        args.data_npz, args.seed
    )
    if args.use_xlag_mean and args.learn_xlag_mean:
        raise ValueError("Choose only one of --use-xlag-mean and --learn-xlag-mean")
    beta = None
    phi_train = None
    phi_test = None
    if args.learn_xlag_mean:
        if args.model == "st_vgp":
            raise ValueError("Learned X-lag mean is currently implemented for ST-SVGP paths")
        if "xlag_phi_train" not in auxiliary or "xlag_phi_test" not in auxiliary:
            raise ValueError("--learn-xlag-mean requires xlag_phi_train/test arrays in --data-npz")
        phi_train = np.asarray(auxiliary["xlag_phi_train"], dtype=np.float64)
        phi_test = np.asarray(auxiliary["xlag_phi_test"], dtype=np.float64)
        if phi_train.shape[:2] != y_train.shape or phi_test.shape[:2] != y_test.shape:
            raise ValueError("X-lag Phi arrays must match y_train/y_test leading dimensions")
        design = phi_train.reshape((-1, phi_train.shape[-1]))
        precision = np.dot(design.T, design) + args.xlag_ridge * np.eye(design.shape[1])
        beta = np.linalg.solve(precision, np.dot(design.T, y_train.reshape(-1)))
        mean_train = np.dot(design, beta).reshape(y_train.shape)
        mean_test = np.dot(phi_test.reshape((-1, phi_test.shape[-1])), beta).reshape(y_test.shape)
        model_y_train = y_train - mean_train
        model_y_test = y_test - mean_test
    elif args.use_xlag_mean:
        if "xlag_mean_train" not in auxiliary or "xlag_mean_test" not in auxiliary:
            raise ValueError("--use-xlag-mean requires xlag_mean_train/test arrays in --data-npz")
        mean_train = np.asarray(auxiliary["xlag_mean_train"], dtype=np.float64)
        mean_test = np.asarray(auxiliary["xlag_mean_test"], dtype=np.float64)
        if mean_train.shape != y_train.shape or mean_test.shape != y_test.shape:
            raise ValueError("X-lag mean arrays must match y_train/y_test shapes")
        model_y_train = y_train - mean_train
        model_y_test = y_test - mean_test
    else:
        mean_train = np.zeros_like(y_train)
        mean_test = np.zeros_like(y_test)
        model_y_train = y_train
        model_y_test = y_test
    if args.model == "st_vgp":
        # The official non-sparse spatial path keeps the full spatial grid in
        # the Markov state. Held-out labels are masked, not removed from R.
        model_coords = np.vstack([train_coords, test_coords])
        masked_test = np.full_like(model_y_test, np.nan)
        model_y = np.concatenate([model_y_train, masked_test], axis=1)
    else:
        model_coords = train_coords
        model_y = model_y_train
    t_train, r_train, y_train_grid = tiled_grid(times, model_coords, model_y)
    t_test, r_test, y_test_grid = tiled_grid(times, test_coords, y_test)
    if args.learn_xlag_mean:
        beta_update_count = min(args.xlag_update_spatial_count, train_coords.shape[0])
        beta_update_idx = np.linspace(
            0, train_coords.shape[0] - 1, beta_update_count
        ).round().astype(int)
    else:
        beta_update_count = 0
        beta_update_idx = None

    sparse = args.model in ("st_svgp", "mf_st_svgp", "st_dsvgp")
    temporal_sparse = args.model == "st_dsvgp"
    if sparse:
        num_z = min(args.num_spatial_inducing, train_coords.shape[0])
        shared_z_key = "inducing_coords_ms%d" % args.num_spatial_inducing
        if shared_z_key in auxiliary:
            z = np.asarray(auxiliary[shared_z_key], dtype=np.float64)
            if z.shape != (num_z, train_coords.shape[1]):
                raise ValueError("%s has incompatible shape %r" % (shared_z_key, z.shape))
        else:
            z = kmeans2(train_coords, num_z, minit="points", seed=args.seed)[0]
    else:
        z = model_coords.copy()

    kern_time = bayesnewton.kernels.Matern32(
        variance=1.0, lengthscale=args.temporal_lengthscale
    )
    kern_space0 = bayesnewton.kernels.Matern32(
        variance=1.0, lengthscale=args.spatial_lengthscale
    )
    kern_space1 = bayesnewton.kernels.Matern32(
        variance=1.0, lengthscale=args.spatial_lengthscale
    )
    kern_space = bayesnewton.kernels.Separable([kern_space0, kern_space1])
    kernel = bayesnewton.kernels.SpatioTemporalKernel(
        temporal_kernel=kern_time,
        spatial_kernel=kern_space,
        z=z,
        sparse=sparse,
        opt_z=sparse and not args.fixed_spatial_inducing,
        conditional="Full",
    )
    likelihood = bayesnewton.likelihoods.Gaussian(variance=args.likelihood_variance)
    if temporal_sparse:
        if args.num_temporal_inducing is None or args.num_temporal_inducing < 2:
            raise ValueError("st_dsvgp requires --num-temporal-inducing >= 2")
        temporal_z = np.linspace(
            float(times.min()), float(times.max()), args.num_temporal_inducing
        )[:, None]
        model = bayesnewton.models.SparseMarkovVariationalGP(
            kernel=kernel,
            likelihood=likelihood,
            X=t_train,
            R=r_train,
            Y=y_train_grid,
            Z=temporal_z,
            parallel=True if args.parallel else None,
        )
    else:
        temporal_z = None
        model_class = (
            bayesnewton.models.MarkovVariationalMeanFieldGP
            if args.model == "mf_st_svgp"
            else bayesnewton.models.MarkovVariationalGP
        )
        model = model_class(
            kernel=kernel,
            likelihood=likelihood,
            X=t_train,
            R=r_train,
            Y=y_train_grid,
            parallel=True if args.parallel else None,
        )

    optimizer = objax.optimizer.Adam(model.vars())
    energy = objax.GradValues(model.energy, model.vars())

    @objax.Function.with_vars(model.vars() + optimizer.vars())
    def train_op(current_y):
        model.Y = jnp.asarray(current_y)
        model.inference(lr=args.newton_rate)
        gradients, value = energy()
        optimizer(args.learning_rate, gradients)
        return value

    if args.jit:
        train_op = objax.Jit(train_op)

    if args.hlo_cost_output is not None:
        if not args.jit:
            raise ValueError("--hlo-cost-output requires --jit")
        from jaxlib import xla_extension

        computation = jax.xla_computation(train_op._call)(
            train_op.vc.tensors(), {}, jnp.asarray(y_train_grid)
        )
        cost = xla_extension.hlo_module_cost_analysis(
            jax.lib.xla_bridge.get_backend(), computation.as_hlo_module()
        )
        payload = {
            "scope": "one_jitted_train_op",
            "includes": "Bayes-Newton inference, energy gradient, and Objax Adam update",
            "model": args.model,
            "num_time_steps": int(times.size),
            "num_train_space": int(train_coords.shape[0]),
            "num_spatial_inducing": int(z.shape[0]),
            "uses_xlag": bool(args.use_xlag_mean or args.learn_xlag_mean),
            "note": (
                "The alternating closed-form X-lag beta update is outside train_op "
                "and is therefore excluded."
            ),
            "xla_cost_analysis": {key: float(value) for key, value in cost.items()},
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
        }
        cost_dir = os.path.dirname(os.path.abspath(args.hlo_cost_output))
        if cost_dir and not os.path.exists(cost_dir):
            os.makedirs(cost_dir)
        with open(args.hlo_cost_output, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(json.dumps(payload, indent=2))
        return

    def prediction_metrics():
        current_mean, current_var = model.predict_y(X=t_test, R=r_test)
        if sparse:
            current_mean = np.asarray(current_mean).reshape(y_test.shape)
            current_var = np.asarray(current_var).reshape(y_test.shape)
        else:
            full_shape = (times.size, model_coords.shape[0])
            current_mean = np.asarray(current_mean).reshape(full_shape)[:, train_coords.shape[0]:]
            current_var = np.asarray(current_var).reshape(full_shape)[:, train_coords.shape[0]:]
        current_mean = current_mean + mean_test
        current_var = np.maximum(current_var, 1e-10)
        current_rmse = float(np.sqrt(np.nanmean((y_test - current_mean) ** 2)))
        current_nll = float(
            0.5
            * np.nanmean(
                np.log(2.0 * np.pi * current_var)
                + (y_test - current_mean) ** 2 / current_var
            )
        )
        return current_mean, current_var, current_rmse, current_nll

    started = time.time()
    losses = []
    beta_updates = []
    trajectory = []
    cumulative_update_seconds = 0.0
    stable_iterations = 0
    stop_reason = "maximum_iterations"
    for iteration in range(1, args.iterations + 1):
        update_started = time.time()
        if (
            args.learn_xlag_mean
            and iteration > 1
            and (iteration - 1) % max(args.xlag_update_every, 1) == 0
        ):
            precision = args.xlag_ridge * np.eye(phi_train.shape[-1])
            rhs = np.zeros(phi_train.shape[-1])
            chunk_size = max(args.xlag_update_time_chunk_size, 1)
            for start in range(0, times.size, chunk_size):
                time_idx = np.arange(start, min(start + chunk_size, times.size))
                residual_chunk, _ = model.conditional_posterior_to_data(batch_ind=time_idx)
                residual_chunk = np.asarray(residual_chunk).reshape(
                    time_idx.size, train_coords.shape[0]
                )
                target_chunk = (
                    y_train[np.ix_(time_idx, beta_update_idx)]
                    - residual_chunk[:, beta_update_idx]
                )
                design_chunk = phi_train[np.ix_(time_idx, beta_update_idx)].reshape(
                    (-1, phi_train.shape[-1])
                )
                precision += np.dot(design_chunk.T, design_chunk)
                rhs += np.dot(design_chunk.T, target_chunk.reshape(-1))
            beta_new = np.linalg.solve(precision, rhs)
            beta = (
                args.xlag_update_damping * beta_new
                + (1.0 - args.xlag_update_damping) * beta
            )
            full_design = phi_train.reshape((-1, phi_train.shape[-1]))
            mean_train = np.dot(full_design, beta).reshape(y_train.shape)
            mean_test = np.dot(
                phi_test.reshape((-1, phi_test.shape[-1])), beta
            ).reshape(y_test.shape)
            model_y_train = y_train - mean_train
            y_train_grid = model_y_train[:, :, None]
            beta_updates.append(
                {
                    "iteration": iteration,
                    "beta_norm": float(np.linalg.norm(beta)),
                    "mean_rmse": float(np.sqrt(np.mean((y_train - mean_train) ** 2))),
                }
            )
        loss = train_op(y_train_grid)
        cumulative_update_seconds += time.time() - update_started
        loss_value = float(np.asarray(loss[0]))
        if losses and args.early_stop_relative_tol is not None:
            relative_change = abs(loss_value - losses[-1]) / max(abs(losses[-1]), 1.0)
            if relative_change < args.early_stop_relative_tol:
                stable_iterations += 1
            else:
                stable_iterations = 0
        losses.append(loss_value)
        if iteration == 1 or iteration == args.iterations or iteration % max(args.log_every, 1) == 0:
            print("iter %d: energy: %.8f" % (iteration, loss_value), flush=True)
        if (
            args.trajectory_every > 0
            and (iteration == 1 or iteration % args.trajectory_every == 0 or iteration == args.iterations)
        ):
            trajectory_started = time.time()
            _, _, trajectory_rmse, trajectory_nll = prediction_metrics()
            trajectory.append(
                {
                    "iteration": iteration,
                    "cumulative_update_seconds": cumulative_update_seconds,
                    "diagnostic_elapsed_seconds": time.time() - started,
                    "prediction_seconds": time.time() - trajectory_started,
                    "test_rmse": trajectory_rmse,
                    "test_nll": trajectory_nll,
                }
            )
            print("trajectory iter %d: rmse %.8f nll %.8f" % (iteration, trajectory_rmse, trajectory_nll), flush=True)
        if (
            args.early_stop_relative_tol is not None
            and iteration >= args.early_stop_min_iterations
            and stable_iterations >= args.early_stop_patience
        ):
            stop_reason = "relative_energy_convergence"
            print(
                "early stop at iter %d after %d stable relative-energy updates"
                % (iteration, stable_iterations),
                flush=True,
            )
            break
    train_seconds = time.time() - started

    pred_mean, pred_var, rmse, nll = prediction_metrics()
    pred_residual_mean = pred_mean - mean_test
    half = 1.6448536269514722 * np.sqrt(pred_var)
    coverage90 = float(np.nanmean((y_test >= pred_mean - half) & (y_test <= pred_mean + half)))

    result = {
        "implementation": "official AaltoML Bayes-Newton API",
        "model": args.model,
        "paper_role": (
            "ST-VGP full spatial"
            if not sparse
            else (
                "Doubly sparse Markov ST-SVGP extension: temporal and spatial inducing states"
                if temporal_sparse
                else (
                "MF-ST-SVGP spatial sparse mean-field, temporal full Markov state"
                if args.model == "mf_st_svgp"
                else "ST-SVGP spatial sparse, temporal full Markov state"
                )
            )
        ),
        "direct_target": not (args.use_xlag_mean or args.learn_xlag_mean),
        "prediction_target": "original_y",
        "training_target": "original_y_with_learned_xlag_mean" if args.learn_xlag_mean else ("fixed_residual" if args.use_xlag_mean else "original_y"),
        "latent_gp_target": "jointly updated xlag residual" if args.learn_xlag_mean else ("fixed xlag residual" if args.use_xlag_mean else "y"),
        "uses_xlag": bool(args.use_xlag_mean or args.learn_xlag_mean),
        "xlag_mean_protocol": (
            "alternating learned linear mean within original-y training"
            if args.learn_xlag_mean
            else ("shared fixed ridge mean from data NPZ" if args.use_xlag_mean else None)
        ),
        "xlag_beta": np.asarray(beta).tolist() if beta is not None else None,
        "xlag_beta_updates": beta_updates,
        "xlag_ridge": args.xlag_ridge if args.learn_xlag_mean else None,
        "xlag_update_spatial_count": beta_update_count if args.learn_xlag_mean else None,
        "xlag_update_time_chunk_size": args.xlag_update_time_chunk_size if args.learn_xlag_mean else None,
        "sparse_space": sparse,
        "mean_field": args.model == "mf_st_svgp",
        "num_train_space": int(train_coords.shape[0]),
        "num_test_space": int(test_coords.shape[0]),
        "num_model_space": int(model_coords.shape[0]),
        "heldout_target_masking": "NaN on full grid" if not sparse else "coordinates excluded from training grid",
        "num_spatial_inducing": int(z.shape[0]),
        "spatial_inducing_optimized": bool(sparse and not args.fixed_spatial_inducing),
        "num_time_steps": int(times.size),
        "temporal_inducing_count": int(args.num_temporal_inducing) if temporal_sparse else None,
        "temporal_state_count": (
            int(args.num_temporal_inducing) if temporal_sparse else int(times.size)
        ),
        "temporal_inducing_locations": np.asarray(temporal_z).reshape(-1).tolist() if temporal_sparse else None,
        "paper_faithful_temporal_representation": not temporal_sparse,
        "temporal_protocol": (
            "sparse Markov inducing states with pairwise transitions"
            if temporal_sparse
            else "full Markov state at every observed time"
        ),
        "iterations": len(losses),
        "iterations_requested": args.iterations,
        "stop_reason": stop_reason,
        "early_stop_relative_tol": args.early_stop_relative_tol,
        "early_stop_patience": args.early_stop_patience,
        "jit": bool(args.jit),
        "parallel": bool(args.parallel),
        "seed": args.seed,
        "rmse": rmse,
        "nll": nll,
        "coverage90": coverage90,
        "mean_predictive_std": float(np.mean(np.sqrt(pred_var))),
        "train_seconds": train_seconds,
        "cumulative_update_seconds": cumulative_update_seconds,
        "diagnostic_trajectory": trajectory,
        "losses": losses,
        "learned_temporal_lengthscale": float(np.asarray(kern_time.lengthscale)),
        "learned_temporal_variance": float(np.asarray(kern_time.variance)),
        "learned_spatial_lengthscales": [
            float(np.asarray(kern_space0.lengthscale)),
            float(np.asarray(kern_space1.lengthscale)),
        ],
        "learned_spatial_variances": [
            float(np.asarray(kern_space0.variance)),
            float(np.asarray(kern_space1.variance)),
        ],
        "learned_likelihood_variance": float(np.asarray(likelihood.variance)),
        "learned_spatial_inducing_locations": np.asarray(kernel.z.value).tolist(),
        "data_source": data_source,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "objax": objax.__version__,
        "bayesnewton": getattr(bayesnewton, "__version__", "1.1"),
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    if args.trajectory_output is not None:
        trajectory_dir = os.path.dirname(os.path.abspath(args.trajectory_output))
        if trajectory_dir and not os.path.exists(trajectory_dir):
            os.makedirs(trajectory_dir)
        with open(args.trajectory_output, "w") as handle:
            json.dump(trajectory, handle, indent=2)
    if args.predictions_output is not None:
        prediction_dir = os.path.dirname(os.path.abspath(args.predictions_output))
        if prediction_dir and not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        np.savez_compressed(
            args.predictions_output,
            y_true=y_test,
            pred_mean=pred_mean,
            pred_residual_mean=pred_residual_mean,
            pred_var=pred_var,
            xlag_mean=mean_test,
            test_coords=test_coords,
            times=times,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
