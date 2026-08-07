#!/usr/bin/env python3
"""Official Bayes-Newton ST-VGP/ST-SVGP runner.

The protocol-aware path selects an iteration on 80 validation locations after
fitting 720 locations, then constructs a fresh model and refits on all 800
training locations before evaluating the 200-location test set.  The legacy
NPZ schema remains a direct train/test path.  This file is Python 3.7
compatible and keeps the heavy legacy dependencies lazy for protocol tests.
"""

from __future__ import print_function

import argparse
import json
import os
import platform
import time

import numpy as np


bayesnewton = None
jax = None
jnp = None
jaxlib = None
objax = None
kmeans2 = None


def import_backend():
    global bayesnewton, jax, jnp, jaxlib, objax, kmeans2
    if bayesnewton is not None:
        return
    import bayesnewton as bayesnewton_module
    import jax as jax_module
    import jax.numpy as jnp_module
    import jaxlib as jaxlib_module
    import objax as objax_module
    from scipy.cluster.vq import kmeans2 as kmeans2_function

    bayesnewton = bayesnewton_module
    jax = jax_module
    jnp = jnp_module
    jaxlib = jaxlib_module
    objax = objax_module
    kmeans2 = kmeans2_function


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
    with np.load(path) as data:
        values = (
            np.asarray(data["times"], dtype=np.float64),
            np.asarray(data["train_coords"], dtype=np.float64),
            np.asarray(data["test_coords"], dtype=np.float64),
            np.asarray(data["y_train"], dtype=np.float64),
            np.asarray(data["y_test"], dtype=np.float64),
        )
        auxiliary = {
            key: np.asarray(data[key])
            for key in data.files
            if key not in {"times", "train_coords", "test_coords", "y_train", "y_test"}
        }
    return values, os.path.abspath(path), auxiliary


def tiled_grid(times, coords, values):
    return (
        np.asarray(times, dtype=np.float64)[:, None],
        np.tile(np.asarray(coords, dtype=np.float64)[None, :, :], (len(times), 1, 1)),
        np.asarray(values, dtype=np.float64)[:, :, None],
    )


def _record(coords, values, offset):
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if values.ndim != 2 or offset.shape != values.shape:
        raise ValueError("target and offset arrays must have matching time/location shapes")
    if coords.ndim != 2 or coords.shape[0] != values.shape[1]:
        raise ValueError("coordinate count must match target location count")
    return {
        "coords": coords,
        "y": values,
        "offset": offset,
        "model_y": values - offset,
    }


def _has_validation_protocol(auxiliary):
    keys = {
        "fit_coords", "validation_coords", "y_fit", "y_validation",
        "xlag_mean_fit", "xlag_mean_validation",
    }
    present = keys.intersection(auxiliary)
    if present and present != keys:
        raise ValueError("validation protocol arrays are incomplete: %s" % sorted(present))
    return bool(present)


def prepare_splits(
    values,
    auxiliary,
    use_xlag_mean=False,
    learn_xlag_mean=False,
    ridge=1e-3,
    update_spatial_count=200,
):
    """Prepare pure NumPy split records for either protocol or legacy inputs."""
    if use_xlag_mean and learn_xlag_mean:
        raise ValueError("Choose only one of --use-xlag-mean and --learn-xlag-mean")
    times, train_coords, test_coords, y_train, y_test = values
    has_protocol = _has_validation_protocol(auxiliary)
    if has_protocol and learn_xlag_mean:
        raise ValueError("--learn-xlag-mean is only supported for legacy train/test NPZ inputs")

    if has_protocol:
        fit_coords = auxiliary["fit_coords"]
        validation_coords = auxiliary["validation_coords"]
        y_fit = auxiliary["y_fit"]
        y_validation = auxiliary["y_validation"]
    else:
        fit_coords = train_coords
        validation_coords = None
        y_fit = y_train
        y_validation = None

    if use_xlag_mean:
        if has_protocol:
            mean_fit = auxiliary["xlag_mean_fit"]
            mean_validation = auxiliary["xlag_mean_validation"]
        else:
            if "xlag_mean_train" not in auxiliary or "xlag_mean_test" not in auxiliary:
                raise ValueError("--use-xlag-mean requires xlag_mean_train/test arrays in --data-npz")
            mean_fit = auxiliary["xlag_mean_train"]
            mean_validation = None
        mean_train = auxiliary.get("xlag_mean_train", np.zeros_like(y_train))
        mean_test = auxiliary.get("xlag_mean_test", np.zeros_like(y_test))
    elif learn_xlag_mean:
        if "xlag_phi_train" not in auxiliary or "xlag_phi_test" not in auxiliary:
            raise ValueError("--learn-xlag-mean requires xlag_phi_train/test arrays in --data-npz")
        phi_train = np.asarray(auxiliary["xlag_phi_train"], dtype=np.float64)
        phi_test = np.asarray(auxiliary["xlag_phi_test"], dtype=np.float64)
        if phi_train.shape[:2] != y_train.shape or phi_test.shape[:2] != y_test.shape:
            raise ValueError("X-lag Phi arrays must match y_train/y_test leading dimensions")
        design = phi_train.reshape((-1, phi_train.shape[-1]))
        precision = np.dot(design.T, design) + ridge * np.eye(design.shape[1])
        beta = np.linalg.solve(precision, np.dot(design.T, y_train.reshape(-1)))
        mean_train = np.dot(design, beta).reshape(y_train.shape)
        mean_test = np.dot(phi_test.reshape((-1, phi_test.shape[-1])), beta).reshape(y_test.shape)
        mean_fit = mean_train
        mean_validation = None
    else:
        mean_fit = np.zeros_like(y_fit)
        mean_validation = None if y_validation is None else np.zeros_like(y_validation)
        mean_train = np.zeros_like(y_train)
        mean_test = np.zeros_like(y_test)

    splits = {
        "fit": _record(fit_coords, y_fit, mean_fit),
        "train": _record(train_coords, y_train, mean_train),
        "test": _record(test_coords, y_test, mean_test),
        "validation": (
            None
            if y_validation is None
            else _record(validation_coords, y_validation, mean_validation)
        ),
        "has_validation": has_protocol,
    }
    learned = None
    if learn_xlag_mean:
        beta_update_count = min(int(update_spatial_count), train_coords.shape[0])
        beta_update_idx = np.linspace(
            0, train_coords.shape[0] - 1, beta_update_count
        ).round().astype(int)
        learned = {
            "beta": beta,
            "phi_train": phi_train,
            "phi_test": phi_test,
            "beta_update_idx": beta_update_idx,
            "beta_updates": [],
            "y_train": y_train,
            "y_test": y_test,
        }
        learned["test_record"] = splits["test"]
    return splits, learned


def make_phase_data(model_name, times, observed, masked_records=()):
    """Build the training grid; ST-VGP retains masked locations in its state."""
    sparse = model_name in ("st_svgp", "mf_st_svgp", "st_dsvgp")
    masked_records = list(masked_records)
    slices = {}
    if sparse:
        model_coords = observed["coords"]
        model_y = observed["model_y"]
    else:
        coord_parts = [observed["coords"]]
        y_parts = [observed["model_y"]]
        start = observed["coords"].shape[0]
        for name, record in masked_records:
            coord_parts.append(record["coords"])
            y_parts.append(np.full_like(record["model_y"], np.nan))
            stop = start + record["coords"].shape[0]
            slices[name] = slice(start, stop)
            start = stop
        model_coords = np.vstack(coord_parts)
        model_y = np.concatenate(y_parts, axis=1)
    t_train, r_train, y_train_grid = tiled_grid(times, model_coords, model_y)
    return {
        "model_name": model_name,
        "sparse": sparse,
        "observed": observed,
        "model_coords": model_coords,
        "slices": slices,
        "t_train": t_train,
        "r_train": r_train,
        "y_train_grid": y_train_grid,
    }


def gaussian_metrics(y_true, mean, variance):
    y_true = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-10)
    error = y_true - mean
    half = 1.6448536269514722 * np.sqrt(variance)
    return {
        "rmse": float(np.sqrt(np.nanmean(error ** 2))),
        "nll": float(np.nanmean(0.5 * (np.log(2.0 * np.pi * variance) + error ** 2 / variance))),
        "coverage90": float(np.nanmean((y_true >= mean - half) & (y_true <= mean + half))),
        "mean_predictive_std": float(np.nanmean(np.sqrt(variance))),
    }


def validation_checkpoint(iteration, iterations, validation_every):
    return iteration % max(int(validation_every), 1) == 0 or iteration == iterations


def taskwise_metrics(y_true, mean, variance):
    """Return the fixed Task 2--10 slices for the 1674-time long stream."""
    y_true = np.asarray(y_true)
    if y_true.ndim < 1 or y_true.shape[0] != 1674:
        return []
    rows = []
    for task_offset in range(9):
        start = task_offset * 186
        stop = start + 186
        row = gaussian_metrics(y_true[start:stop], mean[start:stop], variance[start:stop])
        row.update({"task": task_offset + 2, "start": start, "stop": stop})
        rows.append(row)
    return rows


def _predict_record(model, phase, times, record, name):
    t_test, r_test, _ = tiled_grid(times, record["coords"], record["y"])
    prediction_started = time.perf_counter()
    current_mean, current_var = model.predict_y(X=t_test, R=r_test)
    if phase["sparse"]:
        current_mean = np.asarray(current_mean).reshape(record["y"].shape)
        current_var = np.asarray(current_var).reshape(record["y"].shape)
    else:
        full_shape = (times.size, phase["model_coords"].shape[0])
        current_mean = np.asarray(current_mean).reshape(full_shape)[:, phase["slices"][name]]
        current_var = np.asarray(current_var).reshape(full_shape)[:, phase["slices"][name]]
    current_mean = current_mean + record["offset"]
    current_var = np.maximum(current_var, 1e-10)
    return (
        current_mean,
        current_var,
        gaussian_metrics(record["y"], current_mean, current_var),
        time.perf_counter() - prediction_started,
    )


def _build_phase(args, times, phase, auxiliary):
    import_backend()
    if phase["sparse"]:
        num_z = min(args.num_spatial_inducing, phase["observed"]["coords"].shape[0])
        shared_z_key = "inducing_coords_ms%d" % args.num_spatial_inducing
        if shared_z_key in auxiliary:
            z = np.asarray(auxiliary[shared_z_key], dtype=np.float64).copy()
            if z.shape != (num_z, phase["observed"]["coords"].shape[1]):
                raise ValueError("%s has incompatible shape %r" % (shared_z_key, z.shape))
        else:
            z = kmeans2(
                phase["observed"]["coords"], num_z, minit="points", seed=args.seed
            )[0]
    else:
        z = phase["model_coords"].copy()

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
        sparse=phase["sparse"],
        opt_z=phase["sparse"] and not args.fixed_spatial_inducing,
        conditional="Full",
    )
    likelihood = bayesnewton.likelihoods.Gaussian(variance=args.likelihood_variance)
    temporal_sparse = args.model == "st_dsvgp"
    if temporal_sparse:
        if args.num_temporal_inducing is None or args.num_temporal_inducing < 2:
            raise ValueError("st_dsvgp requires --num-temporal-inducing >= 2")
        temporal_z = np.linspace(
            float(times.min()), float(times.max()), args.num_temporal_inducing
        )[:, None]
        model = bayesnewton.models.SparseMarkovVariationalGP(
            kernel=kernel,
            likelihood=likelihood,
            X=phase["t_train"],
            R=phase["r_train"],
            Y=phase["y_train_grid"],
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
            X=phase["t_train"],
            R=phase["r_train"],
            Y=phase["y_train_grid"],
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
    return {
        "model": model,
        "train_op": train_op,
        "kernel": kernel,
        "kern_time": kern_time,
        "kern_space0": kern_space0,
        "kern_space1": kern_space1,
        "likelihood": likelihood,
        "z": z,
        "temporal_z": temporal_z,
        "phase": phase,
    }


def _update_learned_xlag(args, built, phase, state, iteration):
    if iteration <= 1 or (iteration - 1) % max(args.xlag_update_every, 1) != 0:
        return 0.0
    started = time.perf_counter()
    phi_train = state["phi_train"]
    beta_update_idx = state["beta_update_idx"]
    precision = args.xlag_ridge * np.eye(phi_train.shape[-1])
    rhs = np.zeros(phi_train.shape[-1])
    chunk_size = max(args.xlag_update_time_chunk_size, 1)
    for start in range(0, phi_train.shape[0], chunk_size):
        time_idx = np.arange(start, min(start + chunk_size, phi_train.shape[0]))
        residual_chunk, _ = built["model"].conditional_posterior_to_data(batch_ind=time_idx)
        residual_chunk = np.asarray(residual_chunk).reshape(
            time_idx.size, phase["observed"]["coords"].shape[0]
        )
        target_chunk = state["y_train"][np.ix_(time_idx, beta_update_idx)] - residual_chunk[:, beta_update_idx]
        design_chunk = phi_train[np.ix_(time_idx, beta_update_idx)].reshape(
            (-1, phi_train.shape[-1])
        )
        precision += np.dot(design_chunk.T, design_chunk)
        rhs += np.dot(design_chunk.T, target_chunk.reshape(-1))
    beta_new = np.linalg.solve(precision, rhs)
    state["beta"] = (
        args.xlag_update_damping * beta_new
        + (1.0 - args.xlag_update_damping) * state["beta"]
    )
    mean_train = np.dot(
        phi_train.reshape((-1, phi_train.shape[-1])), state["beta"]
    ).reshape(state["y_train"].shape)
    mean_test = np.dot(
        state["phi_test"].reshape((-1, state["phi_test"].shape[-1])), state["beta"]
    ).reshape(state["y_test"].shape)
    phase["observed"]["offset"] = mean_train
    phase["observed"]["model_y"] = state["y_train"] - mean_train
    phase["y_train_grid"] = phase["observed"]["model_y"][:, :, None]
    state["test_record"]["offset"] = mean_test
    state["test_record"]["model_y"] = state["y_test"] - mean_test
    state["beta_updates"].append(
        {
            "iteration": iteration,
            "beta_norm": float(np.linalg.norm(state["beta"])),
            "mean_rmse": float(np.sqrt(np.mean((state["y_train"] - mean_train) ** 2))),
        }
    )
    return time.perf_counter() - started


def _train_phase(args, built, times, iterations, validation_record=None, learned_state=None, trajectory_record=None):
    if int(iterations) < 1:
        raise ValueError("iterations must be >= 1")
    if validation_record is not None and trajectory_record is not None:
        raise ValueError("test diagnostics are not allowed during validation selection")
    losses = []
    trace = []
    trajectory = []
    cumulative_update_seconds = 0.0
    iteration_times = []
    best_nll = float("inf")
    best_iteration = None
    best_elapsed = None
    stable_iterations = 0
    stop_reason = "maximum_iterations"
    started = time.perf_counter()
    for iteration in range(1, int(iterations) + 1):
        update_seconds = 0.0
        if learned_state is not None:
            update_seconds = _update_learned_xlag(
                args, built, built["phase"], learned_state, iteration
            )
        update_started = time.perf_counter()
        loss = built["train_op"](built["phase"]["y_train_grid"])
        iteration_seconds = time.perf_counter() - update_started
        cumulative_update_seconds += update_seconds + iteration_seconds
        loss_value = float(np.asarray(loss[0]))
        if losses and args.early_stop_relative_tol is not None:
            relative_change = abs(loss_value - losses[-1]) / max(abs(losses[-1]), 1.0)
            stable_iterations = stable_iterations + 1 if relative_change < args.early_stop_relative_tol else 0
        losses.append(loss_value)
        iteration_times.append(iteration_seconds)
        row = {
            "iteration": iteration,
            "training_loss": loss_value,
            "iteration_seconds": iteration_seconds,
        }
        if validation_record is not None and validation_checkpoint(
            iteration, int(iterations), args.validation_every
        ):
            _, _, current, prediction_seconds = _predict_record(
                built["model"], built["phase"], times, validation_record, "validation"
            )
            row.update(
                {
                    "validation_nll": current["nll"],
                    "validation_rmse": current["rmse"],
                    "validation_prediction_seconds": prediction_seconds,
                }
            )
            if current["nll"] < best_nll:
                best_nll = current["nll"]
                best_iteration = iteration
                best_elapsed = time.perf_counter() - started
        trace.append(row)
        if trajectory_record is not None and args.trajectory_every > 0 and (
            iteration == 1 or iteration % args.trajectory_every == 0 or iteration == iterations
        ):
            _, _, current, prediction_seconds = _predict_record(
                built["model"], built["phase"], times, trajectory_record, "test"
            )
            trajectory.append(
                {
                    "iteration": iteration,
                    "cumulative_update_seconds": cumulative_update_seconds,
                    "diagnostic_elapsed_seconds": time.perf_counter() - started,
                    "prediction_seconds": prediction_seconds,
                    "test_rmse": current["rmse"],
                    "test_nll": current["nll"],
                }
            )
        if iteration == 1 or iteration == iterations or iteration % max(args.log_every, 1) == 0:
            print("iter %d: energy: %.8f" % (iteration, loss_value), flush=True)
        if (
            args.early_stop_relative_tol is not None
            and iteration >= args.early_stop_min_iterations
            and stable_iterations >= args.early_stop_patience
            and (validation_record is None or validation_checkpoint(iteration, int(iterations), args.validation_every))
        ):
            stop_reason = "relative_energy_convergence"
            print(
                "early stop at iter %d after %d stable relative-energy updates"
                % (iteration, stable_iterations),
                flush=True,
            )
            break
    if validation_record is not None and best_iteration is None:
        raise RuntimeError("validation protocol produced no validation checkpoint")
    if best_iteration is None:
        best_iteration = len(losses)
    return {
        "trace": trace,
        "validation_trace": [row for row in trace if "validation_nll" in row],
        "best_iteration": int(best_iteration),
        "best_step": int(best_iteration),
        "best_validation_nll": None if validation_record is None else float(best_nll),
        "time_to_best_validation_seconds": best_elapsed,
        "training_seconds": time.perf_counter() - started,
        "mean_iteration_seconds": float(np.mean(iteration_times)),
        "median_iteration_seconds": float(np.median(iteration_times)),
        "first_iteration_seconds": float(iteration_times[0]),
        "mean_steady_state_iteration_seconds": float(np.mean(iteration_times[1:] or iteration_times)),
        "iterations": len(losses),
        "losses": losses,
        "cumulative_update_seconds": cumulative_update_seconds,
        "trajectory": trajectory,
        "stop_reason": stop_reason,
    }


def _write_json(path, payload):
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


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
    parser.add_argument("--validation-every", type=int, default=10)
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

    if args.validation_every < 1:
        raise ValueError("--validation-every must be >= 1")
    process_started = time.perf_counter()
    np.random.seed(args.seed)
    values, data_source, auxiliary = load_data(args.data_npz, args.seed)
    times, train_coords, test_coords, y_train, y_test = values
    if args.model == "st_vgp" and args.learn_xlag_mean:
        raise ValueError("Learned X-lag mean is currently implemented for ST-SVGP paths")
    splits, learned_state = prepare_splits(
        values,
        auxiliary,
        use_xlag_mean=args.use_xlag_mean,
        learn_xlag_mean=args.learn_xlag_mean,
        ridge=args.xlag_ridge,
        update_spatial_count=args.xlag_update_spatial_count,
    )

    if splits["has_validation"]:
        selection_observed = splits["fit"]
        selection_masks = [("validation", splits["validation"]), ("test", splits["test"])]
    else:
        selection_observed = splits["train"]
        selection_masks = [("test", splits["test"])]
    selection_phase = make_phase_data(args.model, times, selection_observed, selection_masks)
    np.random.seed(args.seed)
    selection_built = _build_phase(args, times, selection_phase, auxiliary)

    if args.hlo_cost_output is not None:
        if not args.jit:
            raise ValueError("--hlo-cost-output requires --jit")
        from jaxlib import xla_extension

        computation = jax.xla_computation(selection_built["train_op"]._call)(
            selection_built["train_op"].vc.tensors(),
            {},
            jnp.asarray(selection_phase["y_train_grid"]),
        )
        cost = xla_extension.hlo_module_cost_analysis(
            jax.lib.xla_bridge.get_backend(), computation.as_hlo_module()
        )
        payload = {
            "scope": "one_jitted_train_op",
            "includes": "Bayes-Newton inference, energy gradient, and Objax Adam update",
            "model": args.model,
            "num_time_steps": int(times.size),
            "num_train_space": int(selection_observed["coords"].shape[0]),
            "num_spatial_inducing": int(selection_built["z"].shape[0]),
            "uses_xlag": bool(args.use_xlag_mean or args.learn_xlag_mean),
            "validation_protocol": bool(splits["has_validation"]),
            "note": "The alternating closed-form X-lag beta update is outside train_op and is therefore excluded.",
            "xla_cost_analysis": {key: float(value) for key, value in cost.items()},
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
        }
        _write_json(args.hlo_cost_output, payload)
        print(json.dumps(payload, indent=2))
        return

    validation_record = splits["validation"] if splits["has_validation"] else None
    trajectory_record = None if splits["has_validation"] else splits["test"]
    selection_result = _train_phase(
        args,
        selection_built,
        times,
        args.iterations,
        validation_record=validation_record,
        learned_state=learned_state,
        trajectory_record=trajectory_record,
    )

    if splits["has_validation"]:
        # Build a new model with the same deterministic protocol inducing points.
        refit_phase = make_phase_data(
            args.model, times, splits["train"], [("test", splits["test"]) ]
        )
        np.random.seed(args.seed)
        refit_built = _build_phase(args, times, refit_phase, auxiliary)
        refit_result = _train_phase(
            args,
            refit_built,
            times,
            selection_result["best_iteration"],
        )
        final_built = refit_built
        final_phase = refit_phase
        final_result = refit_result
    else:
        refit_result = None
        final_built = selection_built
        final_phase = selection_phase
        final_result = selection_result

    test_mean, test_var, final_metrics, prediction_seconds = _predict_record(
        final_built["model"], final_phase, times, splits["test"], "test"
    )
    task_metrics = taskwise_metrics(splits["test"]["y"], test_mean, test_var)
    selection_seconds = selection_result["training_seconds"]
    refit_seconds = 0.0 if refit_result is None else refit_result["training_seconds"]
    total_training_seconds = selection_seconds + refit_seconds
    final_kernel = final_built["kernel"]
    final_kern_time = final_built["kern_time"]
    final_kern_space0 = final_built["kern_space0"]
    final_kern_space1 = final_built["kern_space1"]
    final_likelihood = final_built["likelihood"]
    sparse = final_phase["sparse"]
    temporal_sparse = args.model == "st_dsvgp"
    if splits["has_validation"]:
        masking = (
            "validation/test NaN on full ST-VGP grid during selection; "
            "test NaN on full grid during refit"
            if not sparse
            else "validation/test coordinates excluded during selection; test coordinates excluded during refit"
        )
    else:
        masking = "NaN on full grid" if not sparse else "coordinates excluded from training grid"
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
            else ("shared fixed batch ridge mean from protocol NPZ" if args.use_xlag_mean and splits["has_validation"] else ("shared fixed ridge mean from data NPZ" if args.use_xlag_mean else None))
        ),
        "xlag_beta": None if learned_state is None else np.asarray(learned_state["beta"]).tolist(),
        "xlag_beta_updates": [] if learned_state is None else learned_state["beta_updates"],
        "xlag_ridge": args.xlag_ridge if args.learn_xlag_mean else None,
        "xlag_update_spatial_count": None if learned_state is None else int(learned_state["beta_update_idx"].size),
        "xlag_update_time_chunk_size": args.xlag_update_time_chunk_size if args.learn_xlag_mean else None,
        "sparse_space": sparse,
        "mean_field": args.model == "mf_st_svgp",
        "validation_protocol": bool(splits["has_validation"]),
        "num_fit_space": int(splits["fit"]["coords"].shape[0]),
        "num_validation_space": 0 if splits["validation"] is None else int(splits["validation"]["coords"].shape[0]),
        "num_train_space": int(splits["train"]["coords"].shape[0]),
        "num_test_space": int(splits["test"]["coords"].shape[0]),
        "num_model_space": int(final_phase["model_coords"].shape[0]),
        "num_selection_model_space": int(selection_phase["model_coords"].shape[0]),
        "num_refit_model_space": int(final_phase["model_coords"].shape[0]),
        "heldout_target_masking": masking,
        "num_spatial_inducing": int(final_built["z"].shape[0]),
        "spatial_inducing_optimized": bool(sparse and not args.fixed_spatial_inducing),
        "num_time_steps": int(times.size),
        "temporal_inducing_count": int(args.num_temporal_inducing) if temporal_sparse else None,
        "temporal_state_count": int(args.num_temporal_inducing) if temporal_sparse else int(times.size),
        "temporal_inducing_locations": np.asarray(final_built["temporal_z"]).reshape(-1).tolist() if temporal_sparse else None,
        "paper_faithful_temporal_representation": not temporal_sparse,
        "temporal_protocol": "sparse Markov inducing states with pairwise transitions" if temporal_sparse else "full Markov state at every observed time",
        "iterations": int(final_result["iterations"]),
        "iterations_requested": args.iterations,
        "best_iteration": int(selection_result["best_iteration"]) if splits["has_validation"] else None,
        "best_step": int(selection_result["best_iteration"]) if splits["has_validation"] else None,
        "best_validation_nll": selection_result["best_validation_nll"],
        "validation_every": args.validation_every,
        "stop_reason": final_result["stop_reason"],
        "early_stop_relative_tol": args.early_stop_relative_tol,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_iterations": args.early_stop_min_iterations,
        "jit": bool(args.jit),
        "parallel": bool(args.parallel),
        "seed": args.seed,
        "rmse": final_metrics["rmse"],
        "nll": final_metrics["nll"],
        "coverage90": final_metrics["coverage90"],
        "mean_predictive_std": final_metrics["mean_predictive_std"],
        "train_seconds": total_training_seconds,
        "cumulative_update_seconds": selection_result["cumulative_update_seconds"] + final_result["cumulative_update_seconds"] if refit_result is not None else final_result["cumulative_update_seconds"],
        "diagnostic_trajectory": selection_result["trajectory"],
        "losses": final_result["losses"],
        "selection": selection_result,
        "refit": refit_result,
        "timing": {
            "selection_training_seconds": selection_seconds,
            "refit_training_seconds": refit_seconds,
            "end_to_end_training_seconds": total_training_seconds,
            "prediction_seconds": prediction_seconds,
            "process_total_seconds": 0.0,
        },
        "taskwise_metrics": task_metrics,
        "learned_temporal_lengthscale": float(np.asarray(final_kern_time.lengthscale)),
        "learned_temporal_variance": float(np.asarray(final_kern_time.variance)),
        "learned_spatial_lengthscales": [
            float(np.asarray(final_kern_space0.lengthscale)),
            float(np.asarray(final_kern_space1.lengthscale)),
        ],
        "learned_spatial_variances": [
            float(np.asarray(final_kern_space0.variance)),
            float(np.asarray(final_kern_space1.variance)),
        ],
        "learned_likelihood_variance": float(np.asarray(final_likelihood.variance)),
        "learned_spatial_inducing_locations": np.asarray(final_kernel.z.value).tolist(),
        "data_source": data_source,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "objax": objax.__version__,
        "bayesnewton": getattr(bayesnewton, "__version__", "1.1"),
    }
    result["timing"]["process_total_seconds"] = time.perf_counter() - process_started
    _write_json(args.output, result)
    if args.trajectory_output is not None:
        _write_json(args.trajectory_output, selection_result["trajectory"])
    if args.predictions_output is not None:
        prediction_dir = os.path.dirname(os.path.abspath(args.predictions_output))
        if prediction_dir and not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        np.savez_compressed(
            args.predictions_output,
            y_true=splits["test"]["y"],
            pred_mean=test_mean,
            pred_residual_mean=test_mean - splits["test"]["offset"],
            pred_var=test_var,
            xlag_mean=splits["test"]["offset"],
            test_coords=splits["test"]["coords"],
            test_indices=np.asarray(auxiliary.get("test_indices", np.arange(test_coords.shape[0])), dtype=int),
            times=times,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
