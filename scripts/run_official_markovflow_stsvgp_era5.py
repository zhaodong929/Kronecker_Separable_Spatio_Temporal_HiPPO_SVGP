#!/usr/bin/env python3
"""ERA5 wrapper for Markovflow's official sparse spatio-temporal models.

This script is intentionally Python 3.7 compatible for Markovflow v0.0.13. It
keeps the upstream inference code unchanged and only supplies the shared ERA5
protocol, optimisation loop, metrics, and resource accounting.
"""

from __future__ import print_function

import argparse
import copy
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# The legacy TensorFlow compatibility stack is retained as a CPU baseline. It
# must not reserve or time the AutoDL GPU used by the modern methods.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
MARKOVFLOW_ROOT = (
    ROOT / "baselines" / "external" / "secondmind_labs_markovflow_v0.0.13"
)
if str(MARKOVFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(MARKOVFLOW_ROOT))

import numpy as np
import tensorflow as tf
import gpflow

from markovflow.kernels import Matern32 as MarkovflowMatern32
from markovflow.models import SpatioTemporalSparseCVI
from markovflow.models import SpatioTemporalSparseVariational
from markovflow.ssm_natgrad import SSMNaturalGradient

try:
    from scripts.era5_ncu_ranges import pop_range, profile_this_index, push_range
except ImportError:
    from era5_ncu_ranges import pop_range, profile_this_index, push_range


def flatten_inputs(times, coordinates, spatial_indices):
    selected = np.asarray(coordinates[spatial_indices], dtype=np.float64)
    time_values = np.asarray(times, dtype=np.float64)
    # Markovflow expects [space..., time] and non-decreasing time order.
    return np.column_stack(
        [
            np.tile(selected[:, 0], time_values.size),
            np.tile(selected[:, 1], time_values.size),
            np.repeat(time_values, selected.shape[0]),
        ]
    )


def flatten_targets(values, spatial_indices):
    return np.asarray(values[:, spatial_indices], dtype=np.float64).reshape(-1, 1)


def metrics(y_true, mean, variance):
    y = np.asarray(y_true).reshape(-1)
    mu = np.asarray(mean).reshape(-1)
    var = np.maximum(np.asarray(variance).reshape(-1), 1e-10)
    half = 1.6448536269514722 * np.sqrt(var)
    return {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "nll": float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var))),
        "coverage90": float(np.mean((y >= mu - half) & (y <= mu + half))),
        "mean_predictive_std": float(np.mean(np.sqrt(var))),
    }


def build_model(
    model_kind, z_time, z_space, num_data, initial, cvi_rate, temporal_jitter
):
    spatial_0 = gpflow.kernels.Matern32(
        variance=float(initial["kernel_variance"]),
        lengthscales=float(initial["ell_s"][0]),
        active_dims=[0],
    )
    spatial_1 = gpflow.kernels.Matern32(
        variance=1.0,
        lengthscales=float(initial["ell_s"][1]),
        active_dims=[1],
    )
    kernel_space = spatial_0 * spatial_1
    kernel_time = MarkovflowMatern32(
        variance=1.0,
        lengthscale=float(initial["ell_t"]),
        jitter=float(temporal_jitter),
    )
    likelihood = gpflow.likelihoods.Gaussian(
        variance=float(initial["noise_variance"])
    )
    common = dict(
        inducing_time=tf.identity(np.asarray(z_time, dtype=np.float64)),
        inducing_space=tf.identity(np.asarray(z_space, dtype=np.float64)),
        kernel_space=kernel_space,
        kernel_time=kernel_time,
        likelihood=likelihood,
        num_data=int(num_data),
    )
    if model_kind == "sparse_variational":
        model = SpatioTemporalSparseVariational(**common)
    else:
        model = SpatioTemporalSparseCVI(learning_rate=float(cvi_rate), **common)
    return model, likelihood


def predict(model, likelihood, x, chunk_size):
    means = []
    variances = []
    started = time.perf_counter()
    for start in range(0, x.shape[0], chunk_size):
        x_chunk = tf.convert_to_tensor(x[start : start + chunk_size], dtype=tf.float64)
        mean, latent_variance = model.space_time_predict_f(x_chunk)
        means.append(np.asarray(mean).reshape(-1, 1))
        variances.append(
            np.asarray(latent_variance).reshape(-1, 1)
            + float(likelihood.variance.numpy())
        )
    return (
        np.concatenate(means),
        np.concatenate(variances),
        time.perf_counter() - started,
    )


def parameter_values(module):
    return [np.array(variable.numpy(), copy=True) for variable in module.variables]


def assign_parameter_values(module, values):
    if len(module.variables) != len(values):
        raise ValueError("Checkpoint variable count changed")
    for variable, value in zip(module.variables, values):
        variable.assign(value)


def state_bytes(module):
    return int(sum(np.asarray(variable.numpy()).nbytes for variable in module.variables))


def train(
    model,
    likelihood,
    model_kind,
    x,
    y,
    validation,
    iterations,
    batch_size,
    learning_rate,
    natgrad_gamma,
    validation_every,
    seed,
    prediction_chunk_size,
    variational_optimizer,
    profile_enabled=True,
):
    rng = np.random.RandomState(seed)
    adam = tf.optimizers.Adam(float(learning_rate))
    hyper_variables = list(model.kernel.trainable_variables) + list(
        likelihood.trainable_variables
    )
    natgrad = None
    if model_kind == "sparse_variational":
        natgrad = SSMNaturalGradient(gamma=float(natgrad_gamma), momentum=False)

    def ordered_minibatch():
        count = min(int(batch_size), x.shape[0])
        indices = np.sort(rng.choice(x.shape[0], size=count, replace=False))
        return (
            tf.convert_to_tensor(x[indices], dtype=tf.float64),
            tf.convert_to_tensor(y[indices], dtype=tf.float64),
        )

    def one_step(x_batch, y_batch):
        data = (x_batch, y_batch)
        if model_kind == "sparse_cvi":
            model.update_sites(data)
            adam.minimize(lambda: model.loss(data), hyper_variables)
        elif variational_optimizer == "natgrad":
            natgrad.minimize(lambda: model.loss(data), model.dist_q)
            adam.minimize(lambda: model.loss(data), hyper_variables)
        else:
            adam.minimize(lambda: model.loss(data), model.trainable_variables)
        return model.loss(data)

    trace = []
    best_nll = float("inf")
    best_iteration = 1
    best_elapsed = None
    best_values = None
    iteration_times = []
    started = time.perf_counter()
    for iteration in range(1, int(iterations) + 1):
        x_batch, y_batch = ordered_minibatch()
        iteration_started = time.perf_counter()
        profile_range = profile_enabled and profile_this_index(iteration - 1, int(iterations))
        profile_open = push_range("era5_batch_update", profile_range)
        try:
            loss = float(np.asarray(one_step(x_batch, y_batch)))
        finally:
            pop_range(profile_open)
        iteration_seconds = time.perf_counter() - iteration_started
        iteration_times.append(iteration_seconds)
        row = {
            "iteration": iteration,
            "training_loss": loss,
            "iteration_seconds": iteration_seconds,
        }
        if validation is not None and (
            iteration == 1
            or iteration % int(validation_every) == 0
            or iteration == int(iterations)
        ):
            x_val, y_val, offset_val = validation
            mean, variance, prediction_seconds = predict(
                model, likelihood, x_val, prediction_chunk_size
            )
            current = metrics(y_val, mean + offset_val, variance)
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
                best_values = parameter_values(model)
        trace.append(row)
        if iteration == 1 or iteration % int(validation_every) == 0 or iteration == iterations:
            print(json.dumps(row), flush=True)
    if best_values is not None:
        assign_parameter_values(model, best_values)
    return {
        "trace": trace,
        "best_iteration": int(best_iteration),
        "best_validation_nll": None if validation is None else float(best_nll),
        "time_to_best_validation_seconds": best_elapsed,
        "training_seconds": time.perf_counter() - started,
        "mean_iteration_seconds": float(np.mean(iteration_times)),
        "median_iteration_seconds": float(np.median(iteration_times)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output", default=None)
    parser.add_argument(
        "--model-kind", choices=["sparse_variational", "sparse_cvi"], required=True
    )
    parser.add_argument(
        "--target-mode", choices=["direct", "shared_xlag_residual"], required=True
    )
    parser.add_argument("--mt", type=int, required=True)
    parser.add_argument("--ms", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--natgrad-gamma", type=float, default=0.1)
    parser.add_argument(
        "--variational-optimizer", choices=["natgrad", "adam"], default="adam"
    )
    parser.add_argument("--cvi-rate", type=float, default=0.1)
    parser.add_argument("--temporal-jitter", type=float, default=1e-6)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--prediction-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-times", type=int, default=0)
    args = parser.parse_args()
    if args.temporal_jitter < 0.0:
        raise ValueError("--temporal-jitter must be non-negative")

    process_started = time.perf_counter()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    gpflow.config.set_default_float(np.float64)
    arrays = np.load(args.protocol_npz)
    times = np.asarray(arrays["stream_times"], dtype=np.float64)
    y_original = np.asarray(arrays["stream_y"], dtype=np.float64)
    offset = (
        np.zeros_like(y_original)
        if args.target_mode == "direct"
        else np.asarray(arrays["batch_stream_mean"], dtype=np.float64)
    )
    if args.max_times > 0:
        times = times[: args.max_times]
        y_original = y_original[: args.max_times]
        offset = offset[: args.max_times]
    y_model = y_original - offset
    coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
    fit_indices = np.asarray(arrays["fit_indices"], dtype=int)
    validation_indices = np.asarray(arrays["validation_indices"], dtype=int)
    train_indices = np.asarray(arrays["train_indices"], dtype=int)
    test_indices = np.asarray(arrays["test_indices"], dtype=int)
    inducing_key = "inducing_coords_ms%d" % args.ms
    if inducing_key not in arrays:
        raise KeyError("Protocol does not contain %s" % inducing_key)
    z_space = np.asarray(arrays[inducing_key], dtype=np.float64)
    z_time = np.linspace(float(times.min()), float(times.max()), args.mt)
    initial = {
        "ell_t": 0.05,
        "ell_s": [0.35, 0.35],
        "kernel_variance": 1.0,
        "noise_variance": 0.01,
    }

    x_fit = flatten_inputs(times, coordinates, fit_indices)
    y_fit = flatten_targets(y_model, fit_indices)
    x_validation = flatten_inputs(times, coordinates, validation_indices)
    y_validation = flatten_targets(y_original, validation_indices)
    validation_offset = flatten_targets(offset, validation_indices)
    selection_model, selection_likelihood = build_model(
        args.model_kind,
        z_time,
        z_space,
        x_fit.shape[0],
        initial,
        args.cvi_rate,
        args.temporal_jitter,
    )
    selection = train(
        selection_model,
        selection_likelihood,
        args.model_kind,
        x_fit,
        y_fit,
        (x_validation, y_validation, validation_offset),
        args.iterations,
        args.batch_size,
        args.learning_rate,
        args.natgrad_gamma,
        args.validation_every,
        args.seed,
        args.prediction_chunk_size,
        args.variational_optimizer,
        False,
    )

    x_train = flatten_inputs(times, coordinates, train_indices)
    y_train = flatten_targets(y_model, train_indices)
    final_model, final_likelihood = build_model(
        args.model_kind,
        z_time,
        z_space,
        x_train.shape[0],
        initial,
        args.cvi_rate,
        args.temporal_jitter,
    )
    refit = train(
        final_model,
        final_likelihood,
        args.model_kind,
        x_train,
        y_train,
        None,
        selection["best_iteration"],
        args.batch_size,
        args.learning_rate,
        args.natgrad_gamma,
        args.validation_every,
        args.seed,
        args.prediction_chunk_size,
        args.variational_optimizer,
        True,
    )
    x_test = flatten_inputs(times, coordinates, test_indices)
    y_test = flatten_targets(y_original, test_indices)
    test_offset = flatten_targets(offset, test_indices)
    mean, variance, prediction_seconds = predict(
        final_model, final_likelihood, x_test, args.prediction_chunk_size
    )
    mean += test_offset
    final = metrics(y_test, mean, variance)
    persistent_bytes = state_bytes(final_model)
    payload = {
        "implementation": "official Markovflow v0.0.13 API with thin ERA5 wrapper",
        "source_repository": "https://github.com/secondmind-labs/markovflow",
        "source_commit": "v0.0.13",
        "model": args.model_kind,
        "protocol": "controlled batch/full-history with spatial held-out validation",
        "target_mode": args.target_mode,
        "split_seed": args.seed,
        "mt": args.mt,
        "ms": args.ms,
        "num_time": int(times.size),
        "num_train_space": int(train_indices.size),
        "num_test_space": int(test_indices.size),
        "temporal_representation": "Markovflow sparse temporal state chain",
        "spatial_representation": "fixed k-means inducing locations",
        "optimizer": (
            (
                "SSMNaturalGradient for q plus Adam for kernel/noise"
                if args.variational_optimizer == "natgrad"
                else "Adam over the official model trainable variables"
            )
            if args.model_kind == "sparse_variational"
            else "official CVI site update plus Adam for kernel/noise"
        ),
        "selection": selection,
        "refit": refit,
        "final": final,
        "timing": {
            "selection_training_seconds": selection["training_seconds"],
            "refit_training_seconds": refit["training_seconds"],
            "end_to_end_training_seconds": selection["training_seconds"]
            + refit["training_seconds"],
            "prediction_seconds": prediction_seconds,
            "process_total_seconds": time.perf_counter() - process_started,
        },
        "resources": {
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "persistent_model_state_bytes": persistent_bytes,
            "persistent_model_state_mib": persistent_bytes / 1024.0 ** 2,
            "history_replay_buffer_bytes": int(y_train.nbytes),
            "device": "CPU",
            "dtype": "float64",
        },
        "versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "gpflow": gpflow.__version__,
            "numpy": np.__version__,
        },
        "args": vars(args),
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)
    if args.predictions_output is not None:
        prediction_dir = os.path.dirname(os.path.abspath(args.predictions_output))
        if prediction_dir and not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        np.savez_compressed(
            args.predictions_output,
            y_true=y_test,
            pred_mean=mean,
            pred_var=variance,
            test_indices=test_indices,
            times=times,
        )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
