#!/usr/bin/env python3
"""Thin ERA5 wrapper around the official Aalto GPflow SVGP baseline path."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gpflow
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.benchmark_runtime import (  # noqa: E402
    configure_tensorflow,
    host_snapshot,
    tensorflow_memory,
)


NP_DTYPE = np.float64


def flatten_inputs(times, coordinates, spatial_indices):
    selected = np.asarray(coordinates[spatial_indices], dtype=NP_DTYPE)
    return np.column_stack(
        [
            np.repeat(np.asarray(times, dtype=NP_DTYPE), selected.shape[0]),
            np.tile(selected[:, 0], len(times)),
            np.tile(selected[:, 1], len(times)),
        ]
    )


def flatten_targets(values, spatial_indices):
    return np.asarray(values[:, spatial_indices], dtype=NP_DTYPE).reshape(-1, 1)


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


def product_inducing(times, spatial_inducing, mt):
    temporal = np.linspace(float(np.min(times)), float(np.max(times)), int(mt))
    return np.asarray(
        [[time_value, coord[0], coord[1]] for time_value in temporal for coord in spatial_inducing],
        dtype=NP_DTYPE,
    )


def build_model(z, num_data, initial):
    kernel = None
    for dimension, lengthscale in enumerate(initial["lengthscales"]):
        component = gpflow.kernels.Matern32(
            variance=1.0,
            lengthscales=float(lengthscale),
            active_dims=[dimension],
        )
        kernel = component if kernel is None else kernel * component
    likelihood = gpflow.likelihoods.Gaussian(variance=float(initial["noise_variance"]))
    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=np.asarray(z, dtype=NP_DTYPE),
        num_latent_gps=1,
        num_data=int(num_data),
        whiten=True,
        q_diag=False,
    )
    gpflow.set_trainable(model.q_mu, False)
    gpflow.set_trainable(model.q_sqrt, False)
    gpflow.set_trainable(model.inducing_variable, True)
    return model


def state_bytes(model):
    return int(sum(np.asarray(variable.numpy()).nbytes for variable in model.variables))


def predict(model, x, chunk_size):
    means = []
    variances = []
    started = time.perf_counter()
    for start in range(0, x.shape[0], chunk_size):
        mean, variance = model.predict_y(x[start : start + chunk_size])
        means.append(np.asarray(mean))
        variances.append(np.asarray(variance))
    return np.concatenate(means), np.concatenate(variances), time.perf_counter() - started


def train(
    *,
    model,
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
):
    rng = np.random.default_rng(seed)
    natgrad = gpflow.optimizers.NaturalGradient(gamma=natgrad_gamma)
    adam = tf.optimizers.Adam(learning_rate)
    variational_parameters = [(model.q_mu, model.q_sqrt)]
    adam_variables = list(model.trainable_variables)

    @tf.function
    def step(x_batch, y_batch):
        closure = lambda: model.training_loss((x_batch, y_batch))
        natgrad.minimize(closure, var_list=variational_parameters)
        adam.minimize(closure, var_list=adam_variables)
        return closure()

    trace = []
    best_nll = float("inf")
    best_iteration = 1
    best_elapsed = None
    best_values = None
    iteration_times = []
    started = time.perf_counter()
    for iteration in range(1, int(iterations) + 1):
        indices = rng.choice(x.shape[0], size=min(batch_size, x.shape[0]), replace=False)
        iteration_started = time.perf_counter()
        loss = float(step(x[indices], y[indices]).numpy())
        iteration_seconds = time.perf_counter() - iteration_started
        iteration_times.append(iteration_seconds)
        row = {
            "iteration": iteration,
            "training_loss": loss,
            "iteration_seconds": iteration_seconds,
        }
        if validation is not None and (
            iteration == 1 or iteration % validation_every == 0 or iteration == iterations
        ):
            x_val, y_val, offset_val = validation
            mean, variance, validation_prediction_seconds = predict(
                model, x_val, prediction_chunk_size
            )
            mean = mean + offset_val
            validation_metrics = metrics(y_val, mean, variance)
            row.update(
                {
                    "validation_nll": validation_metrics["nll"],
                    "validation_rmse": validation_metrics["rmse"],
                    "validation_prediction_seconds": validation_prediction_seconds,
                }
            )
            if validation_metrics["nll"] < best_nll:
                best_nll = validation_metrics["nll"]
                best_iteration = iteration
                best_elapsed = time.perf_counter() - started
                best_values = {
                    name: np.array(parameter.numpy(), copy=True)
                    for name, parameter in gpflow.utilities.parameter_dict(model).items()
                }
        trace.append(row)
        if iteration == 1 or iteration % validation_every == 0 or iteration == iterations:
            print(json.dumps(row), flush=True)
    if best_values is not None:
        gpflow.utilities.multiple_assign(model, best_values)
    return {
        "trace": trace,
        "best_iteration": int(best_iteration),
        "best_validation_nll": float(best_nll) if validation is not None else None,
        "time_to_best_validation_seconds": best_elapsed,
        "training_seconds": time.perf_counter() - started,
        "mean_iteration_seconds": float(np.mean(iteration_times)),
        "median_iteration_seconds": float(np.median(iteration_times)),
        "first_iteration_seconds": float(iteration_times[0]),
        "mean_steady_state_iteration_seconds": float(
            np.mean(iteration_times[1:] or iteration_times)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--target-mode", choices=["direct", "shared_xlag_residual"], required=True)
    parser.add_argument("--mt", type=int, default=8)
    parser.add_argument("--ms", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--natgrad-gamma", type=float, default=0.1)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--prediction-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()

    global NP_DTYPE
    runtime = configure_tensorflow(tf, device=args.device, dtype=args.dtype)
    NP_DTYPE = np.float32 if args.dtype == "float32" else np.float64
    process_started = time.perf_counter()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    gpflow.config.set_default_float(NP_DTYPE)
    if runtime["device"].startswith("cuda"):
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except (RuntimeError, ValueError):
            pass
    data = np.load(args.protocol_npz)
    times = np.asarray(data["stream_times"], dtype=NP_DTYPE)
    coordinates = np.asarray(data["coordinates"], dtype=NP_DTYPE)
    y_original = np.asarray(data["stream_y"], dtype=NP_DTYPE)
    offset = (
        np.zeros_like(y_original)
        if args.target_mode == "direct"
        else np.asarray(data["batch_stream_mean"], dtype=NP_DTYPE)
    )
    y_model = y_original - offset
    fit_indices = np.asarray(data["fit_indices"], dtype=int)
    validation_indices = np.asarray(data["validation_indices"], dtype=int)
    train_indices = np.asarray(data["train_indices"], dtype=int)
    test_indices = np.asarray(data["test_indices"], dtype=int)
    inducing_key = f"inducing_coords_ms{args.ms}"
    if inducing_key not in data:
        raise KeyError(f"{args.protocol_npz} does not contain {inducing_key}")
    z = product_inducing(times, np.asarray(data[inducing_key]), args.mt)
    initial = {"lengthscales": [0.05, 0.35, 0.35], "noise_variance": 0.01}

    x_fit = flatten_inputs(times, coordinates, fit_indices)
    y_fit = flatten_targets(y_model, fit_indices)
    x_validation = flatten_inputs(times, coordinates, validation_indices)
    y_validation = flatten_targets(y_original, validation_indices)
    validation_offset = flatten_targets(offset, validation_indices)
    selection_model = build_model(z, x_fit.shape[0], initial)
    selection = train(
        model=selection_model,
        x=x_fit,
        y=y_fit,
        validation=(x_validation, y_validation, validation_offset),
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        natgrad_gamma=args.natgrad_gamma,
        validation_every=args.validation_every,
        seed=args.seed,
        prediction_chunk_size=args.prediction_chunk_size,
    )

    x_train = flatten_inputs(times, coordinates, train_indices)
    y_train = flatten_targets(y_model, train_indices)
    final_model = build_model(z, x_train.shape[0], initial)
    refit = train(
        model=final_model,
        x=x_train,
        y=y_train,
        validation=None,
        iterations=selection["best_iteration"],
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        natgrad_gamma=args.natgrad_gamma,
        validation_every=args.validation_every,
        seed=args.seed,
        prediction_chunk_size=args.prediction_chunk_size,
    )
    x_test = flatten_inputs(times, coordinates, test_indices)
    y_test = flatten_targets(y_original, test_indices)
    test_offset = flatten_targets(offset, test_indices)
    prediction_mean, prediction_variance, prediction_seconds = predict(
        final_model, x_test, args.prediction_chunk_size
    )
    prediction_mean += test_offset
    final_metrics = metrics(y_test, prediction_mean, prediction_variance)
    model_state_bytes = state_bytes(final_model)
    total_iterations = args.iterations + selection["best_iteration"]
    estimated_flops = float(
        total_iterations
        * (
            4.0 * args.batch_size * z.shape[0] ** 2
            + (10.0 / 3.0) * z.shape[0] ** 3
        )
    )
    payload = {
        "implementation": "Aalto GPflow SVGP training path (thin ERA5 wrapper)",
        "source_repository": "https://github.com/AaltoML/spatio-temporal-GPs",
        "source_commit": "c5b929e",
        "target_mode": args.target_mode,
        "split_seed": args.seed,
        "num_time": int(times.size),
        "num_train_space": int(train_indices.size),
        "num_validation_space": int(validation_indices.size),
        "num_test_space": int(test_indices.size),
        "temporal_inducing": args.mt,
        "spatial_inducing": args.ms,
        "joint_inducing": int(z.shape[0]),
        "inducing_initialization": "Cartesian product of global linspace time and protocol k-means spatial points",
        "inducing_trainable": True,
        "kernel": "product of three one-dimensional Matern-3/2 kernels",
        "optimizer": "GPflow NaturalGradient(q) plus Adam(kernel, noise, inducing locations)",
        "selection": selection,
        "refit": refit,
        "final": final_metrics,
        "timing": {
            "selection_training_seconds": selection["training_seconds"],
            "refit_training_seconds": refit["training_seconds"],
            "end_to_end_training_seconds": selection["training_seconds"] + refit["training_seconds"],
            "prediction_seconds": prediction_seconds,
            "process_total_seconds": time.perf_counter() - process_started,
        },
        "resources": {
            **tensorflow_memory(tf, runtime["device"]),
            "persistent_model_state_bytes": model_state_bytes,
            "persistent_model_state_mib": model_state_bytes / 1024.0**2,
            "history_replay_buffer_bytes": 0,
            "estimated_training_flops": estimated_flops,
            "flops_scope": "dominant dense MxM minibatch solves and natural-gradient algebra; analytic estimate",
            "device": runtime["device"],
            "dtype": runtime["dtype"],
            "physical_gpus": runtime["physical_gpus"],
        },
        "environment": host_snapshot(ROOT),
        "versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "gpflow": gpflow.__version__,
            "numpy": np.__version__,
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions_output,
            y_true=y_test.reshape(times.size, test_indices.size),
            pred_mean=prediction_mean.reshape(times.size, test_indices.size),
            pred_var=prediction_variance.reshape(times.size, test_indices.size),
            test_indices=test_indices,
            times=times,
        )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
