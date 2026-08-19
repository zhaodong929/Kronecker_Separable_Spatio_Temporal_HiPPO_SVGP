#!/usr/bin/env python3
"""COVID strict-online OHSVGP baseline with its own Task-1 calibration.

The upstream multidimensional OHSVGP has one HiPPO inducing state of size M,
not separate Kronecker temporal and spatial inducing systems.  This runner
therefore uses M=32, matching Route B's temporal state size but not claiming a
nonexistent OHSVGP ``M_s`` parameter.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "baselines/external/harrisonzhu508_HIPPOSVGP"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from hipposvgp.hippo import HiPPO_LegS  # noqa: E402
from hipposvgp.likelihood import GaussianLikelihood  # noqa: E402
from hipposvgp.multidim import HIPPOOSVGP, SE_kernel  # noqa: E402
from baselines.covid_long_setting_b.archive import PredictionArchive  # noqa: E402
from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol  # noqa: E402
from scripts.run_epidemiology_pilot import predictive_metrics  # noqa: E402
from stvgp_kronecker.benchmark_runtime import host_snapshot, resolve_torch_runtime  # noqa: E402
from stvgp_kronecker.temporal_kernel_config import load_spectral_mixture_config  # noqa: E402


class TemporalQ2SpectralMixtureKernel(nn.Module):
    """OHSVGP-compatible RFF kernel: Q=2 in time and RBF in space.

    The mixture shape is fixed before calibration.  The OHSVGP calibration
    still learns the three ARD lengthscales, signal variance, likelihood noise,
    and variational state, just as in the RBF condition.
    """

    def __init__(
        self,
        *,
        lengthscales: tuple[float, float, float],
        variance: float,
        rff_sample_size: int,
        mixture: dict[str, tuple[float, ...]],
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if len(mixture["weights"]) != 2:
            raise ValueError("This OHSVGP extension implements exactly Q=2")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        weights = torch.as_tensor(mixture["weights"], dtype=dtype)
        components = torch.multinomial(
            weights / weights.sum(), int(rff_sample_size), replacement=True, generator=generator
        )
        self.register_buffer("components", components.to(device=device))
        self.register_buffer(
            "temporal_standard_normal",
            torch.randn(int(rff_sample_size), generator=generator, dtype=dtype).to(device=device),
        )
        self.register_buffer(
            "spatial_standard_normal",
            torch.randn(int(rff_sample_size), 2, generator=generator, dtype=dtype).to(device=device),
        )
        self.register_buffer("means", torch.as_tensor(mixture["means"], dtype=dtype, device=device))
        self.register_buffer("scales", torch.as_tensor(mixture["scales"], dtype=dtype, device=device))
        self.log_ls = nn.Parameter(torch.log(torch.as_tensor(lengthscales, dtype=dtype, device=device)))
        self.log_sf = nn.Parameter(torch.log(torch.as_tensor([variance], dtype=dtype, device=device)))
        self.rff_sample_size = int(rff_sample_size)
        self.device = device

    def sample_from_spectral(self, num_rff: int) -> torch.Tensor:
        if int(num_rff) != self.rff_sample_size:
            raise ValueError("OHSVGP Q=2 kernel uses one fixed RFF sample size")
        temporal = self.means[self.components] + self.scales[self.components] * self.temporal_standard_normal
        frequency = torch.cat([temporal[:, None], self.spatial_standard_normal], dim=1)
        return frequency / torch.exp(self.log_ls)[None, :]


def flatten_inputs(times: np.ndarray, coordinates: np.ndarray, indices: np.ndarray, block: slice) -> np.ndarray:
    selected_times = np.asarray(times[block], dtype=np.float64)
    selected_space = np.asarray(coordinates[indices], dtype=np.float64)
    return np.column_stack(
        [
            np.repeat(selected_times, selected_space.shape[0]),
            np.tile(selected_space[:, 0], selected_times.shape[0]),
            np.tile(selected_space[:, 1], selected_times.shape[0]),
        ]
    )


def flatten_values(values: np.ndarray, indices: np.ndarray, block: slice) -> np.ndarray:
    return np.asarray(values[block][:, indices], dtype=np.float64).reshape(-1, 1)


def ridge_mean(phi: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    design = np.asarray(phi[:, indices], dtype=np.float64).reshape(-1, phi.shape[-1])
    target = np.asarray(y[:, indices], dtype=np.float64).reshape(-1)
    beta = np.linalg.solve(design.T @ design + 1e-3 * np.eye(design.shape[1]), design.T @ target)
    return np.einsum("tsp,p->ts", phi, beta), beta


def sorted_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((x[:, 2], x[:, 1], x[:, 0]))
    return x[order], y[order]


def make_kernel(args: argparse.Namespace, runtime, mixture: dict[str, tuple[float, ...]] | None) -> nn.Module:
    initial_lengthscales = (args.initial_ell_t, *args.initial_ell_s)
    if args.kernel == "rbf":
        kernel = SE_kernel(3, device=runtime.device).to(device=runtime.device, dtype=runtime.dtype)
        with torch.no_grad():
            kernel.log_ls.copy_(torch.log(torch.as_tensor(initial_lengthscales, dtype=runtime.dtype, device=runtime.device)))
            kernel.log_sf.copy_(torch.log(torch.as_tensor([args.initial_kernel_variance], dtype=runtime.dtype, device=runtime.device)))
        return kernel
    assert mixture is not None
    return TemporalQ2SpectralMixtureKernel(
        lengthscales=initial_lengthscales,
        variance=args.initial_kernel_variance,
        rff_sample_size=args.rff_sample_size,
        mixture=mixture,
        seed=args.seed,
        device=runtime.device,
        dtype=runtime.dtype,
    )


def make_model(
    *,
    kernel: nn.Module,
    likelihood: GaussianLikelihood,
    z_interpolate: np.ndarray,
    rff_sample_size: int,
    previous_steps: int,
    hippo: HiPPO_LegS,
    inducing_size: int,
    old_state: dict[str, torch.Tensor] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> HIPPOOSVGP:
    old_kwargs = {}
    if old_state is not None:
        old_kwargs = {
            "num_inducing_old": inducing_size,
            "mv_old": old_state["mv"],
            "Lv_old": old_state["Lv"],
            "Kaa_old": old_state["Kaa"],
            "Z_old": old_state["Z"],
        }
    model = HIPPOOSVGP(
        kernel=deepcopy(kernel),
        likelihood=deepcopy(likelihood),
        Z_interpolate=torch.as_tensor(z_interpolate, dtype=dtype, device=device),
        rff_sample_size=rff_sample_size,
        prev_discrete_steps=previous_steps,
        hippo=hippo,
        inducing_size=inducing_size,
        device=device,
        flag_update_kernel=False,
        **old_kwargs,
    ).to(device=device, dtype=dtype)
    if old_state is not None:
        model.mv = nn.Parameter(old_state["mv"].clone())
        model.Lv = nn.Parameter(old_state["Lv"].clone())
    return model


def export_state(model: HIPPOOSVGP, frequencies: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        z, _, kuu, _, _ = model.Kuu_se(frequencies)
        kuu = torch.exp(model.kernel.log_sf) * kuu
    return {
        "Z": z.detach().clone(),
        "Kaa": kuu.detach().clone(),
        "mv": model.mv.detach().clone(),
        "Lv": model.Lv.detach().clone(),
    }


def predict(model: HIPPOOSVGP, frequencies: torch.Tensor, x: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        x_tensor = torch.as_tensor(x, dtype=dtype, device=device)
        mean, latent_variance = model.pred_f(x_tensor, frequencies, full_cov=False)
        variance = latent_variance + model.likelihood.variance
    return mean.detach().cpu().numpy(), variance.detach().cpu().numpy()


def kernel_summary(model: HIPPOOSVGP, kernel_name: str, mixture: dict[str, tuple[float, ...]] | None) -> dict[str, object]:
    return {
        "kernel": kernel_name,
        "ell": [float(value) for value in torch.exp(model.kernel.log_ls).detach().cpu()],
        "kernel_variance": float(torch.exp(model.kernel.log_sf).detach().cpu()),
        "noise_std": float(torch.sqrt(model.likelihood.variance).detach().cpu()),
        "fixed_spectral_mixture": None if mixture is None else {key: list(values) for key, values in mixture.items()},
    }


def clamp_hyperparameters(model: HIPPOOSVGP) -> None:
    with torch.no_grad():
        model.kernel.log_ls[0].clamp_(math.log(0.003), math.log(2.0))
        model.kernel.log_ls[1:].clamp_(math.log(0.02), math.log(5.0))
        model.kernel.log_sf.clamp_(math.log(0.005), math.log(10.0))
        model.likelihood.log_variance.clamp_(math.log(0.01**2), math.log(1.0**2))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel", choices=["rbf", "spectral_mixture_q2"], required=True)
    parser.add_argument("--spectral-mixture-json", type=Path)
    parser.add_argument("--inducing-size", type=int, default=32)
    parser.add_argument("--rff-sample-size", type=int, default=64)
    parser.add_argument("--calibration-iterations", type=int, default=50000)
    parser.add_argument("--calibration-batch-size", type=int, default=128)
    parser.add_argument("--task1-check-interval", type=int, default=250)
    parser.add_argument("--task1-min-steps", type=int, default=2500)
    parser.add_argument("--task1-plateau-checks", type=int, default=10)
    parser.add_argument("--task1-plateau-relative-improvement", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--update-steps", type=int, default=1)
    parser.add_argument("--initial-ell-t", type=float, default=0.05)
    parser.add_argument("--initial-ell-s", type=float, nargs=2, default=[0.35, 0.35])
    parser.add_argument("--initial-kernel-variance", type=float, default=1.0)
    parser.add_argument("--initial-noise", type=float, default=0.1)
    parser.add_argument(
        "--delayed-observations",
        action="store_true",
        help="Absorb each scored hidden block once before the next current-block update.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()

    mixture = load_spectral_mixture_config(args.spectral_mixture_json)
    if args.kernel == "spectral_mixture_q2" and mixture is None:
        raise ValueError("--spectral-mixture-json is required for spectral_mixture_q2")
    if args.kernel == "rbf" and mixture is not None:
        raise ValueError("--spectral-mixture-json is only valid for spectral_mixture_q2")
    runtime = resolve_torch_runtime(args.device, args.dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if runtime.uses_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.perf_counter()

    protocol = COVIDSettingBProtocol(args.protocol_npz, args.protocol_json)
    arrays = np.load(args.protocol_npz)
    metadata = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    if int(metadata["split_seed"]) != args.seed:
        raise ValueError("Protocol split seed does not match --seed")
    calibration_y = np.asarray(arrays["calibration_y"], dtype=np.float64)
    stream_y = np.asarray(arrays["stream_y"], dtype=np.float64)
    calibration_phi = np.asarray(arrays["calibration_phi"], dtype=np.float64)
    stream_phi = np.asarray(arrays["stream_phi"], dtype=np.float64)
    calibration_times = np.asarray(arrays["calibration_times"], dtype=np.float64)
    stream_times = np.asarray(arrays["stream_times"], dtype=np.float64)
    coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
    fit_indices = np.asarray(arrays["fit_indices"], dtype=int)
    validation_indices = np.asarray(arrays["validation_indices"], dtype=int)
    train_indices = np.asarray(arrays["train_indices"], dtype=int)
    test_indices = np.asarray(arrays["test_indices"], dtype=int)
    blocks = tuple(slice(int(start), int(stop)) for start, stop in zip(arrays["block_start"], arrays["block_stop"]))
    if len(blocks) != protocol.online_weeks or any(block.stop - block.start != 1 for block in blocks):
        raise ValueError("The OHSVGP Setting B adapter requires one real weekly update per protocol block")

    fit_mean, fit_beta = ridge_mean(calibration_phi, calibration_y, fit_indices)
    x_fit, y_fit = sorted_xy(
        flatten_inputs(calibration_times, coordinates, fit_indices, slice(None)),
        flatten_values(calibration_y - fit_mean, fit_indices, slice(None)),
    )
    x_validation, y_validation = sorted_xy(
        flatten_inputs(calibration_times, coordinates, validation_indices, slice(None)),
        flatten_values(calibration_y - fit_mean, validation_indices, slice(None)),
    )
    validation_offset = flatten_values(fit_mean, validation_indices, slice(None))

    kernel = make_kernel(args, runtime, mixture)
    likelihood = GaussianLikelihood(args.initial_noise**2).to(device=runtime.device, dtype=runtime.dtype)
    calibration_hippo = HiPPO_LegS(args.inducing_size, runtime.device, max_length=x_fit.shape[0] + 1).to(device=runtime.device, dtype=runtime.dtype)
    calibration_model = make_model(
        kernel=kernel,
        likelihood=likelihood,
        z_interpolate=x_fit,
        rff_sample_size=args.rff_sample_size,
        previous_steps=0,
        hippo=calibration_hippo,
        inducing_size=args.inducing_size,
        old_state=None,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    optimizer = torch.optim.Adam(calibration_model.parameters(), lr=args.learning_rate)
    trace: list[dict[str, object]] = []
    num_batches = int(math.ceil(x_fit.shape[0] / args.calibration_batch_size))
    fixed_calibration_frequencies = calibration_model.kernel.sample_from_spectral(args.rff_sample_size).detach()
    validation_frequencies = fixed_calibration_frequencies
    completed_iterations = 0
    convergence_status = "max_budget_not_converged"
    losses_in_check: list[float] = []
    for iteration in range(1, args.calibration_iterations + 1):
        batch_index = (iteration - 1) % num_batches
        start = batch_index * args.calibration_batch_size
        stop = min(x_fit.shape[0], start + args.calibration_batch_size)
        x_batch = torch.as_tensor(x_fit[start:stop], dtype=runtime.dtype, device=runtime.device)
        y_batch = torch.as_tensor(y_fit[start:stop], dtype=runtime.dtype, device=runtime.device)
        optimizer.zero_grad(set_to_none=True)
        elbo, _, _ = calibration_model.ELBO(
            x_batch,
            y_batch,
            fixed_calibration_frequencies,
            recompute_k=True,
            cache_k=False,
        )
        loss = -elbo
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Task-1 OHSVGP loss at iteration {iteration}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibration_model.parameters(), 20.0)
        optimizer.step()
        clamp_hyperparameters(calibration_model)
        losses_in_check.append(float(loss.detach()))
        completed_iterations = iteration
        if iteration % args.task1_check_interval == 0 or iteration == args.calibration_iterations:
            row: dict[str, object] = {
                "iteration": iteration,
                "chunk_elbo_median": float(-np.median(losses_in_check)),
                "chunk_elbo_mean": float(-np.mean(losses_in_check)),
                **kernel_summary(calibration_model, args.kernel, mixture),
            }
            mean, variance = predict(calibration_model, validation_frequencies, x_validation, device=runtime.device, dtype=runtime.dtype)
            validation = predictive_metrics(y_validation + validation_offset, mean + validation_offset, variance)
            row.update({f"validation_{key}": value for key, value in validation.items()})
            args.output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = args.output_dir / "task1_checkpoints" / f"checkpoint_{iteration:05d}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"iteration": iteration, "model_state_dict": calibration_model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, checkpoint)
            row["checkpoint"] = str(checkpoint)
            window = int(args.task1_plateau_checks)
            if iteration >= args.task1_min_steps and len(trace) >= 2 * window - 1:
                combined_trace = trace + [row]
                earlier = float(np.median([entry["chunk_elbo_median"] for entry in combined_trace[-2 * window : -window]]))
                recent = float(np.median([entry["chunk_elbo_median"] for entry in combined_trace[-window:]]))
                plateau_change = abs(recent - earlier) / max(abs(earlier), 1e-12)
                row["moving_median_relative_change"] = plateau_change
                if plateau_change < args.task1_plateau_relative_improvement:
                    convergence_status = "converged_elbo_plateau"
                    trace.append(row)
                    break
            trace.append(row)
            losses_in_check.clear()

    full_calibration_mean, full_beta = ridge_mean(calibration_phi, calibration_y, train_indices)
    stream_mean = np.einsum("tsp,p->ts", stream_phi, full_beta)
    x_calibration, y_calibration = sorted_xy(
        flatten_inputs(calibration_times, coordinates, train_indices, slice(None)),
        flatten_values(calibration_y - full_calibration_mean, train_indices, slice(None)),
    )
    all_online_observations = sum((block.stop - block.start) * train_indices.size for block in blocks)
    if args.delayed_observations:
        all_online_observations += sum(
            (block.stop - block.start) * test_indices.size for block in blocks[:-1]
        )
    online_hippo = HiPPO_LegS(
        args.inducing_size, runtime.device, max_length=x_calibration.shape[0] + all_online_observations + 1
    ).to(device=runtime.device, dtype=runtime.dtype)
    for parameter in calibration_model.kernel.parameters():
        parameter.requires_grad_(False)
    for parameter in calibration_model.likelihood.parameters():
        parameter.requires_grad_(False)
    refit_model = make_model(
        kernel=calibration_model.kernel,
        likelihood=calibration_model.likelihood,
        z_interpolate=x_calibration,
        rff_sample_size=args.rff_sample_size,
        previous_steps=0,
        hippo=online_hippo,
        inducing_size=args.inducing_size,
        old_state=None,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    fixed_frequencies = refit_model.kernel.sample_from_spectral(args.rff_sample_size).detach()
    refit_optimizer = torch.optim.Adam([refit_model.mv, refit_model.Lv], lr=args.learning_rate)
    for iteration in range(completed_iterations):
        batch_index = iteration % int(math.ceil(x_calibration.shape[0] / args.calibration_batch_size))
        start = batch_index * args.calibration_batch_size
        stop = min(x_calibration.shape[0], start + args.calibration_batch_size)
        x_batch = torch.as_tensor(x_calibration[start:stop], dtype=runtime.dtype, device=runtime.device)
        y_batch = torch.as_tensor(y_calibration[start:stop], dtype=runtime.dtype, device=runtime.device)
        refit_optimizer.zero_grad(set_to_none=True)
        elbo, _, _ = refit_model.ELBO(x_batch, y_batch, fixed_frequencies, recompute_k=True, cache_k=False)
        loss = -elbo
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Task-1 refit loss at iteration {iteration + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([refit_model.mv, refit_model.Lv], 20.0)
        refit_optimizer.step()
    old_state = export_state(refit_model, fixed_frequencies)
    previous_steps = x_calibration.shape[0]

    rows: list[dict[str, object]] = []
    true_grid = np.empty((stream_y.shape[0], test_indices.size), dtype=np.float64)
    mean_grid = np.empty_like(true_grid)
    variance_grid = np.empty_like(true_grid)
    delayed_rows = 0
    archive = PredictionArchive(protocol, method="official_ohsvgp_multidimensional_setting_b_adapter", seed=args.seed)
    for block_id, block in enumerate(blocks):
        model = None
        update_started = time.perf_counter()
        updates = [(block, train_indices, "current_visible")]
        if args.delayed_observations and block_id > 0:
            updates.insert(0, (blocks[block_id - 1], test_indices, "delayed_hidden"))
        for observation_block, spatial_indices, update_kind in updates:
            x_train, y_train = sorted_xy(
                flatten_inputs(stream_times, coordinates, spatial_indices, observation_block),
                flatten_values(stream_y - stream_mean, spatial_indices, observation_block),
            )
            if update_kind == "delayed_hidden":
                delayed_rows += int(x_train.shape[0])
            for start in range(0, x_train.shape[0], args.calibration_batch_size):
                stop = min(x_train.shape[0], start + args.calibration_batch_size)
                model = make_model(
                    kernel=refit_model.kernel,
                    likelihood=refit_model.likelihood,
                    z_interpolate=x_train[start:stop],
                    rff_sample_size=args.rff_sample_size,
                    previous_steps=previous_steps,
                    hippo=online_hippo,
                    inducing_size=args.inducing_size,
                    old_state=old_state,
                    device=runtime.device,
                    dtype=runtime.dtype,
                )
                optimizer = torch.optim.Adam([model.mv, model.Lv], lr=args.learning_rate)
                x_batch = torch.as_tensor(x_train[start:stop], dtype=runtime.dtype, device=runtime.device)
                y_batch = torch.as_tensor(y_train[start:stop], dtype=runtime.dtype, device=runtime.device)
                for step in range(args.update_steps):
                    optimizer.zero_grad(set_to_none=True)
                    elbo, _, _ = model.ELBO(x_batch, y_batch, fixed_frequencies, recompute_k=step == 0, cache_k=step == 0)
                    loss = -elbo
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite online OHSVGP loss at block {block_id}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([model.mv, model.Lv], 20.0)
                    optimizer.step()
                old_state = export_state(model, fixed_frequencies)
                previous_steps += stop - start
        assert model is not None
        update_seconds = time.perf_counter() - update_started
        x_test = flatten_inputs(stream_times, coordinates, test_indices, block)
        y_test = flatten_values(stream_y, test_indices, block)
        test_offset = flatten_values(stream_mean, test_indices, block)
        prediction_started = time.perf_counter()
        mean, variance = predict(model, fixed_frequencies, x_test, device=runtime.device, dtype=runtime.dtype)
        prediction_seconds = time.perf_counter() - prediction_started
        mean = mean + test_offset
        metric = predictive_metrics(y_test, mean, variance)
        if not np.isfinite(variance).all() or np.any(variance <= 0.0):
            raise FloatingPointError(f"Invalid predictive variance at block {block_id}")
        true_grid[block] = y_test.reshape(block.stop - block.start, test_indices.size)
        mean_grid[block] = mean.reshape(block.stop - block.start, test_indices.size)
        variance_grid[block] = variance.reshape(block.stop - block.start, test_indices.size)
        rows.append({"block_id": block_id, "block_start": block.start, "block_stop": block.stop, "update_seconds": update_seconds, "prediction_seconds": prediction_seconds, **metric})
        archive.append(protocol.week(block_id), mean.reshape(-1), variance.reshape(-1))

    overall = predictive_metrics(true_grid, mean_grid, variance_grid)
    payload = {
        "status": "complete",
        "implementation": "official OHSVGP core adapted to multidimensional COVID Setting B",
        "source_repository": "https://github.com/harrisonzhu508/HIPPOSVGP",
        "source_commit": "a1bff1b",
        "kernel": args.kernel,
        "kernel_strategy": "own Task-1 hyperparameter learning; frozen before strict online updates",
        "capacity": {"hippo_inducing_size": args.inducing_size, "rff_sample_size": args.rff_sample_size, "spatial_inducing_size": None},
        "capacity_note": "Official multidimensional OHSVGP has one M-dimensional HiPPO state, not Route B's separate Mt and Ms Kronecker state.",
        "target_mode": "two-stage causal X-lag ridge residual; fixed mean after Task-1 refit",
        "delayed_observations": bool(args.delayed_observations),
        "delayed_observation_rows": delayed_rows,
        "split_seed": args.seed,
        "task1_convergence": {
            "status": convergence_status,
            "steps_completed": completed_iterations,
            "max_steps": int(args.calibration_iterations),
            "check_interval": int(args.task1_check_interval),
            "minimum_steps": int(args.task1_min_steps),
            "moving_median_checks": int(args.task1_plateau_checks),
            "relative_improvement_threshold": float(args.task1_plateau_relative_improvement),
            "trace": trace,
        },
        "learned_theta": kernel_summary(calibration_model, args.kernel, mixture),
        "calibration_beta": full_beta.tolist(),
        "overall_current_block": overall,
        "num_calibration_times": int(calibration_y.shape[0]),
        "num_stream_times": int(stream_y.shape[0]),
        "num_blocks": len(blocks),
        "num_fit_space": int(fit_indices.size),
        "num_validation_space": int(validation_indices.size),
        "num_train_space": int(train_indices.size),
        "num_test_space": int(test_indices.size),
        "timing": {"process_total_seconds": time.perf_counter() - started, "mean_block_update_seconds": float(np.mean([row["update_seconds"] for row in rows])), "mean_block_prediction_seconds": float(np.mean([row["prediction_seconds"] for row in rows]) )},
        "resources": runtime.resources(),
        "environment": host_snapshot(ROOT),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(trace, args.output_dir / "calibration_trace.csv")
    write_csv(rows, args.output_dir / "blocks.csv")
    audit = archive.write(
        args.output_dir / "predictions.npz",
        extra_metadata={
            "adapter": "official_ohsvgp_multidimensional_setting_b",
            "source_commit": "a1bff1b",
            "inducing_size": args.inducing_size,
            "rff_sample_size": args.rff_sample_size,
            "rff_frequencies": "fixed deterministic per seeded run",
            "task1_convergence": payload["task1_convergence"],
        },
    )
    payload["audit"] = audit
    (args.output_dir / "calibration.json").write_text(json.dumps({"task1_convergence": payload["task1_convergence"], "learned_theta": kernel_summary(calibration_model, args.kernel, mixture), "fit_beta": fit_beta.tolist()}, indent=2), encoding="utf-8")
    (args.output_dir / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
