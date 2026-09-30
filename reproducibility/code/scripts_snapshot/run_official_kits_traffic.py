#!/usr/bin/env python3
"""Task-1-only adapter for the pinned official KITS model core."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "baselines/external/Sam1224_KITS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from lib.nn.models.kits import KITS  # noqa: E402
from stvgp_kronecker.data.traffic import load_spatial_split, load_traffic_dataset  # noqa: E402
from stvgp_kronecker.traffic_spatial_kernels import load_road_laplacian  # noqa: E402


def make_model(adjacency: np.ndarray, hidden_dim: int, device: torch.device) -> KITS:
    official_args = SimpleNamespace(dataset_name="bay_point", use_adj_drop=False, use_init=False)
    return KITS(adjacency.astype(np.float32), 1, hidden_dim, official_args).to(device)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=prediction.dtype)
    return torch.sum((prediction - target).square() * weights) / torch.clamp(weights.sum(), min=1.0)


def evaluate(
    model: KITS,
    values: np.ndarray,
    observed: np.ndarray,
    targets: np.ndarray,
    *,
    window: int,
    stride: int,
    device: torch.device,
) -> float:
    mask = np.zeros(values.shape[1], dtype=bool)
    mask[observed] = True
    errors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for stop in range(window, values.shape[0] + 1, stride):
            block = values[stop - window : stop].copy()
            input_mask = np.broadcast_to(mask[None, :, None], (window, values.shape[1], 1)).copy()
            block[~mask[None, :].repeat(window, axis=0)] = 0.0
            prediction = model(
                torch.as_tensor(block[None, :, :, None], dtype=torch.float32, device=device),
                torch.as_tensor(input_mask[None], dtype=torch.uint8, device=device),
            )
            estimate = prediction[0, -1, targets, 0].detach().cpu().numpy()
            errors.append(estimate - values[stop - 1, targets])
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def train_task1(
    model: KITS,
    values: np.ndarray,
    calibration: np.ndarray,
    validation: np.ndarray,
    *,
    window: int,
    epochs: int,
    samples_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    whiten_probability: float,
    validation_every: int,
    validation_stride: int,
    patience_checks: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = deepcopy(model.state_dict())
    best_rmse = float("inf")
    checks_without_improvement = 0
    trace: list[dict[str, float | int]] = []
    batches = max(1, samples_per_epoch // batch_size)
    observed_ratio = calibration.size / values.shape[1]

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for _ in range(batches):
            starts = rng.integers(0, values.shape[0] - window + 1, size=batch_size)
            target = np.stack([values[start : start + window, calibration] for start in starts]).astype(np.float32)
            mask = rng.random(target.shape) >= whiten_probability
            inputs = target.copy()
            inputs[~mask] = 0.0
            dynamic_ratio = observed_ratio + 0.2 * float(rng.random())
            augmented_nodes = int(calibration.size / dynamic_ratio)
            virtual_nodes = max(1, augmented_nodes - calibration.size)
            zeros = np.zeros((batch_size, window, virtual_nodes), dtype=np.float32)
            input_augmented = np.concatenate([inputs, zeros], axis=2)[..., None]
            target_augmented = np.concatenate([target, zeros], axis=2)[..., None]
            mask_augmented = np.concatenate([mask, zeros.astype(bool)], axis=2)[..., None]

            prediction, cycle_prediction, cycle_target = model(
                torch.as_tensor(input_augmented, device=device),
                torch.as_tensor(mask_augmented, dtype=torch.uint8, device=device),
                known_set=calibration.tolist(),
                sub_entry_num=virtual_nodes,
                reset=True,
            )
            target_tensor = torch.as_tensor(target_augmented, device=device)
            mask_tensor = torch.as_tensor(mask_augmented, dtype=torch.bool, device=device)
            loss = masked_mse(prediction, target_tensor, mask_tensor) + torch.mean((cycle_prediction - cycle_target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % validation_every == 0:
            rmse = evaluate(
                model,
                values,
                calibration,
                validation,
                window=window,
                stride=validation_stride,
                device=device,
            )
            trace.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_rmse": rmse})
            if rmse < best_rmse - 1e-4:
                best_rmse = rmse
                best_state = deepcopy(model.state_dict())
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            if epoch >= 50 and checks_without_improvement >= patience_checks:
                break
    return best_state, trace


def predict_stream(
    model: KITS,
    values: np.ndarray,
    visible: np.ndarray,
    heldout: np.ndarray,
    *,
    task1_steps: int,
    window: int,
    batch_size: int,
    max_stream_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stream = np.arange(task1_steps, values.shape[0], dtype=int)
    if max_stream_steps:
        stream = stream[:max_stream_steps]
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, stream.size, batch_size):
            current = stream[start : start + batch_size]
            blocks = np.stack([values[t - window + 1 : t + 1].copy() for t in current]).astype(np.float32)
            masks = np.ones(blocks.shape, dtype=bool)
            blocks[:, -1, heldout] = 0.0
            masks[:, -1, heldout] = False
            prediction = model(
                torch.as_tensor(blocks[..., None], device=device),
                torch.as_tensor(masks[..., None], dtype=torch.uint8, device=device),
            )
            predictions.append(prediction[:, -1, heldout, 0].detach().cpu().numpy())
    mean = np.vstack(predictions)
    return stream, values[stream][:, heldout], mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/traffic/raw")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "results/traffic/protocols/pems_bay/pems_bay_seed0_spatial_split.json")
    parser.add_argument("--road-distance-csv", type=Path, default=ROOT / "data/traffic/raw/pems_bay/distances_bay_2017.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/traffic/seed0_external_baselines_v1/kits")
    parser.add_argument("--task1-steps", type=int, default=2016)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples-per-epoch", type=int, default=5120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--whiten-probability", type=float, default=0.05)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--validation-stride", type=int, default=24)
    parser.add_argument("--patience-checks", type=int, default=10)
    parser.add_argument("--prediction-batch-size", type=int, default=16)
    parser.add_argument("--max-stream-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = load_traffic_dataset(args.data_root, "pems_bay", task1_steps=args.task1_steps)
    split = load_spatial_split(args.split_manifest)
    calibration = np.asarray(split.visible_calibration_indices, dtype=int)
    validation = np.asarray(split.visible_validation_indices, dtype=int)
    visible = np.asarray(split.visible_indices, dtype=int)
    heldout = np.asarray(split.heldout_indices, dtype=int)
    laplacian, road_metadata = load_road_laplacian(args.road_distance_csv, dataset.sensor_ids)
    adjacency = np.maximum(np.eye(dataset.num_sensors) - laplacian, 0.0)
    np.fill_diagonal(adjacency, 0.0)

    started = time.perf_counter()
    model = make_model(adjacency, args.hidden_dim, device)
    best_state, trace = train_task1(
        model,
        dataset.values_standardised[: args.task1_steps],
        calibration,
        validation,
        window=args.window,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        whiten_probability=args.whiten_probability,
        validation_every=args.validation_every,
        validation_stride=args.validation_stride,
        patience_checks=args.patience_checks,
        seed=args.seed,
        device=device,
    )
    model.load_state_dict(best_state)
    stream, y, mean = predict_stream(
        model,
        dataset.values_standardised,
        visible,
        heldout,
        task1_steps=args.task1_steps,
        window=args.window,
        batch_size=args.prediction_batch_size,
        max_stream_steps=args.max_stream_steps,
        device=device,
    )
    error = mean - y
    speed_error = dataset.target_to_speed(mean) - dataset.target_to_speed(y)
    metrics = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse_speed": float(np.sqrt(np.mean(speed_error**2))),
        "mae_speed": float(np.mean(np.abs(speed_error))),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "predictions.npz", stream_indices=stream, y=y, mean=mean)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "official_repository": "https://github.com/Sam1224/KITS",
        "official_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=OFFICIAL, text=True).strip(),
        "official_core": "lib.nn.models.KITS",
        "adaptation": "official increment/cycle model core with Task-1-only training and strict Protocol-N rolling windows",
        "selection_boundary": "Task-1 visible-validation sensors only",
        "deterministic_baseline": True,
        "probabilistic_metrics": "not applicable",
        "device": str(device),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "road_graph": road_metadata,
        "training_trace": trace,
        "best_validation_rmse": min(row["validation_rmse"] for row in trace),
        "metrics": metrics,
        "causal_audit": {
            "current_hidden_reads_before_prediction": 0,
            "current_visible_values_present": True,
            "heldout_history_latest_available_lag": 1,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (args.output / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "status.json").write_text(json.dumps({"status": "complete", "finite_predictions": bool(np.all(np.isfinite(mean)))}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_rmse": payload["best_validation_rmse"], "metrics": metrics, "wall_clock_seconds": payload["wall_clock_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
