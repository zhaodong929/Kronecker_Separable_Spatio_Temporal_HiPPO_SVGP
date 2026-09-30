#!/usr/bin/env python3
"""Task-1-only adapter for the pinned official IGNNK model core."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "baselines/external/Kaimaoge_IGNNK"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from basic_structure import IGNNK  # noqa: E402
from utils import calculate_random_walk_matrix  # noqa: E402
from stvgp_kronecker.data.traffic import load_spatial_split, load_traffic_dataset  # noqa: E402
from stvgp_kronecker.traffic_spatial_kernels import load_road_laplacian  # noqa: E402


def graph_support(adjacency: np.ndarray, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    selected = adjacency[np.ix_(indices, indices)]
    forward = calculate_random_walk_matrix(selected).T.astype(np.float32)
    backward = calculate_random_walk_matrix(selected.T).T.astype(np.float32)
    return torch.as_tensor(forward, device=device), torch.as_tensor(backward, device=device)


def validation_rmse(
    model: IGNNK,
    values: np.ndarray,
    adjacency: np.ndarray,
    calibration: np.ndarray,
    validation: np.ndarray,
    *,
    window: int,
    stride: int,
    device: torch.device,
) -> float:
    nodes = np.concatenate([calibration, validation])
    forward, backward = graph_support(adjacency, nodes, device)
    errors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for stop in range(window, values.shape[0] + 1, stride):
            block = values[stop - window : stop, nodes].copy()
            block[:, calibration.size :] = 0.0
            prediction = model(torch.as_tensor(block[None], dtype=torch.float32, device=device), forward, backward)
            estimate = prediction[0, -1, calibration.size :].detach().cpu().numpy()
            errors.append(estimate - values[stop - 1, validation])
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def train_task1(
    model: IGNNK,
    values: np.ndarray,
    adjacency: np.ndarray,
    calibration: np.ndarray,
    validation: np.ndarray,
    *,
    window: int,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    sampled_nodes: int,
    masked_nodes: int,
    learning_rate: float,
    validation_every: int,
    validation_stride: int,
    patience_checks: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    best_state = deepcopy(model.state_dict())
    best_rmse = float("inf")
    checks_without_improvement = 0
    trace: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for _ in range(batches_per_epoch):
            nodes = np.sort(rng.choice(calibration, size=min(sampled_nodes, calibration.size), replace=False))
            forward, backward = graph_support(adjacency, nodes, device)
            starts = rng.integers(0, values.shape[0] - window + 1, size=batch_size)
            targets = np.stack([values[start : start + window, nodes] for start in starts]).astype(np.float32)
            inputs = targets.copy()
            for batch in range(batch_size):
                masked = rng.choice(nodes.size, size=min(masked_nodes, nodes.size - 1), replace=False)
                inputs[batch, :, masked] = 0.0
            target_tensor = torch.as_tensor(targets, device=device)
            prediction = model(torch.as_tensor(inputs, device=device), forward, backward)
            loss = criterion(prediction, target_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % validation_every == 0:
            rmse = validation_rmse(
                model,
                values,
                adjacency,
                calibration,
                validation,
                window=window,
                stride=validation_stride,
                device=device,
            )
            trace.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "validation_rmse": rmse})
            if rmse < best_rmse - 1e-4:
                best_rmse = rmse
                best_state = deepcopy(model.state_dict())
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            if epoch >= 100 and checks_without_improvement >= patience_checks:
                break
    return best_state, trace


def predict_stream(
    model: IGNNK,
    values: np.ndarray,
    adjacency: np.ndarray,
    visible: np.ndarray,
    heldout: np.ndarray,
    *,
    task1_steps: int,
    window: int,
    batch_size: int,
    max_stream_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.arange(values.shape[1], dtype=int)
    forward, backward = graph_support(adjacency, nodes, device)
    stream = np.arange(task1_steps, values.shape[0], dtype=int)
    if max_stream_steps:
        stream = stream[:max_stream_steps]
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, stream.size, batch_size):
            current = stream[start : start + batch_size]
            blocks = np.stack([values[t - window + 1 : t + 1].copy() for t in current]).astype(np.float32)
            blocks[:, -1, heldout] = 0.0
            output = model(torch.as_tensor(blocks, device=device), forward, backward)
            predictions.append(output[:, -1, heldout].detach().cpu().numpy())
    mean = np.vstack(predictions)
    return stream, values[stream][:, heldout], mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/traffic/raw")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "results/traffic/protocols/pems_bay/pems_bay_seed0_spatial_split.json")
    parser.add_argument("--road-distance-csv", type=Path, default=ROOT / "data/traffic/raw/pems_bay/distances_bay_2017.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/traffic/seed0_external_baselines_v1/ignnk")
    parser.add_argument("--task1-steps", type=int, default=2016)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--diffusion-order", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=750)
    parser.add_argument("--batches-per-epoch", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sampled-nodes", type=int, default=180)
    parser.add_argument("--masked-nodes", type=int, default=45)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-every", type=int, default=25)
    parser.add_argument("--validation-stride", type=int, default=24)
    parser.add_argument("--patience-checks", type=int, default=5)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    parser.add_argument("--max-stream-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
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
    model = IGNNK(args.window, args.hidden_dim, args.diffusion_order).to(device)
    best_state, trace = train_task1(
        model,
        dataset.values_standardised[: args.task1_steps],
        adjacency,
        calibration,
        validation,
        window=args.window,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        sampled_nodes=args.sampled_nodes,
        masked_nodes=args.masked_nodes,
        learning_rate=args.learning_rate,
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
        adjacency,
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
        "official_repository": "https://github.com/Kaimaoge/IGNNK",
        "official_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=OFFICIAL, text=True).strip(),
        "official_core": "basic_structure.IGNNK",
        "adaptation": "Task-1-only training and strict Protocol-N rolling windows",
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
