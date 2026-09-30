#!/usr/bin/env python3
"""Task-1-trained graph baselines for strict PEMS-BAY Protocol F."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stvgp_kronecker.data.traffic import TrafficDataset, load_spatial_split, load_traffic_dataset
from stvgp_kronecker.traffic_spatial_kernels import load_road_laplacian


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (3, 6, 12)
SEQUENCE_LENGTH = 12
SOURCE = {
    "graph_wavenet": {
        "repository": "https://github.com/nnzhan/Graph-WaveNet",
        "commit": "6b162e80c59a1d494809252eca055cff93dc66b1",
        "core": "baselines/external/nnzhan_GraphWaveNet/model.py",
    },
    "dcrnn": {
        "repository": "https://github.com/chnsh/DCRNN_Pytorch",
        "commit": "d92490b808ba5c5be2f23d427d96e9a56b066d7f",
        "official_tensorflow_repository": "https://github.com/liyaguang/DCRNN",
        "core": "baselines/external/chnsh_DCRNN_Pytorch/model/pytorch",
    },
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def time_of_day(timestamps) -> np.ndarray:
    minutes = timestamps.hour.to_numpy() * 60 + timestamps.minute.to_numpy()
    return (minutes / (24.0 * 60.0)).astype(np.float32)


def road_adjacency(dataset: TrafficDataset, road_distance_csv: Path) -> tuple[np.ndarray, dict[str, object]]:
    laplacian, metadata = load_road_laplacian(road_distance_csv, dataset.sensor_ids)
    adjacency = np.eye(dataset.num_sensors, dtype=np.float64) - laplacian
    adjacency = np.maximum(adjacency, 0.0)
    np.fill_diagonal(adjacency, np.maximum(np.diag(adjacency), 1.0))
    return adjacency.astype(np.float32), metadata


def row_transition(adjacency: np.ndarray) -> np.ndarray:
    denominator = np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-12)
    return (adjacency / denominator).astype(np.float32)


class Task1Windows(Dataset):
    """Task-1 windows whose targets are restricted to one declared node set."""

    def __init__(
        self,
        dataset: TrafficDataset,
        *,
        origins: np.ndarray,
        calibration: np.ndarray,
        target_nodes: np.ndarray,
        permanently_hidden: np.ndarray,
        training: bool,
        mask_fraction: float,
        seed: int,
    ) -> None:
        self.values = np.asarray(dataset.values_standardised, dtype=np.float32)
        self.tod = time_of_day(dataset.timestamps)
        self.origins = np.asarray(origins, dtype=int)
        self.calibration = np.asarray(calibration, dtype=int)
        self.target_nodes = np.asarray(target_nodes, dtype=int)
        self.permanently_hidden = np.asarray(permanently_hidden, dtype=int)
        self.training = bool(training)
        self.mask_fraction = float(mask_fraction)
        self.seed = int(seed)
        if np.intersect1d(self.calibration, self.permanently_hidden).size:
            raise ValueError("Calibration and permanently hidden nodes must be disjoint")

    def __len__(self) -> int:
        return int(self.origins.size)

    def __getitem__(self, item: int):
        origin = int(self.origins[item])
        window = np.arange(origin - SEQUENCE_LENGTH + 1, origin + 1, dtype=int)
        speed = np.zeros((self.values.shape[1], SEQUENCE_LENGTH), dtype=np.float32)
        observed = np.zeros_like(speed)
        available = self.calibration.copy()
        if self.training and self.mask_fraction > 0.0:
            rng = np.random.default_rng(self.seed * 1_000_003 + origin)
            count = max(1, int(round(self.mask_fraction * available.size)))
            masked = rng.choice(available, size=count, replace=False)
            available = np.setdiff1d(available, masked, assume_unique=True)
        speed[available] = self.values[window][:, available].T
        observed[available] = 1.0
        tod = np.broadcast_to(self.tod[window][None, :], speed.shape).copy()
        features = np.stack((speed, observed, tod), axis=0)

        target_times = origin + np.arange(1, max(HORIZONS) + 1, dtype=int)
        targets = self.values[target_times][:, self.target_nodes]
        decoder_targets = np.zeros((max(HORIZONS), self.values.shape[1]), dtype=np.float32)
        if self.training:
            decoder_targets[:, self.calibration] = self.values[target_times][:, self.calibration]
        return (
            torch.from_numpy(features),
            torch.from_numpy(targets),
            torch.from_numpy(decoder_targets),
        )


def build_online_features(
    dataset: TrafficDataset,
    *,
    origins: np.ndarray,
    visible: np.ndarray,
    heldout: np.ndarray,
) -> np.ndarray:
    """Build legal formal-stream inputs without reading a future target."""

    values = np.asarray(dataset.values_standardised, dtype=np.float32)
    tod_values = time_of_day(dataset.timestamps)
    features = np.zeros((len(origins), 3, dataset.num_sensors, SEQUENCE_LENGTH), dtype=np.float32)
    for row, origin_value in enumerate(np.asarray(origins, dtype=int)):
        origin = int(origin_value)
        window = np.arange(origin - SEQUENCE_LENGTH + 1, origin + 1, dtype=int)
        features[row, 0, visible] = values[window][:, visible].T
        features[row, 1, visible] = 1.0
        legal_hidden = window[(window >= dataset.task1_steps) & (window < origin)]
        if legal_hidden.size:
            positions = np.flatnonzero((window >= dataset.task1_steps) & (window < origin))
            features[row, 0, heldout[:, None], positions[None, :]] = values[legal_hidden][:, heldout].T
            features[row, 1, heldout[:, None], positions[None, :]] = 1.0
        features[row, 2] = np.broadcast_to(tod_values[window][None, :], (dataset.num_sensors, SEQUENCE_LENGTH))
    return features


@dataclass
class ModelAdapter:
    method: str
    model: nn.Module
    device: torch.device

    def predict(self, features: torch.Tensor, decoder_targets: torch.Tensor | None = None, batches_seen: int | None = None) -> torch.Tensor:
        features = features.to(self.device, non_blocking=True)
        if self.method == "graph_wavenet":
            return self.model(features)[..., -1]
        inputs = features.permute(3, 0, 2, 1).reshape(SEQUENCE_LENGTH, features.shape[0], -1)
        labels = None
        if decoder_targets is not None:
            labels = decoder_targets.to(self.device, non_blocking=True).permute(1, 0, 2)
        outputs = self.model(inputs, labels=labels, batches_seen=batches_seen)
        return outputs.permute(1, 0, 2)


def make_model(method: str, adjacency: np.ndarray, device: torch.device) -> ModelAdapter:
    if method == "graph_wavenet":
        from baselines.external.nnzhan_GraphWaveNet.model import gwnet

        supports = [
            torch.as_tensor(row_transition(adjacency), device=device),
            torch.as_tensor(row_transition(adjacency.T), device=device),
        ]
        model = gwnet(
            device,
            adjacency.shape[0],
            dropout=0.3,
            supports=supports,
            gcn_bool=True,
            addaptadj=True,
            aptinit=None,
            in_dim=3,
            out_dim=12,
        ).to(device)
    else:
        from baselines.external.chnsh_DCRNN_Pytorch.model.pytorch.dcrnn_model import DCRNNModel

        model = DCRNNModel(
            adjacency,
            logging.getLogger("dcrnn"),
            max_diffusion_step=2,
            cl_decay_steps=2000,
            filter_type="dual_random_walk",
            num_nodes=adjacency.shape[0],
            num_rnn_layers=2,
            rnn_units=64,
            input_dim=3,
            seq_len=SEQUENCE_LENGTH,
            output_dim=1,
            horizon=12,
            use_curriculum_learning=True,
        ).to(device)
    adapter = ModelAdapter(method=method, model=model, device=device)
    dummy = torch.zeros((1, 3, adjacency.shape[0], SEQUENCE_LENGTH), dtype=torch.float32)
    decoder = torch.zeros((1, 12, adjacency.shape[0]), dtype=torch.float32)
    with torch.no_grad():
        adapter.predict(dummy, decoder, batches_seen=0)
    return adapter


def evaluate(
    adapter: ModelAdapter,
    loader: DataLoader,
    target_nodes: np.ndarray,
) -> float:
    adapter.model.eval()
    squared_error = 0.0
    count = 0
    with torch.no_grad():
        for features, targets, _ in loader:
            predictions = adapter.predict(features)[:, :, target_nodes].cpu()
            selected = predictions[:, np.asarray(HORIZONS) - 1]
            target_selected = targets[:, np.asarray(HORIZONS) - 1]
            squared_error += float(torch.sum((selected - target_selected) ** 2))
            count += int(selected.numel())
    return math.sqrt(squared_error / max(count, 1))


def fit_task1(
    adapter: ModelAdapter,
    dataset: TrafficDataset,
    split,
    *,
    output: Path,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    mask_fraction: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    selected_path = output / "selected_checkpoint.pt"
    calibration = np.asarray(split.visible_calibration_indices, dtype=int)
    validation = np.asarray(split.visible_validation_indices, dtype=int)
    heldout = np.asarray(split.heldout_indices, dtype=int)
    origins = np.arange(SEQUENCE_LENGTH - 1, dataset.task1_steps - max(HORIZONS), dtype=int)
    train_data = Task1Windows(
        dataset,
        origins=origins,
        calibration=calibration,
        target_nodes=calibration,
        permanently_hidden=np.concatenate((validation, heldout)),
        training=True,
        mask_fraction=mask_fraction,
        seed=seed,
    )
    validation_data = Task1Windows(
        dataset,
        origins=origins[::3],
        calibration=calibration,
        target_nodes=validation,
        permanently_hidden=np.concatenate((validation, heldout)),
        training=False,
        mask_fraction=0.0,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if selected_path.exists():
        selected = torch.load(selected_path, map_location=adapter.device, weights_only=False)
        adapter.model.load_state_dict(selected["model_state"])
        return list(selected["training_trace"]), dict(selected["selection"])

    learning_rate = 1e-3 if adapter.method == "graph_wavenet" else 1e-2
    optimizer = torch.optim.Adam(adapter.model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=4, min_lr=1e-5)
    resume_path = output / "training_checkpoint.pt"
    start_epoch = 0
    batches_seen = 0
    best_score = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    trace: list[dict[str, object]] = []
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=adapter.device, weights_only=False)
        adapter.model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        start_epoch = int(saved["next_epoch"])
        batches_seen = int(saved["batches_seen"])
        best_score = float(saved["best_score"])
        best_epoch = int(saved["best_epoch"])
        best_state = saved["best_state"]
        wait = int(saved["wait"])
        trace = list(saved["training_trace"])
        if "loader_rng_state" in saved:
            generator.set_state(saved["loader_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if torch.cuda.is_available() and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])

    target_nodes = torch.as_tensor(calibration, dtype=torch.long, device=adapter.device)
    for epoch in range(start_epoch, max_epochs):
        adapter.model.train()
        total_loss = 0.0
        total_count = 0
        for features, targets, decoder_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = adapter.predict(features, decoder_targets, batches_seen=batches_seen)
            target = targets.to(adapter.device, non_blocking=True)
            loss = torch.mean(torch.abs(predictions[:, :, target_nodes] - target))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * int(target.numel())
            total_count += int(target.numel())
            batches_seen += 1
        validation_rmse = evaluate(adapter, validation_loader, validation)
        scheduler.step(validation_rmse)
        row = {
            "epoch": epoch + 1,
            "train_mae": total_loss / max(total_count, 1),
            "visible_validation_rmse": validation_rmse,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        trace.append(row)
        if validation_rmse < best_score - 1e-5:
            best_score = validation_rmse
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in adapter.model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        torch.save(
            {
                "next_epoch": epoch + 1,
                "batches_seen": batches_seen,
                "model_state": adapter.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "wait": wait,
                "training_trace": trace,
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loader_rng_state": generator.get_state(),
            },
            resume_path,
        )
        print(json.dumps(row), flush=True)
        if epoch + 1 >= 10 and wait >= patience:
            break
    if best_state is None:
        raise RuntimeError("Task-1 training produced no selectable checkpoint")
    adapter.model.load_state_dict(best_state)
    selection = {
        "criterion": "mean visible-validation RMSE over horizons 3, 6 and 12",
        "validation_stride": 3,
        "best_epoch": best_epoch,
        "best_visible_validation_rmse": best_score,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
    }
    torch.save(
        {
            "model_state": best_state,
            "training_trace": trace,
            "selection": selection,
        },
        selected_path,
    )
    return trace, selection


def point_metrics(y: np.ndarray, mean: np.ndarray, scale: float) -> dict[str, float]:
    error = np.asarray(mean, dtype=float) - np.asarray(y, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse_speed": float(scale * np.sqrt(np.mean(error**2))),
        "mae_speed": float(scale * np.mean(np.abs(error))),
    }


def predict_stream(
    adapter: ModelAdapter,
    dataset: TrafficDataset,
    split,
    *,
    batch_size: int,
    max_stream_origins: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, dict[str, float]], dict[str, int]]:
    origins = np.arange(dataset.task1_steps, dataset.num_time - max(HORIZONS), dtype=int)
    if max_stream_origins:
        origins = origins[:max_stream_origins]
    visible = np.asarray(split.visible_indices, dtype=int)
    heldout = np.asarray(split.heldout_indices, dtype=int)
    predictions = {horizon: [] for horizon in HORIZONS}
    rows: list[dict[str, object]] = []
    adapter.model.eval()
    with torch.no_grad():
        for start in range(0, origins.size, batch_size):
            batch_origins = origins[start : start + batch_size]
            features = build_online_features(
                dataset,
                origins=batch_origins,
                visible=visible,
                heldout=heldout,
            )
            outputs = adapter.predict(torch.from_numpy(features)).cpu().numpy()
            for horizon in HORIZONS:
                predictions[horizon].append(outputs[:, horizon - 1, heldout])

    archive: dict[str, np.ndarray] = {"stream_indices": origins}
    final: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS:
        mean = np.concatenate(predictions[horizon], axis=0)
        y = np.asarray(dataset.values_standardised[origins + horizon][:, heldout], dtype=np.float32)
        metrics = point_metrics(y, mean, dataset.target_scale)
        archive[f"y_h{horizon}"] = y
        archive[f"mean_h{horizon}"] = mean
        final[str(horizon)] = metrics
        for row_index, origin in enumerate(origins):
            rows.append(
                {
                    "step": int(row_index),
                    "origin_time_index": int(origin),
                    "target_time_index": int(origin + horizon),
                    "horizon_steps": horizon,
                    "unknown_future_exogenous_reads": 0,
                    **point_metrics(y[row_index], mean[row_index], dataset.target_scale),
                }
            )
    guard = {
        "current_hidden_reads_before_prediction": 0,
        "current_hidden_reveals": int(origins.size),
        "current_visible_reads": int(origins.size),
        "unique_delayed_hidden_absorptions": max(int(origins.size) - 1, 0),
    }
    return archive, rows, final, guard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["graph_wavenet", "dcrnn"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/traffic/raw"))
    parser.add_argument("--road-distance-csv", type=Path, default=Path("data/traffic/raw/pems_bay/distances_bay_2017.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--training-mask-fraction", type=float, default=0.2)
    parser.add_argument("--max-stream-origins", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "status.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete":
        print(f"SKIP complete: {args.output}")
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal graph baseline run")
    device = torch.device(args.device)
    seed_everything(args.seed)
    dataset = load_traffic_dataset(args.data_root, "pems_bay", task1_steps=2016)
    split = load_spatial_split(args.split_manifest)
    if split.seed != args.seed:
        raise ValueError("Split seed does not match --seed")
    adjacency, road_metadata = road_adjacency(dataset, args.road_distance_csv)
    adapter = make_model(args.method, adjacency, device)
    started = time.perf_counter()
    trace, selection = fit_task1(
        adapter,
        dataset,
        split,
        output=args.output,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        mask_fraction=args.training_mask_fraction,
    )
    archive, rows, final, guard = predict_stream(
        adapter,
        dataset,
        split,
        batch_size=args.batch_size,
        max_stream_origins=args.max_stream_origins,
    )
    if not all(np.isfinite(value).all() for key, value in archive.items() if key.startswith("mean_h")):
        raise RuntimeError("Non-finite formal predictions")
    np.savez_compressed(args.output / "predictions.npz", **archive)
    write_csv(args.output / "per_step_metrics.csv", rows)
    write_csv(args.output / "training_trace.csv", trace)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "method": args.method,
        "deterministic_baseline": True,
        "probabilistic_metrics": "not_applicable",
        "upstream": SOURCE[args.method],
        "dataset": "pems_bay",
        "seed": args.seed,
        "task1_steps": dataset.task1_steps,
        "target_standardisation": {"mean": dataset.target_mean, "scale": dataset.target_scale, "fit_prefix": "Task-1 only"},
        "split_manifest": str(args.split_manifest),
        "training_protocol": {
            "loss_nodes": "visible_calibration only",
            "selection_nodes": "visible_validation only",
            "formal_heldout_task1_input_reads": 0,
            "formal_heldout_task1_loss_reads": 0,
            "input_channels": ["standardised_speed", "observation_mask", "time_of_day_fraction"],
            "training_spatial_mask_fraction": args.training_mask_fraction,
            "parameters_after_task1": "frozen",
        },
        "selection": selection,
        "road_graph": road_metadata,
        "result": {
            "protocol": "F",
            "final": final,
            "nowcasting_guard": guard,
            "wall_clock_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        },
    }
    (args.output / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    status_path.write_text(
        json.dumps({"status": "complete", "finite_predictions": True, "result": "result.json"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
