#!/usr/bin/env python3
"""Evaluate the isolated RTX 4090 Route B five-seed confirmation run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_covid_long_final_metric_system import gaussian_metrics_on_common_scale, mean_sd


METHODS = (
    ("routeb_ordinary", "Route B ordinary inducing"),
    ("routeb_cumulative", "Route B cumulative HiPPO"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_archive(path: Path, expected_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key], dtype=np.float64)
            for key in ("y_true", "pred_mean", "pred_var")
        }
    if any(array.shape != expected_shape for array in arrays.values()):
        shapes = {key: array.shape for key, array in arrays.items()}
        raise ValueError(f"Invalid archive shape in {path}: {shapes}, expected {expected_shape}")
    if not all(np.isfinite(array).all() for array in arrays.values()) or np.any(arrays["pred_var"] <= 0.0):
        raise ValueError(f"Non-finite mean or non-positive variance in {path}")
    return arrays


def verify_causal_result(result: dict[str, object], expected_shape: tuple[int, int], seed: int, method: str) -> None:
    """Accept the current Route B result record and the newer adapter audit record."""

    audit = result.get("audit")
    if isinstance(audit, dict) and audit:
        if audit.get("passed") is True and audit.get("current_hidden_labels_read") == 0:
            return
        raise ValueError(f"Causal audit failed for seed {seed}, {method}: {audit}")
    expected_delayed = max(0, expected_shape[0] - 1) * expected_shape[1]
    if (
        result.get("delayed_observations") is True
        and int(result.get("delayed_observation_rows", -1)) == expected_delayed
    ):
        return
    raise ValueError(f"Delayed-observation record failed for seed {seed}, {method}")


def main() -> None:
    args = parse_args()
    results_root = absolute(args.results_root)
    protocol_root = absolute(args.protocol_root)
    output = absolute(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, object]] = []

    for seed in args.seeds:
        protocol_path = protocol_root / f"seed{seed}" / "protocol.json"
        metadata = json.loads(protocol_path.read_text(encoding="utf-8"))
        expected_shape = (int(metadata["num_stream_times"]), int(metadata["num_test_locations"]))
        for method, label in METHODS:
            root = results_root / f"seed{seed}" / method / "online"
            result_path = root / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            verify_causal_result(result, expected_shape, seed, method)
            arrays = read_archive(root / "predictions.npz", expected_shape)
            metrics = gaussian_metrics_on_common_scale(
                arrays,
                metadata["target_standardization"],
                ece_seed=30_000_000 + 1000 * seed + (0 if method == "routeb_ordinary" else 1),
            )
            per_seed.append({"seed": seed, "method": method, "label": label, **metrics})

    aggregate: list[dict[str, object]] = []
    for method, label in METHODS:
        rows = [row for row in per_seed if row["method"] == method]
        entry: dict[str, object] = {"method": method, "label": label, "seeds": len(rows)}
        for metric in ("rmse", "crps", "native_gaussian_nlpd", "ece", "coverage90"):
            mean, sd = mean_sd([float(row[metric]) for row in rows])
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_sd"] = sd
        aggregate.append(entry)

    write_csv(per_seed, output / "per_seed_metrics.csv")
    write_csv(aggregate, output / "aggregate_metrics.csv")
    (output / "audit.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "methods": [method for method, _ in METHODS],
                "seeds": args.seeds,
                "metrics": ["RMSE", "CRPS", "Gaussian NLPD", "ECE", "Coverage90"],
                "result_root": str(results_root),
                "protocol_root": str(protocol_root),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
