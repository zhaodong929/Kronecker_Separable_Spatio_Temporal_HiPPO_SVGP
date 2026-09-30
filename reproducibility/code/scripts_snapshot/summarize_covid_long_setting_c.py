#!/usr/bin/env python3
"""Audit the paired Setting B/C Route B long-stream comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_iclr_era5_routeb_strict_online import metrics


METHODS = (
    ("routeb_ordinary", "Route B ordinary inducing"),
    ("routeb_cumulative", "Route B cumulative HiPPO"),
)
METRICS = ("rmse", "nll", "coverage90")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_run(
    *,
    root: Path,
    protocol_root: Path,
    seed: int,
    method: str,
    setting_c: bool,
) -> dict[str, object]:
    run = root / f"seed{seed}" / method / "online"
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    with np.load(run / "predictions.npz") as archive, np.load(
        protocol_root / f"seed{seed}" / "protocol.npz"
    ) as protocol:
        y_true = np.asarray(archive["y_true"], dtype=np.float64)
        mean = np.asarray(archive["pred_mean"], dtype=np.float64)
        variance = np.asarray(archive["pred_var"], dtype=np.float64)
        test_indices = np.asarray(archive["test_indices"], dtype=np.int64)
        expected_indices = np.asarray(protocol["test_indices"], dtype=np.int64)
        expected_target = np.asarray(protocol["stream_y"], dtype=np.float64)[:, expected_indices]
        train_size = int(np.asarray(protocol["train_indices"]).size)

    if not np.array_equal(test_indices, expected_indices):
        raise ValueError(f"{run}: hidden-state mask differs from the protocol")
    if not np.allclose(y_true, expected_target, atol=1e-12, rtol=0.0):
        raise ValueError(f"{run}: prediction archive targets differ from the protocol")
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance <= 0.0).any():
        raise ValueError(f"{run}: prediction archive contains invalid moments")
    recomputed = metrics(y_true, mean, variance)
    for name in METRICS:
        if not np.isclose(recomputed[name], result["overall_current_block"][name], atol=1e-10, rtol=0.0):
            raise ValueError(f"{run}: recomputed {name} differs from result.json")

    expected_delayed_hidden = (y_true.shape[0] - 1) * y_true.shape[1]
    if int(result["delayed_observation_rows"]) != expected_delayed_hidden:
        raise ValueError(f"{run}: hidden labels were not absorbed exactly once")
    if setting_c:
        expected_delayed_visible = (y_true.shape[0] - 1) * train_size
        if not bool(result.get("forecast_without_current_visible_observations")):
            raise ValueError(f"{run}: missing Setting C flag")
        if int(result.get("current_visible_observation_rows", -1)) != 0:
            raise ValueError(f"{run}: current visible labels entered the Setting C posterior")
        if int(result.get("delayed_visible_observation_rows", -1)) != expected_delayed_visible:
            raise ValueError(f"{run}: visible history was not absorbed exactly once")
    elif bool(result.get("forecast_without_current_visible_observations", False)):
        raise ValueError(f"{run}: Setting B archive incorrectly declares Setting C")

    return {
        "seed": seed,
        "method": method,
        "label": dict(METHODS)[method],
        "rmse": float(recomputed["rmse"]),
        "nll": float(recomputed["nll"]),
        "coverage90": float(recomputed["coverage90"]),
        "weeks": int(y_true.shape[0]),
        "hidden_states": int(y_true.shape[1]),
        "runtime_seconds": float(result["timing"]["process_total_seconds"]),
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for setting in ("B", "C"):
        for method, label in METHODS:
            selected = [row for row in rows if row["setting"] == setting and row["method"] == method]
            if not selected:
                continue
            record: dict[str, object] = {"setting": setting, "method": method, "label": label, "n_splits": len(selected)}
            for name in METRICS:
                values = np.asarray([float(row[name]) for row in selected], dtype=np.float64)
                record[f"{name}_mean"] = float(values.mean())
                record[f"{name}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-b-root", type=Path, required=True)
    parser.add_argument("--setting-c-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-seed", type=int, default=0)
    parser.add_argument("--formal-seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for split_name, seeds in (("development", [args.development_seed]), ("formal", args.formal_seeds)):
        for setting, root, is_setting_c in (
            ("B", args.setting_b_root, False),
            ("C", args.setting_c_root, True),
        ):
            for seed in seeds:
                for method, _ in METHODS:
                    row = read_run(
                        root=root.resolve(),
                        protocol_root=args.protocol_root.resolve(),
                        seed=seed,
                        method=method,
                        setting_c=is_setting_c,
                    )
                    row["split"] = split_name
                    row["setting"] = setting
                    rows.append(row)

    write_csv(output_dir / "setting_b_c_per_seed.csv", rows)
    formal_rows = [row for row in rows if row["split"] == "formal"]
    aggregates = aggregate(formal_rows)
    (output_dir / "setting_b_c_aggregate.json").write_text(
        json.dumps(aggregates, indent=2) + "\n", encoding="utf-8"
    )
    summary = {(row["setting"], row["method"]): row for row in aggregates}
    lines = [
        "# Long-stream Setting B vs. Setting C",
        "",
        "Setting B conditions each prediction on delayed labels plus current visible-state labels. "
        "Setting C defers both visible and hidden labels by one week, so its prediction at week t uses only labels available through t-1.",
        "",
        "| Setting | Representation | RMSE | NLL | Coverage90 |",
        "|---|---|---:|---:|---:|",
    ]
    for setting in ("B", "C"):
        for method, label in METHODS:
            row = summary[(setting, method)]
            lines.append(
                f"| {setting} | {label} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | "
                f"{row['nll_mean']:.4f} +/- {row['nll_sd']:.4f} | "
                f"{row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} |"
            )
    lines.extend(
        [
            "",
            "All entries are recomputed from prediction archives for paired formal spatial splits 5-9. "
            "Setting C audit requires zero current-visible update rows and exactly one delayed update for each available visible and hidden label.",
        ]
    )
    (output_dir / "setting_b_c_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "artifact_audit.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "formal_seeds": args.formal_seeds,
                "development_seed": args.development_seed,
                "prediction_metrics_recomputed": True,
                "setting_c_current_visible_rows_required": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
