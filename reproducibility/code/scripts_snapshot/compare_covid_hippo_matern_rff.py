#!/usr/bin/env python3
"""Compare Matérn cumulative-HiPPO RFF capacities for the COVID stream."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.compare_covid_routeb_objectives import score_archive


SEEDS = (5, 6, 7, 8, 9)
METRICS = ("rmse", "crps", "native_gaussian_nlpd", "ece", "coverage90")
CONFIGS = (
    ("Kron-STGP VFE", "ordinary", None),
    ("KronHiPPO-STGP VFE (Matern, RFF=128)", "rff128", 128),
    ("KronHiPPO-STGP VFE (Matern, RFF=256)", "rff256", 256),
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary-root", type=Path, default=Path("results/diagnostics/covid_long_stream_2020_2024_mandatory_vfe"))
    parser.add_argument("--rff128-root", type=Path, required=True)
    parser.add_argument("--rff256-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    roots = {"ordinary": args.ordinary_root.resolve(), "rff128": args.rff128_root.resolve(), "rff256": args.rff256_root.resolve()}
    per_seed: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for label, key, rff in CONFIGS:
        scores: list[dict[str, float]] = []
        for seed in SEEDS:
            method_dir = "routeb_ordinary" if key == "ordinary" else "routeb_cumulative"
            archive = roots[key] / f"seed{seed}/{method_dir}/online/predictions.npz"
            protocol = args.protocol_root.resolve() / f"seed{seed}/protocol.json"
            score = score_archive(archive, protocol, seed)
            scores.append(score)
            per_seed.append({"configuration": label, "seed": seed, **score})
        row: dict[str, object] = {"configuration": label, "rff_sample_size": rff, "seeds": len(scores)}
        for metric in METRICS:
            mean, sd = aggregate([score[metric] for score in scores])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        summaries.append(row)

    rff128 = next(row for row in summaries if row["configuration"].endswith("RFF=128)"))
    rff256 = next(row for row in summaries if row["configuration"].endswith("RFF=256)"))
    delta: dict[str, object] = {"comparison": "RFF256_minus_RFF128", "seeds": len(SEEDS)}
    for metric in METRICS:
        delta[f"{metric}_mean"] = float(rff256[f"{metric}_mean"]) - float(rff128[f"{metric}_mean"])

    write_csv(output / "hippo_matern_rff_per_seed.csv", per_seed)
    write_csv(output / "hippo_matern_rff_aggregate.csv", summaries)
    write_csv(output / "hippo_matern_rff_delta_256_minus_128.csv", [delta])

    lines = [
        "# COVID Matérn cumulative-HiPPO capacity comparison",
        "",
        "All rows use VFE, the same COVID protocol, Mt=32, Ms=32, Matérn-3/2, and identical RFF counts in calibration and online prediction.",
        "",
        "| Configuration | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['configuration']} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | "
            f"{row['crps_mean']:.4f} +/- {row['crps_sd']:.4f} | "
            f"{row['native_gaussian_nlpd_mean']:.4f} +/- {row['native_gaussian_nlpd_sd']:.4f} | "
            f"{row['ece_mean']:.4f} +/- {row['ece_sd']:.4f} | "
            f"{row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The primary capacity contrast is RFF=256 versus RFF=128; Kron-STGP is included only as the point-inducing VFE reference.",
            "",
            f"RFF=256 minus RFF=128: RMSE {delta['rmse_mean']:+.6f}, CRPS {delta['crps_mean']:+.6f}, Gaussian NLPD {delta['native_gaussian_nlpd_mean']:+.6f}, ECE {delta['ece_mean']:+.6f}, Coverage90 {delta['coverage90_mean']:+.6f}.",
        ]
    )
    (output / "hippo_matern_rff_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "status": "complete",
        "protocol_root": str(args.protocol_root.resolve()),
        "sources": {key: str(path) for key, path in roots.items()},
        "common_settings": {"training_objective": "vfe", "temporal_kernel": "matern32", "mt": 32, "ms": 32, "formal_seeds": list(SEEDS)},
        "rff_policy": "Calibration and online runner use the same RFF sample size within each HiPPO configuration.",
        "output_files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
