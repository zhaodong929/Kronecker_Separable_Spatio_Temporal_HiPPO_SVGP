#!/usr/bin/env python3
"""Diagnose exact and approximate ranks of the shared ERA5 X-lag design."""

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

from scripts.run_iclr_era5_routeb_batch import load_joint_phi
from scripts.run_iclr_era5_routeb_strict_online import TaskPhiCache


TOLERANCES = (1e-12, 1e-10, 1e-8)
DEFAULT_CANDIDATE_RANKS = (133, 73, 64, 48)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative_projection_error(matrix: np.ndarray, basis: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    denominator = float(np.linalg.norm(matrix, ord="fro"))
    if denominator == 0.0:
        return 0.0
    residual = matrix - (matrix @ basis) @ basis.T
    return float(np.linalg.norm(residual, ord="fro") / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data/era5/processed_timeseries_4_task1_10_extension",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--xlag-length", type=int, default=10)
    parser.add_argument(
        "--candidate-ranks",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_RANKS),
        help="Shared Task-1 SVD ranks for the approximate compression ablation.",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    all_rank_rows: list[dict[str, object]] = []
    all_projection_rows: list[dict[str, object]] = []
    all_spectrum_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        protocol_dir = args.benchmark_root / "protocol" / "task1_10" / f"seed{seed}"
        protocol_npz = protocol_dir / "protocol.npz"
        protocol_json = protocol_dir / "protocol.json"
        metadata = json.loads(protocol_json.read_text(encoding="utf-8"))
        with np.load(protocol_npz) as arrays:
            fit_indices = np.asarray(arrays["fit_indices"], dtype=int)
            stream_y = np.asarray(arrays["stream_y"], dtype=np.float64)
            blocks = [
                slice(int(start), int(stop))
                for start, stop in zip(arrays["block_start"], arrays["block_stop"])
            ]
            calibration_phi, _ = load_joint_phi(
                arrays=arrays,
                protocol_json=protocol_json,
                data_root=args.data_root,
                xlag_length=args.xlag_length,
                data_part="calibration",
            )
        design = np.asarray(
            calibration_phi[:, fit_indices, :], dtype=np.float64
        ).reshape(-1, calibration_phi.shape[-1])
        left_vectors, singular_values, vh = np.linalg.svd(design, full_matrices=False)
        del left_vectors
        relative = singular_values / singular_values[0]
        ranks = {
            tolerance: int(np.count_nonzero(relative > tolerance))
            for tolerance in TOLERANCES
        }
        seed_dir = args.output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        spectrum_rows = [
            {
                "seed": seed,
                "index": index + 1,
                "singular_value": float(value),
                "relative_singular_value": float(rel),
            }
            for index, (value, rel) in enumerate(zip(singular_values, relative))
        ]
        all_spectrum_rows.extend(spectrum_rows)
        write_csv(seed_dir / "singular_values.csv", spectrum_rows)
        for tolerance, rank in ranks.items():
            basis = np.asarray(vh[:rank].T, dtype=np.float64)
            tag = f"tol_{tolerance:.0e}".replace("-", "m")
            np.savez_compressed(
                seed_dir / f"feature_basis_{tag}.npz",
                basis=basis,
                singular_values=singular_values,
                relative_tolerance=np.asarray(tolerance),
                numerical_rank=np.asarray(rank),
                fit_indices=fit_indices,
                seed=np.asarray(seed),
            )
            all_rank_rows.append(
                {
                    "seed": seed,
                    "selection": "relative_tolerance",
                    "relative_tolerance": tolerance,
                    "numerical_rank": rank,
                    "candidate_rank": None,
                    "original_dimension": int(design.shape[1]),
                    "task1_fit_rows": int(design.shape[0]),
                    "task1_relative_projection_error": float(
                        np.sqrt(np.square(singular_values[rank:]).sum() / np.square(singular_values).sum())
                    ),
                }
            )

        candidate_ranks = sorted(
            {min(max(1, int(rank)), design.shape[1]) for rank in args.candidate_ranks},
            reverse=True,
        )
        candidate_bases: dict[int, np.ndarray] = {}
        for rank in candidate_ranks:
            basis = np.asarray(vh[:rank].T, dtype=np.float64)
            candidate_bases[rank] = basis
            np.savez_compressed(
                seed_dir / f"basis_rank{rank}.npz",
                basis=basis,
                singular_values=singular_values,
                candidate_rank=np.asarray(rank),
                fit_indices=fit_indices,
                seed=np.asarray(seed),
                fitted_on=np.asarray("task1_fit_only"),
            )
            task1_error = float(
                np.sqrt(np.square(singular_values[rank:]).sum() / np.square(singular_values).sum())
            )
            all_rank_rows.append(
                {
                    "seed": seed,
                    "selection": "candidate_rank",
                    "relative_tolerance": None,
                    "numerical_rank": ranks[1e-12],
                    "candidate_rank": rank,
                    "original_dimension": int(design.shape[1]),
                    "task1_fit_rows": int(design.shape[0]),
                    "task1_relative_projection_error": task1_error,
                    "compression_class": "exact" if task1_error <= 1e-12 else "approximate",
                }
            )

        norm_sq: dict[tuple[int, float], float] = {}
        residual_sq: dict[tuple[int, float], float] = {}
        candidate_residual_sq: dict[tuple[int, int], float] = {}
        cache = TaskPhiCache(args.data_root, stream_y, args.xlag_length)
        for block in blocks:
            block_phi, task = cache.block(block)
            flat = np.asarray(block_phi, dtype=np.float64).reshape(
                -1, design.shape[1]
            )
            total = float(np.square(flat).sum())
            for tolerance, rank in ranks.items():
                basis = vh[:rank].T
                projected = flat @ basis
                residual = flat - projected @ basis.T
                key = (task, tolerance)
                norm_sq[key] = norm_sq.get(key, 0.0) + total
                residual_sq[key] = residual_sq.get(key, 0.0) + float(
                    np.square(residual).sum()
                )
            for rank, basis in candidate_bases.items():
                projected = flat @ basis
                residual = flat - projected @ basis.T
                key = (task, rank)
                candidate_residual_sq[key] = candidate_residual_sq.get(key, 0.0) + float(
                    np.square(residual).sum()
                )
        for (task, tolerance), total in sorted(norm_sq.items()):
            all_projection_rows.append(
                {
                    "seed": seed,
                    "task": f"task_{task}",
                    "spatial_subset": "all_locations",
                    "selection": "relative_tolerance",
                    "relative_tolerance": tolerance,
                    "numerical_rank": ranks[tolerance],
                    "candidate_rank": None,
                    "relative_projection_error": float(
                        np.sqrt(residual_sq[(task, tolerance)] / total)
                    ),
                }
            )
        for (task, rank), residual in sorted(candidate_residual_sq.items()):
            total = norm_sq[(task, TOLERANCES[0])]
            error = float(np.sqrt(residual / total))
            all_projection_rows.append(
                {
                    "seed": seed,
                    "task": f"task_{task}",
                    "spatial_subset": "all_locations",
                    "selection": "candidate_rank",
                    "relative_tolerance": None,
                    "numerical_rank": ranks[1e-12],
                    "candidate_rank": rank,
                    "relative_projection_error": error,
                    "compression_class": "exact" if error <= 1e-12 else "approximate",
                }
            )

        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.semilogy(np.arange(1, relative.size + 1), relative, marker=".")
        for tolerance in TOLERANCES:
            axis.axhline(tolerance, linestyle="--", linewidth=1, label=f"{tolerance:.0e}")
        axis.set(
            xlabel="Feature singular-value index",
            ylabel="Relative singular value",
            title=f"Task-1 X-lag design spectrum (seed {seed})",
        )
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(seed_dir / "singular_value_spectrum.png", dpi=220)
        plt.close(fig)

    write_csv(args.output_dir / "rank_summary.csv", all_rank_rows)
    write_csv(args.output_dir / "future_projection_error.csv", all_projection_rows)
    write_csv(args.output_dir / "singular_value_spectrum.csv", all_spectrum_rows)
    payload = {
        "protocol": "Task-1 fit-subset SVD; shared basis evaluated causally on Task 2--10 blocks",
        "seeds": args.seeds,
        "relative_tolerances": TOLERANCES,
        "candidate_ranks": args.candidate_ranks,
        "rank_summary": all_rank_rows,
        "future_projection_error": all_projection_rows,
    }
    (args.output_dir / "rank_diagnostic.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    candidate_future = [
        row for row in all_projection_rows if row.get("selection") == "candidate_rank"
    ]
    exact_future = [
        row for row in candidate_future if float(row["relative_projection_error"]) <= 1e-12
    ]
    report = [
        "# Route B feature-rank diagnostic",
        "",
        "The feature basis is fitted only on the Task-1 fit design and then frozen for all future tasks.",
        "Candidate ranks 73, 64 and 48 are accuracy-efficiency ablations, not exact optimizations.",
        "",
        f"Exact future projections at tolerance 1e-12: {len(exact_future)}/{len(candidate_future)}.",
        "Any non-zero future projection error means the corresponding compression changes the model.",
        "",
    ]
    (args.output_dir / "rank_diagnostic_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
