#!/usr/bin/env python3
"""Validate and summarize the budget-relaxed 4090 Setting-B runs.

This deliberately writes an exploratory report rather than a main-table
fairness lock.  A valid archive is necessary for the report, but Task-1 fits
that did not satisfy the original convergence or capacity gates remain marked
as exploratory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.formalization import validate_prediction_archive
from scripts.compute_covid_long_final_metric_system import gaussian_metrics_on_common_scale, mean_sd


SEEDS = (5, 6, 7)
CONVERGED = {"converged_elbo_plateau", "converged_objective_plateau"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("baselines/covid_long_setting_b/results/formal_repaired_4090_v5"),
    )
    parser.add_argument(
        "--lmc-root",
        type=Path,
        default=Path(
            "baselines/covid_long_setting_b/results/convergence_repair_v1/"
            "gpu_execution_plan/prelock_lmc_gpu_s5_s7_v1"
        ),
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("baselines/covid_long_setting_b/reports/exploratory_gpu_4090_s5_s7"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def method_specs(output_root: Path, lmc_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": "lmc_svgp",
            "label": "LMC-SVGP",
            "root": lmc_root,
            "child": "lmc_svgp",
            "status_name": "result.json",
            "qualification": "exploratory_budget_limited",
        },
        {
            "id": "st_svgp",
            "label": "ST-SVGP causal refit",
            "root": output_root / "st_svgp_short_budget_2500_from_checkpoint_cuda118_v3",
            "child": "st_svgp",
            "status_name": "status.json",
            "qualification": "exploratory_short_budget_capacity_unresolved",
        },
        {
            "id": "imc_svgp",
            "label": "IMC-SVGP",
            "root": output_root / "imc_accelerated_budget_relaxed_gpu_v1",
            "child": "imc_svgp",
            "status_name": "result.json",
            "qualification": "exploratory_budget_relaxed",
        },
        {
            "id": "fsde_svi",
            "label": "FSDE-SVI",
            "root": output_root / "fsde_accelerated_budget_relaxed_gpu_v1",
            "child": "fsde_svi",
            "status_name": "result.json",
            "qualification": "exploratory_budget_relaxed",
        },
    ]


def audit_passed(audit: dict[str, Any]) -> bool:
    return (
        audit.get("passed") is True
        and audit.get("current_hidden_labels_read") == 0
        and audit.get("delayed_hidden_labels") == audit.get("expected_delayed_hidden_labels")
        and audit.get("hidden_predictions") == audit.get("expected_hidden_predictions")
    )


def collect_method(
    spec: dict[str, Any],
    *,
    seeds: list[int],
    protocol_root: Path,
    method_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        output = Path(spec["root"]) / f"seed{seed}" / str(spec["child"])
        archive = output / "predictions.npz"
        status_path = output / str(spec["status_name"])
        base = {
            "method": spec["id"],
            "label": spec["label"],
            "seed": seed,
            "qualification": spec["qualification"],
            "archive": str(archive),
            "status_path": str(status_path),
        }
        try:
            validate_prediction_archive(archive)
            status = json.loads(status_path.read_text(encoding="utf-8"))
            audit = status.get("audit", {})
            if not audit_passed(audit):
                raise ValueError(f"causal audit did not pass: {audit}")
            convergence = status.get("task1_convergence", {}).get("status")
            protocol = json.loads(
                (protocol_root / f"seed{seed}" / "protocol.json").read_text(encoding="utf-8")
            )
            with np.load(archive, allow_pickle=False) as arrays:
                values = {
                    key: np.asarray(arrays[key], dtype=np.float64)
                    for key in ("y_true", "pred_mean", "pred_var")
                }
            metrics = gaussian_metrics_on_common_scale(
                values,
                protocol["target_standardization"],
                ece_seed=80_000_000 + 10_000 * method_index + seed,
            )
            rows.append(
                {
                    **base,
                    "archive_status": "valid",
                    "task1_convergence": convergence,
                    "strict_convergence_passed": convergence in CONVERGED,
                    **metrics,
                }
            )
        except Exception as error:  # The report records evidence instead of hiding a failed run.
            failures.append({**base, "archive_status": "invalid_or_missing", "error": repr(error)})
    return rows, failures


def render_markdown(
    aggregate: list[dict[str, Any]], failures: list[dict[str, Any]], path: Path
) -> None:
    lines = [
        "# Accelerated 4090 Exploratory Results",
        "",
        "All rows below use the common Gaussian metric evaluator on restored `log1p(admissions per 100k)`.",
        "They are not admitted to the strict main table: the original capacity/convergence gates were relaxed for this cloud-budget run.",
        "",
        "| Method | Qualification | Seeds | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['label']} | {row['qualification']} | {row['seeds']} | "
            f"{row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | "
            f"{row['crps_mean']:.4f} +/- {row['crps_sd']:.4f} | "
            f"{row['native_gaussian_nlpd_mean']:.4f} +/- {row['native_gaussian_nlpd_sd']:.4f} | "
            f"{row['ece_mean']:.4f} +/- {row['ece_sd']:.4f} | "
            f"{row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} |"
        )
    if failures:
        lines.extend(["", "## Missing Or Invalid Archives", ""])
        for row in failures:
            lines.append(f"- `{row['method']}` seed {row['seed']}: `{row['error']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = absolute(args.output_root)
    lmc_root = absolute(args.lmc_root)
    protocol_root = absolute(args.protocol_root)
    report_dir = absolute(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, spec in enumerate(method_specs(output_root, lmc_root)):
        rows, invalid = collect_method(
            spec,
            seeds=list(args.seeds),
            protocol_root=protocol_root,
            method_index=index,
        )
        per_seed.extend(rows)
        failures.extend(invalid)

    aggregate: list[dict[str, Any]] = []
    for spec in method_specs(output_root, lmc_root):
        rows = [row for row in per_seed if row["method"] == spec["id"]]
        if len(rows) != len(args.seeds):
            continue
        record: dict[str, Any] = {
            "method": spec["id"],
            "label": spec["label"],
            "qualification": spec["qualification"],
            "seeds": len(rows),
            "all_strict_convergence_passed": all(bool(row["strict_convergence_passed"]) for row in rows),
        }
        for metric in ("rmse", "crps", "native_gaussian_nlpd", "ece", "coverage90"):
            record[f"{metric}_mean"], record[f"{metric}_sd"] = mean_sd(
                [float(row[metric]) for row in rows]
            )
        aggregate.append(record)

    status = {
        "status": "complete" if not failures else "complete_with_invalid_or_missing_archives",
        "strict_main_table_admission": False,
        "seeds": list(args.seeds),
        "valid_archives": len(per_seed),
        "invalid_or_missing_archives": len(failures),
        "aggregate_methods": [row["method"] for row in aggregate],
        "note": "This is an expedited exploratory report; it does not replace BASELINE_FAIRNESS_PROTOCOL.",
    }
    write_csv(per_seed, report_dir / "per_seed_metrics.csv")
    write_csv(aggregate, report_dir / "aggregate_metrics.csv")
    (report_dir / "validation_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    (report_dir / "invalid_or_missing_archives.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    render_markdown(aggregate, failures, report_dir / "REPORT.md")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
