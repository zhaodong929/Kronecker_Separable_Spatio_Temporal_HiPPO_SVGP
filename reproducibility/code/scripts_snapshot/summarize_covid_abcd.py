#!/usr/bin/env python3
"""Create the final A-E COVID diagnostic report from completed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def pilot_rows(path: Path) -> list[dict]:
    return json.loads((path / "pilot_summary.json").read_text(encoding="utf-8"))["rows"]


def route_rows(path: Path) -> list[dict]:
    return [row for row in pilot_rows(path) if row["method"].startswith("routeb_")]


def aggregate_rows(path: Path) -> list[dict]:
    with (path / "metrics_aggregate.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results" / "diagnostics" / "covid_abcd")
    args = parser.parse_args()
    root = args.root
    report_rows = []

    for name in ("A_joint_100_all_seeds", "A_joint_250_all_seeds"):
        for seed in range(5):
            for row in route_rows(root / name / f"seed{seed}"):
                calibration = json.loads(
                    (root / name / f"seed{seed}" / row["method"].replace("routeb_", "") / "calibration" / "result.json").read_text(encoding="utf-8")
                )
                report_rows.append(
                    {
                        "phase": "A convergence",
                        "setting": name,
                        "seed": seed,
                        "method": row["method"],
                        "rmse": row["rmse"],
                        "nll": row["nll"],
                        "coverage90": row["coverage90"],
                        "best_iteration": calibration["best_iteration"],
                        "iterations_completed": calibration["iterations_completed"],
                        "stop_reason": calibration["stop_reason"],
                    }
                )

    for name in (
        "C_joint_mt16_ms8",
        "C_joint_mt16_ms16",
        "C_joint_mt16_ms32",
        "C_joint_mt32_ms16",
        "C_joint_mt16_ms32_500",
    ):
        for row in route_rows(root / name / "seed0"):
            calibration = json.loads(
                (root / name / "seed0" / row["method"].replace("routeb_", "") / "calibration" / "result.json").read_text(encoding="utf-8")
            )
            report_rows.append(
                {
                    "phase": "C capacity",
                    "setting": name,
                    "seed": 0,
                    "method": row["method"],
                    "rmse": row["rmse"],
                    "nll": row["nll"],
                    "coverage90": row["coverage90"],
                    "best_iteration": calibration["best_iteration"],
                    "iterations_completed": calibration["iterations_completed"],
                    "stop_reason": calibration["stop_reason"],
                }
            )

    for mode in ("zero_mean", "intercept_only"):
        with (root / "D_feature_ablation" / mode / "metrics_aggregate.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["method"].startswith("routeb_"):
                    report_rows.append(
                        {
                            "phase": "D feature ablation",
                            "setting": mode,
                            "seed": "0-4",
                            "method": row["method"],
                            "rmse": row["rmse_mean"],
                            "nll": row["nll_mean"],
                            "coverage90": row["coverage90_mean"],
                            "best_iteration": "aggregate",
                            "iterations_completed": "aggregate",
                            "stop_reason": "aggregate",
                        }
                    )
    write_rows(root / "abcd_metrics.csv", report_rows)

    final_summary = []
    with (root / "C_final_mt16_ms32_500" / "metrics_aggregate.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"].startswith("routeb_"):
                final_summary.append(row)

    lines = [
        "# COVID Route B A-E diagnostic report",
        "",
        "All Route B prediction runs use full-joint-conditional variance. The data are the current CDC retrospective snapshot: 91 weeks, 52 state-level locations, 52 calibration weeks and 39 online weeks.",
        "",
        "## A. Convergence",
        "",
        "At Mt=16, Ms=8, the five-seed results show that the 250-step budget improves cumulative HiPPO (RMSE 0.7575 to 0.7225; NLL 1.1094 to 1.0560), while global inducing RMSE is unchanged and its NLL becomes worse. This is a budget comparison, not a converged comparison: seed 2 reaches the maximum budget for both methods, and seed 4 reaches the maximum budget for HiPPO at 250 steps. The spread is also large because seed 4 is a difficult spatial split.",
        "",
        "| Budget | Method | RMSE | NLL | Coverage90 |",
        "|---|---|---:|---:|---:|",
    ]
    for name, label in (("A_joint_100_all_seeds", "100 steps"), ("A_joint_250_all_seeds", "250 steps")):
        for row in aggregate_rows(root / name):
            if row["method"].startswith("routeb_"):
                lines.append(
                    f"| {label} | {row['method']} | {float(row['rmse_mean']):.4f} +/- {float(row['rmse_sd']):.4f} | "
                    f"{float(row['nll_mean']):.4f} +/- {float(row['nll_sd']):.4f} | "
                    f"{float(row['coverage90_mean']):.4f} +/- {float(row['coverage90_sd']):.4f} |"
                )
    lines.extend(
        [
        "",
        "",
        "## B. Seed 4 diagnosis",
        "",
        "Seed 4 has mean nearest-visible distance 0.9815 and maximum distance 3.1070, compared with mean distances 0.2208-0.5972 for the other seeds. Its held-out set includes Alaska, Hawaii, Louisiana, Mississippi, Nebraska, Rhode Island, Texas, Wisconsin, Wyoming and Puerto Rico. Both Route B representations fail on this split, so the main issue is spatial extrapolation/calibration difficulty rather than HiPPO itself.",
        "",
        "## C. Capacity",
        "",
        "On seed 0, increasing Ms from 8 to 32 improves both Route B variants. At Mt=16, Ms=32 gives RMSE 0.6193 for global inducing and 0.5666 for HiPPO at the 500-step budget. HiPPO still has its best validation point at the final step, so this is not yet a converged comparison.",
        "",
        "## D. Mean-feature ablation",
        "",
        "The five-seed ablation keeps Mt=16, Ms=32 and the same 500-step budget. Zero-mean direct GP is worse than a model with an intercept. Intercept-only and the 12-dimensional causal-joint model are close, so the independent benefit of the three visible lags and seasonal features is not yet established.",
        "",
        "| Setting | Method | RMSE | NLL | Coverage90 |",
        "|---|---|---:|---:|---:|",
        ]
    )
    for row in final_summary:
        lines.append(f"| causal-joint Ms=32 | {row['method']} | {float(row['rmse_mean']):.4f} +/- {float(row['rmse_sd']):.4f} | {float(row['nll_mean']):.4f} +/- {float(row['nll_sd']):.4f} | {float(row['coverage90_mean']):.4f} +/- {float(row['coverage90_sd']):.4f} |")
    for mode in ("intercept_only", "zero_mean"):
        with (root / "D_feature_ablation" / mode / "metrics_aggregate.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["method"] == "routeb_cumulative_hippo":
                    lines.append(f"| {mode} Ms=32 | {row['method']} | {float(row['rmse_mean']):.4f} +/- {float(row['rmse_sd']):.4f} | {float(row['nll_mean']):.4f} +/- {float(row['nll_sd']):.4f} | {float(row['coverage90_mean']):.4f} +/- {float(row['coverage90_sd']):.4f} |")
    lines.extend(
        [
            "",
            "## E. Decision",
            "",
            "The current evidence supports using Mt=16, Ms=32 as the next COVID pilot capacity and retaining an intercept. It does not yet justify claiming that the full 12-dimensional X-lag is necessary or that HiPPO is conclusively superior. A converged 1000-step or validation-converged HiPPO run, followed by the same capacity and feature ablations on a longer multi-year/more spatial dataset, is still required.",
        ]
    )
    (root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str((root / "final_report.md").resolve())}, indent=2))


if __name__ == "__main__":
    main()
