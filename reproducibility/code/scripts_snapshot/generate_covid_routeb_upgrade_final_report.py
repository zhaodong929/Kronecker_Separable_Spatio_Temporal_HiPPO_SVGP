#!/usr/bin/env python3
"""Create the final, artifact-audited report for the COVID Route B pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from scripts.run_epidemiology_pilot import predictive_metrics
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_epidemiology_pilot import predictive_metrics


METRICS = ("rmse", "nll", "coverage90")


def archive_metrics(path: Path) -> dict[str, float]:
    with np.load(path) as archive:
        y_true = np.asarray(archive["y_true"], dtype=np.float64)
        pred_mean = np.asarray(archive["pred_mean"], dtype=np.float64)
        pred_var = np.asarray(archive["pred_var"], dtype=np.float64)
    if not np.isfinite(y_true).all() or not np.isfinite(pred_mean).all():
        raise ValueError(f"{path}: non-finite target or predictive mean")
    if not np.isfinite(pred_var).all() or (pred_var <= 0.0).any():
        raise ValueError(f"{path}: predictive variance is not finite and positive")
    return {key: float(value) for key, value in predictive_metrics(y_true, pred_mean, pred_var).items()}


def audited_run(
    root: Path,
    relative_run: str,
    candidate: str,
    phase: str,
    decision: str,
    notes: str,
) -> dict[str, object]:
    run = root / relative_run
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    metrics = archive_metrics(run / "predictions.npz")
    reported = result["overall_current_block"]
    for key in METRICS:
        if abs(metrics[key] - float(reported[key])) > 1e-8:
            raise ValueError(f"{candidate}: {key} differs from result.json")
    return {
        "phase": phase,
        "candidate": candidate,
        "decision": decision,
        "rmse": metrics["rmse"],
        "nll": metrics["nll"],
        "coverage90": metrics["coverage90"],
        "runtime_seconds": float(result["timing"]["process_total_seconds"]),
        "steady_update_ms": 1000.0 * float(result["timing"]["mean_steady_state_block_update_seconds"]),
        "peak_allocated_mib": float(result["resources"]["peak_cuda_allocated_mib"]),
        "peak_reserved_mib": float(result["resources"]["peak_cuda_reserved_mib"]),
        "persistent_state_mib": float(result["resources"]["persistent_state_mib"]),
        "artifact": str((run / "predictions.npz").relative_to(root)),
        "notes": notes,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Phase | Candidate | Decision | RMSE | NLL | Coverage90 | Runtime (s) | Steady update (ms) | Peak alloc (MiB) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['candidate']} | {row['decision']} | {row['rmse']:.6f} | "
            f"{row['nll']:.6f} | {row['coverage90']:.4f} | {row['runtime_seconds']:.3f} | "
            f"{row['steady_update_ms']:.3f} | {row['peak_allocated_mib']:.3f} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("results/diagnostics/covid_routeb_upgrade")
    )
    args = parser.parse_args()
    root = args.root
    output = root / "final"
    output.mkdir(parents=True, exist_ok=True)

    seed0_rows = [
        audited_run(root, "phase5_delayed_history_mean_seed0", "Current D", "Reference", "reference", "Delayed labels enter mean features only."),
        audited_run(root, "phase6_task1_init_seed0", "P0 visible-state Task-1 posterior init", "P0", "retain", "2184 visible Task-1 rows initialize posterior."),
        audited_run(root, "phase7_p1_seed0/lag4", "P1a reduced lag4 mean", "P1", "retain", "No-regression simplification; 11 active features."),
        audited_run(root, "phase7_p1_seed0/lag8", "P1a lag8 mean", "P1", "discard", "Worse RMSE and NLL."),
        audited_run(root, "phase7_p1_seed0/state_intercept", "P1b state intercept", "P1", "discard", "High-dimensional state effects destabilize this pilot."),
        audited_run(root, "phase7_p1_seed0/state_lag1", "P1b state lag1", "P1", "discard", "Worse than global lag mean."),
        audited_run(root, "phase7_p1_seed0/state_lag1_growth", "P1b state lag1 plus growth", "P1", "discard", "Worse than global lag mean."),
        audited_run(root, "phase8_p2_seed0/neighbour_lag4", "P2 neighbour exposure", "P2", "diagnostic_only", "0.64% RMSE gain is below the predeclared 1-2% promotion threshold."),
        audited_run(root, "phase11_p4_seed0_spatial/graph", "P4 graph kernel", "P4", "discard", "Fixed-theta representation audit; worse than geographic kernel."),
        audited_run(root, "phase11_p4_seed0_spatial/geo_graph", "P4 geo plus graph kernel", "P4", "discard", "Fixed-theta representation audit; worse than geographic kernel."),
    ]
    write_csv(output / "seed0_ablation.csv", seed0_rows)

    reference = audited_run(
        root,
        "phase7_p1_seed0/lag4",
        "Mt=32, Ms=32 geographic",
        "P6",
        "retain",
        "Confirmed capacity; train-only inducing layout.",
    )
    capacity_rows = [
        reference,
        audited_run(root, "phase13_p6_capacity_seed0/mt32_ms16", "Mt=32, Ms=16", "P6", "discard", "No material gain over reference."),
        audited_run(root, "phase13_p6_capacity_seed0/mt48_ms32", "Mt=48, Ms=32", "P6", "discard", "No material gain over reference."),
        audited_run(root, "phase13_p6_capacity_seed0/mt64_ms32", "Mt=64, Ms=32", "P6", "development_only", "Best P6 seed-0 RMSE, but 0.68% gain is unconfirmed and below 1% material-gain threshold."),
        audited_run(root, "phase13_p6_capacity_seed0/mt32_ms52", "Mt=32, Ms=52 all-state coordinates", "P6", "discard_unstable", "Transductive-coordinate layout; finite values but catastrophic numerical divergence."),
    ]
    for row in capacity_rows:
        row["rmse_delta_vs_mt32_ms32"] = float(row["rmse"]) - float(reference["rmse"])
        row["rmse_improvement_percent"] = 100.0 * (float(reference["rmse"]) - float(row["rmse"])) / float(reference["rmse"])
    write_csv(output / "capacity.csv", capacity_rows)

    p3_rows = []
    with (root / "phase10_p3_seed0_state_noise/metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metrics = archive_metrics(root / "phase10_p3_seed0_state_noise" / f"predictions_{row['mode']}.npz")
            for key in METRICS:
                if abs(metrics[key] - float(row[key])) > 1e-8:
                    raise ValueError(f"P3 {row['mode']}: recomputed {key} differs from CSV")
            p3_rows.append({"mode": row["mode"], **metrics, "mean_state_noise_variance": float(row["mean_state_noise_variance"])})
    write_csv(output / "p3_calibration_audit.csv", p3_rows)

    confirmation_rows = []
    with (root / "phase9_confirmation_seeds5_9/metrics_per_seed.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            metrics = archive_metrics(root / f"phase9_confirmation_seeds5_9/seed{seed}/online/predictions.npz")
            for key in METRICS:
                if abs(metrics[key] - float(row[key])) > 1e-8:
                    raise ValueError(f"confirmation seed {seed}: recomputed {key} differs from CSV")
            confirmation_rows.append({"seed": seed, **metrics})
    write_csv(output / "confirmation_seeds5_9.csv", confirmation_rows)
    aggregate = {
        key: {"mean": float(np.mean([row[key] for row in confirmation_rows])), "sample_sd": float(np.std([row[key] for row in confirmation_rows], ddof=1))}
        for key in METRICS
    }

    p5 = json.loads((root / "phase12_p5_drift_seed0/drift.json").read_text(encoding="utf-8"))
    failure = root / "phase8_p2_seed0/failure_diagnostic/heldout_per_state.csv"
    lines = [
        "# Final COVID Route B Optimization Audit",
        "",
        "## Scope",
        "",
        "This is a 52-state, 91-week COVID feasibility pilot: 52 calibration weeks, 39 strict-online one-week blocks, and a 42/4/10 visible/validation/held-out state split. Results are float64 on an RTX 5070 Laptop GPU. It is not a final epidemiology benchmark.",
        "",
        "The retained model is cumulative HiPPO Route B with Q=2 fixed Task-1 spectral-mixture theta, delayed-observation online updates, full-joint-conditional predictive variance including conditional residual, visible-state Task-1 posterior initialization, and the reduced global lag4 mean. It absorbs 2184 visible Task-1 rows and 380 delayed held-out online rows. This is not an all-52-state Task-1 posterior because the present single-Kronecker state cannot exactly mix visible and held-out spatial projection matrices.",
        "",
        "## Seed-0 Structure Screen",
        "",
        *markdown_table(seed0_rows),
        "",
        "P3 is deliberately not in the model table: it is a causal Task-1 residual, post-hoc calibration audit. It leaves the posterior and predictive mean unchanged, so it cannot improve RMSE. `state_raw` reduces NLL from 0.246885 to 0.237234 but raises Coverage90 from 0.9385 to 0.9590; it is not promoted because coverage is already conservative.",
        "",
        "P4 used the seed-0 geographic theta without re-optimizing after substituting graph representations. It is a fixed-theta representation diagnostic, not a fully tuned kernel comparison. Both graph and geo-plus-graph are worse, so the geographic kernel is retained.",
        "",
        "## Drift Decision",
        "",
        f"P5 was skipped by its preregistered trigger: mean block RMSE was {p5['summary']['rmse']['early']:.6f} early, {p5['summary']['rmse']['middle']:.6f} middle, and {p5['summary']['rmse']['late']:.6f} late (slope {p5['summary']['rmse']['linear_slope_per_block']:.6f} per block). There is no late-stream RMSE deterioration warranting periodic theta adaptation.",
        "",
        "## Capacity Screen",
        "",
        "| Candidate | Decision | RMSE | Delta RMSE | Improvement | NLL | Coverage90 | Steady update (ms) | Persistent state (MiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in capacity_rows:
        lines.append(
            f"| {row['candidate']} | {row['decision']} | {row['rmse']:.6f} | {row['rmse_delta_vs_mt32_ms32']:+.6f} | "
            f"{row['rmse_improvement_percent']:+.2f}% | {row['nll']:.6f} | {row['coverage90']:.4f} | "
            f"{row['steady_update_ms']:.3f} | {row['persistent_state_mib']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The P6 `Mt=64, Ms=32` point has the lowest seed-0 RMSE, but it is only a 0.68% development improvement and was not re-confirmed on new spatial splits. The retained capacity is therefore `Mt=32, Ms=32`, which has the independent seeds 5-9 confirmation below.",
            "The `Ms=52` point uses all state coordinates but no held-out labels. It is transductive in coordinates, and its massive finite outputs demonstrate numerical divergence rather than a usable accuracy result.",
            "",
            "## New-Split Confirmation",
            "",
            "| Seeds | RMSE mean +/- SD | NLL mean +/- SD | Coverage90 mean +/- SD |",
            "|---|---:|---:|---:|",
            f"| 5-9 | {aggregate['rmse']['mean']:.4f} +/- {aggregate['rmse']['sample_sd']:.4f} | {aggregate['nll']['mean']:.4f} +/- {aggregate['nll']['sample_sd']:.4f} | {aggregate['coverage90']['mean']:.4f} +/- {aggregate['coverage90']['sample_sd']:.4f} |",
            "",
            "These five seeds were not used for model-structure selection. Seed 9 is materially harder (RMSE 0.435124, Coverage90 0.8000), so this pilot retains meaningful split sensitivity.",
            "",
            "## Failure-State Diagnostic",
            "",
            "The retained seed-0 model's most influential held-out failure is North Dakota: RMSE 0.533191, NLL 0.994671, Coverage90 0.6667, and 30.7% of held-out squared error. Hawaii has the next-largest error contribution but very conservative intervals (Coverage90 0.9744). This pattern argues against a single global variance fix. The full per-state table and five-state prediction curves are retained under `phase8_p2_seed0/failure_diagnostic/`.",
            "",
            "## Audit Outputs",
            "",
            "All rows in `seed0_ablation.csv`, `capacity.csv`, `p3_calibration_audit.csv`, and `confirmation_seeds5_9.csv` were recomputed from prediction archives. The audit requires finite predictive means and strictly positive, finite variances; Route B result.json metrics are checked against those recomputations.",
            "",
            f"Failure-state source: `{failure.relative_to(root)}`.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "artifact_audit.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "prediction_metrics_recomputed": True,
                "seed0_variants": len(seed0_rows),
                "capacity_variants": len(capacity_rows),
                "p3_variants": len(p3_rows),
                "confirmation_seeds": [row["seed"] for row in confirmation_rows],
                "retained_configuration": "Mt=32, Ms=32, geographic, P0 visible Task-1 posterior init, lag4 mean",
                "development_only_best_capacity": "Mt=64, Ms=32",
                "periodic_theta": p5["decision"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "report": str((output / "report.md").resolve()), "confirmation": aggregate}, indent=2))


if __name__ == "__main__":
    main()
