#!/usr/bin/env python3
"""Audit every local Official ST-SVGP artifact without modifying raw archives."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist, mean, stdev

import numpy as np


ROOT = Path("results")
OUTDIR = ROOT / "official_st_svgp_all_versions_audit_20260828"
REPORT = ROOT / "OFFICIAL_ST_SVGP_ALL_VERSIONS_AUDIT_20260828.md"
NORMAL = NormalDist()
ECE_LEVELS = tuple(round(x, 2) for x in np.arange(0.05, 1.0, 0.10))


def rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def mean_sd(values: list[float]) -> str:
    if not values:
        return "NA"
    if len(values) == 1:
        return fmt(values[0])
    return f"{fmt(mean(values))} +/- {fmt(stdev(values))}"


def gaussian_metrics(y_true: np.ndarray, pred_mean: np.ndarray, pred_var: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    pred_mean = np.asarray(pred_mean, dtype=float).reshape(-1)
    pred_var = np.asarray(pred_var, dtype=float).reshape(-1)
    sigma = np.sqrt(pred_var)
    error = y_true - pred_mean
    z = error / sigma
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    crps = np.mean(sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi)))
    nlpd = np.mean(0.5 * (np.log(2.0 * math.pi * pred_var) + error * error / pred_var))
    coverage = []
    for level in ECE_LEVELS:
        half_width = NORMAL.inv_cdf((1.0 + level) / 2.0) * sigma
        coverage.append(float(np.mean(np.abs(error) <= half_width)))
    half90 = NORMAL.inv_cdf(0.95) * sigma
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "crps": float(crps),
        "nlpd": float(nlpd),
        "ece": float(np.mean(np.abs(np.asarray(coverage) - np.asarray(ECE_LEVELS)))),
        "coverage90": float(np.mean(np.abs(error) <= half90)),
    }


def audit_prediction(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"archive_status": "missing_predictions"}
    try:
        with np.load(path) as archive:
            needed = ("y_true", "pred_mean", "pred_var")
            missing = [key for key in needed if key not in archive]
            if missing:
                return {"archive_status": f"missing_keys:{','.join(missing)}"}
            arrays = {key: np.asarray(archive[key]) for key in needed}
    except Exception as exc:  # The audit should report corrupt archives rather than stop.
        return {"archive_status": f"unreadable:{type(exc).__name__}"}
    shapes = {tuple(value.shape) for value in arrays.values()}
    if len(shapes) != 1:
        return {"archive_status": f"shape_mismatch:{sorted(map(str, shapes))}"}
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        return {"archive_status": "nonfinite"}
    if np.any(arrays["pred_var"] <= 0.0):
        return {"archive_status": "nonpositive_variance"}
    return {"archive_status": "valid", "prediction_shape": "x".join(map(str, arrays["y_true"].shape)), **gaussian_metrics(**arrays)}


def version_for(path: Path, result: dict) -> str:
    text = rel(path)
    if "p0b_official_full_era5" in text:
        return "ERA5 short direct-target, no X-lag (P0b)"
    if "phase_d_joint_xlag_controlled" in text:
        return "ERA5 short direct-target, joint X-lag (Phase D)"
    if "phase_h_xlag_temporal_sparse_markov" in text:
        return "ERA5 short direct-target, full Markov no X-lag (Phase H)"
    return "Unclassified ST-SVGP result"


def collect_official_archives() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result_path in sorted(ROOT.rglob("result.json")):
        try:
            result = json_read(result_path)
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("model") != "st_svgp":
            continue
        run_dir = result_path.parent
        audit = audit_prediction(run_dir / "predictions.npz")
        row: dict[str, object] = {
            "category": "official_full_era5",
            "version": version_for(result_path, result),
            "path": rel(run_dir),
            "seed": result.get("seed"),
            "Ms": result.get("num_spatial_inducing"),
            "uses_xlag": result.get("uses_xlag"),
            "temporal_protocol": result.get("temporal_protocol"),
            "num_time_steps": result.get("num_time_steps"),
            "num_train_space": result.get("num_train_space"),
            "num_test_space": result.get("num_test_space"),
            "reported_rmse": result.get("rmse"),
            "reported_nlpd": result.get("nll"),
            "reported_coverage90": result.get("coverage90"),
            "iterations": result.get("iterations"),
            "train_seconds": result.get("train_seconds"),
            **audit,
        }
        if audit["archive_status"] == "valid":
            for metric, reported in (("rmse", result.get("rmse")), ("nlpd", result.get("nll")), ("coverage90", result.get("coverage90"))):
                row[f"{metric}_abs_delta"] = abs(float(row[metric]) - float(reported))
        rows.append(row)
    return rows


def collect_subset_smokes() -> list[dict[str, object]]:
    base = ROOT / "experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/post_meeting_priority_2026-07-14/p0_official_stvgp"
    specs = (("Synthetic smoke", base / "synthetic_st_svgp.json", None), ("ERA5 16-location subset smoke", base / "era5_subset_st_svgp.json", base / "era5_subset_st_svgp_predictions.npz"))
    rows = []
    for name, result_path, prediction_path in specs:
        result = json_read(result_path)
        row: dict[str, object] = {
            "category": "official_subset_or_smoke",
            "version": name,
            "path": rel(result_path.parent),
            "seed": result.get("seed"),
            "Ms": result.get("num_spatial_inducing"),
            "uses_xlag": result.get("uses_xlag"),
            "temporal_protocol": result.get("temporal_protocol"),
            "num_time_steps": result.get("num_time_steps"),
            "num_train_space": result.get("num_train_space"),
            "num_test_space": result.get("num_test_space"),
            "reported_rmse": result.get("rmse"),
            "reported_nlpd": result.get("nll"),
            "reported_coverage90": result.get("coverage90"),
            "iterations": result.get("iterations"),
            "train_seconds": result.get("train_seconds"),
        }
        if prediction_path:
            row.update(audit_prediction(prediction_path))
        else:
            row["archive_status"] = "no_prediction_archive_recorded"
        rows.append(row)
    return rows


def collect_failures() -> list[dict[str, object]]:
    rows = []
    for status_path in sorted(ROOT.rglob("status.json")):
        path_text = rel(status_path).lower()
        if "official_st_svgp" not in path_text:
            continue
        status = json_read(status_path)
        run_dir = status_path.parent
        if (run_dir / "result.json").is_file():
            continue
        rows.append({
            "category": "official_failed_or_unstarted",
            "version": "ERA5 candidate",
            "path": rel(run_dir),
            "scope": status.get("scope"),
            "seed": status.get("seed"),
            "Ms": status.get("num_spatial_inducing"),
            "status": status.get("status"),
            "reason": status.get("failure_reason") or status.get("note") or status.get("consequence"),
            "has_prediction_archive": (run_dir / "predictions.npz").is_file(),
        })
    interrupted = ROOT / "experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison/phase_d_shared_xlag_controlled/seed0/official_st_svgp_Ms64"
    if interrupted.is_dir() and not (interrupted / "result.json").is_file():
        rows.append({
            "category": "official_failed_or_unstarted",
            "version": "ERA5 candidate",
            "path": rel(interrupted),
            "scope": "task1_2",
            "seed": 0,
            "Ms": 64,
            "status": "interrupted_without_result",
            "reason": "Official CPU run reached the first iteration and was interrupted; no result or prediction archive was written.",
            "has_prediction_archive": False,
        })
    return rows


def collect_covid_diagnostics() -> list[dict[str, object]]:
    root = ROOT / "diagnostics/covid_long_external_baselines/st_svgp"
    rows = []
    for status_path in sorted(root.glob("*/status.json")):
        status = json_read(status_path)
        audit = status.get("audit", {})
        run_dir = status_path.parent
        rows.append({
            "category": "covid_partial_diagnostic",
            "version": status.get("method"),
            "path": rel(run_dir),
            "seed": status.get("seed"),
            "weeks": status.get("weeks"),
            "status": status.get("status"),
            "archive_status": audit_prediction(run_dir / "predictions.npz").get("archive_status"),
            "current_hidden_labels_read": audit.get("current_hidden_labels_read"),
            "hidden_predictions": audit.get("hidden_predictions"),
            "expected_hidden_predictions": audit.get("expected_hidden_predictions"),
            "protocol_audit_passed": audit.get("passed"),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("archive_status") == "valid":
            groups[(row["version"], row["Ms"])].append(row)
    summary = []
    for (version, ms), group in sorted(groups.items()):
        record: dict[str, object] = {"version": version, "Ms": ms, "valid_seeds": len(group)}
        for metric in ("rmse", "crps", "nlpd", "ece", "coverage90", "train_seconds"):
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        record["prediction_shapes"] = ";".join(sorted({str(row["prediction_shape"]) for row in group}))
        summary.append(record)
    return summary


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    body = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    body.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(body)


def build_report(archives: list[dict[str, object]], summary: list[dict[str, object]], smokes: list[dict[str, object]], failures: list[dict[str, object]], covid: list[dict[str, object]]) -> str:
    formal_rows = []
    for row in summary:
        formal_rows.append([
            str(row["version"]), str(row["Ms"]), str(row["valid_seeds"]), row["prediction_shapes"],
            mean_sd([float(item["rmse"]) for item in archives if item.get("version") == row["version"] and item.get("Ms") == row["Ms"] and item.get("archive_status") == "valid"]),
            mean_sd([float(item["crps"]) for item in archives if item.get("version") == row["version"] and item.get("Ms") == row["Ms"] and item.get("archive_status") == "valid"]),
            mean_sd([float(item["nlpd"]) for item in archives if item.get("version") == row["version"] and item.get("Ms") == row["Ms"] and item.get("archive_status") == "valid"]),
            mean_sd([float(item["ece"]) for item in archives if item.get("version") == row["version"] and item.get("Ms") == row["Ms"] and item.get("archive_status") == "valid"]),
            mean_sd([float(item["coverage90"]) for item in archives if item.get("version") == row["version"] and item.get("Ms") == row["Ms"] and item.get("archive_status") == "valid"]),
        ])
    smoke_rows = [[str(row["version"]), str(row["Ms"]), str(row["num_time_steps"]), f'{row["num_train_space"]}/{row["num_test_space"]}', fmt(row.get("reported_rmse")), fmt(row.get("reported_nlpd")), fmt(row.get("reported_coverage90")), str(row["archive_status"])] for row in smokes]
    failure_rows = [[str(row.get("scope") or "NA"), str(row.get("seed") if row.get("seed") is not None else "NA"), str(row.get("Ms") if row.get("Ms") is not None else "NA"), str(row.get("status")), str(row.get("path"))] for row in failures]
    covid_rows = [[str(row["path"]).rsplit("/", 1)[-1], str(row["weeks"]), str(row["archive_status"]), str(row["current_hidden_labels_read"]), f'{row["hidden_predictions"]}/{row["expected_hidden_predictions"]}', str(row["protocol_audit_passed"])] for row in covid]
    return f"""# Official ST-SVGP: All Local Result Versions\n\nGenerated from the current `results/` tree on 2026-08-28. This audit does not modify any source archive. A result is counted only when its `result.json` declares `model = st_svgp` and the co-located `predictions.npz` has finite `y_true`, `pred_mean`, `pred_var`, matching shapes, and strictly positive predictive variance.\n\n## Scope\n\n- **Counted official full-ERA5 archives:** {len(archives)} run archives, all valid.\n- **Not equivalent:** different X-lag mean settings are distinct task/model versions and must not be pooled.\n- **Not included as Official ST-SVGP:** `mf_st_svgp`, `st_dsvgp` temporal-sparse extensions, `stvgp_exact_*` local Kronecker refits, and `official_ohsvgp_*`.\n- **Metric convention:** CRPS is analytic Gaussian CRPS; NLPD is Gaussian negative log predictive density; ECE averages central-interval coverage errors at 5%, 15%, ..., 95%.\n\n## Valid Full ERA5 Results\n\n{markdown_table(["Version", "Ms", "Seeds", "Prediction shape", "RMSE", "CRPS", "Gaussian NLPD", "ECE", "Coverage90"], formal_rows)}\n\nAll rows above use 186 time steps with an 800/200 held-out spatial split and have zero archive-integrity failures. The differences are intentional protocol differences:\n\n- **P0b:** historical direct-target official run, full Markov temporal state, no X-lag mean.\n- **Phase D:** direct-target official run with the joint learned X-lag mean.\n- **Phase H:** full-Markov official run with no X-lag, included to isolate the X-lag condition from the temporal-sparse extension.\n\nThe per-archive CSV and machine-readable aggregate are in `{rel(OUTDIR)}/`.\n\n## Official Smoke/Subsets\n\n{markdown_table(["Run", "Ms", "Time steps", "Train/test space", "Reported RMSE", "Reported NLPD", "Coverage90", "Archive check"], smoke_rows)}\n\nThese tests establish environment/API behavior only. They are not full ERA5 benchmark rows and must not be compared to the 800/200 formal results.\n\n## Failed Or Unstarted Official Full-Scale Attempts\n\n{markdown_table(["Scope", "Seed", "Ms", "Status", "Path"], failure_rows)}\n\nNo metrics are inferred for these attempts. In particular, the long `task1_10` probe failed before its first iteration because the official full-Markov computation requested 20.27 GiB; the new short/long batch addendum has no official archive; and the local SM-VFE-era wrapper attempt was stopped after resource-limited execution.\n\n## COVID Official-Core Diagnostics\n\n{markdown_table(["Diagnostic", "Weeks", "Prediction archive", "Current hidden reads", "Predictions/required", "Full protocol passed"], covid_rows)}\n\nThese are partial Setting-B causal-refit diagnostics, not completed 143-week formal ST-SVGP runs. Their `passed=false` status reflects incomplete horizons/counts, even though recorded current-hidden reads are zero. They must not be placed in a COVID main table.\n\n## Explicit Exclusions\n\n- `mf_st_svgp_Ms30` in P0b is the mean-field ST-SVGP variant, not the standard Official ST-SVGP.\n- `temporal_sparse_*` has `model = st_dsvgp`; it uses a temporal-inducing doubly-sparse extension rather than the official full-Markov ST-SVGP representation.\n- `stvgp_exact_zero` and `stvgp_exact_xlag_ridge` are local separable-Kronecker refit baselines. Their reports reference the official repository but they are not official Bayes-Newton results.\n- `official_ohsvgp_*` is OHSVGP, a different method.\n\n## Artifact Paths\n\n- Historical P0b: `results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/post_meeting_priority_2026-07-14/p0b_official_full_era5`\n- Phase D: `results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison/phase_d_joint_xlag_controlled`\n- Phase H: `results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison/phase_h_xlag_temporal_sparse_markov`\n- Detailed audit: `{rel(OUTDIR)}`\n"""


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    archives = collect_official_archives()
    smokes = collect_subset_smokes()
    failures = collect_failures()
    covid = collect_covid_diagnostics()
    summary = summarize(archives)
    write_csv(OUTDIR / "per_archive.csv", archives)
    write_csv(OUTDIR / "summary.csv", summary)
    write_csv(OUTDIR / "subset_and_smoke.csv", smokes)
    write_csv(OUTDIR / "failed_or_unstarted.csv", failures)
    write_csv(OUTDIR / "covid_partial_diagnostics.csv", covid)
    REPORT.write_text(build_report(archives, summary, smokes, failures, covid), encoding="utf-8")
    print(json.dumps({"archives": len(archives), "valid_archives": sum(row["archive_status"] == "valid" for row in archives), "failures": len(failures), "report": rel(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
