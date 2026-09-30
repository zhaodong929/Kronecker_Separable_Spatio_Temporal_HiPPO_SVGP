#!/usr/bin/env python3
"""Audit and summarize the ERA5 batch addendum result directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.special import erf
from scipy.stats import norm


METHODS = {
    "official_st_svgp": "Official ST-SVGP",
    "kron_stgp_vfe": "Kron-STGP (VFE)",
    "kronhippo_stgp_vfe": "KronHiPPO-STGP (VFE, 256 RFF)",
}

NOMINAL_COVERAGES = np.arange(0.05, 1.0, 0.10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def common_gaussian_metrics(y_true: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-12)
    sigma = np.sqrt(variance)
    standardized = (y_true - mean) / sigma
    normal_cdf = 0.5 * (1.0 + erf(standardized / np.sqrt(2.0)))
    normal_pdf = np.exp(-0.5 * standardized**2) / np.sqrt(2.0 * np.pi)
    crps = (y_true - mean) * (2.0 * normal_cdf - 1.0) + 2.0 * sigma * normal_pdf - sigma / np.sqrt(np.pi)
    coverages = []
    for nominal in NOMINAL_COVERAGES:
        half = norm.ppf(0.5 + nominal / 2.0) * sigma
        coverages.append(np.mean((y_true >= mean - half) & (y_true <= mean + half)))
    return {
        "rmse": float(np.sqrt(np.mean((y_true - mean) ** 2))),
        "nll": float(np.mean(0.5 * (np.log(2.0 * np.pi * variance) + (y_true - mean) ** 2 / variance))),
        "crps": float(np.mean(crps)),
        "ece": float(np.mean(np.abs(np.asarray(coverages) - NOMINAL_COVERAGES))),
        "coverage90": float(coverages[-1]),
    }


def inspect_result(path: Path, scope: str, method: str, seed: int) -> dict:
    status_path = path.parent / "status.json"
    status_payload = {}
    if status_path.is_file():
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    row = {
        "scope": scope,
        "method_id": method,
        "method": METHODS[method],
        "seed": seed,
        "status": status_payload.get("status", "missing"),
        "rmse": "",
        "crps": "",
        "nll": "",
        "ece": "",
        "coverage90": "",
        "prediction_shape": "",
        "result_path": str(path),
        "status_path": str(status_path),
    }
    if not path.is_file():
        return row
    try:
        if status_payload.get("status") not in {"complete", "complete_existing"}:
            row["status"] = status_payload.get("status", "unverified_result")
            return row
        payload = json.loads(path.read_text(encoding="utf-8"))
        final = payload.get("final", payload)
        row["rmse"] = final.get("rmse", payload.get("rmse", ""))
        row["crps"] = final.get("crps", payload.get("crps", ""))
        row["nll"] = final.get("nll", payload.get("nll", ""))
        row["coverage90"] = final.get("coverage90", payload.get("coverage90", ""))
        prediction_path = path.parent / "predictions.npz"
        expected_time = 186 if scope == "task1_2" else 1674
        if not prediction_path.is_file():
            row["status"] = "incomplete_missing_predictions"
        else:
            with np.load(prediction_path) as arrays:
                required = {"y_true", "pred_mean", "pred_var"}
                if not required.issubset(arrays.files):
                    row["status"] = "invalid_missing_prediction_keys"
                else:
                    shape = tuple(arrays["y_true"].shape)
                    row["prediction_shape"] = str(shape)
                    valid = (
                        shape == (expected_time, 200)
                        and arrays["pred_mean"].shape == shape
                        and arrays["pred_var"].shape == shape
                        and np.isfinite(arrays["y_true"]).all()
                        and np.isfinite(arrays["pred_mean"]).all()
                        and np.isfinite(arrays["pred_var"]).all()
                        and (arrays["pred_var"] > 0).all()
                    )
                    if valid:
                        common = common_gaussian_metrics(
                            arrays["y_true"], arrays["pred_mean"], arrays["pred_var"]
                        )
                        row["rmse"] = common["rmse"]
                        row["nll"] = common["nll"]
                        row["crps"] = common["crps"]
                        row["ece"] = common["ece"]
                        row["coverage90"] = common["coverage90"]
                    row["status"] = "complete" if valid and all(
                        finite(row[key]) for key in ("rmse", "nll", "coverage90")
                    ) else "invalid_nonfinite_or_shape"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        row["status"] = f"invalid_{type(exc).__name__}"
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["scope", "method_id", "seed", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scopes", nargs="+", default=["task1_2", "task1_10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    rows = []
    for scope in args.scopes:
        for method in METHODS:
            for seed in args.seeds:
                path = args.output_root / "runs" / scope / method / f"seed{seed}/result.json"
                rows.append(inspect_result(path, scope, method, seed))
    report = args.output_root / "report"
    write_csv(report / "per_seed.csv", rows)

    summary_rows = []
    for scope in args.scopes:
        for method in METHODS:
            complete = [
                row for row in rows
                if row["scope"] == scope and row["method_id"] == method and row["status"] == "complete"
            ]
            summary = {
                "scope": scope,
                "method_id": method,
                "method": METHODS[method],
                "n_complete": len(complete),
                "n_requested": len(args.seeds),
                "status": "complete" if len(complete) == len(args.seeds) else "incomplete",
            }
            for metric in ("rmse", "crps", "nll", "ece", "coverage90"):
                values = [float(row[metric]) for row in complete if finite(row[metric])]
                summary[f"{metric}_mean"] = statistics.mean(values) if values else ""
                summary[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
            summary_rows.append(summary)
    write_csv(report / "summary.csv", summary_rows)

    manifest = {
        "schema_version": 1,
        "purpose": "audited ERA5 batch addendum summary",
        "output_root": str(args.output_root),
        "requested_scopes": args.scopes,
        "requested_seeds": args.seeds,
        "methods": METHODS,
        "old_archives_modified": False,
        "per_seed_path": str(report / "per_seed.csv"),
        "summary_path": str(report / "summary.csv"),
        "run_manifest_sha256": (
            sha256(args.output_root / "run_manifest.json")
            if (args.output_root / "run_manifest.json").is_file()
            else None
        ),
    }
    (report / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    readme = report / "README.md"
    readme.write_text(
        "# ERA5 batch addendum\n\n"
        "This report audits only results produced under the independent addendum root. "
        "The short scope is `task1_2` (Task 1 + Task 2; 186 stream times) and the long "
        "scope is `task1_10` (Task 1 + Tasks 2--10; 1674 stream times). All methods use "
        "the existing 800/200 spatial split, shared Gaussian ERA5 protocol, and fixed "
        "held-out locations. Official ST-SVGP uses the official AaltoML/Bayes-Newton "
        "wrapper with the shared fixed X-lag mean. Kron-STGP and KronHiPPO-STGP use "
        "the Route-B VFE runner with the shared X-lag residual, Matern-3/2 kernel, "
        "`M_t=M_s=128`, and 256 RFFs for HiPPO.\n\n"
        "A row is `complete` only when `result.json` and a finite positive-variance "
        "`(time, 200)` prediction archive pass the audit. Missing, failed, OOM, and "
        "resource-limited jobs remain explicit and are not filled from previous archives.\n",
        encoding="utf-8",
    )
    print(json.dumps({"per_seed": str(report / "per_seed.csv"), "summary": str(report / "summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
