#!/usr/bin/env python3
"""Phase-0 data audit for the COVID, Dengue, and Malaria candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _date_span(dates: pd.Series) -> tuple[str | None, str | None]:
    parsed = pd.to_datetime(dates, errors="coerce")
    if not parsed.notna().any():
        return None, None
    return parsed.min().date().isoformat(), parsed.max().date().isoformat()


def audit_covid(root: Path) -> dict[str, Any]:
    path = root / "covid_hospital_admissions.csv"
    locations_path = root / "locations.csv"
    if not path.exists() or not locations_path.exists():
        return {"dataset": "covid", "status": "missing", "missing": [str(path), str(locations_path)]}
    frame = pd.read_csv(path, dtype={"location": str})
    locations = pd.read_csv(locations_path, dtype={"location": str})
    frame = frame[frame["location"].astype(str) != "US"].copy()
    duplicate_count = int(frame.duplicated(["location", "target_end_date"]).sum())
    pivot = frame.pivot_table(
        index="target_end_date", columns="location", values="value", aggfunc="first"
    )
    start, end = _date_span(frame["target_end_date"])
    expected = int(pivot.shape[0] * pivot.shape[1])
    observed = int(pivot.notna().sum().sum())
    missing_fraction = 1.0 - observed / max(expected, 1)
    location_set = set(frame["location"].astype(str))
    known_locations = set(locations["location"].astype(str)) - {"US"}
    minimum_pilot = pivot.shape[0] >= 52 and pivot.shape[1] >= 30 and missing_fraction <= 0.05
    preferred_primary = pivot.shape[0] >= 200 and pivot.shape[1] >= 100 and missing_fraction <= 0.05
    return {
        "dataset": "covid",
        "status": "pass_primary" if preferred_primary else ("pass_pilot" if minimum_pilot else "no_go"),
        "target": "weekly confirmed COVID-19 hospital admissions",
        "source_kind": "official reported target data",
        "temporal_frequency": "weekly",
        "start_date": start,
        "end_date": end,
        "num_times": int(pivot.shape[0]),
        "num_locations": int(pivot.shape[1]),
        "observed_cells": observed,
        "expected_cells": expected,
        "missing_fraction": float(missing_fraction),
        "duplicate_location_time_rows": duplicate_count,
        "unknown_location_codes": sorted(location_set - known_locations),
        "negative_target_count": int((pd.to_numeric(frame["value"], errors="coerce") < 0).sum()),
        "primary_gates": {
            "at_least_200_times": bool(pivot.shape[0] >= 200),
            "at_least_100_locations": bool(pivot.shape[1] >= 100),
            "missing_at_most_5pct": bool(missing_fraction <= 0.05),
        },
        "pilot_gates": {
            "at_least_52_times": bool(pivot.shape[0] >= 52),
            "at_least_30_locations": bool(pivot.shape[1] >= 30),
            "missing_at_most_5pct": bool(missing_fraction <= 0.05),
        },
        "leakage_risk": (
            "The repository target is a current retrospective snapshot. Strict real-time claims require historical vintages."
        ),
    }


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def audit_dengue(root: Path, download_manifest: dict[str, Any]) -> dict[str, Any]:
    path = root / "infodengue.jsonl"
    if not path.exists():
        download_status = (
            download_manifest.get("results", {}).get("dengue", {}).get("status", "missing")
        )
        return {
            "dataset": "dengue",
            "status": download_status,
            "required_action": "Create a Mosqlimate API key and set MOSQLIMATE_API_KEY.",
            "registration_url": "https://mosqlimate.org/profile/auth",
        }
    frame = _read_jsonl(path)
    required = {"data_iniSE", "casos", "municipio_geocodigo", "pop"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        return {"dataset": "dengue", "status": "invalid_schema", "missing_columns": missing_columns}
    frame["data_iniSE"] = pd.to_datetime(frame["data_iniSE"], errors="coerce")
    frame["municipio_geocodigo"] = frame["municipio_geocodigo"].astype(str)
    pivot = frame.pivot_table(
        index="data_iniSE", columns="municipio_geocodigo", values="casos", aggfunc="first"
    )
    expected = int(pivot.shape[0] * pivot.shape[1])
    observed = int(pivot.notna().sum().sum())
    missing_fraction = 1.0 - observed / max(expected, 1)
    start, end = _date_span(frame["data_iniSE"])
    preferred_primary = pivot.shape[0] >= 200 and pivot.shape[1] >= 100 and missing_fraction <= 0.05
    return {
        "dataset": "dengue",
        "status": "pass_primary" if preferred_primary else "no_go",
        "target": "weekly notified dengue cases (casos)",
        "source_kind": "notification surveillance; retrospectively revised",
        "temporal_frequency": "weekly",
        "start_date": start,
        "end_date": end,
        "num_times": int(pivot.shape[0]),
        "num_locations": int(pivot.shape[1]),
        "observed_cells": observed,
        "expected_cells": expected,
        "missing_fraction": float(missing_fraction),
        "duplicate_location_time_rows": int(
            frame.duplicated(["municipio_geocodigo", "data_iniSE"]).sum()
        ),
        "nonpositive_population_count": int((pd.to_numeric(frame["pop"], errors="coerce") <= 0).sum()),
        "negative_target_count": int((pd.to_numeric(frame["casos"], errors="coerce") < 0).sum()),
        "primary_gates": {
            "at_least_200_times": bool(pivot.shape[0] >= 200),
            "at_least_100_locations": bool(pivot.shape[1] >= 100),
            "missing_at_most_5pct": bool(missing_fraction <= 0.05),
        },
        "leakage_risk": (
            "Notified cases are retrospectively updated. This snapshot supports retrospective strict-block processing, not an as-of real-time claim."
        ),
    }


def audit_malaria(root: Path) -> dict[str, Any]:
    path = root / "printables_catalog.json"
    if not path.exists():
        return {"dataset": "malaria", "status": "missing", "missing": [str(path)]}
    catalog = json.loads(path.read_text(encoding="utf-8"))
    ranges = []
    predicted_metrics = []
    for metric in catalog:
        description = str(metric.get("description", ""))
        if "predict" in description.lower():
            predicted_metrics.append(str(metric.get("id")))
        for version in metric.get("versions", []):
            ranges.append((int(version["startYear"]), int(version["endYear"])))
    minimum_year = min((start for start, _ in ranges), default=None)
    maximum_year = max((end for _, end in ranges), default=None)
    maximum_steps = (
        maximum_year - minimum_year + 1
        if minimum_year is not None and maximum_year is not None
        else 0
    )
    return {
        "dataset": "malaria",
        "status": "no_go_strict_online",
        "source_kind": "model-derived annual raster/printable catalog",
        "temporal_frequency": "annual",
        "start_year": minimum_year,
        "end_year": maximum_year,
        "maximum_annual_steps": maximum_steps,
        "num_catalog_metrics": len(catalog),
        "predicted_metric_ids": sorted(predicted_metrics),
        "primary_gates": {
            "at_least_200_times": bool(maximum_steps >= 200),
            "raw_observations": False,
        },
        "recommended_use": (
            "Supplementary spatial experiment only. Obtain a public monthly DHIS2 routine panel before a Malaria strict-online experiment."
        ),
    }


def write_outputs(audits: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for audit in audits:
        (output / f"{audit['dataset']}_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
    summary_fields = [
        "dataset", "status", "source_kind", "temporal_frequency", "num_times", "num_locations", "missing_fraction"
    ]
    with (output / "audit_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for audit in audits:
            writer.writerow({key: audit.get(key) for key in summary_fields})
    lines = [
        "# Epidemiology Phase 0 data audit",
        "",
        "| Dataset | Status | Time points | Locations | Missing fraction | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    decisions = {
        "pass_primary": "Proceed to full pilot protocol",
        "pass_pilot": "Proceed to small pilot; not a primary spatial benchmark",
        "no_go_strict_online": "Do not run strict-online Route B",
        "blocked_by_auth": "Register and provide API key",
    }
    for audit in audits:
        missing = audit.get("missing_fraction")
        lines.append(
            f"| {audit['dataset']} | {audit['status']} | {audit.get('num_times', audit.get('maximum_annual_steps', ''))} "
            f"| {audit.get('num_locations', '')} | {'' if missing is None else f'{missing:.4f}'} "
            f"| {decisions.get(audit['status'], 'Stop and inspect')} |"
        )
    lines.extend(
        [
            "",
            "The audit distinguishes a local feasibility pilot from a primary long-stream benchmark.",
            "A current retrospective snapshot does not by itself establish real-time vintage correctness.",
        ]
    )
    (output / "phase0_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/epidemiology/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/diagnostics/epidemiology_phase0")
    )
    args = parser.parse_args()
    manifest_path = args.raw_root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    audits = [
        audit_dengue(args.raw_root / "dengue", manifest),
        audit_covid(args.raw_root / "covid"),
        audit_malaria(args.raw_root / "malaria"),
    ]
    write_outputs(audits, args.output)
    print(json.dumps({"audits": audits, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
