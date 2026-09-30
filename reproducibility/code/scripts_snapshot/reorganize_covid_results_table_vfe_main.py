#!/usr/bin/env python3
"""Build the COVID paper table with the VFE KronHiPPO result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    ("rmse", "RMSE"),
    ("crps", "CRPS"),
    ("native_gaussian_nlpd", "Gaussian NLPD"),
    ("ece", "ECE"),
)
METHOD_ORDER = (
    ("persistence", "Last-value persistence"),
    ("routeb_ordinary", "Kron-STGP"),
    ("routeb_cumulative_hippo", "KronHiPPO-STGP"),
    ("bui_adaptive", "Streaming sparse GP"),
    ("st_svgp", "ST-SVGP"),
    ("lmc_svgp", "LMC-SVGP"),
    ("imc_svgp", "ICM-SVGP"),
    ("fsde_svi", "FSDE-SVI"),
)


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def value(row: dict[str, str], metric: str, suffix: str) -> float:
    return float(row[f"{metric}_{suffix}"])


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_rows(archived: dict[str, dict[str, str]], vfe: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, label in METHOD_ORDER:
        source = vfe if method == "routeb_cumulative_hippo" else archived[method]
        row: dict[str, object] = {
            "method": method,
            "label": label,
            "source_kind": "routeb_vfe_aggregate" if method == "routeb_cumulative_hippo" else "retained_audited_archive_aggregate",
            "replications": int(source["seeds"]),
        }
        for metric, _ in METRICS:
            row[f"{metric}_mean"] = value(source, metric, "mean")
            row[f"{metric}_sd"] = value(source, metric, "sd")
        rows.append(row)
    return rows


def display(row: dict[str, object], metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_sd']):.4f}"


def write_tables(rows: list[dict[str, object]], output: Path) -> None:
    csv_fields = ["method", "label", "source_kind", "replications"]
    csv_fields.extend(part for metric, _ in METRICS for part in (f"{metric}_mean", f"{metric}_sd"))
    write_csv(output / "formal_results_table.csv", rows, csv_fields)

    markdown = [
        "| Method | RMSE | CRPS | Gaussian NLPD | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append("| " + str(row["label"]) + " | " + " | ".join(display(row, metric) for metric, _ in METRICS) + " |")
    markdown.extend(
        [
            "",
            "Values are mean +/- sample SD over the retained repetitions.",
            "KronHiPPO-STGP uses the VFE Task-1 empirical-Bayes objective with the cumulative HiPPO representation.",
            "Coverage90 is omitted from this main table by design; row provenance and actual repetition counts are recorded in `source_manifest.json`.",
        ]
    )
    (output / "formal_results_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    tex = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & RMSE $\downarrow$ & CRPS $\downarrow$ & Gaussian NLPD $\downarrow$ & ECE $\downarrow$ " + r"\\",
        r"\midrule",
    ]
    for row in rows:
        tex.append(str(row["label"]) + " & " + " & ".join(
            f"{float(row[f'{metric}_mean']):.4f} $\\pm$ {float(row[f'{metric}_sd']):.4f}"
            for metric, _ in METRICS
        ) + r" \\")
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"% KronHiPPO-STGP uses the VFE Task-1 empirical-Bayes objective.",
            r"% Coverage90 is intentionally omitted from the main table.",
        ]
    )
    (output / "formal_results_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archived-table",
        type=Path,
        default=ROOT / "baselines/covid_long_setting_b/reports/reorganized_results_20260821/formal_results_table.csv",
    )
    parser.add_argument(
        "--vfe-summary",
        type=Path,
        default=ROOT / "results/diagnostics/covid_routeb_finite_dtc_vs_vfe_20260824/aggregate_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "baselines/covid_long_setting_b/reports/reorganized_results_20260824_vfe_main",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archived = read_rows(args.archived_table.resolve())
    vfe_rows = read_rows(args.vfe_summary.resolve())
    if "vfe" not in vfe_rows:
        raise KeyError("VFE aggregate row is missing")
    rows = build_rows(archived, vfe_rows["vfe"])
    write_tables(rows, output)
    manifest = {
        "status": "complete",
        "archived_table_source": str(args.archived_table.resolve()),
        "vfe_comparison_source": str(args.vfe_summary.resolve()),
        "replacement": "routeb_cumulative_hippo finite-DTC aggregate replaced by VFE aggregate",
        "excluded_methods": ["bui_controlled"],
        "omitted_metrics": ["coverage90"],
        "visible_label_policy": "Remove parenthetical implementation descriptors from method names.",
        "method_rows": [
            {
                "method": row["method"],
                "label": row["label"],
                "source_kind": row["source_kind"],
                "replications": row["replications"],
            }
            for row in rows
        ],
        "output_files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# COVID Formal Main Table: VFE KronHiPPO-STGP\n\n"
        "This revision replaces the cumulative HiPPO Route B row with the VFE objective result. "
        "The controlled-transfer Streaming sparse GP row and Coverage90 column are omitted. "
        "The previous main-table directory remains unchanged for provenance.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
