#!/usr/bin/env python3
"""Build the current COVID Setting-B table from retained result evidence.

The earlier five-seed LMC, ICM and FSDE-SVI values are archival only. This
script excludes them from the current table and replaces them with the latest
three-repetition 4090 aggregate. OHSVGP and Task-1 lag ridge are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_TABLE = ROOT / "baselines/covid_long_setting_b/reports/formal_gaussian_task1_selected_strict/aggregate_metrics.csv"
UPDATED_TABLE = ROOT / "baselines/covid_long_setting_b/reports/exploratory_gpu_4090_s5_s7/aggregate_metrics.csv"
UPDATED_COMMIT = "341e05332f72d9edcbbde95c4e656b275c8ef048"

METRICS = (
    ("rmse", "RMSE"),
    ("crps", "CRPS"),
    ("native_gaussian_nlpd", "Gaussian NLPD"),
    ("ece", "ECE"),
    ("coverage90", "Coverage90"),
)

# Keep OHSVGP and Task-1 lag ridge out of the current paper table. Only these
# three rows may read from UPDATED_TABLE; all other rows retain their archive.
METHOD_ORDER = (
    ("persistence", "Last-value persistence"),
    ("routeb_ordinary", "Kron-STGP (point-inducing temporal control)"),
    ("routeb_cumulative_hippo", "KronHiPPO-STGP (cumulative HiPPO)"),
    ("bui_controlled", "Streaming sparse GP (Bui et al.; controlled transfer)"),
    ("bui_adaptive", "Streaming sparse GP (Bui et al.; adaptive update, CPU)"),
    ("st_svgp", "ST-SVGP"),
    ("lmc_svgp", "LMC-SVGP"),
    ("imc_svgp", "ICM-SVGP"),
    ("fsde_svi", "FSDE-SVI"),
)
REPLACED_METHODS = frozenset(("lmc_svgp", "imc_svgp", "fsde_svi"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "baselines/covid_long_setting_b/reports/reorganized_results_20260821",
    )
    parser.add_argument("--archived-table", type=Path, default=ARCHIVED_TABLE)
    parser.add_argument("--updated-table", type=Path, default=UPDATED_TABLE)
    return parser.parse_args()


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["method"]): row for row in rows}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], name: str, suffix: str) -> float:
    return float(row[f"{name}_{suffix}"])


def build_current_rows(
    archived: dict[str, dict[str, str]], updated: dict[str, dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    for method, label in METHOD_ORDER:
        if method in REPLACED_METHODS:
            source = updated[method]
            source_kind = "latest_4090_aggregate"
        else:
            source = archived[method]
            source_kind = "retained_audited_archive_aggregate"

        row: dict[str, Any] = {
            "method": method,
            "label": label,
            "source_kind": source_kind,
            "replications": int(source["seeds"]),
        }
        for metric, _ in METRICS:
            row[f"{metric}_mean"] = number(source, metric, "mean")
            row[f"{metric}_sd"] = number(source, metric, "sd")
        rows.append(row)

        if method in REPLACED_METHODS:
            old = archived[method]
            change: dict[str, Any] = {
                "method": method,
                "label": label,
                "archived_replications": int(old["seeds"]),
                "current_replications": int(source["seeds"]),
            }
            for metric, _ in METRICS:
                old_value = number(old, metric, "mean")
                new_value = row[f"{metric}_mean"]
                change[f"archived_{metric}"] = old_value
                change[f"current_{metric}"] = new_value
                change[f"delta_{metric}"] = new_value - old_value
            replacements.append(change)
    return rows, replacements


def display_value(row: dict[str, Any], metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_sd']):.4f}"


def write_table(rows: list[dict[str, Any]], output: Path) -> None:
    write_csv(
        output / "formal_results_table.csv",
        rows,
        ["method", "label", "source_kind", "replications"]
        + [part for metric, _ in METRICS for part in (f"{metric}_mean", f"{metric}_sd")],
    )

    markdown = [
        "| Method | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['label']} | "
            + " | ".join(display_value(row, metric) for metric, _ in METRICS)
            + " |"
        )
    markdown.extend(
        [
            "",
            "Values are mean +/- sample SD over the retained repetitions for each method. "
            "The provenance and replication count for every row are recorded in `source_manifest.json`.",
            "OHSVGP, Task-1 lag ridge and the previous five-seed LMC-SVGP, ICM-SVGP and FSDE-SVI values are excluded from this table.",
        ]
    )
    (output / "formal_results_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    tex = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & RMSE $\downarrow$ & CRPS $\downarrow$ & Gaussian NLPD $\downarrow$ & ECE $\downarrow$ & Coverage90 \\",
        r"\midrule",
    ]
    for row in rows:
        label = str(row["label"])
        if row["method"] == "routeb_cumulative_hippo":
            label = r"\textbf{" + label + "}"
        values = [
            f"{float(row[f'{metric}_mean']):.4f} $\\pm$ {float(row[f'{metric}_sd']):.4f}"
            for metric, _ in METRICS
        ]
        tex.append(label + " & " + " & ".join(values) + r" \\")
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"% Per-row replication counts and sources are recorded in source_manifest.json.",
            r"% OHSVGP and the superseded five-seed LMC/ICM/FSDE rows are excluded.",
        ]
    )
    (output / "formal_results_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def write_replacement_audit(rows: list[dict[str, Any]], output: Path) -> None:
    fields = ["method", "label", "archived_replications", "current_replications"]
    for metric, _ in METRICS:
        fields.extend((f"archived_{metric}", f"current_{metric}", f"delta_{metric}"))
    write_csv(output / "superseded_rows_audit.csv", rows, fields)
    lines = [
        "# Superseded Baseline Rows",
        "",
        "This audit is historical evidence only. The archived five-seed LMC-SVGP, ICM-SVGP and FSDE-SVI rows below are not present in `formal_results_table.*`.",
        "",
        "| Method | Archived RMSE | Current RMSE | Delta | Archived CRPS | Current CRPS | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['archived_rmse']:.4f} | {row['current_rmse']:.4f} | {row['delta_rmse']:+.4f} | "
            f"{row['archived_crps']:.4f} | {row['current_crps']:.4f} | {row['delta_crps']:+.4f} |"
        )
    (output / "superseded_rows_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(output: Path) -> None:
    text = """# Current COVID Long-Stream Results

`formal_results_table.*` is the current paper table. It excludes OHSVGP and
Task-1 lag ridge, and replaces the previous five-seed LMC-SVGP, ICM-SVGP and FSDE-SVI rows with the
latest cloud aggregate. The old rows remain only in the immutable archival
report tree and in `superseded_rows_audit.*`; they are not a source for any
current table or figure.

`source_manifest.json` is the authoritative per-row provenance record. It
retains the actual number of repetitions used to compute each mean and sample
standard deviation. The table intentionally has no seed-count column.

`figures/` contains regenerated metric, trajectory, Route-B error, long-memory
and calibration figures. The three cloud methods have aggregate metrics but no
local prediction archives, so no trajectory or uncertainty plot is fabricated
for them.

`reproducibility/` contains the two generators used to build this package.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archived_path = args.archived_table.resolve()
    updated_path = args.updated_table.resolve()
    archived = read_csv(archived_path)
    updated = read_csv(updated_path)
    missing = [method for method in REPLACED_METHODS if method not in updated]
    if missing:
        raise KeyError(f"Latest aggregate is missing replacement methods: {missing}")

    rows, replacements = build_current_rows(archived, updated)
    write_table(rows, output)
    write_replacement_audit(replacements, output)
    write_readme(output)
    manifest = {
        "status": "current_table_complete",
        "archived_aggregate_source": str(archived_path),
        "latest_4090_aggregate_source": str(updated_path),
        "latest_4090_source_commit": UPDATED_COMMIT,
        "excluded_methods": ["ohsvgp_rbf", "task1_lag_ridge"],
        "superseded_archived_methods": sorted(REPLACED_METHODS),
        "policy": "Use latest_4090_aggregate for LMC-SVGP, ICM-SVGP (underlying archive id imc_svgp) and FSDE-SVI; retain audited archived values for all other rows.",
        "method_sources": [
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
    print(json.dumps({"output": str(output), "rows": len(rows), "replaced": sorted(REPLACED_METHODS)}, indent=2))


if __name__ == "__main__":
    main()
