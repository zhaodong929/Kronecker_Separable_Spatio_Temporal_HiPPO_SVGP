#!/usr/bin/env python3
"""Build the thesis COVID table with VFE for both Route B methods."""

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


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def archived_value(row: dict[str, str], metric: str, suffix: str) -> float:
    return float(row[f"{metric}_{suffix}"])


def display(row: dict[str, object], metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_sd']):.4f}"


def build_vfe_rows(
    vfe_root: Path,
    protocol_root: Path,
    hippo_vfe_root: Path | None = None,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    aggregate_rows: dict[str, dict[str, object]] = {}
    per_seed: list[dict[str, object]] = []
    hippo_root = vfe_root if hippo_vfe_root is None else hippo_vfe_root
    for method in ("routeb_ordinary", "routeb_cumulative_hippo"):
        method_dir = "routeb_ordinary" if method == "routeb_ordinary" else "routeb_cumulative"
        result_root = vfe_root if method == "routeb_ordinary" else hippo_root
        scores: list[dict[str, float]] = []
        for seed in SEEDS:
            protocol = protocol_root / f"seed{seed}" / "protocol.json"
            archive = result_root / f"seed{seed}" / method_dir / "online" / "predictions.npz"
            score = score_archive(archive, protocol, seed)
            scores.append(score)
            per_seed.append({"method": method, "seed": seed, **score})
        row: dict[str, object] = {
            "method": method,
            "source_kind": "routeb_vfe_aggregate",
            "replications": len(scores),
        }
        for metric, _ in METRICS:
            mean, sd = aggregate([score[metric] for score in scores])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        aggregate_rows[method] = row
    return aggregate_rows, per_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archived-table",
        type=Path,
        default=ROOT / "baselines/covid_long_setting_b/reports/reorganized_results_20260821/formal_results_table.csv",
    )
    parser.add_argument(
        "--vfe-root",
        type=Path,
        default=ROOT / "results/diagnostics/covid_long_stream_2020_2024_mandatory_vfe",
    )
    parser.add_argument(
        "--hippo-vfe-root",
        type=Path,
        help="Optional replacement root for the cumulative HiPPO archive.",
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=ROOT / "data/epidemiology/protocol/covid_long_2020_2024_mandatory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "thesis/routeb_mres_thesis/tables/covid_formal_results_vfe",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    archived = read_csv(args.archived_table.resolve())
    vfe_rows, per_seed = build_vfe_rows(
        args.vfe_root.resolve(),
        args.protocol_root.resolve(),
        None if args.hippo_vfe_root is None else args.hippo_vfe_root.resolve(),
    )
    rows: list[dict[str, object]] = []
    for method, label in METHOD_ORDER:
        if method in vfe_rows:
            row = dict(vfe_rows[method])
        else:
            source = archived[method]
            row = {
                "method": method,
                "source_kind": "retained_audited_archive_aggregate",
                "replications": int(source["replications"]),
            }
            for metric, _ in METRICS:
                row[f"{metric}_mean"] = archived_value(source, metric, "mean")
                row[f"{metric}_sd"] = archived_value(source, metric, "sd")
        row["label"] = label
        rows.append(row)

    write_csv(output / "formal_results_table.csv", rows)
    write_csv(output / "routeb_vfe_metrics_per_seed.csv", per_seed)

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
            "Both Kron-STGP and KronHiPPO-STGP use the VFE Task-1 empirical-Bayes objective; the latter uses the cumulative HiPPO temporal representation.",
            "Coverage90 and parenthetical implementation descriptors are omitted from this main table.",
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
        values = " & ".join(
            f"{float(row[f'{metric}_mean']):.4f} $\\pm$ {float(row[f'{metric}_sd']):.4f}"
            for metric, _ in METRICS
        )
        tex.append(str(row["label"]) + " & " + values + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "formal_results_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    manifest = {
        "status": "complete",
        "archived_table_source": str(args.archived_table.resolve()),
        "vfe_results_root": str(args.vfe_root.resolve()),
        "hippo_vfe_results_root": str(
            args.vfe_root.resolve()
            if args.hippo_vfe_root is None
            else args.hippo_vfe_root.resolve()
        ),
        "protocol_root": str(args.protocol_root.resolve()),
        "routeb_objectives": {
            "routeb_ordinary": "vfe",
            "routeb_cumulative_hippo": "vfe",
        },
        "formal_seeds": list(SEEDS),
        "excluded_methods": ["bui_controlled"],
        "omitted_metrics": ["coverage90"],
        "label_policy": "No parenthetical implementation descriptors in visible method names.",
        "retained_non_routeb_rows": ["persistence", "bui_adaptive", "st_svgp", "lmc_svgp", "imc_svgp", "fsde_svi"],
        "output_files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# COVID Formal Thesis Results: VFE Route B\n\n"
        "This is the thesis-ready main table. Both Route B methods use the VFE Task-1 empirical-Bayes objective. "
        "The controlled-transfer Streaming sparse GP row and Coverage90 column are omitted as requested. "
        "The previous formal result directories remain unchanged for provenance.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
