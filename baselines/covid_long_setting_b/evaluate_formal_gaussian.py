#!/usr/bin/env python3
"""Score completed Setting B Gaussian prediction archives with one metric contract."""

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

from baselines.covid_long_setting_b.formalization import formal_catalog_methods
from scripts.compute_covid_long_final_metric_system import gaussian_metrics_on_common_scale, mean_sd


INTERNAL = {
    "persistence": ("Persistence", "deterministic/persistence/predictions.npz"),
    "task1_lag_ridge": ("Task-1 lag ridge", "deterministic/lag_ridge/predictions.npz"),
    "routeb_ordinary": ("Route B ordinary inducing", "routeb_ordinary/online/predictions.npz"),
    "routeb_cumulative_hippo": ("Route B cumulative HiPPO", "routeb_cumulative/online/predictions.npz"),
}
RETAINED_BUI = {
    "bui_osgpr_controlled": ("Bui OSGPR (controlled)", "bui_controlled"),
    "bui_osgpr_adaptive": ("Bui OSGPR (adaptive, CPU)", "bui_adaptive"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-root", type=Path,
        default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"),
    )
    parser.add_argument(
        "--historical-root", type=Path,
        default=Path("results/diagnostics/covid_long_stream_2020_2024_mandatory"),
    )
    parser.add_argument(
        "--bui-selected-root", type=Path,
        default=Path("baselines/covid_long_setting_b/results/formal_selected_bui_ovc_shared_m8"),
        help="Task-1-selected 8x32 Bui formal archives.",
    )
    parser.add_argument(
        "--catalog", type=Path, default=Path("baselines/covid_long_setting_b/catalog.json"),
    )
    parser.add_argument(
        "--capacity-policy",
        type=Path,
        default=Path("baselines/covid_long_setting_b/capacity_policy.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("baselines/covid_long_setting_b/reports/formal_gaussian_repaired_pending_20260819"),
    )
    parser.add_argument(
        "--fairness-protocol",
        type=Path,
        help="Include repaired methods only from a completed new formal root locked by this file.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & RMSE $\\downarrow$ & CRPS $\\downarrow$ & Gaussian NLPD $\\downarrow$ & ECE $\\downarrow$ & Cov.90 $\\uparrow$ " + r"\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['rmse_mean']:.4f} $\\pm$ {row['rmse_sd']:.4f} & "
            f"{row['crps_mean']:.4f} $\\pm$ {row['crps_sd']:.4f} & "
            f"{row['native_gaussian_nlpd_mean']:.4f} $\\pm$ {row['native_gaussian_nlpd_sd']:.4f} & "
            f"{row['ece_mean']:.4f} $\\pm$ {row['ece_sd']:.4f} & "
            f"{row['coverage90_mean']:.4f} $\\pm$ {row['coverage90_sd']:.4f} " + r"\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def methods_for_evaluation(catalog: dict[str, Any], fairness: dict[str, Any] | None) -> dict[str, str]:
    """Do not let preliminary adapters re-enter a final table by path alone."""

    allowed = formal_catalog_methods(catalog)
    methods = {
        key: label
        for key, (label, _) in INTERNAL.items()
        if key in allowed
    }
    methods.update(
        {catalog_id: label for catalog_id, (label, _) in RETAINED_BUI.items() if catalog_id in allowed}
    )
    if fairness is None:
        return methods
    if fairness.get("status") != "locked_before_formal_seeds":
        raise ValueError("A repaired result requires an intact pre-formal fairness lock")
    formal_root = Path(fairness["formal_result_root"])
    manifest_path = formal_root / "formal_run_manifest.json"
    if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError("Repaired methods may be scored only after a complete locked formal run")
    catalog_names = {entry["id"]: entry["display_name"] for entry in catalog["methods"]}
    for method in fairness["methods"]:
        if method not in catalog_names:
            raise ValueError(f"Fairness lock names unknown catalog method: {method}")
        methods[method] = catalog_names[method]
    return methods


def archive_path(
    *,
    method: str,
    seed: int,
    historical_root: Path,
    bui_selected_root: Path,
    fairness: dict[str, Any] | None,
) -> Path:
    if method in INTERNAL:
        return historical_root / f"seed{seed}" / INTERNAL[method][1]
    if method in RETAINED_BUI:
        return bui_selected_root / f"seed{seed}" / RETAINED_BUI[method][1] / "predictions.npz"
    if fairness is not None and method in fairness["methods"]:
        return Path(fairness["formal_result_root"]) / f"seed{seed}" / method / "predictions.npz"
    raise KeyError(f"No formal archive resolver for {method}")


def main() -> None:
    args = parse_args()
    protocol_root, historical_root, bui_selected_root = map(
        absolute,
        (
            args.protocol_root,
            args.historical_root,
            args.bui_selected_root,
        ),
    )
    catalog = json.loads(absolute(args.catalog).read_text(encoding="utf-8"))
    fairness = None if args.fairness_protocol is None else json.loads(absolute(args.fairness_protocol).read_text(encoding="utf-8"))
    methods = methods_for_evaluation(catalog, fairness)
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for method, label in methods.items():
        for seed in args.seeds:
            path = archive_path(
                method=method,
                seed=seed,
                historical_root=historical_root,
                bui_selected_root=bui_selected_root,
                fairness=fairness,
            )
            if not path.is_file():
                missing.append({"method": method, "label": label, "seed": seed, "status": "archive_missing", "path": str(path)})
                continue
            with np.load(path, allow_pickle=False) as arrays:
                archive = {name: np.asarray(arrays[name]) for name in ("y_true", "pred_mean", "pred_var")}
                shapes = {name: value.shape for name, value in archive.items()}
                if any(shape != (143, 10) for shape in shapes.values()):
                    missing.append({"method": method, "label": label, "seed": seed, "status": f"invalid_shapes_{shapes}", "path": str(path)})
                    continue
                if not all(np.isfinite(value).all() for value in archive.values()) or np.any(archive["pred_var"] < 0.0):
                    missing.append({"method": method, "label": label, "seed": seed, "status": "nonfinite_or_negative_variance", "path": str(path)})
                    continue
                protocol = json.loads((protocol_root / f"seed{seed}" / "protocol.json").read_text(encoding="utf-8"))
                metrics = gaussian_metrics_on_common_scale(
                    archive,
                    protocol["target_standardization"],
                    ece_seed=50_000_000 + 10_000 * seed + len(rows),
                )
            rows.append({"method": method, "label": label, "seed": seed, **metrics, "path": str(path)})
    aggregate: list[dict[str, Any]] = []
    for method, label in methods.items():
        group = [row for row in rows if row["method"] == method]
        if len(group) != len(args.seeds):
            continue
        record: dict[str, Any] = {"method": method, "label": label, "seeds": len(group)}
        for metric in ("rmse", "crps", "native_gaussian_nlpd", "ece", "coverage90"):
            record[f"{metric}_mean"], record[f"{metric}_sd"] = mean_sd([float(row[metric]) for row in group])
        aggregate.append(record)
    capacity_policy = json.loads(absolute(args.capacity_policy).read_text(encoding="utf-8"))
    failures = [
        {
            "method": item["id"], "label": item["display_name"], "status": item["setting_b_status"],
            "reason": item.get(
                "validation_pending_reason",
                item.get(
                    "formal_failure_reason",
                    item.get("covid_adapter_status", item.get("official_reproduction_status", "not available")),
                ),
            ),
        }
        for item in catalog["methods"]
        if item["id"] in ("ohsvgp_rbf", "st_svgp", "ovc_svgp", "lmc_svgp", "imc_svgp", "fsde_svi", "earth")
        and item["setting_b_status"] != "formal_result_available"
    ]
    output = absolute(args.output_dir)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing report root: {output}. Choose a new --output-dir."
        )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "per_seed_metrics.csv")
    write_csv(aggregate, output / "aggregate_metrics.csv")
    write_csv(missing, output / "incomplete_archives.csv")
    write_csv(failures, output / "failure_table.csv")
    write_latex_table(aggregate, output / "main_table.tex")
    report = [
        "# COVID Long-Stream Setting B Gaussian Preliminary Audit",
        "",
        "All metric rows use the same delayed-observation information set and the common log1p(per-100k) target scale.",
        "",
        "## Capacity Selection",
        "",
        capacity_policy["selection_protocol"],
        "",
        "Route B ordinary and cumulative HiPPO remain exactly matched at "
        "`Mt=32, Ms=32`. Repaired external adapters are deliberately absent "
        "until their three blocked seed-0 validation windows, method-specific "
        "gates and a new locked seed-5--9 formal run pass. Prior adapter archives "
        "are preliminary audit evidence and cannot be reintroduced by this evaluator.",
        "",
        "| Method | Splits | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        report.append(
            f"| {row['label']} | {row['seeds']} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | "
            f"{row['crps_mean']:.4f} +/- {row['crps_sd']:.4f} | {row['native_gaussian_nlpd_mean']:.4f} +/- {row['native_gaussian_nlpd_sd']:.4f} | "
            f"{row['ece_mean']:.4f} +/- {row['ece_sd']:.4f} | {row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} |"
        )
    report.extend(["", "## Excluded Candidates", "", "| Method | Status | Reason |", "|---|---|---|"])
    for row in failures:
        report.append(f"| {row['label']} | {row['status']} | {row['reason']} |")
    if missing:
        report.extend(["", "## Pending Archives", ""])
        report.extend(f"- {row['label']} seed {row['seed']}: {row['status']}" for row in missing)
    report.extend([
        "",
        "## Repaired Adapter Provenance",
        "",
        "OHSVGP, OVC-SVGP, ST-SVGP, LMC-SVGP, IMC-SVGP and FSDE-SVI remain validation-pending unless explicitly admitted through a completed fairness-locked run. "
        "Their unmodified upstream entrypoints use different protocols and are never represented as completed Setting B results.",
        "",
        "## Completion Audit",
        "",
        "Archive coverage, causal-update counters, capacity selection and external-source status are itemized in `completion_audit.md` in this report directory.",
    ])
    completion_audit = [
        "# Completion Audit",
        "",
        "This audit is generated without modifying any prediction archive.",
        "",
        f"- Complete retained rows: {len(aggregate)}",
        f"- Pending or excluded methods: {len(failures)}",
        f"- Missing archives among admitted rows: {len(missing)}",
        "- Pending external adapters are excluded unless a completed fairness-locked formal run is supplied.",
    ]
    (output / "completion_audit.md").write_text("\n".join(completion_audit) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"completed_rows": len(aggregate), "missing_archives": len(missing), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
