"""Run and report the ERA5 online-GP baseline suite.

This is an orchestration layer around the existing ERA5 runners. It keeps the
paper protocol explicit:

- task_1 calibration with full-GP MLL grid for kernel hyperparameters;
- task_2 online held-out seen-history evaluation;
- Route B reference: Rich-v3 Phi + Matern-3/2 + Mt=32 + Ms=256;
- local fair baselines: persistence, climatology, ridge, SGPR/SVGP with the
  same Matern-3/2 frozen-hyperparameter protocol;
- external online-GP repositories are audited and reported separately until they
  are wrapped by the local OnlineBaseline interface.

The external OSVGP/OVC/OHSVGP/Markovflow repositories have incompatible
framework and data-loop assumptions, so this script does not pretend that their
example scripts are fair ERA5 held-out seen-history results.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_ROUTEB_REFERENCE = (
    ROOT
    / "results"
    / "experiments_era5_ohsvgp_heldout_fullspace"
    / "paper_ready"
    / "kernel_family_fullgp_wide_diagnostic"
    / "matern32_rich_v4_fullgp_wide"
    / "era5_routeb_summary.csv"
)


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"cmd": cmd, "returncode": None, "status": "dry_run"}
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_external_status(manifest_path: Path, outdir: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for repo in manifest.get("repositories", []):
        local_folder = repo.get("local_folder")
        local_path = ROOT / "baselines" / "external" / str(local_folder) if local_folder else None
        rows.append(
            {
                "name": repo.get("name"),
                "intended_methods": "; ".join(repo.get("intended_methods", [])),
                "url": repo.get("url") or "",
                "local_folder": local_folder or "",
                "download_status": repo.get("download_status"),
                "checked_commit": repo.get("checked_commit", ""),
                "local_exists": bool(local_path and local_path.exists()),
                "unified_era5_runner_status": repo.get("unified_era5_runner_status"),
                "paper_table_status": (
                    "excluded_from_main_metrics_until_wrapped"
                    if repo.get("unified_era5_runner_status") != "integrated"
                    else "included"
                ),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "external_online_gp_repo_status.csv", index=False)
    return df


def summarize_for_report(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["result_group"] = group
    return out


def plot_seen_history(combined: pd.DataFrame, outdir: Path) -> list[Path]:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if combined.empty:
        return paths
    sub = combined[combined["eval_mode"].astype(str).str.contains("seen_history", na=False)].copy()
    if sub.empty:
        return paths
    if "display_method" not in sub.columns:
        sub["display_method"] = sub["method"].astype(str)
    method_order = list(dict.fromkeys(sub["display_method"].astype(str)))
    for metric, ylabel in [
        ("nll", "NLL / NLPD"),
        ("rmse", "RMSE"),
        ("coverage90", "90% coverage"),
        ("runtime_per_block", "runtime/block"),
    ]:
        if metric not in sub.columns:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(method_order)), 3.8))
        values = []
        errors = []
        for method in method_order:
            row = sub[sub["display_method"].eq(method)].iloc[0]
            values.append(float(row.get(metric, np.nan)))
            errors.append(float(row.get(f"{metric}_se", np.nan)) if f"{metric}_se" in row else np.nan)
        x = np.arange(len(method_order))
        yerr = np.array(errors)
        yerr = None if np.all(~np.isfinite(yerr)) else yerr
        ax.bar(x, values, yerr=yerr, capsize=3, color="#4E79A7", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"ERA5 held-out seen-history {ylabel}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = plot_dir / f"online_gp_suite_seen_history_{metric}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def format_metric(row: pd.Series, metric: str) -> str:
    value = row.get(metric, np.nan)
    if pd.isna(value):
        return ""
    se = row.get(f"{metric}_se", np.nan)
    if pd.notna(se):
        return f"{float(value):.4f} +/- {float(se):.4f}"
    return f"{float(value):.4f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    rows = [[str(value) for value in row] for row in df[cols].to_numpy(dtype=object)]
    widths = [len(str(col)) for col in cols]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    header = "| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_markdown_report(
    outdir: Path,
    args: argparse.Namespace,
    combined: pd.DataFrame,
    external_status: pd.DataFrame,
    command_log: list[dict[str, Any]],
    figure_paths: list[Path],
) -> Path:
    lines: list[str] = []
    lines.append("# ERA5 online GP baseline suite addendum\n")
    lines.append("## Protocol\n")
    lines.append("- Calibration task: `task_1`.")
    lines.append("- Online task: `task_2`.")
    lines.append("- Main metric mode: OHSVGP-style held-out seen-history.")
    lines.append("- Hyperparameters: task-1 initial full-GP MLL grid, then frozen on task_2.")
    lines.append("- Route B reference: Rich-v3 Phi + Matern-3/2 + `Mt=32`, `Ms=256`.")
    lines.append(f"- Location setting requested: `{args.location_label}`.")
    lines.append(f"- Seeds: `{args.seeds}`.")
    lines.append("")
    lines.append("## Main comparable results\n")
    if combined.empty:
        lines.append("No comparable summary rows are available yet. Run the generated commands with `--run`.")
    else:
        sub = combined[combined["eval_mode"].astype(str).str.contains("seen_history", na=False)].copy()
        if sub.empty:
            lines.append("No seen-history rows were found.")
        else:
            rows = []
            for _, row in sub.iterrows():
                rows.append(
                    {
                        "group": row.get("result_group", ""),
                        "method": row.get("display_method", row.get("method", "")),
                        "NLL": format_metric(row, "nll"),
                        "RMSE": format_metric(row, "rmse"),
                        "Cov90": format_metric(row, "coverage90"),
                        "ECE": format_metric(row, "ece"),
                        "runtime/block": format_metric(row, "runtime_per_block"),
                    }
                )
            lines.append(markdown_table(pd.DataFrame(rows)))
    lines.append("")
    lines.append("## Full-space fairness status\n")
    lines.append(
        "- Deterministic baselines completed on the requested full-space task_2 held-out seen-history protocol."
    )
    lines.append(
        "- Route B Rich-v3 + Matern-3/2 + Mt=32 + Ms=256 was re-launched with the subset-safe task_1 full-GP MLL grid branch. The original dense-memory failure was removed, but the requested `3 seeds x 3 heldout splits` run exceeded the local runtime budget and was stopped rather than reported as a completed metric."
    )
    lines.append(
        "- SGPR/SVGP + Rich-v3 + Matern-3/2 with frozen task_1 hyperparameters was also launched on full-space task_2. It exceeded the local runtime budget and was stopped before a summary file was produced."
    )
    lines.append(
        "- Therefore, the fair full-space SGPR/SVGP/Route-B comparison remains pending on a larger compute node or a reduced-location diagnostic. The table below includes only completed rows."
    )
    if CONTEXT_ROUTEB_REFERENCE.exists():
        ref = read_csv(CONTEXT_ROUTEB_REFERENCE)
        if not ref.empty:
            ref_rows = []
            for _, row in ref.iterrows():
                if str(row.get("eval_mode", "")) != "seen_history":
                    continue
                ref_rows.append(
                    {
                        "source": "prior context, not 3-seed fair rerun",
                        "method": "Route B rich-v3 Matern32 32/256",
                        "NLL": format_metric(row, "nll"),
                        "RMSE": format_metric(row, "rmse"),
                        "Cov90": format_metric(row, "coverage90"),
                        "runtime/block": format_metric(row, "runtime_per_block"),
                    }
                )
            if ref_rows:
                lines.append("")
                lines.append("Context-only Route B reference from the previous completed diagnostic:")
                lines.append(markdown_table(pd.DataFrame(ref_rows)))
    lines.append("")
    lines.append("## External online GP baseline status\n")
    if not external_status.empty:
        lines.append(
            markdown_table(
                external_status[
                    [
                        "name",
                        "intended_methods",
                        "download_status",
                        "checked_commit",
                        "local_exists",
                        "unified_era5_runner_status",
                        "paper_table_status",
                    ]
                ]
            )
        )
    lines.append("")
    lines.append(
        "The external repositories are downloaded for provenance and future wrapping. They are not included in the main metric table until they expose the local `OnlineBaseline` interface and use the same held-out seen-history loop."
    )
    lines.append("")
    lines.append("## Figures\n")
    for path in figure_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Command log\n")
    for item in command_log:
        lines.append(f"- `{item['status']}` returncode={item.get('returncode')}: `{' '.join(item['cmd'])}`")
        if item.get("status") == "failed":
            lines.append(f"  - stderr tail: `{item.get('stderr_tail', '')[-500:]}`")
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append(
        "This addendum gives the fair comparison layer for baselines that can be executed by the local runner. It does not yet claim a full Markovian/s2VGP/sparse-Markovian baseline suite, because those external projects need framework-specific wrappers before their results are comparable. It also does not claim completed full-space same-kernel SGPR/SVGP metrics because the requested run did not finish locally."
    )
    path = outdir / "era5_online_gp_baseline_suite_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_pdf_report(md_path: Path, outdir: Path, combined: pd.DataFrame, external_status: pd.DataFrame, figures: list[Path]) -> Path:
    pdf_path = outdir / "era5_online_gp_baseline_suite_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "ERA5 online GP baseline suite addendum", fontsize=17, weight="bold", va="top")
        fig.text(
            0.08,
            0.89,
            "Continuation of the ERA5 Route B report. Main comparable rows use the task_1 calibration / task_2 held-out seen-history protocol.",
            fontsize=10,
            va="top",
            wrap=True,
        )
        y = 0.82
        if not combined.empty:
            sub = combined[combined["eval_mode"].astype(str).str.contains("seen_history", na=False)].copy()
            if not sub.empty:
                fig.text(0.08, y, "Comparable seen-history summary", fontsize=12, weight="bold", va="top")
                y -= 0.03
                rows = []
                for _, row in sub.head(12).iterrows():
                    rows.append(
                        [
                            str(row.get("display_method", row.get("method", "")))[:24],
                            format_metric(row, "nll"),
                            format_metric(row, "rmse"),
                            format_metric(row, "coverage90"),
                            format_metric(row, "runtime_per_block"),
                        ]
                    )
                txt = pd.DataFrame(rows, columns=["method", "NLL", "RMSE", "Cov90", "runtime"]).to_string(index=False)
                fig.text(0.08, y, txt, fontsize=7.4, family="monospace", va="top")
                y -= 0.25
        fig.text(0.08, y, "External online GP status", fontsize=12, weight="bold", va="top")
        y -= 0.03
        if not external_status.empty:
            ext = external_status[["name", "download_status", "checked_commit", "unified_era5_runner_status"]].copy()
            txt = ext.to_string(index=False)
            fig.text(0.08, y, txt[:2500], fontsize=7.2, family="monospace", va="top")
        fig.text(0.08, 0.06, f"Markdown: {md_path}", fontsize=8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        fig = plt.figure(figsize=(8.5, 11))
        y = 0.95
        fig.text(0.08, y, "Full-space fairness status", fontsize=15, weight="bold", va="top")
        y -= 0.06
        status_text = (
            "Completed: full-space deterministic baselines under the held-out seen-history protocol.\n\n"
            "Attempted but not completed locally: Route B Rich-v3 + Matern-3/2 + Mt=32 + Ms=256 "
            "with 3 seeds x 3 heldout splits; SGPR/SVGP + Rich-v3 + Matern-3/2 with frozen task-1 "
            "hyperparameters. Both exceeded the local runtime budget after the dense calibration-memory "
            "issue was avoided, so their full-space fair metrics are not reported as completed results.\n\n"
            "External OSVGP/OVC/OHSVGP/Markovflow repositories were downloaded and audited, but are excluded "
            "from the main metric table until wrapped into the local OnlineBaseline interface."
        )
        fig.text(0.08, y, status_text, fontsize=10, va="top", wrap=True)
        y -= 0.30
        if CONTEXT_ROUTEB_REFERENCE.exists():
            ref = read_csv(CONTEXT_ROUTEB_REFERENCE)
            ref = ref[ref["eval_mode"].astype(str).eq("seen_history")] if not ref.empty and "eval_mode" in ref.columns else ref
            if not ref.empty:
                row = ref.iloc[0]
                txt = (
                    "Context-only prior Route B reference (not the completed 3-seed fair rerun):\n"
                    f"NLL={format_metric(row, 'nll')}, RMSE={format_metric(row, 'rmse')}, "
                    f"Cov90={format_metric(row, 'coverage90')}, runtime/block={format_metric(row, 'runtime_per_block')}"
                )
                fig.text(0.08, y, txt, fontsize=10, family="monospace", va="top", wrap=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for path in figures:
            if not path.exists():
                continue
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(path.name)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return pdf_path


def merge_existing_and_addendum(existing_pdf: Path, addendum_pdf: Path, out_pdf: Path) -> bool:
    try:
        from PyPDF2 import PdfMerger  # type: ignore
    except Exception:
        try:
            from pypdf import PdfWriter as PdfMerger  # type: ignore
        except Exception:
            return False
    try:
        merger = PdfMerger()
        merger.append(str(existing_pdf))
        merger.append(str(addendum_pdf))
        merger.write(str(out_pdf))
        merger.close()
        return True
    except Exception:
        return False


def build_commands(args: argparse.Namespace, outdir: Path) -> list[tuple[str, list[str]]]:
    python = sys.executable
    loc_args: list[str] = []
    if args.random_n_locations:
        loc_args = ["--random-n-locations", str(args.random_n_locations)]
    deterministic_out = outdir / "local_deterministic_baselines"
    gp_out = outdir / "local_same_kernel_gp_baselines"
    routeb_out = outdir / "routeb_reference"
    common = [
        "--root",
        args.root,
        "--tasks",
        "task_2",
        "--split",
        "all",
        "--seeds",
        *[str(s) for s in args.seeds],
        "--block-size",
        str(args.block_size),
        "--eval-modes",
        "seen_history",
        "--phi-mode",
        "rich_v3",
        *loc_args,
    ]
    fair_gp_grid = [
        "--ell-t-grid",
        "0.03",
        "0.05",
        "0.10",
        "--noise-grid",
        "0.30",
        "0.50",
        "0.80",
        "--kernel-variance-grid",
        "0.50",
        "1.00",
        "--hyperparam-fit-max-time",
        str(args.hyperparam_fit_max_time),
        "--hyperparam-fit-max-locations",
        str(args.hyperparam_fit_max_locations),
    ]
    commands = [
        (
            "deterministic",
            [
                python,
                "scripts/run_hipposvgp_era5_baselines.py",
                *common,
                "--methods",
                "climatology",
                "persistence",
                "ridge",
                "--outdir",
                str(deterministic_out),
                "--output-prefix",
                "era5_deterministic",
            ],
        ),
        (
            "same_kernel_gp",
            [
                python,
                "scripts/run_hipposvgp_era5_baselines.py",
                *common,
                "--calibration-tasks",
                "task_1",
                "--methods",
                "sgpr_phi",
                "svgp_phi",
                "--gp-kernel-type",
                "matern32",
                "--gp-hyperparam-fit-mode",
                "routeb_initial_task_fullgp_grid",
                *fair_gp_grid,
                "--gp-training-iterations",
                str(args.gp_training_iterations),
                "--gp-learning-rate",
                str(args.gp_learning_rate),
                "--gp-inducing-points",
                str(args.gp_inducing_points),
                "--gp-minibatch-size",
                str(args.gp_minibatch_size),
                "--gp-inducing-init",
                "random",
                "--outdir",
                str(gp_out),
                "--output-prefix",
                "era5_same_kernel_gp",
            ],
        ),
        (
            "routeb",
            [
                python,
                "scripts/run_hipposvgp_era5_routeb.py",
                "--root",
                args.root,
                "--tasks",
                "task_2",
                "--calibration-tasks",
                "task_1",
                "--split",
                "all",
                "--seeds",
                *[str(s) for s in args.seeds],
                "--heldout-split-seeds",
                *[str(s) for s in args.seeds],
                "--block-size",
                str(args.block_size),
                "--routeb-methods",
                "structured_joint",
                "--mt",
                "32",
                "--ms",
                "256",
                "--kernel-type",
                "matern32",
                "--phi-mode",
                "rich_v3",
                "--ell-t-fit-mode",
                "initial_task_fullgp",
                "--hyperparam-fit-mode",
                "initial_task_fullgp_grid",
                *fair_gp_grid,
                "--ohsvgp-heldout-eval",
                "--heldout-test-fraction",
                "0.2",
                *loc_args,
                "--outdir",
                str(routeb_out),
            ],
        ),
    ]
    requested = set(args.stages)
    return [(label, cmd) for label, cmd in commands if label in requested]


def collect_results(outdir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    det = read_csv(outdir / "local_deterministic_baselines" / "era5_deterministic_summary.csv")
    if not det.empty:
        det["result_group"] = "deterministic"
        det["display_method"] = det["method"]
        frames.append(det)
    gp = read_csv(outdir / "local_same_kernel_gp_baselines" / "era5_same_kernel_gp_summary.csv")
    if not gp.empty:
        gp["result_group"] = "same_kernel_frozen_gp"
        gp["display_method"] = gp["method"].replace(
            {
                "gpytorch_sgpr_phi": "SGPR + Rich-Phi + Matern32 frozen",
                "gpytorch_svgp_phi": "SVGP + Rich-Phi + Matern32 frozen",
            }
        )
        frames.append(gp)
    routeb = read_csv(outdir / "routeb_reference" / "era5_ohsvgp_heldout_summary.csv")
    if not routeb.empty:
        routeb["result_group"] = "routeb_reference"
        routeb["display_method"] = routeb["method"].replace({"structured_joint": "Route B structured_joint"})
        frames.append(routeb)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(outdir / "combined_online_gp_suite_summary.csv", index=False)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/era5/processed_timeseries_4")
    parser.add_argument("--outdir", default="results/experiments_era5_ohsvgp_heldout_fullspace/online_gp_baseline_suite")
    parser.add_argument("--original-report-pdf", default="docs/era5_ohsvgp_heldout_experiment_report_nature_style_updated.pdf")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--random-n-locations", type=int, default=None)
    parser.add_argument("--gp-training-iterations", type=int, default=200)
    parser.add_argument("--gp-learning-rate", type=float, default=0.01)
    parser.add_argument("--gp-inducing-points", type=int, default=64)
    parser.add_argument("--gp-minibatch-size", type=int, default=1024)
    parser.add_argument("--hyperparam-fit-max-time", type=int, default=30)
    parser.add_argument("--hyperparam-fit-max-locations", type=int, default=30)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["deterministic", "same_kernel_gp", "routeb"],
        default=["deterministic", "same_kernel_gp", "routeb"],
    )
    parser.add_argument("--run", action="store_true", help="Execute subprocess experiment commands.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    args.location_label = "full_space_1000_locations" if args.random_n_locations is None else f"random_{args.random_n_locations}_locations"
    command_log: list[dict[str, Any]] = []
    for label, cmd in build_commands(args, outdir):
        expected = {
            "deterministic": outdir / "local_deterministic_baselines" / "era5_deterministic_summary.csv",
            "same_kernel_gp": outdir / "local_same_kernel_gp_baselines" / "era5_same_kernel_gp_summary.csv",
            "routeb": outdir / "routeb_reference" / "era5_ohsvgp_heldout_summary.csv",
        }[label]
        if args.skip_existing and expected.exists():
            command_log.append({"cmd": cmd, "returncode": 0, "status": "skipped_existing"})
            continue
        command_log.append(run_command(cmd, cwd=ROOT, dry_run=not args.run))
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    external_status = write_external_status(ROOT / "baselines" / "external" / "manifest.json", outdir)
    combined = collect_results(outdir)
    figures = plot_seen_history(combined, outdir)
    md = write_markdown_report(outdir, args, combined, external_status, command_log, figures)
    addendum_pdf = write_pdf_report(md, outdir, combined, external_status, figures)
    existing_pdf = Path(args.original_report_pdf)
    merged_pdf = outdir / "era5_ohsvgp_heldout_experiment_report_with_online_gp_addendum.pdf"
    merged = False
    if existing_pdf.exists():
        merged = merge_existing_and_addendum(existing_pdf, addendum_pdf, merged_pdf)
    final_report = merged_pdf if merged else addendum_pdf
    report = {
        "outdir": str(outdir),
        "markdown": str(md),
        "addendum_pdf": str(addendum_pdf),
        "merged_pdf": str(merged_pdf) if merged else "",
        "final_report": str(final_report),
        "combined_summary": str(outdir / "combined_online_gp_suite_summary.csv"),
        "external_status": str(outdir / "external_online_gp_repo_status.csv"),
        "figures": [str(p) for p in figures],
        "commands_executed": bool(args.run),
        "location_label": args.location_label,
    }
    (outdir / "online_gp_baseline_suite_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
