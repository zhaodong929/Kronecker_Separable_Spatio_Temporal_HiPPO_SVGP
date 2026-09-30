"""Summarize ERA5 Route B future horizon and block-size diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_run(run_dir: Path, label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report = json.loads((run_dir / "era5_routeb_report.json").read_text(encoding="utf-8"))
    args = report["args"]
    block_size = int(args["block_size"])
    future_basis_mode = str(args.get("future_basis_mode", "observed"))
    summary_rows = read_csv(run_dir / "era5_routeb_summary.csv")
    horizon_rows = read_csv(run_dir / "era5_routeb_future_horizon_summary.csv")
    for rows in (summary_rows, horizon_rows):
        for row in rows:
            row["run_label"] = label
            row["block_size"] = block_size
            row["future_basis_mode"] = row.get("future_basis_mode") or future_basis_mode
            row["source_dir"] = str(run_dir)
    return summary_rows, horizon_rows


def f(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def plot_blocksize(summary_rows: list[dict[str, Any]], outdir: Path) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in summary_rows
        if row.get("eval_mode") == "future"
    ]
    modes = sorted({str(row.get("future_basis_mode", "observed")) for row in rows})
    for metric in ["rmse", "nll", "coverage90", "ece", "avg_predictive_variance"]:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for mode in modes:
            mode_rows = sorted([row for row in rows if row.get("future_basis_mode", "observed") == mode], key=lambda row: int(row["block_size"]))
            x = np.asarray([int(row["block_size"]) for row in mode_rows], dtype=int)
            y = np.asarray([f(row, metric) for row in mode_rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=2, label=mode)
        if metric == "coverage90":
            ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("block_size")
        ax.set_ylabel(metric)
        ax.set_title(f"Future prediction vs block size: {metric}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"era5_future_blocksize_{metric}.png", dpi=180)
        plt.close(fig)


def plot_horizon(horizon_rows: list[dict[str, Any]], outdir: Path) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    series = sorted({(int(row["block_size"]), str(row.get("future_basis_mode", "observed"))) for row in horizon_rows})
    for metric in ["rmse", "nll", "coverage90", "ece", "avg_predictive_variance", "avg_nu_star"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for block_size, future_basis_mode in series:
            rows = [
                row
                for row in horizon_rows
                if int(row["block_size"]) == block_size and row.get("future_basis_mode", "observed") == future_basis_mode
            ]
            rows = sorted(rows, key=lambda row: int(row["horizon_index"]))
            x = np.asarray([int(row["horizon_index"]) for row in rows], dtype=int)
            y = np.asarray([f(row, metric) for row in rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.8, label=f"bs={block_size}, {future_basis_mode}")
        if metric == "coverage90":
            ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("future horizon inside block")
        ax.set_ylabel(metric)
        ax.set_title(f"Future horizon breakdown: {metric}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"era5_future_horizon_blocksize_{metric}.png", dpi=180)
        plt.close(fig)


def write_report(summary_rows: list[dict[str, Any]], horizon_rows: list[dict[str, Any]], outdir: Path) -> None:
    future_rows = [row for row in summary_rows if row.get("eval_mode") == "future"]
    future_rows = sorted(future_rows, key=lambda row: (int(row["block_size"]), str(row.get("future_basis_mode", "observed"))))
    bs10_rows = [
        row
        for row in summary_rows
        if row.get("eval_mode") == "future" and int(row["block_size"]) == 10
    ]
    lines = [
        "# ERA5 Future Horizon and Block-Size Diagnostic",
        "",
        "This diagnostic tests whether weak future performance is caused by block-ahead multi-step prediction.",
        "The Route B core formulas, old-likelihood transfer, R_beta_u transfer, Schur complement, and Sylvester/D_u solve are unchanged.",
        "",
        "## Shared Setting",
        "",
        "- calibration task: task_1",
        "- online task: task_2",
        "- full spatial grid: 1000 locations",
        "- method: structured_joint",
        "- phi mode: base",
        "- M_t=8, M_s=64",
        "- model ell_t=0.05, selected from the earlier task_1 calibration protocol; full all-location dense MLL is infeasible",
        "- evaluation mode: future only",
        "",
            "## Block-Size and Future-Basis Sweep",
            "",
            "| block_size | future_basis | RMSE | NLL | Cov90 | ECE | Avg var |",
            "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in future_rows:
        lines.append(
            f"| {int(row['block_size'])} | {row.get('future_basis_mode', 'observed')} | "
            f"{f(row, 'rmse'):.4f} | {f(row, 'nll'):.4f} | {f(row, 'coverage90'):.4f} | "
            f"{f(row, 'ece'):.4f} | {f(row, 'avg_predictive_variance'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Observed vs Extended Future Basis at block_size=10",
            "",
            "| future_basis | RMSE | NLL | Cov90 | ECE | Avg var | avg_nu_star |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(bs10_rows, key=lambda row: str(row.get("future_basis_mode", "observed"))):
        lines.append(
            f"| {row.get('future_basis_mode', 'observed')} | {f(row, 'rmse'):.4f} | {f(row, 'nll'):.4f} | "
            f"{f(row, 'coverage90'):.4f} | {f(row, 'ece'):.4f} | {f(row, 'avg_predictive_variance'):.4f} | "
            f"{f(row, 'avg_nu_star'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- block_size=1 is the one-step prequential-style setting; it is much easier than predicting a whole future block without intermediate updates.",
            "- The observed-basis results improve strongly as block_size decreases, so the poor block_size=10 future result is mainly a block-ahead multi-step forecasting issue.",
            "- Extended future basis improves the harder block-ahead settings more than the one-step setting, which supports the diagnosis that the observed prediction basis did not fully cover longer prediction intervals.",
            "- Neither diagnostic uses future labels for model fitting; labels are used only after prediction to compute metrics.",
            "",
            "## Output Files",
            "",
            "- `era5_future_blocksize_summary.csv`",
            "- `era5_future_horizon_blocksize_summary.csv`",
            "- `plots/era5_future_blocksize_rmse.png`",
            "- `plots/era5_future_blocksize_nll.png`",
            "- `plots/era5_future_horizon_blocksize_rmse.png`",
            "- `plots/era5_future_horizon_blocksize_nll.png`",
        ]
    )
    outdir.joinpath("era5_future_horizon_blocksize_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/experiments_era5_future_horizon_blocksize_diagnostic")
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIR"), required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for label, run_dir in args.run:
        run_summary, run_horizon = load_run(Path(run_dir), label)
        summary_rows.extend(run_summary)
        horizon_rows.extend(run_horizon)

    write_csv(summary_rows, outdir / "era5_future_blocksize_summary.csv")
    write_csv(horizon_rows, outdir / "era5_future_horizon_blocksize_summary.csv")
    plot_blocksize(summary_rows, outdir)
    plot_horizon(horizon_rows, outdir)
    write_report(summary_rows, horizon_rows, outdir)
    print(json.dumps({"outdir": str(outdir), "summary_rows": len(summary_rows), "horizon_rows": len(horizon_rows)}, indent=2))


if __name__ == "__main__":
    main()
