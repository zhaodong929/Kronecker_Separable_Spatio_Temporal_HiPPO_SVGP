from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "routeB_experiment_report"
PLOTDIR = OUTDIR / "plots"
TABLEDIR = OUTDIR / "tables"
METHODS = ["no_transfer", "mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]
LABELS = {
    "no_transfer": "no_transfer",
    "mean_field_ssgp_transfer": "mean-field",
    "structured_joint_ssgp_transfer": "Route B",
}


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_level(rows: list[dict[str, str]], method: str, metric: str) -> tuple[float, float]:
    vals = []
    for seed in sorted({int(r["seed"]) for r in rows if r["method"] == method}):
        seed_vals = [
            float(r[metric])
            for r in rows
            if r["method"] == method and int(r["seed"]) == seed and r["eval_mode"] == "seen_history" and r[metric] != "nan"
        ]
        if seed_vals:
            vals.append(float(np.mean(seed_vals)))
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), se


def summarize_long_memory() -> list[dict[str, object]]:
    base = ROOT / "results" / "experiments_routeB_long_memory_initial_task_ellt_fit"
    settings = [
        ("mismatch", base / "mismatch_model_ellt_025"),
        ("oracle", base / "oracle_model_ellt_08"),
        ("initial_task_fullgp_mll", base / "initial_task_fullgp"),
    ]
    rows_out: list[dict[str, object]] = []
    for setting, setting_dir in settings:
        all_rows = []
        metrics_paths = sorted(setting_dir.glob("mt_*/joint_ssgp_kron_synthetic_metrics.csv"))
        direct_path = setting_dir / "joint_ssgp_kron_synthetic_metrics.csv"
        if direct_path.exists():
            metrics_paths.append(direct_path)
        for metrics_path in metrics_paths:
            mt = int(metrics_path.parent.name.split("_")[1]) if metrics_path.parent.name.startswith("mt_") else -1
            for row in read_metrics(metrics_path):
                row = dict(row)
                row["mt"] = mt
                all_rows.append(row)
        for method in METHODS:
            row_out: dict[str, object] = {"regime": "long_memory", "setting": setting, "method": LABELS[method]}
            for metric in ["rmse", "nll", "coverage90", "rmse_forgetting", "nll_forgetting"]:
                mean, se = seed_level(all_rows, method, metric)
                row_out[f"{metric}_mean"] = mean
                row_out[f"{metric}_se"] = se
            fitted_vals = sorted({float(r["fitted_ell_t"]) for r in all_rows if r["method"] == method and r["fitted_ell_t"] not in ("", "nan")})
            model_vals = sorted({float(r["model_ell_t"]) for r in all_rows if r["method"] == method and r["model_ell_t"] not in ("", "nan")})
            row_out["model_ell_t_values"] = " ".join(f"{x:g}" for x in model_vals)
            row_out["fitted_ell_t_values"] = " ".join(f"{x:g}" for x in fitted_vals)
            rows_out.append(row_out)
    return rows_out


def summarize_standard() -> list[dict[str, object]]:
    settings = [
        ("fixed_0.25", ROOT / "results" / "experiments_routeB_standard_confirmatory_all" / "joint_ssgp_kron_synthetic_metrics.csv"),
        ("initial_task_fullgp_mll", ROOT / "results" / "experiments_routeB_standard_initial_task_fullgp_ellt_fit" / "joint_ssgp_kron_synthetic_metrics.csv"),
    ]
    rows_out: list[dict[str, object]] = []
    for setting, path in settings:
        if not path.exists():
            continue
        rows = read_metrics(path)
        for method in METHODS:
            row_out: dict[str, object] = {"regime": "standard", "setting": setting, "method": LABELS[method]}
            for metric in ["rmse", "nll", "coverage90", "rmse_forgetting", "nll_forgetting"]:
                mean, se = seed_level(rows, method, metric)
                row_out[f"{metric}_mean"] = mean
                row_out[f"{metric}_se"] = se
            model_vals = sorted({float(r["model_ell_t"]) for r in rows if r["method"] == method and r.get("model_ell_t", "") not in ("", "nan")})
            fitted_vals = sorted({float(r["fitted_ell_t"]) for r in rows if r["method"] == method and r.get("fitted_ell_t", "") not in ("", "nan")})
            row_out["model_ell_t_values"] = " ".join(f"{x:g}" for x in model_vals)
            row_out["fitted_ell_t_values"] = " ".join(f"{x:g}" for x in fitted_vals)
            rows_out.append(row_out)
    return rows_out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_selected_ell_t() -> list[dict[str, object]]:
    configs = [
        (
            "long_memory",
            "initial_task_fullgp_mll",
            ROOT / "results" / "experiments_routeB_long_memory_initial_task_ellt_fit" / "initial_task_fullgp" / "joint_ssgp_kron_synthetic_metrics.csv",
        ),
        (
            "standard",
            "initial_task_fullgp_mll",
            ROOT / "results" / "experiments_routeB_standard_initial_task_fullgp_ellt_fit" / "joint_ssgp_kron_synthetic_metrics.csv",
        ),
    ]
    out: list[dict[str, object]] = []
    for regime, setting, path in configs:
        if not path.exists():
            continue
        rows = read_metrics(path)
        for seed in sorted({int(r["seed"]) for r in rows}):
            seed_rows = [
                r
                for r in rows
                if int(r["seed"]) == seed
                and r["method"] == "structured_joint_ssgp_transfer"
                and r["eval_mode"] == "seen_history"
            ]
            fitted = sorted({float(r["fitted_ell_t"]) for r in seed_rows if r.get("fitted_ell_t", "") not in ("", "nan")})
            score = sorted({float(r["selected_candidate_score"]) for r in seed_rows if r.get("selected_candidate_score", "") not in ("", "nan")})
            grid = sorted({r.get("ell_t_grid", "") for r in seed_rows if r.get("ell_t_grid", "")})
            out.append(
                {
                    "regime": regime,
                    "setting": setting,
                    "seed": seed,
                    "fitted_ell_t": " ".join(f"{x:g}" for x in fitted),
                    "selected_candidate_score": score[0] if score else float("nan"),
                    "ell_t_grid": grid[0] if grid else "",
                }
            )
    return out


def plot_summary(rows: list[dict[str, object]], regime: str) -> None:
    PLOTDIR.mkdir(parents=True, exist_ok=True)
    subset = [r for r in rows if r["regime"] == regime]
    settings = list(dict.fromkeys(str(r["setting"]) for r in subset))
    methods = ["no_transfer", "mean-field", "Route B"]
    x = np.arange(len(settings))
    width = 0.24
    for metric, ylabel in [
        ("rmse", "RMSE"),
        ("nll", "NLL"),
        ("coverage90", "90% coverage"),
        ("rmse_forgetting", "RMSE forgetting"),
        ("nll_forgetting", "NLL forgetting"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, method in enumerate(methods):
            vals = []
            ses = []
            for setting in settings:
                row = next(r for r in subset if r["setting"] == setting and r["method"] == method)
                vals.append(float(row[f"{metric}_mean"]))
                ses.append(float(row[f"{metric}_se"]))
            ax.bar(x + (i - 1) * width, vals, width, yerr=ses, capsize=3, label=method)
        if metric == "coverage90":
            ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(settings, rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{regime}: ell_t fitting protocol, {ylabel}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTDIR / f"{regime}_ellt_fit_{metric}.png", dpi=220)
        plt.close(fig)


def main() -> None:
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    rows = summarize_long_memory() + summarize_standard()
    write_csv(TABLEDIR / "ellt_fit_protocol_summary.csv", rows)
    write_csv(TABLEDIR / "ellt_fit_selected_values.csv", summarize_selected_ell_t())
    plot_summary(rows, "long_memory")
    plot_summary(rows, "standard")
    for row in rows:
        if row["method"] == "Route B":
            print(
                row["regime"],
                row["setting"],
                row["method"],
                f"rmse={row['rmse_mean']:.4f}",
                f"nll={row['nll_mean']:.4f}",
                f"cov90={row['coverage90_mean']:.4f}",
                f"rmse_forget={row['rmse_forgetting_mean']:.4f}",
                f"nll_forget={row['nll_forgetting_mean']:.4f}",
                f"fitted={row['fitted_ell_t_values']}",
            )


if __name__ == "__main__":
    main()
