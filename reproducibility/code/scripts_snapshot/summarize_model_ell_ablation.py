from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "routeB_experiment_report"
PLOTDIR = OUTDIR / "plots"
TABLEDIR = OUTDIR / "tables"
METHOD_LABELS = {
    "no_transfer": "no_transfer",
    "mean_field_ssgp_transfer": "mean-field",
    "structured_joint_ssgp_transfer": "Route B",
}


def parse_key(key: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in key.split("|"))


def collect_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = ROOT / "results" / "experiments_routeB_model_ell_ablation"
    for report_path in sorted(base.glob("mt_*/joint_ssgp_kron_synthetic_report.json")):
        mt_match = re.search(r"mt_(\d+)", str(report_path))
        if not mt_match:
            continue
        mt = int(mt_match.group(1))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for key, metrics in report["seed_level_summary_by_ablation"].items():
            parts = parse_key(key)
            if parts.get("eval_mode") != "seen_history":
                continue
            row: dict[str, object] = {
                "mt": mt,
                "data_ell_t": float(parts["ell_t"]),
                "model_ell_t": float(parts["model_ell_t"]),
                "method": METHOD_LABELS.get(parts["method"], parts["method"]),
            }
            for metric in ["rmse", "nll", "coverage90", "rmse_forgetting", "nll_forgetting", "avg_predictive_variance"]:
                row[f"{metric}_mean"] = metrics[metric]["mean"]
                row[f"{metric}_se"] = metrics[metric]["se"]
            rows.append(row)
    return rows


def collect_first_batch_fit() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = ROOT / "results" / "experiments_routeB_model_ell_first_batch_fit"
    for metrics_path in sorted(base.glob("mt_*/joint_ssgp_kron_synthetic_metrics.csv")):
        mt_match = re.search(r"mt_(\d+)", str(metrics_path))
        if not mt_match:
            continue
        mt = int(mt_match.group(1))
        with metrics_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw = [r for r in reader if r["eval_mode"] == "seen_history"]
        for method in ["no_transfer", "mean_field_ssgp_transfer", "structured_joint_ssgp_transfer"]:
            method_rows = [r for r in raw if r["method"] == method]
            chosen = sorted({float(r["model_ell_t"]) for r in method_rows})
            for metric in ["rmse", "nll", "coverage90", "rmse_forgetting", "nll_forgetting"]:
                seed_vals = []
                for seed in sorted({int(r["seed"]) for r in method_rows}):
                    vals = [float(r[metric]) for r in method_rows if int(r["seed"]) == seed and r[metric] != "nan"]
                    if vals:
                        seed_vals.append(float(np.mean(vals)))
                arr = np.asarray(seed_vals, dtype=float)
                rows.append(
                    {
                        "mt": mt,
                        "method": METHOD_LABELS[method],
                        "metric": metric,
                        "mean": float(np.mean(arr)) if arr.size else float("nan"),
                        "se": float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
                        "num_seeds": int(arr.size),
                        "chosen_model_ell_t_values": " ".join(str(x) for x in chosen),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_grid(rows: list[dict[str, object]]) -> None:
    PLOTDIR.mkdir(parents=True, exist_ok=True)
    methods = ["no_transfer", "mean-field", "Route B"]
    colors = {"no_transfer": "#8da0cb", "mean-field": "#66c2a5", "Route B": "#fc8d62"}
    for metric, ylabel in [
        ("rmse", "Seen-history RMSE"),
        ("nll", "Seen-history NLL"),
        ("coverage90", "90% coverage"),
        ("rmse_forgetting", "RMSE forgetting"),
        ("nll_forgetting", "NLL forgetting"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
        for ax, mt in zip(axes.ravel(), [5, 8, 12, 16]):
            for method in methods:
                subset = [r for r in rows if r["mt"] == mt and r["method"] == method]
                subset = sorted(subset, key=lambda r: float(r["model_ell_t"]))
                ax.errorbar(
                    [float(r["model_ell_t"]) for r in subset],
                    [float(r[f"{metric}_mean"]) for r in subset],
                    yerr=[float(r[f"{metric}_se"]) for r in subset],
                    marker="o",
                    label=method,
                    color=colors[method],
                )
            if metric == "coverage90":
                ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"M_t={mt}")
            ax.set_xlabel("model ell_t")
            ax.set_ylabel(ylabel)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.suptitle(f"Long-memory model temporal lengthscale ablation: {ylabel}")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(PLOTDIR / f"model_ell_ablation_{metric}.png", dpi=220)
        plt.close(fig)


def main() -> None:
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    rows = collect_grid()
    write_csv(TABLEDIR / "model_ell_ablation_seen_history.csv", rows)
    averaged_rows: list[dict[str, object]] = []
    for method in ["no_transfer", "mean-field", "Route B"]:
        for ell in [0.25, 0.5, 0.8]:
            subset = [r for r in rows if r["method"] == method and abs(float(r["model_ell_t"]) - ell) < 1e-12]
            averaged_rows.append(
                {
                    "method": method,
                    "model_ell_t": ell,
                    "rmse_mean_over_mt": float(np.mean([float(r["rmse_mean"]) for r in subset])),
                    "nll_mean_over_mt": float(np.mean([float(r["nll_mean"]) for r in subset])),
                    "coverage90_mean_over_mt": float(np.mean([float(r["coverage90_mean"]) for r in subset])),
                    "rmse_forgetting_mean_over_mt": float(np.mean([float(r["rmse_forgetting_mean"]) for r in subset])),
                    "nll_forgetting_mean_over_mt": float(np.mean([float(r["nll_forgetting_mean"]) for r in subset])),
                }
            )
    write_csv(TABLEDIR / "model_ell_ablation_average_over_mt.csv", averaged_rows)
    fit_rows = collect_first_batch_fit()
    write_csv(TABLEDIR / "model_ell_first_batch_fit_seen_history.csv", fit_rows)
    plot_grid(rows)

    routeb = [r for r in rows if r["method"] == "Route B"]
    print("Average over M_t")
    for row in averaged_rows:
        print(
            f"{row['method']} model_ell_t={row['model_ell_t']}: "
            f"rmse={row['rmse_mean_over_mt']:.4f}, "
            f"nll={row['nll_mean_over_mt']:.4f}, "
            f"cov90={row['coverage90_mean_over_mt']:.4f}, "
            f"rmse_forget={row['rmse_forgetting_mean_over_mt']:.4f}, "
            f"nll_forget={row['nll_forgetting_mean_over_mt']:.4f}"
        )
    print("Route B summary by M_t and model ell_t")
    for mt in [5, 8, 12, 16]:
        subset = sorted([r for r in routeb if r["mt"] == mt], key=lambda r: float(r["model_ell_t"]))
        for r in subset:
            print(
                f"mt={mt} model_ell_t={r['model_ell_t']}: "
                f"rmse={float(r['rmse_mean']):.4f}±{float(r['rmse_se']):.4f}, "
                f"nll={float(r['nll_mean']):.4f}±{float(r['nll_se']):.4f}, "
                f"cov90={float(r['coverage90_mean']):.4f}, "
                f"rmse_forget={float(r['rmse_forgetting_mean']):.4f}, "
                f"nll_forget={float(r['nll_forgetting_mean']):.4f}"
            )


if __name__ == "__main__":
    main()
