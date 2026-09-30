#!/usr/bin/env python3
"""Audit delayed-history COVID baselines and create paper-ready tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.run_epidemiology_pilot import predictive_metrics
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_epidemiology_pilot import predictive_metrics


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5, 6, 7, 8, 9)
METRICS = ("rmse", "nll", "coverage90")
COLORS = {
    "persistence": "#7B8794",
    "lag_ridge": "#999999",
    "ohsvgp_rbf": "#D55E00",
    "bui_osgpr_controlled": "#E69F00",
    "routeb_ordinary": "#009E73",
    "routeb_cumulative": "#0072B2",
}
METHODS = (
    ("persistence", "Persistence", "deterministic/persistence", "overall_current_block"),
    ("lag_ridge", "Task-1 lag ridge", "deterministic/lag_ridge", "overall_current_block"),
    ("ohsvgp_rbf", "Official OHSVGP (RBF)", "ohsvgp_rbf", "overall_current_block"),
    ("bui_osgpr_controlled", "Official Bui OSGPR (controlled)", "bui_osgpr_controlled", "final"),
    ("routeb_ordinary", "Route B ordinary inducing", "routeb_ordinary/online", "overall_current_block"),
    ("routeb_cumulative", "Route B cumulative HiPPO", None, "overall_current_block"),
)


def prediction_metrics(path: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    with np.load(path) as archive:
        arrays = {key: np.asarray(archive[key]) for key in ("y_true", "pred_mean", "pred_var", "test_indices")}
    if not np.isfinite(arrays["y_true"]).all() or not np.isfinite(arrays["pred_mean"]).all():
        raise ValueError(f"{path}: non-finite targets or means")
    if not np.isfinite(arrays["pred_var"]).all() or (arrays["pred_var"] <= 0.0).any():
        raise ValueError(f"{path}: non-positive or non-finite predictive variance")
    return predictive_metrics(arrays["y_true"], arrays["pred_mean"], arrays["pred_var"]), arrays


def report_metrics(payload: dict[str, object], key: str) -> dict[str, float]:
    return {metric: float(payload[key][metric]) for metric in METRICS}


def result_path(baseline_root: Path, routeb_root: Path, seed: int, relative: str | None) -> Path:
    if relative is None:
        return routeb_root / f"seed{seed}/online"
    return baseline_root / f"seed{seed}" / relative


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def paired_bootstrap(values: list[float]) -> tuple[float, float, float]:
    samples = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(0)
    boot = samples[rng.integers(0, samples.size, size=(20_000, samples.size))].mean(axis=1)
    return float(samples.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def save_figure(fig: plt.Figure, target: Path) -> None:
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_main_metrics(summary: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharey=True)
    y = np.arange(len(summary))[::-1]
    labels = [str(row["label"]) for row in summary]
    config = (("rmse", "RMSE", None), ("nll", "NLL", None), ("coverage90", "Coverage90", 0.90))
    for axis, (metric, title, reference) in zip(axes, config):
        for ypos, row in zip(y, summary):
            color = COLORS[str(row["method"])]
            axis.errorbar(
                float(row[f"{metric}_mean"]), ypos,
                xerr=float(row[f"{metric}_sd"]),
                fmt="o", color=color, ecolor=color, capsize=3, markersize=6,
            )
        if reference is not None:
            axis.axvline(reference, color="#333333", lw=1, ls="--", zorder=0)
        axis.set_title(title)
        axis.grid(axis="x", color="#D9D9D9", lw=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("mean +/- sample SD")
    axes[1].set_xlabel("mean +/- sample SD")
    axes[2].set_xlabel("mean +/- sample SD")
    axes[0].set_xlim(left=0.0)
    fig.suptitle("Delayed-history strict-online COVID nowcasting (seeds 5-9)", y=1.03, fontsize=12)
    save_figure(fig, output / "fig_covid_main_metrics")


def plot_blockwise(per_seed: dict[str, dict[int, dict[str, object]]], output: Path) -> None:
    method_order = ("persistence", "ohsvgp_rbf", "bui_osgpr_controlled", "routeb_ordinary", "routeb_cumulative")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), sharex=True)
    for axis, metric, title, reference in zip(
        axes, ("rmse", "nll", "coverage90"), ("RMSE", "NLL", "Coverage90"), (None, None, 0.90)
    ):
        for method in method_order:
            curves = np.asarray([per_seed[method][seed]["block_metrics"][metric] for seed in SEEDS], dtype=np.float64)
            mean = curves.mean(axis=0)
            sd = curves.std(axis=0, ddof=1)
            x = np.arange(1, mean.size + 1)
            axis.plot(x, mean, lw=1.7, color=COLORS[method], label=next(label for key, label, _, _ in METHODS if key == method))
            axis.fill_between(x, mean - sd, mean + sd, color=COLORS[method], alpha=0.13, linewidth=0)
        if reference is not None:
            axis.axhline(reference, color="#333333", lw=1, ls="--")
        axis.set_title(title)
        axis.set_xlabel("Online week")
        axis.grid(color="#D9D9D9", lw=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean across spatial splits")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.17), fontsize=8)
    fig.suptitle("Online trajectory under the same delayed-observation protocol", y=1.03, fontsize=12)
    save_figure(fig, output / "fig_covid_online_curves")


def plot_efficiency(summary: list[dict[str, object]], output: Path) -> None:
    learned = [row for row in summary if row["method"] not in {"persistence", "lag_ridge"}]
    labels = {
        "ohsvgp_rbf": "OHSVGP\nRBF",
        "bui_osgpr_controlled": "Bui\nOSGPR\nCPU",
        "routeb_ordinary": "Route B\nordinary",
        "routeb_cumulative": "Route B\ncumulative\nHiPPO",
    }
    tick_labels = [labels[str(row["method"])] for row in learned]
    update = [1000.0 * (float(row["update_seconds_mean"]) + float(row["prediction_seconds_mean"])) for row in learned]
    state = [float(row["persistent_state_mib_mean"]) for row in learned]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    x = np.arange(len(learned))
    colors = [COLORS[str(row["method"])] for row in learned]
    bars = axes[0].bar(x, update, color=colors, edgecolor="#333333", linewidth=0.4)
    bars[1].set_hatch("///")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, tick_labels, fontsize=8)
    axes[0].set_ylabel("update + prediction (ms / week, log scale)")
    axes[0].set_title("Online latency")
    axes[0].grid(axis="y", color="#D9D9D9", lw=0.7)
    axes[0].set_axisbelow(True)
    axes[0].text(1, update[1], "CPU only", ha="center", va="bottom", fontsize=8)

    plotted = np.nan_to_num(state, nan=0.0)
    bars = axes[1].bar(x, plotted, color=colors, edgecolor="#333333", linewidth=0.4)
    bars[1].set_hatch("///")
    for index, value in enumerate(state):
        if not np.isfinite(value):
            axes[1].text(index, 0.01, "not\ninstrumented", ha="center", va="bottom", fontsize=8, rotation=90)
    axes[1].set_xticks(x, tick_labels, fontsize=8)
    axes[1].set_ylabel("persistent state (MiB)")
    axes[1].set_title("Stored online state")
    axes[1].grid(axis="y", color="#D9D9D9", lw=0.7)
    axes[1].set_axisbelow(True)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Efficiency context: Bui OSGPR is CPU-only on this host", y=1.03, fontsize=12)
    save_figure(fig, output / "fig_covid_efficiency")


def plot_trajectory(per_seed: dict[str, dict[int, dict[str, object]]], protocol_root: Path, output: Path) -> dict[str, object]:
    seed = 5
    routeb = per_seed["routeb_cumulative"][seed]["arrays"]
    state_rmse = np.sqrt(np.mean((routeb["y_true"] - routeb["pred_mean"]) ** 2, axis=0))
    selected_position = int(np.argsort(state_rmse)[len(state_rmse) // 2])
    with np.load(protocol_root / f"seed{seed}/protocol.npz") as arrays:
        test_indices = np.asarray(arrays["test_indices"], dtype=int)
    metadata = json.loads((protocol_root / f"seed{seed}/protocol.json").read_text(encoding="utf-8"))
    state_name = metadata["location_names"][int(test_indices[selected_position])]
    x = np.arange(1, routeb["y_true"].shape[0] + 1)
    fig, axis = plt.subplots(figsize=(10.2, 3.6))
    axis.plot(x, routeb["y_true"][:, selected_position], color="#202020", lw=2.1, label="Observed")
    for method in ("persistence", "ohsvgp_rbf", "routeb_ordinary", "routeb_cumulative"):
        arrays = per_seed[method][seed]["arrays"]
        axis.plot(x, arrays["pred_mean"][:, selected_position], color=COLORS[method], lw=1.5, label=next(label for key, label, _, _ in METHODS if key == method))
    std = np.sqrt(routeb["pred_var"][:, selected_position])
    axis.fill_between(x, routeb["pred_mean"][:, selected_position] - 1.6448536269514722 * std, routeb["pred_mean"][:, selected_position] + 1.6448536269514722 * std, color=COLORS["routeb_cumulative"], alpha=0.18, label="Route B 90% interval")
    axis.set_xlabel("Online week")
    axis.set_ylabel("log1p admissions per 100k")
    axis.grid(color="#D9D9D9", lw=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", ncol=2, frameon=False, fontsize=8)
    axis.set_title(f"Diagnostic trajectory: {state_name} (seed 5, median Route B per-state RMSE)")
    save_figure(fig, output / "fig_covid_trajectory")
    return {"seed": seed, "state": state_name, "test_position": selected_position, "routeb_state_rmse": float(state_rmse[selected_position])}


def write_latex(summary: list[dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & RMSE $\\downarrow$ & NLL $\\downarrow$ & Coverage90 \\\\",
        "\\midrule",
    ]
    for row in summary:
        label = str(row["label"])
        if row["method"] == "routeb_cumulative":
            label = "\\textbf{" + label + "}"
        lines.append(
            f"{label} & {row['rmse_mean']:.4f} $\\pm$ {row['rmse_sd']:.4f} & "
            f"{row['nll_mean']:.4f} $\\pm$ {row['nll_sd']:.4f} & "
            f"{row['coverage90_mean']:.4f} $\\pm$ {row['coverage90_sd']:.4f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "% Mean +/- sample SD over independent spatial split seeds 5-9.",
            "% Bui OSGPR is CPU-only and uses frozen Route-B Task-1 theta for a controlled posterior-transfer comparison.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=Path("results/diagnostics/covid_delayed_baseline_comparison"))
    parser.add_argument("--routeb-root", type=Path, default=Path("results/diagnostics/covid_routeb_upgrade/phase9_confirmation_seeds5_9"))
    parser.add_argument("--protocol-root", type=Path, default=Path("data/epidemiology/protocol/covid_history_aware"))
    args = parser.parse_args()
    baseline_root = (ROOT / args.baseline_root).resolve()
    routeb_root = (ROOT / args.routeb_root).resolve()
    protocol_root = (ROOT / args.protocol_root).resolve()
    output = baseline_root / "final"
    output.mkdir(parents=True, exist_ok=True)

    per_seed: dict[str, dict[int, dict[str, object]]] = {key: {} for key, _, _, _ in METHODS}
    audited = []
    for seed in SEEDS:
        reference_arrays = None
        for method, label, relative, result_key in METHODS:
            run = result_path(baseline_root, routeb_root, seed, relative)
            payload = json.loads((run / "result.json").read_text(encoding="utf-8"))
            metrics, arrays = prediction_metrics(run / "predictions.npz")
            reported = report_metrics(payload, result_key)
            for metric in METRICS:
                if abs(metrics[metric] - reported[metric]) > 1e-8:
                    raise ValueError(f"{method} seed {seed}: {metric} differs from result.json")
            if int(payload["split_seed"]) != seed:
                raise ValueError(f"{method} seed {seed}: result split seed mismatch")
            if reference_arrays is None:
                reference_arrays = arrays
            else:
                if not np.array_equal(arrays["test_indices"], reference_arrays["test_indices"]):
                    raise ValueError(f"{method} seed {seed}: held-out state identities differ")
                if not np.allclose(arrays["y_true"], reference_arrays["y_true"], rtol=0.0, atol=1e-12):
                    raise ValueError(f"{method} seed {seed}: held-out labels differ")
            if method in {"ohsvgp_rbf", "bui_osgpr_controlled", "routeb_ordinary", "routeb_cumulative"}:
                if not payload.get("delayed_observations") or int(payload.get("delayed_observation_rows", -1)) != 380:
                    raise ValueError(f"{method} seed {seed}: delayed observation audit failed")
            block_metrics = {
                metric: np.asarray(
                    [predictive_metrics(arrays["y_true"][block], arrays["pred_mean"][block], arrays["pred_var"][block])[metric] for block in range(arrays["y_true"].shape[0])],
                    dtype=np.float64,
                )
                for metric in METRICS
            }
            timing = payload.get("timing", {})
            resources = payload.get("resources", {})
            update = float(timing.get("mean_steady_state_block_update_seconds", timing.get("mean_block_update_seconds", 0.0)))
            prediction = float(timing.get("mean_block_prediction_seconds", 0.0))
            state_mib = float(resources["persistent_state_mib"]) if "persistent_state_mib" in resources else float("nan")
            per_seed[method][seed] = {
                "metrics": metrics,
                "arrays": arrays,
                "block_metrics": block_metrics,
                "update_seconds": update,
                "prediction_seconds": prediction,
                "persistent_state_mib": state_mib,
                "device": str(resources.get("device", "not_instrumented")),
                "artifact": str(run.relative_to(ROOT)),
            }
            audited.append({"seed": seed, "method": method, "artifact": str(run.relative_to(ROOT)), "delayed_checked": method not in {"persistence", "lag_ridge"}})

    summary = []
    for method, label, _, _ in METHODS:
        row: dict[str, object] = {"method": method, "label": label}
        for metric in METRICS:
            mean, sd = mean_sd([float(per_seed[method][seed]["metrics"][metric]) for seed in SEEDS])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        for key in ("update_seconds", "prediction_seconds", "persistent_state_mib"):
            values = [float(per_seed[method][seed][key]) for seed in SEEDS]
            finite = [value for value in values if np.isfinite(value)]
            row[f"{key}_mean"] = float(np.mean(finite)) if finite else float("nan")
        row["device"] = ";".join(sorted({str(per_seed[method][seed]["device"]) for seed in SEEDS}))
        summary.append(row)
    write_latex(summary, output / "table_covid_delayed_baselines.tex")
    with (output / "metrics_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "method", *METRICS, "update_seconds", "prediction_seconds", "persistent_state_mib", "device", "artifact"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, _, _, _ in METHODS:
            for seed in SEEDS:
                record = per_seed[method][seed]
                writer.writerow({"seed": seed, "method": method, **{metric: record["metrics"][metric] for metric in METRICS}, "update_seconds": record["update_seconds"], "prediction_seconds": record["prediction_seconds"], "persistent_state_mib": record["persistent_state_mib"], "device": record["device"], "artifact": record["artifact"]})
    with (output / "metrics_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    contrasts = []
    for method, label, _, _ in METHODS:
        if method == "routeb_cumulative":
            continue
        for metric in METRICS:
            values = [float(per_seed[method][seed]["metrics"][metric]) - float(per_seed["routeb_cumulative"][seed]["metrics"][metric]) for seed in SEEDS]
            mean, low, high = paired_bootstrap(values)
            contrasts.append({"method": method, "label": label, "metric": metric, "baseline_minus_routeb_mean": mean, "bootstrap95_low": low, "bootstrap95_high": high})
    with (output / "paired_contrasts_vs_routeb.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0].keys()))
        writer.writeheader()
        writer.writerows(contrasts)

    plot_main_metrics(summary, output)
    plot_blockwise(per_seed, output)
    plot_efficiency(summary, output)
    trajectory = plot_trajectory(per_seed, protocol_root, output)

    routeb = next(row for row in summary if row["method"] == "routeb_cumulative")
    persistence = next(row for row in summary if row["method"] == "persistence")
    ordinary = next(row for row in summary if row["method"] == "routeb_ordinary")
    report = [
        "# Delayed-History Strict-Online COVID Baseline Comparison",
        "",
        "## Protocol",
        "",
        "All main rows use independent spatial split seeds 5-9, 52 Task-1 calibration weeks, 39 one-week strict-online blocks, 42 visible states, 4 validation states, and 10 scored hidden states. At week t, a learned method first absorbs the previously scored hidden labels exactly once, updates on the 42 current visible labels, and then predicts the 10 current hidden labels. All figures use prediction archives after recomputation, not runner summaries alone.",
        "",
        "Route B cumulative HiPPO uses Mt=32, Ms=32, geographic kernel, fixed Task-1 Q=2 spectral-mixture theta, visible-state Task-1 posterior initialization, and full-joint-conditional variance. Ordinary Route B has Mt=32, Ms=32 and identical delayed schedule. OHSVGP is the pinned upstream multidimensional implementation with one M=32 state and RFF=64; this is not equivalent to a separate Ms=32 spatial state. Bui OSGPR is the pinned official VFE update with 32x32 fixed Cartesian inducing points, visible-state Task-1 posterior warm start, and frozen Route-B Task-1 theta, so it is explicitly a controlled posterior-transfer comparison rather than Bui-owned empirical Bayes.",
        "",
        "## Main Results",
        "",
        "| Method | RMSE | NLL | Coverage90 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        report.append(f"| {row['label']} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | {row['nll_mean']:.4f} +/- {row['nll_sd']:.4f} | {row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} |")
    report.extend(
        [
            "",
            f"Route B cumulative has RMSE {routeb['rmse_mean']:.4f} +/- {routeb['rmse_sd']:.4f}. Persistence has {persistence['rmse_mean']:.4f} +/- {persistence['rmse_sd']:.4f}; ordinary Route B has {ordinary['rmse_mean']:.4f} +/- {ordinary['rmse_sd']:.4f}. Paired bootstrap contrasts are saved separately and must be used for method-to-method claims.",
            "",
            "## Efficiency Scope",
            "",
            "Route B and OHSVGP were run on the RTX 5070 Laptop GPU. The installed TensorFlow 2.16.2 build does not recognize that GPU, so Bui OSGPR is a CPU-only accuracy baseline and is excluded from a GPU runtime ranking. OHSVGP persistent state is not instrumented by its upstream runner and is marked accordingly rather than estimated. This is a feasibility pilot, not a common-backend speed proof.",
            "",
            "## Figures",
            "",
            "- `fig_covid_main_metrics.pdf`: main 5-split accuracy and calibration comparison.",
            "- `fig_covid_online_curves.pdf`: weekly RMSE, NLL and Coverage90 trajectories.",
            "- `fig_covid_efficiency.pdf`: online latency and reported persistent state scope.",
            "- `fig_covid_trajectory.pdf`: diagnostic held-out state trajectory.",
            "",
            f"Trajectory selection: {trajectory['state']} in seed {trajectory['seed']}, the median Route-B per-state RMSE among the ten held-out states; Route-B state RMSE {trajectory['routeb_state_rmse']:.4f}.",
            "",
            "## Boundary",
            "",
            "COVID has only 52 locations and 91 weekly observations. These results support a controlled feasibility comparison under the stated delayed-history protocol, not a final epidemiological benchmark or a general long-memory claim.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "artifact_audit.json").write_text(json.dumps({"status": "complete", "seeds": list(SEEDS), "runs_audited": audited, "prediction_metrics_recomputed": True, "trajectory": trajectory}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "routeb_rmse": routeb["rmse_mean"]}, indent=2))


if __name__ == "__main__":
    main()
