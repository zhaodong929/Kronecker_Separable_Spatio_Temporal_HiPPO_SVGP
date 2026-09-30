#!/usr/bin/env python3
"""Create the paper-ready PEMS-BAY three-seed nowcasting figure and table."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from stvgp_kronecker.data.traffic import load_spatial_split, load_traffic_dataset


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/traffic/formal_locked_sm_q2_road_context_v1"
OUTPUT = RESULTS / "paper_ready"
PER_SEED = RESULTS / "per_seed_results_1_2_3.csv"
AGGREGATE = RESULTS / "aggregate_results_1_2_3.csv"

LABELS = {
    "persistence": "Delayed-target last-value persistence",
    "ignnk": "IGNNK",
    "kron_stgp": "Kron-STGP",
    "joint_fixed_global": "Joint + fixed-global",
    "decoupled_fixed_global": "Decoupled + fixed-global",
    "kronhippo_stgp": "KronHiPPO-STGP",
}
PLOT_ORDER = (
    "kron_stgp",
    "joint_fixed_global",
    "decoupled_fixed_global",
    "ignnk",
    "kronhippo_stgp",
)
TABLE_BLOCKS = (
    ("Delayed-target control", ("persistence",)),
    ("External baseline", ("ignnk",)),
    (
        "Our method and mechanism controls",
        ("kron_stgp", "joint_fixed_global", "decoupled_fixed_global", "kronhippo_stgp"),
    ),
)
METRICS = ("rmse_mph", "crps", "gaussian_nlpd", "ece", "coverage90")
NOWCAST = RESULTS / "pems_bay" / "nowcast"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_value(row: dict[str, str], metric: str, *, latex: bool = False) -> str:
    mean = row.get(f"{metric}_mean", "")
    sd = row.get(f"{metric}_sd", "")
    if not mean or not sd:
        return "--"
    separator = r" $\pm$ " if latex else " +/- "
    return f"{float(mean):.4f}{separator}{float(sd):.4f}"


def make_paired_plot(rows: list[dict[str, str]]) -> dict[str, object]:
    by_method_seed = {
        (row["method"], int(row["seed"])): float(row["rmse_mph"])
        for row in rows
        if row["method"] in PLOT_ORDER
    }
    seeds = (1, 2, 3)
    x = np.arange(len(PLOT_ORDER))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(7.2, 3.45))
    seed_colors = ("#6B8EAD", "#5A9A84", "#B47760")
    seed_markers = ("o", "s", "^")
    for seed, color, marker in zip(seeds, seed_colors, seed_markers):
        values = [by_method_seed[(method, seed)] for method in PLOT_ORDER]
        axis.plot(
            x,
            values,
            color=color,
            marker=marker,
            markersize=4.0,
            linewidth=0.9,
            alpha=0.62,
            label=f"Split {seed}",
            zorder=2,
        )

    means = np.asarray(
        [np.mean([by_method_seed[(method, seed)] for seed in seeds]) for method in PLOT_ORDER]
    )
    axis.plot(
        x,
        means,
        color="black",
        marker="D",
        markersize=6.2,
        linewidth=1.8,
        label="Mean",
        zorder=4,
    )
    axis.set_xticks(
        x,
        [
            "Kron-STGP\n(point temporal)",
            "Joint +\nfixed-global",
            "Decoupled +\nfixed-global",
            "IGNNK",
            "KronHiPPO-STGP\n(joint + changing)",
        ],
    )
    axis.set_ylabel("RMSE (mph)")
    axis.set_title("Paired strict-online nowcasting performance across spatial splits", loc="left", pad=8)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(0.7)
    axis.tick_params(width=0.7)
    axis.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    axis.text(
        0.015,
        0.035,
        "KronHiPPO improves over all internal controls on 3/3 splits; over IGNNK on 2/3.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )
    fig.tight_layout(pad=0.8)
    png = OUTPUT / "fig_pems_paired_rmse.png"
    pdf = OUTPUT / "fig_pems_paired_rmse.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    comparisons = {
        method: sum(
            by_method_seed[("kronhippo_stgp", seed)] < by_method_seed[(method, seed)]
            for seed in seeds
        )
        for method in PLOT_ORDER
        if method != "kronhippo_stgp"
    }
    return {"seeds": list(seeds), "hippo_pairwise_wins": comparisons}


def make_paired_delta_plot(rows: list[dict[str, str]]) -> dict[str, object]:
    methods = ("kron_stgp", "joint_fixed_global", "decoupled_fixed_global", "ignnk")
    seeds = (1, 2, 3)
    by_method_seed = {
        (row["method"], int(row["seed"])): float(row["rmse_mph"])
        for row in rows
        if row["method"] in (*methods, "kronhippo_stgp")
    }
    deltas = np.asarray(
        [
            [by_method_seed[(method, seed)] - by_method_seed[("kronhippo_stgp", seed)] for seed in seeds]
            for method in methods
        ]
    )

    fig, axis = plt.subplots(figsize=(6.8, 3.15))
    x = np.arange(len(methods), dtype=float)
    offsets = (-0.11, 0.0, 0.11)
    colors = ("#0072B2", "#009E73", "#D55E00")
    markers = ("o", "s", "^")
    for seed_index, (seed, offset, color, marker) in enumerate(zip(seeds, offsets, colors, markers)):
        axis.scatter(
            x + offset,
            deltas[:, seed_index],
            s=30,
            marker=marker,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            label=f"Split {seed}",
            zorder=3,
        )
    means = deltas.mean(axis=1)
    axis.scatter(x, means, s=58, marker="D", color="black", label="Mean", zorder=4)
    for method_index in range(len(methods)):
        axis.vlines(
            x[method_index],
            deltas[method_index].min(),
            deltas[method_index].max(),
            color="#A6A6A6",
            linewidth=0.8,
            zorder=1,
        )
        wins = int(np.sum(deltas[method_index] > 0.0))
        axis.text(
            x[method_index],
            max(deltas[method_index].max(), means[method_index]) + 0.045,
            f"{wins}/3",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )
    axis.axhline(0.0, color="#4D4D4D", linewidth=0.9, zorder=0)
    axis.set_xticks(
        x,
        ["Kron-STGP", "Joint + fixed", "Decoupled + fixed", "IGNNK"],
    )
    axis.set_ylabel(r"$\Delta$RMSE: baseline $-$ KronHiPPO (mph)")
    axis.set_title("Paired improvement over the same spatial split", loc="left", pad=8)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(0.7)
    axis.tick_params(width=0.7)
    axis.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    axis.text(
        0.01,
        0.02,
        "Positive values favour KronHiPPO-STGP.",
        transform=axis.transAxes,
        fontsize=7.5,
        color="#333333",
    )
    fig.tight_layout(pad=0.8)
    png = OUTPUT / "fig_pems_paired_delta_rmse.png"
    pdf = OUTPUT / "fig_pems_paired_delta_rmse.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "definition": "baseline RMSE (mph) minus KronHiPPO-STGP RMSE (mph)",
        "methods": list(methods),
        "seeds": list(seeds),
        "delta_by_method_seed": {
            method: {str(seed): float(deltas[index, seed_index]) for seed_index, seed in enumerate(seeds)}
            for index, method in enumerate(methods)
        },
        "mean_delta": {method: float(means[index]) for index, method in enumerate(methods)},
        "wins": {method: int(np.sum(deltas[index] > 0.0)) for index, method in enumerate(methods)},
    }


def make_prediction_trajectory_plot() -> dict[str, object]:
    methods = ("kronhippo_stgp", "kron_stgp", "ignnk")
    archives = {
        method: np.load(NOWCAST / method / "seed1" / "predictions.npz")
        for method in methods
    }
    stream = np.asarray(archives["kronhippo_stgp"]["stream_indices"], dtype=int)
    truth = np.asarray(archives["kronhippo_stgp"]["y"], dtype=float)
    for method in methods[1:]:
        np.testing.assert_array_equal(archives[method]["stream_indices"], stream)
        np.testing.assert_allclose(archives[method]["y"], truth, rtol=0.0, atol=1e-12)

    result_path = NOWCAST / "kronhippo_stgp" / "seed1" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    scale = float(result["target_standardisation"]["scale"])
    offset = float(result["target_standardisation"]["mean"])
    truth_mph = offset + scale * truth
    means_mph = {
        method: offset + scale * np.asarray(archives[method]["mean"], dtype=float)
        for method in methods
    }
    hippo_variance = np.asarray(archives["kronhippo_stgp"]["variance"], dtype=float)
    hippo_std_mph = scale * np.sqrt(np.maximum(hippo_variance, 1e-12))

    split_path = ROOT / result["split_manifest"]
    split = load_spatial_split(split_path)
    dataset = load_traffic_dataset(ROOT / "data/traffic/raw", "pems_bay", task1_steps=2016)
    heldout = np.asarray(split.heldout_indices, dtype=int)
    if heldout.size != truth.shape[1]:
        raise RuntimeError("Seed-1 split does not match the prediction archive")

    hippo_sensor_rmse = np.sqrt(np.mean((means_mph["kronhippo_stgp"] - truth_mph) ** 2, axis=0))
    order = np.argsort(hippo_sensor_rmse)
    quantiles = (0.15, 0.50, 0.85)
    selected_columns = np.asarray([order[int(round(q * (order.size - 1)))] for q in quantiles], dtype=int)
    difficulty = ("easy", "median", "difficult")

    window_steps = 48 * 12
    starts = np.arange(0, truth_mph.shape[0] - window_steps + 1, 24 * 12, dtype=int)
    scores = []
    for start in starts:
        window = truth_mph[start : start + window_steps]
        sensor_ranges = np.quantile(window, 0.95, axis=0) - np.quantile(window, 0.05, axis=0)
        scores.append(float(np.median(sensor_ranges)))
    window_start = int(starts[int(np.argmax(scores))])
    window_stop = window_start + window_steps
    timestamps = dataset.timestamps[stream[window_start:window_stop]]

    colors = {"kronhippo_stgp": "#0072B2", "kron_stgp": "#D55E00", "ignnk": "#009E73"}
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.5), sharex=True, sharey=True)
    for axis, label, column in zip(axes, difficulty, selected_columns):
        lower = means_mph["kronhippo_stgp"][window_start:window_stop, column] - 1.644853627 * hippo_std_mph[window_start:window_stop, column]
        upper = means_mph["kronhippo_stgp"][window_start:window_stop, column] + 1.644853627 * hippo_std_mph[window_start:window_stop, column]
        axis.fill_between(timestamps, lower, upper, color=colors["kronhippo_stgp"], alpha=0.13, linewidth=0.0, label="KronHiPPO 90% PI")
        axis.plot(timestamps, truth_mph[window_start:window_stop, column], color="black", linewidth=1.25, label="Ground truth", zorder=4)
        axis.plot(timestamps, means_mph["kronhippo_stgp"][window_start:window_stop, column], color=colors["kronhippo_stgp"], linewidth=1.0, label="KronHiPPO-STGP", zorder=3)
        axis.plot(timestamps, means_mph["kron_stgp"][window_start:window_stop, column], color=colors["kron_stgp"], linewidth=0.9, linestyle="--", label="Kron-STGP", zorder=2)
        axis.plot(timestamps, means_mph["ignnk"][window_start:window_stop, column], color=colors["ignnk"], linewidth=0.9, linestyle=":", label="IGNNK", zorder=2)
        sensor_index = int(heldout[column])
        axis.set_title(
            f"{label.capitalize()}: sensor {dataset.sensor_ids[sensor_index]} "
            f"(full-stream KronHiPPO RMSE {hippo_sensor_rmse[column]:.2f} mph)",
            loc="left",
            fontsize=8.5,
            pad=3,
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_linewidth(0.65)
    axes[1].set_ylabel("Traffic speed (mph)")
    axes[-1].set_xlabel("Time")
    handles, labels = axes[0].get_legend_handles_labels()
    order_legend = [1, 2, 3, 4, 0]
    fig.legend(
        [handles[index] for index in order_legend],
        [labels[index] for index in order_legend],
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        fontsize=7.5,
    )
    fig.suptitle("Representative held-out sensor trajectories (seed 1)", x=0.08, ha="left", y=1.035, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=0.75)
    png = OUTPUT / "fig_pems_seed1_prediction_trajectories.png"
    pdf = OUTPUT / "fig_pems_seed1_prediction_trajectories.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    selection = []
    for label, quantile, column in zip(difficulty, quantiles, selected_columns):
        sensor_index = int(heldout[column])
        selection.append(
            {
                "difficulty": label,
                "difficulty_quantile": quantile,
                "archive_column": int(column),
                "sensor_index": sensor_index,
                "sensor_id": dataset.sensor_ids[sensor_index],
                "kronhippo_full_stream_rmse_mph": float(hippo_sensor_rmse[column]),
            }
        )
    return {
        "seed": 1,
        "sensor_selection_rule": "nearest ranks to the 15th, 50th and 85th percentiles of full-stream KronHiPPO per-sensor RMSE; selection is for qualitative difficulty coverage, not comparative ranking",
        "window_selection_rule": "48-hour window, aligned to 24-hour boundaries, maximising the median held-out-sensor ground-truth 95th-to-5th percentile speed range; no model prediction enters this rule",
        "selected_sensors": selection,
        "window_start_stream_row": window_start,
        "window_stop_stream_row_exclusive": window_stop,
        "window_start_timestamp": timestamps[0].isoformat(),
        "window_end_timestamp": timestamps[-1].isoformat(),
        "source_archives": {
            method: sha256(NOWCAST / method / "seed1" / "predictions.npz") for method in methods
        },
    }


def make_tables(rows: list[dict[str, str]]) -> None:
    aggregate = {row["method"]: row for row in rows}
    markdown = [
        "# PEMS-BAY Strict-Online Spatial Nowcasting",
        "",
        "Mean +/- sample standard deviation over spatial splits 1--3. RMSE is reported on the original speed scale. Lower is better for CRPS, Gaussian NLPD and ECE; Coverage90 targets 0.90.",
        "",
        "| Block | Method | RMSE (mph) | CRPS | Gaussian NLPD | ECE | Coverage90 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    csv_rows: list[dict[str, str]] = []
    for block, methods in TABLE_BLOCKS:
        for method in methods:
            row = aggregate[method]
            cells = [format_value(row, metric) for metric in METRICS]
            if method == "kronhippo_stgp":
                cells = [f"**{cell}**" for cell in cells]
            markdown.append(f"| {block} | {LABELS[method]} | " + " | ".join(cells) + " |")
            csv_rows.append(
                {"block": block, "method": LABELS[method]}
                | {metric: format_value(row, metric) for metric in METRICS}
            )
    markdown.extend(
        [
            "",
            "The persistence row is a delayed-target control: Protocol N legally reveals each held-out target after prediction, so it can use y_H,t-1. It is not treated as a competing no-history spatial reconstruction method. IGNNK does not provide Gaussian predictive variances, so probabilistic metrics are not reported.",
            "",
            "The changing-versus-fixed comparison changes temporal coordinates as well as the transfer path. It is mechanism evidence for the changing-coordinate HiPPO representation, not an isolated causal estimate of transfer approximation benefit.",
        ]
    )
    (OUTPUT / "table_pems_nowcasting_3seed.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    with (OUTPUT / "table_pems_nowcasting_3seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("block", "method", *METRICS))
        writer.writeheader()
        writer.writerows(csv_rows)

    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{PEMS-BAY strict-online spatial nowcasting over spatial splits 1--3. RMSE is on the original speed scale. Lower is better for CRPS, Gaussian NLPD, and ECE; Coverage90 targets 0.90.}",
        r"\label{tab:pems_nowcasting}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & RMSE (mph) $\downarrow$ & CRPS $\downarrow$ & NLPD $\downarrow$ & ECE $\downarrow$ & Coverage90 \\",
        r"\midrule",
    ]
    for block_index, (block, methods) in enumerate(TABLE_BLOCKS):
        latex.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{block}}}}} \\")
        for method in methods:
            row = aggregate[method]
            cells = [format_value(row, metric, latex=True) for metric in METRICS]
            if method == "kronhippo_stgp":
                cells = [rf"\textbf{{{cell}}}" for cell in cells]
            latex.append(LABELS[method] + " & " + " & ".join(cells) + r" \\")
        if block_index < len(TABLE_BLOCKS) - 1:
            latex.append(r"\addlinespace[2pt]")
    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}\footnotesize",
            r"Persistence is a delayed-target control using the legally revealed $y_{H,t-1}$ and is not ranked against the spatial reconstruction methods. IGNNK does not output Gaussian predictive variances; its probabilistic cells are therefore omitted.",
            r"\end{minipage}",
            r"\end{table*}",
        ]
    )
    (OUTPUT / "table_pems_nowcasting_3seed.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    per_seed_rows = read_csv(PER_SEED)
    aggregate_rows = read_csv(AGGREGATE)
    if {int(row["seed"]) for row in per_seed_rows} != {1, 2, 3}:
        raise SystemExit("Expected exactly formal seeds 1, 2 and 3")
    if any(row.get("audit") != "passed" for row in per_seed_rows):
        raise SystemExit("At least one source archive failed its common audit")

    plot_audit = make_paired_plot(per_seed_rows)
    delta_plot_audit = make_paired_delta_plot(per_seed_rows)
    trajectory_plot_audit = make_prediction_trajectory_plot()
    make_tables(aggregate_rows)
    audit_path = OUTPUT / "PAPER_ARTIFACT_AUDIT.json"
    outputs = sorted(
        path for path in OUTPUT.iterdir() if path.is_file() and path != audit_path
    )
    audit = {
        "schema_version": 1,
        "formal_seeds": [1, 2, 3],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_files": {str(path.relative_to(ROOT)): sha256(path) for path in (PER_SEED, AGGREGATE)},
        "paired_plot": plot_audit,
        "paired_delta_plot": delta_plot_audit,
        "prediction_trajectory_plot": trajectory_plot_audit,
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
