#!/usr/bin/env python3
"""Generate the three-panel main-text mechanism figure from archived records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter


PAPER = Path(__file__).resolve().parents[1]
RESULTS_CANDIDATES = (
    Path(r"\\wsl.localhost\Ubuntu-24.04\home\zd929\projects\stvgp_kronecker\results"),
    Path("/home/zd929/projects/stvgp_kronecker/results"),
)
for _candidate in RESULTS_CANDIDATES:
    if (_candidate / "aistats2027_controlled_controls_20260916T163843Z").is_dir():
        RESULTS = _candidate
        break
else:
    raise FileNotFoundError("Archived result tree not found")

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#666666"


def read_json_rmse(path: Path) -> float:
    data = json.loads(path.read_text())
    if "aggregate" in data and "rmse" in data["aggregate"]:
        return float(data["aggregate"]["rmse"])
    return float(data["result"]["final"]["rmse"])


def paired_ratios() -> tuple[list[str], list[np.ndarray]]:
    labels = ["ERA5 mechanism", "ERA5 benchmark", "PEMS-BAY"]
    ratios: list[np.ndarray] = []

    mech = RESULTS / "iclr2027_long_mechanism_20260914/formal"
    joint_paths = sorted(mech.glob("structured_changing_ms128_mt128_seed*/result.json"))
    zero_paths = sorted(mech.glob("zero_cross_changing_ms128_mt128_seed*/result.json"))
    if len(joint_paths) != 5 or len(zero_paths) != 5:
        raise ValueError("ERA5 mechanism records must contain five matched seeds")
    ratios.append(np.array([read_json_rmse(z) / read_json_rmse(j) for j, z in zip(joint_paths, zero_paths)]))

    bench = RESULTS / "aistats2027_controlled_controls_20260916T163843Z/aggregate/experiment_a_per_split.csv"
    rows = list(csv.DictReader(bench.open(newline="")))
    joint = {int(r["seed"]): float(r["rmse"]) for r in rows if r["variant"] == "full_joint_changing"}
    zero = {int(r["seed"]): float(r["rmse"]) for r in rows if r["variant"] == "zero_cross"}
    if set(joint) != set(range(5)) or set(zero) != set(range(5)):
        raise ValueError("ERA5 benchmark records must contain five matched seeds")
    ratios.append(np.array([zero[s] / joint[s] for s in range(5)]))

    traffic_joint = RESULTS / "traffic/formal_locked_sm_q2_road_context_v1/pems_bay/nowcast/kronhippo_stgp"
    traffic_zero = RESULTS / "pems_matched_mechanism_20260914/pems_bay/nowcast/zero_cross_changing"
    joint_paths = sorted(traffic_joint.glob("seed*/result.json"))
    zero_paths = sorted(traffic_zero.glob("seed*/result.json"))
    if len(joint_paths) != 3 or len(zero_paths) != 3:
        raise ValueError("PEMS-BAY records must contain three matched seeds")
    ratios.append(np.array([read_json_rmse(z) / read_json_rmse(j) for j, z in zip(joint_paths, zero_paths)]))
    return labels, ratios


def experiment_b_audit() -> pd.DataFrame:
    source = RESULTS / "aistats2027_controlled_controls_20260916T163843Z/aggregate/experiment_b_blockwise_kl.csv"
    frame = pd.read_csv(source, float_precision="round_trip")
    frame = frame[(frame["mt"] == 128) & (frame["block_id"] != 0)].copy()
    if set(frame["variant"]) != {"conditional_transport", "identity_reuse"}:
        raise ValueError("Experiment-B blockwise file is missing a transfer variant")
    grouped = (frame.groupby(["block_id", "variant"])["reference_predictive_gaussian_kl"]
               .agg(mean="mean", sample_sd=lambda x: x.std(ddof=1), splits="count")
               .reset_index())
    if set(grouped["block_id"]) != set(range(1, 19)) or not (grouped["splits"] == 5).all():
        raise ValueError("Experiment-B must contain five splits for blocks 1--18")
    return grouped


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.45, alpha=0.25)
    ax.grid(axis="y", which="minor", visible=False)
    ax.tick_params(direction="out", length=2.4, width=0.55, pad=2)


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.0, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    labels, ratios = paired_ratios()
    transfer = experiment_b_audit()
    solver = pd.read_csv(PAPER / "structured_dense_solver_benchmark.csv").sort_values("Ms")
    if solver["Ms"].astype(int).tolist() != [8, 16, 32, 48, 64]:
        raise ValueError("Unexpected solver sweep")

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), constrained_layout=False,
                             gridspec_kw={"wspace": 0.34})

    # (a) paired split-level RMSE ratios; uncertainty is computed from ratios.
    ax = axes[0]
    x = np.arange(3)
    means = np.array([v.mean() for v in ratios])
    sds = np.array([v.std(ddof=1) for v in ratios])
    rng = np.random.default_rng(0)
    for i, values in enumerate(ratios):
        jitter = rng.uniform(-0.055, 0.055, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=10, color=ORANGE,
                   alpha=0.35, linewidths=0, zorder=2)
    ax.errorbar(x, means, yerr=sds, fmt="o", color=ORANGE, ecolor=ORANGE,
                elinewidth=1.0, capsize=2.2, markersize=4.2, markeredgewidth=0.6,
                markerfacecolor="white", zorder=3)
    ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Relative RMSE")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title("(a) Revision-aware coupling", loc="left", fontsize=8.0, pad=3)
    style_axis(ax)

    # (b) Experiment-B blockwise KL.
    ax = axes[1]
    for variant, color, ls in (("conditional_transport", BLUE, "-"), ("identity_reuse", ORANGE, "--")):
        part = transfer[transfer["variant"] == variant].sort_values("block_id")
        xx = part["block_id"].to_numpy()
        yy = part["mean"].to_numpy()
        sd = part["sample_sd"].to_numpy()
        ax.plot(xx, yy, color=color, linestyle=ls, linewidth=1.2,
                marker="o" if variant == "conditional_transport" else "s", markevery=4,
                markersize=3.0, markerfacecolor="white", markeredgewidth=0.6,
                label="Conditional transport" if variant == "conditional_transport" else "Identity reuse")
        ax.fill_between(xx, yy - sd, yy + sd, color=color, alpha=0.12, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlim(1, 18)
    ax.set_xticks([1, 6, 12, 18])
    ax.set_xlabel("Online block")
    ax.set_ylabel("Predictive KL")
    ax.set_title("(b) Coordinate-consistent transfer", loc="left", fontsize=8.0, pad=3)
    ax.legend(loc="center right", frameon=False, handlelength=1.7, borderpad=0.2,
              labelspacing=0.25, handletextpad=0.4)
    style_axis(ax)

    # (c) Full archived solver sweep.
    ax = axes[2]
    m = solver["Ms"].to_numpy()
    ax.plot(m, solver["dense_time_s"], color=ORANGE, linewidth=1.2, marker="o", markersize=3.0, label="Dense")
    ax.plot(m, solver["sylvester_time_s"], color=BLUE, linewidth=1.2, marker="s", markersize=3.0, label="Schur--Sylvester")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_s=M_t$")
    ax.set_ylabel("Solve time (s)")
    ax.set_xticks([8, 16, 32, 48, 64])
    ax.set_title("(c) Structured recovery", loc="left", fontsize=8.0, pad=3)
    ax.legend(loc="upper left", frameon=False, handlelength=1.7, borderpad=0.2,
              labelspacing=0.25, handletextpad=0.4)
    ax.annotate(r"$1036.7\times$", xy=(64, float(solver.iloc[-1]["sylvester_time_s"])),
                xytext=(-3, 18), textcoords="offset points", ha="right", fontsize=7.4, color=BLUE)
    style_axis(ax)

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.23, top=0.84)
    out_pdf = PAPER / "figures/figure3_main_mechanism.pdf"
    out_png = PAPER / "figures/figure3_main_mechanism.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print("Figure 3 numerical audit (paired split-level ratios):")
    for label, values, mean, sd in zip(labels, ratios, means, sds):
        print(f"{label:18s} values={np.array2string(values, precision=6)} mean={mean:.8f} sample_sd={sd:.8f}")
    print("Experiment-B Mt=128 block range: 1--18; five splits per variant")
    print(f"Figure 3 PDF: {out_pdf}")
    print(f"Figure 3 PNG: {out_png}")


if __name__ == "__main__":
    main()
