#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/traffic/formal_deterministic_v1/pems_bay"
OUT = BASE / "paper_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = {
    "Last-value persistence": BASE / "main/nowcast/persistence",
    "Kron-STGP": BASE / "main/nowcast/kron_stgp",
    "KronHiPPO-STGP": BASE / "main/nowcast/kronhippo_stgp",
}
COLORS = {
    "Last-value persistence": "#666666",
    "Kron-STGP": "#56B4E9",
    "KronHiPPO-STGP": "#D55E00",
}
LEVELS = np.arange(0.1, 1.0, 0.1)
Z = np.array([0.12566135, 0.25334710, 0.38532047, 0.52440051, 0.67448975, 0.84162123, 1.03643339, 1.28155157, 1.64485363])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def metrics(y, mean, variance):
    variance = np.maximum(variance, 1e-10)
    error = y - mean
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "gaussian_nlpd": float(np.mean(0.5 * (np.log(2*np.pi*variance) + error**2 / variance))),
        "coverage90": float(np.mean(np.abs(error) <= 1.644853627 * np.sqrt(variance))),
    }

def load_archive(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}

# P5: equal-count long-stream bins and successive predictive KL.
stability_rows = []
bins = 20
for method, root in METHODS.items():
    for seed in range(5):
        a = load_archive(root / f"seed{seed}/predictions.npz")
        edges = np.linspace(0, len(a["stream_indices"]), bins + 1, dtype=int)
        var = np.maximum(a["variance"], 1e-10)
        kl = np.zeros_like(var)
        kl[1:] = 0.5 * (
            np.log(var[:-1] / var[1:])
            + (var[1:] + (a["mean"][1:] - a["mean"][:-1])**2) / var[:-1]
            - 1.0
        )
        for b in range(bins):
            lo, hi = edges[b], edges[b+1]
            row = {
                "method": method,
                "seed": seed,
                "bin": b + 1,
                "origin_start": int(a["stream_indices"][lo]),
                "origin_end": int(a["stream_indices"][hi-1]),
                "stream_fraction": (lo + hi) / (2 * len(a["stream_indices"])),
                **metrics(a["y"][lo:hi], a["mean"][lo:hi], var[lo:hi]),
                "successive_predictive_kl": float(np.mean(kl[max(lo,1):hi])),
            }
            stability_rows.append(row)
stability = pd.DataFrame(stability_rows)
stability.to_csv(OUT / "long_stream_stability.csv", index=False)

# P6: persistent state and online latency at growing history checkpoints.
scaling_rows = []
checkpoints = [1000, 5000, 10000, 20000, 30000, 40000, 50100]
for method, root in METHODS.items():
    for seed in range(5):
        frame = pd.read_csv(root / f"seed{seed}/per_step_metrics.csv")
        for end in checkpoints:
            subset = frame.iloc[:min(end, len(frame))]
            scaling_rows.append({
                "method": method,
                "seed": seed,
                "history_steps": len(subset),
                "persistent_state_bytes": int(subset["persistent_state_bytes"].iloc[-1]),
                "median_update_prediction_ms": float(1000 * (subset["update_seconds"] + subset["prediction_seconds"]).median()),
                "cumulative_rmse": float(np.sqrt(np.mean(subset["rmse"]**2))),
            })
scaling = pd.DataFrame(scaling_rows)
scaling.to_csv(OUT / "bounded_state_scaling.csv", index=False)

# Main stability figure.
summary = stability.groupby(["method", "bin"], as_index=False).agg(
    stream_fraction=("stream_fraction", "mean"),
    rmse=("rmse", "mean"),
    nlpd=("gaussian_nlpd", "mean"),
    coverage=("coverage90", "mean"),
    kl=("successive_predictive_kl", "mean"),
)
fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.7), constrained_layout=True)
for method in METHODS:
    d = summary[summary.method == method]
    axes[0,0].plot(d.stream_fraction, d.rmse, color=COLORS[method], lw=1.5, label=method)
    axes[0,1].plot(d.stream_fraction, d.nlpd, color=COLORS[method], lw=1.5)
    axes[1,0].plot(d.stream_fraction, d.coverage, color=COLORS[method], lw=1.5)
    if not method.startswith(chr(76)+chr(97)+chr(115)+chr(116)+chr(45)+chr(118)+chr(97)+chr(108)+chr(117)+chr(101)):
        axes[1,1].plot(d.stream_fraction, d.kl, color=COLORS[method], lw=1.5)
axes[0,0].set_ylabel("RMSE")
axes[0,1].set_ylabel("Gaussian NLPD")
axes[1,0].set_ylabel("Coverage90")
axes[1,1].set_ylabel("Successive predictive KL")
axes[1,0].axhline(0.9, color="#999999", ls="--", lw=0.8)
for ax in axes.flat:
    ax.set_xlabel("Fraction of strict-online stream")
    ax.grid(axis="y", color="#E5E5E5", lw=0.5)
axes[0,0].legend(frameon=False, ncol=1)
for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes.flat):
    ax.set_title(label, loc="left", fontweight="bold")
fig.savefig(OUT / "fig_pems_long_stream_stability.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_pems_long_stream_stability.png", dpi=350, bbox_inches="tight")
plt.close(fig)

# Bounded-state evidence.
scale_summary = scaling.groupby(["method", "history_steps"], as_index=False).agg(
    state_bytes=("persistent_state_bytes", "mean"),
    latency_ms=("median_update_prediction_ms", "mean"),
)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)
for method in METHODS:
    d = scale_summary[scale_summary.method == method]
    axes[0].plot(d.history_steps, d.state_bytes / 1024, color=COLORS[method], marker="o", ms=2.8, lw=1.4, label=method)
    axes[1].plot(d.history_steps, d.latency_ms, color=COLORS[method], marker="o", ms=2.8, lw=1.4)
axes[0].set_ylabel("Persistent state (KiB)")
axes[1].set_ylabel("Median update + prediction (ms)")
for ax in axes:
    ax.set_xlabel("Observed stream length (5-min steps)")
    ax.grid(axis="y", color="#E5E5E5", lw=0.5)
axes[0].legend(frameon=False)
axes[0].set_title("(a) Bounded state", loc="left", fontweight="bold")
axes[1].set_title("(b) Online latency", loc="left", fontweight="bold")
fig.savefig(OUT / "fig_pems_bounded_state_scaling.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_pems_bounded_state_scaling.png", dpi=350, bbox_inches="tight")
plt.close(fig)

# P4 future forecasting.
future = pd.read_csv(BASE / "forecast_summary/forecast_summary.csv")
fig, ax = plt.subplots(figsize=(3.55, 2.65), constrained_layout=True)
for method in METHODS:
    d = future[future.method == method].sort_values("horizon_steps")
    if d.empty:
        continue
    ax.errorbar(d.horizon_steps * 5, d.rmse_mean, yerr=d.rmse_sample_sd, color=COLORS[method], marker="o", ms=3.5, lw=1.5, capsize=2, label=method)
ax.set_xlabel("Forecast horizon (min)")
ax.set_ylabel("RMSE")
ax.set_xticks([15, 30, 60])
ax.grid(axis="y", color="#E5E5E5", lw=0.5)
ax.legend(frameon=False)
fig.savefig(OUT / "fig_pems_future_forecasting.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_pems_future_forecasting.png", dpi=350, bbox_inches="tight")
plt.close(fig)

# Error heatmaps, seed 0, time-binned to remain legible.
hippo = load_archive(METHODS["KronHiPPO-STGP"] / "seed0/predictions.npz")
ordinary = load_archive(METHODS["Kron-STGP"] / "seed0/predictions.npz")
block = 100
n = len(hippo["stream_indices"]) // block
err_h = np.abs(hippo["y"][:n*block] - hippo["mean"][:n*block]).reshape(n, block, -1).mean(1).T
err_o = np.abs(ordinary["y"][:n*block] - ordinary["mean"][:n*block]).reshape(n, block, -1).mean(1).T
delta = err_o - err_h
vmax = np.quantile(np.concatenate([err_o.ravel(), err_h.ravel()]), 0.99)
dmax = np.quantile(np.abs(delta), 0.99)
fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.15), constrained_layout=True)
im0 = axes[0].imshow(err_o, aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
axes[1].imshow(err_h, aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
im2 = axes[2].imshow(delta, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax), interpolation="nearest")
axes[0].set_title("(a) Kron-STGP", loc="left", fontweight="bold")
axes[1].set_title("(b) KronHiPPO-STGP", loc="left", fontweight="bold")
axes[2].set_title("(c) Error reduction", loc="left", fontweight="bold")
for ax in axes:
    ax.set_xlabel("Strict-online time")
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].set_ylabel("Held-out sensors")
fig.colorbar(im0, ax=axes[:2], shrink=0.78, label="Mean absolute error")
fig.colorbar(im2, ax=axes[2], shrink=0.78, label="Ordinary - HiPPO")
fig.savefig(OUT / "fig_pems_error_heatmaps.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_pems_error_heatmaps.png", dpi=350, bbox_inches="tight")
plt.close(fig)

# Calibration curve on all seed-0 held-out predictions.
fig, ax = plt.subplots(figsize=(3.3, 3.0), constrained_layout=True)
for method, root in METHODS.items():
    a = load_archive(root / "seed0/predictions.npz")
    residual = np.abs(a["y"] - a["mean"])
    std = np.sqrt(np.maximum(a["variance"], 1e-10))
    empirical = [float(np.mean(residual <= z * std)) for z in Z]
    ax.plot(LEVELS, empirical, color=COLORS[method], marker="o", ms=2.8, lw=1.4, label=method)
ax.plot([0,1], [0,1], color="#999999", ls="--", lw=0.9, label="Ideal")
ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(0,1), ylim=(0,1))
ax.set_aspect("equal")
ax.grid(color="#E5E5E5", lw=0.5)
ax.legend(frameon=False, loc="upper left")
fig.savefig(OUT / "fig_pems_calibration.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_pems_calibration.png", dpi=350, bbox_inches="tight")
plt.close(fig)

# Representative trajectories: first four held-out sensor indices by manifest order rule.
split = json.loads((ROOT / "results/traffic/protocols/pems_bay/pems_bay_seed0_spatial_split.json").read_text())
heldout = np.array(split["split"]["heldout_indices"], dtype=int)
chosen_pos = np.argsort(heldout)[:4]
window = slice(0, 576)
persistence = load_archive(METHODS["Last-value persistence"] / "seed0/predictions.npz")
fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.5), sharex=True, constrained_layout=True)
x = np.arange(576) * 5 / 60
for ax, pos in zip(axes.flat, chosen_pos):
    y = hippo["y"][window, pos]
    mu_h = hippo["mean"][window, pos]
    sd_h = np.sqrt(np.maximum(hippo["variance"][window, pos], 1e-10))
    ax.fill_between(x, mu_h - 1.644853627*sd_h, mu_h + 1.644853627*sd_h, color=COLORS["KronHiPPO-STGP"], alpha=0.16, linewidth=0)
    ax.plot(x, y, color="#111111", lw=1.0, label="Observed")
    ax.plot(x, persistence["mean"][window, pos], color=COLORS["Last-value persistence"], lw=0.9, label="Persistence")
    ax.plot(x, ordinary["mean"][window, pos], color=COLORS["Kron-STGP"], lw=0.9, label="Kron-STGP")
    ax.plot(x, mu_h, color=COLORS["KronHiPPO-STGP"], lw=1.1, label="KronHiPPO-STGP")
    ax.set_title(f"Sensor {heldout[pos]}")
    ax.grid(axis="y", color="#E5E5E5", lw=0.5)
for ax in axes[:,0]: ax.set_ylabel("Standardised speed")
for ax in axes[-1,:]: ax.set_xlabel("Hours from stream start")
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04))
fig.savefig(OUT / "fig_pems_representative_trajectories.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_pems_representative_trajectories.png", dpi=350, bbox_inches="tight")
plt.close(fig)

print(json.dumps({"output": str(OUT), "files": sorted(p.name for p in OUT.iterdir())}, indent=2))