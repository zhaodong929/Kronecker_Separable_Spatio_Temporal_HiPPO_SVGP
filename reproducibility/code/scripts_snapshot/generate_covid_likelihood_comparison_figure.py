#!/usr/bin/env python3
"""Render a Gaussian-versus-NB posterior comparison for long-stream COVID.

The figure uses the formal delayed-history nowcasting protocol (Setting B):
previous hidden labels are assimilated, current visible labels are observed,
and the current hidden labels are predicted.  Both likelihoods use the Route B
cumulative HiPPO representation; only the observation model differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.dates import DateFormatter, YearLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_covid_long_target_likelihood_crps import ensemble_crps, normal_crps


DEFAULT_PROTOCOL_ROOT = ROOT / "data/epidemiology/protocol/covid_long_target_ablation/log1p_per_100k"
DEFAULT_GAUSSIAN_ROOT = ROOT / "results/diagnostics/covid_long_target_likelihood_ablation/gaussian_log1p_per_100k"
DEFAULT_NB_CRPS_ROOT = ROOT / "results/diagnostics/covid_long_target_likelihood_ablation/negative_binomial_crps_s2048"
DEFAULT_NB_ECE_ROOT = ROOT / "results/diagnostics/covid_long_target_likelihood_ablation/negative_binomial_ece_s100"
DEFAULT_GAUSSIAN_ECE_ROOT = ROOT / "results/diagnostics/covid_long_target_likelihood_ablation/ece_formal_seeds5_9"
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/"
    "kronecker+s2vgp/ICLR Formal experiment/covid_delayed_history_baseline_paper_materials/"
    "long_stream_2020_2024_mandatory/likelihood_comparison_visuals"
)

FORMAL_SEEDS = (5, 6, 7, 8, 9)
REPRESENTATIVE_STATES = (
    ("New York", "Northeast"),
    ("Louisiana", "South"),
    ("Nevada", "West"),
)
NORMAL_90_QUANTILE = 1.6448536269514722
GAUSSIAN_COLOR = "#0072B2"
NB_COLOR = "#D55E00"
TRUTH_COLOR = "#1D1D1D"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_python(value: object) -> object:
    """Convert NumPy scalar/array values for the reproducibility JSON."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def empirical_coverage(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    if lower.shape != upper.shape or lower.ndim != 3:
        raise ValueError("Calibration intervals must have shape (levels, weeks, locations)")
    if lower.shape[1:] != truth.shape:
        raise ValueError("Calibration targets and intervals have incompatible shapes")
    return np.mean((truth[None, :, :] >= lower) & (truth[None, :, :] <= upper), axis=(1, 2))


def read_seed(
    *,
    seed: int,
    protocol_root: Path,
    gaussian_root: Path,
    nb_crps_root: Path,
) -> dict[str, object]:
    protocol_path = protocol_root / f"seed{seed}" / "protocol.npz"
    metadata_path = protocol_path.with_suffix(".json")
    gaussian_path = gaussian_root / f"seed{seed}" / "routeb_cumulative" / "online" / "predictions.npz"
    nb_path = nb_crps_root / f"seed{seed}" / "predictions.npz"
    for path in (protocol_path, metadata_path, gaussian_path, nb_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(protocol_path) as protocol:
        test_indices = np.asarray(protocol["test_indices"], dtype=np.int64)
        stream_dates = np.asarray(protocol["stream_week_dates"], dtype="datetime64[D]")
        exposure = np.asarray(protocol["population_per_100k"], dtype=np.float64)[test_indices]
        truth = np.log1p(np.asarray(protocol["stream_counts"], dtype=np.float64)[:, test_indices] / exposure[None, :])
    with np.load(gaussian_path) as gaussian:
        gaussian_truth = np.asarray(gaussian["y_true"], dtype=np.float64)
        mean = np.asarray(gaussian["pred_mean"], dtype=np.float64)
        variance = np.asarray(gaussian["pred_var"], dtype=np.float64)
        archive_test_indices = np.asarray(gaussian["test_indices"], dtype=np.int64)
        mt = int(np.asarray(gaussian["mt"]).item())
        ms = int(np.asarray(gaussian["ms"]).item())
        variance_mode = str(np.asarray(gaussian["variance_mode"]).item())
    if mt != 32 or ms != 32 or variance_mode != "full_joint_conditional":
        raise ValueError("The comparison requires the formal Mt=32, Ms=32 full-joint cumulative HiPPO archive")
    if not np.array_equal(test_indices, archive_test_indices):
        raise ValueError(f"Gaussian seed {seed} test indices differ from the protocol")
    standardization = metadata["target_standardization"]
    scale = float(standardization["scale"])
    offset = float(standardization["mean"])
    gaussian_mean = mean * scale + offset
    gaussian_variance = variance * scale**2
    np.testing.assert_allclose(gaussian_truth * scale + offset, truth, atol=1e-12, rtol=0.0)

    with np.load(nb_path) as nb:
        nb_truth = np.asarray(nb["y_true"], dtype=np.float64)
        nb_mean = np.asarray(nb["pred_mean"], dtype=np.float64)
        nb_variance = np.asarray(nb["pred_variance"], dtype=np.float64)
        samples = np.asarray(nb["common_predictive_samples"], dtype=np.float64)
        saved_samples = int(np.asarray(nb["predictive_sample_count"]).item())
    if samples.shape[0] != 2048 or saved_samples != 2048:
        raise ValueError(f"NB seed {seed} must contain the audited 2048-sample CRPS archive")
    if samples.shape[1:] != truth.shape:
        raise ValueError(f"NB seed {seed} sample shape {samples.shape} does not match target {truth.shape}")
    np.testing.assert_allclose(nb_truth, truth, atol=1e-12, rtol=0.0)
    if not np.isfinite(gaussian_mean).all() or not np.isfinite(gaussian_variance).all() or np.any(gaussian_variance <= 0.0):
        raise FloatingPointError(f"Gaussian seed {seed} has invalid predictive moments")
    if not np.isfinite(samples).all() or not np.isfinite(nb_mean).all() or not np.isfinite(nb_variance).all():
        raise FloatingPointError(f"NB seed {seed} has invalid posterior samples or moments")

    return {
        "seed": seed,
        "metadata": metadata,
        "protocol_path": protocol_path,
        "metadata_path": metadata_path,
        "gaussian_path": gaussian_path,
        "nb_path": nb_path,
        "test_indices": test_indices,
        "names": np.asarray(metadata["location_names"], dtype=str)[test_indices],
        "dates": stream_dates,
        "truth": truth,
        "gaussian_mean": gaussian_mean,
        "gaussian_variance": gaussian_variance,
        "nb_mean": nb_mean,
        "nb_variance": nb_variance,
        "nb_samples": samples,
    }


def read_calibration(
    *,
    seed_data: dict[str, object],
    gaussian_ece_root: Path,
    nb_ece_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = int(seed_data["seed"])
    gaussian_path = gaussian_ece_root / f"gaussian_empirical_intervals_seed{seed}.npz"
    nb_path = nb_ece_root / f"seed{seed}" / "predictions.npz"
    for path in (gaussian_path, nb_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    truth = np.asarray(seed_data["truth"], dtype=np.float64)
    with np.load(gaussian_path) as gaussian:
        gaussian_truth = np.asarray(gaussian["y_true"], dtype=np.float64)
        levels = np.asarray(gaussian["ece_coverage_levels"], dtype=np.float64)
        gaussian_coverage = empirical_coverage(
            truth,
            np.asarray(gaussian["pred_interval_lower"], dtype=np.float64),
            np.asarray(gaussian["pred_interval_upper"], dtype=np.float64),
        )
    with np.load(nb_path) as nb:
        nb_truth = np.asarray(nb["y_true"], dtype=np.float64)
        nb_levels = np.asarray(nb["ece_coverage_levels"], dtype=np.float64)
        nb_coverage = empirical_coverage(
            truth,
            np.asarray(nb["pred_interval_lower"], dtype=np.float64),
            np.asarray(nb["pred_interval_upper"], dtype=np.float64),
        )
    np.testing.assert_allclose(gaussian_truth, truth, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(nb_truth, truth, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(nb_levels, levels, atol=0.0, rtol=0.0)
    return levels, gaussian_coverage, nb_coverage


def ece(levels: np.ndarray, observed: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(observed, dtype=np.float64) - np.asarray(levels, dtype=np.float64))))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.1,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.7,
            "axes.titleweight": "bold",
            "legend.fontsize": 7.1,
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def draw_trajectories(
    axes: list[plt.Axes],
    seed_data: dict[str, object],
) -> list[dict[str, object]]:
    names = np.asarray(seed_data["names"], dtype=str)
    dates = np.asarray(seed_data["dates"]).astype("datetime64[D]").astype(object)
    truth = np.asarray(seed_data["truth"], dtype=np.float64)
    gaussian_mean = np.asarray(seed_data["gaussian_mean"], dtype=np.float64)
    gaussian_std = np.sqrt(np.asarray(seed_data["gaussian_variance"], dtype=np.float64))
    nb_mean = np.asarray(seed_data["nb_mean"], dtype=np.float64)
    nb_samples = np.asarray(seed_data["nb_samples"], dtype=np.float64)
    nb_lower, nb_upper = np.quantile(nb_samples, (0.05, 0.95), axis=0)
    selected: list[dict[str, object]] = []
    for panel_index, (axis, (state, region)) in enumerate(zip(axes, REPRESENTATIVE_STATES)):
        matching = np.flatnonzero(names == state)
        if matching.size != 1:
            raise ValueError(f"Pre-specified representative state {state!r} is not uniquely held out in seed 5")
        column = int(matching[0])
        gaussian_lower = gaussian_mean[:, column] - NORMAL_90_QUANTILE * gaussian_std[:, column]
        gaussian_upper = gaussian_mean[:, column] + NORMAL_90_QUANTILE * gaussian_std[:, column]
        axis.fill_between(dates, gaussian_lower, gaussian_upper, color=GAUSSIAN_COLOR, alpha=0.17, linewidth=0.0, zorder=1)
        axis.fill_between(dates, nb_lower[:, column], nb_upper[:, column], color=NB_COLOR, alpha=0.16, linewidth=0.0, zorder=1)
        axis.plot(dates, truth[:, column], color=TRUTH_COLOR, linewidth=1.2, label="Observed", zorder=4)
        axis.plot(dates, gaussian_mean[:, column], color=GAUSSIAN_COLOR, linewidth=1.05, label="Gaussian mean", zorder=3)
        axis.plot(dates, nb_mean[:, column], color=NB_COLOR, linewidth=1.05, label="NB mean", zorder=3)
        axis.set_title(f"{state} ({region})", pad=3)
        axis.xaxis.set_major_locator(YearLocator())
        axis.xaxis.set_major_formatter(DateFormatter("%Y"))
        axis.tick_params(axis="x", labelsize=7.1, pad=1.5)
        axis.tick_params(axis="y", labelsize=7.1, pad=1.5)
        axis.set_xlim(dates[0], dates[-1])
        axis.set_xlabel("Week")
        if panel_index == 0:
            axis.set_ylabel(r"$z=\log(1 + \mathrm{admissions}/100\mathrm{k})$")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.65)
        selected.append(
            {
                "state": state,
                "region": region,
                "selection_rule": "pre-specified geographic representation; not selected by error",
                "seed5_hidden_column": column,
            }
        )
    return selected


def draw_calibration(
    axis: plt.Axes,
    levels: np.ndarray,
    gaussian_coverage: np.ndarray,
    nb_coverage: np.ndarray,
) -> dict[str, object]:
    gaussian_mean = gaussian_coverage.mean(axis=0)
    gaussian_sd = gaussian_coverage.std(axis=0, ddof=1)
    nb_mean = nb_coverage.mean(axis=0)
    nb_sd = nb_coverage.std(axis=0, ddof=1)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=0.9, label="Ideal")
    axis.fill_between(levels, np.clip(gaussian_mean - gaussian_sd, 0.0, 1.0), np.clip(gaussian_mean + gaussian_sd, 0.0, 1.0), color=GAUSSIAN_COLOR, alpha=0.11, linewidth=0.0)
    axis.fill_between(levels, np.clip(nb_mean - nb_sd, 0.0, 1.0), np.clip(nb_mean + nb_sd, 0.0, 1.0), color=NB_COLOR, alpha=0.11, linewidth=0.0)
    axis.plot(levels, gaussian_mean, marker="o", markersize=3.2, color=GAUSSIAN_COLOR, linewidth=1.35, label="Gaussian")
    axis.plot(levels, nb_mean, marker="s", markersize=3.1, color=NB_COLOR, linewidth=1.35, label="Negative Binomial")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks(np.arange(0.0, 1.01, 0.2))
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.set_xlabel("Nominal central coverage")
    axis.set_ylabel("Empirical coverage")
    axis.grid(color="#D9D9D9", linewidth=0.55, alpha=0.65)
    axis.legend(loc="lower right", handlelength=1.6)
    gaussian_ece = np.asarray([ece(levels, row) for row in gaussian_coverage], dtype=np.float64)
    nb_ece = np.asarray([ece(levels, row) for row in nb_coverage], dtype=np.float64)
    annotation = (
        f"ECE (seeds 5-9)\n"
        f"Gaussian: {gaussian_ece.mean():.3f} +/- {gaussian_ece.std(ddof=1):.3f}\n"
        f"NB: {nb_ece.mean():.3f} +/- {nb_ece.std(ddof=1):.3f}"
    )
    axis.text(0.04, 0.96, annotation, transform=axis.transAxes, ha="left", va="top", fontsize=6.7, color="#262626")
    return {
        "levels": levels,
        "gaussian_coverage_mean": gaussian_mean,
        "gaussian_coverage_sd": gaussian_sd,
        "negative_binomial_coverage_mean": nb_mean,
        "negative_binomial_coverage_sd": nb_sd,
        "gaussian_ece_per_seed": gaussian_ece,
        "negative_binomial_ece_per_seed": nb_ece,
    }


def draw_crps_heatmap(axis: plt.Axes, seed_data: dict[str, object]) -> dict[str, object]:
    truth = np.asarray(seed_data["truth"], dtype=np.float64)
    gaussian = normal_crps(
        truth,
        np.asarray(seed_data["gaussian_mean"], dtype=np.float64),
        np.asarray(seed_data["gaussian_variance"], dtype=np.float64),
    )
    nb = ensemble_crps(truth, np.asarray(seed_data["nb_samples"], dtype=np.float64))
    difference = nb - gaussian
    limit = float(np.quantile(np.abs(difference), 0.99))
    if not np.isfinite(limit) or limit <= 0.0:
        raise FloatingPointError("The paired CRPS heatmap requires non-zero finite differences")
    cmap = LinearSegmentedColormap.from_list("nb_gaussian", [NB_COLOR, "#F8F7F3", GAUSSIAN_COLOR])
    image = axis.imshow(
        difference.T,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    dates = np.asarray(seed_data["dates"], dtype="datetime64[D]")
    ticks = [0]
    for year in (2022, 2023, 2024):
        candidates = np.flatnonzero(dates >= np.datetime64(f"{year}-01-01"))
        if candidates.size:
            ticks.append(int(candidates[0]))
            axis.axvline(float(candidates[0]) - 0.5, color="#262626", linewidth=0.35, alpha=0.5)
    axis.set_xticks(ticks, [str(dates[index])[:4] for index in ticks])
    axis.set_yticks(np.arange(truth.shape[1]), np.asarray(seed_data["names"], dtype=str), fontsize=6.9)
    axis.tick_params(axis="x", labelsize=7.0, pad=1.5)
    axis.tick_params(axis="y", length=0, pad=1.5)
    axis.set_xlabel("Week")
    axis.set_ylabel("Held-out jurisdiction")
    colorbar = plt.colorbar(image, ax=axis, pad=0.012, fraction=0.047, extend="both")
    colorbar.set_label(r"$\Delta$CRPS = NB - Gaussian ($<0$: NB better)", fontsize=7.0, labelpad=2)
    colorbar.ax.tick_params(labelsize=6.5, length=2)
    return {
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "fraction_nb_better": float(np.mean(difference < 0.0)),
        "color_limit_abs_99th_percentile": limit,
        "gaussian_crps_mean": float(gaussian.mean()),
        "negative_binomial_crps_mean": float(nb.mean()),
    }


def generate_figure(
    *,
    protocol_root: Path,
    gaussian_root: Path,
    nb_crps_root: Path,
    gaussian_ece_root: Path,
    nb_ece_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    all_seed_data = [
        read_seed(
            seed=seed,
            protocol_root=protocol_root,
            gaussian_root=gaussian_root,
            nb_crps_root=nb_crps_root,
        )
        for seed in FORMAL_SEEDS
    ]
    seed5 = all_seed_data[0]
    if int(seed5["seed"]) != 5:
        raise AssertionError("The representative trajectories and heatmap must use formal seed 5")

    calibration_rows = [
        read_calibration(seed_data=seed_data, gaussian_ece_root=gaussian_ece_root, nb_ece_root=nb_ece_root)
        for seed_data in all_seed_data
    ]
    levels = calibration_rows[0][0]
    if any(not np.array_equal(row[0], levels) for row in calibration_rows[1:]):
        raise ValueError("Formal seed calibration levels are inconsistent")
    gaussian_coverage = np.stack([row[1] for row in calibration_rows])
    nb_coverage = np.stack([row[2] for row in calibration_rows])

    style()
    figure = plt.figure(figsize=(12.6, 6.55), constrained_layout=False)
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.04, 1.0),
        left=0.06,
        right=0.985,
        top=0.85,
        bottom=0.105,
        hspace=0.40,
    )
    top = outer[0].subgridspec(1, 3, wspace=0.26)
    bottom = outer[1].subgridspec(1, 2, width_ratios=(0.78, 1.35), wspace=0.28)
    trajectory_axes = [figure.add_subplot(top[0, index]) for index in range(3)]
    calibration_axis = figure.add_subplot(bottom[0, 0])
    heatmap_axis = figure.add_subplot(bottom[0, 1])

    representative = draw_trajectories(trajectory_axes, seed5)
    handles, labels = trajectory_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.51, 0.985), ncol=3, frameon=False, handlelength=2.0, columnspacing=1.15)
    figure.text(0.06, 0.916, "(a) Representative posterior forecasts: formal split seed 5", fontsize=9.5, fontweight="bold", ha="left")
    figure.text(0.06, 0.892, "Shaded bands are central 90% posterior intervals; jurisdictions were pre-specified by geographic region, not prediction error.", fontsize=6.7, ha="left", color="#303030")

    calibration = draw_calibration(calibration_axis, levels, gaussian_coverage, nb_coverage)
    calibration_axis.set_title("(b) Interval calibration across formal splits", loc="left", pad=5.0)
    heatmap = draw_crps_heatmap(heatmap_axis, seed5)
    heatmap_axis.set_title("(c) Paired CRPS differences by week and jurisdiction (seed 5)", loc="left", pad=5.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig_covid_gaussian_nb_likelihood_comparison"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(figure)

    input_paths: list[Path] = []
    for seed_data in all_seed_data:
        input_paths.extend(
            [
                Path(seed_data["protocol_path"]),
                Path(seed_data["metadata_path"]),
                Path(seed_data["gaussian_path"]),
                Path(seed_data["nb_path"]),
                gaussian_ece_root / f"gaussian_empirical_intervals_seed{int(seed_data['seed'])}.npz",
                nb_ece_root / f"seed{int(seed_data['seed'])}" / "predictions.npz",
            ]
        )
    audit = {
        "status": "complete",
        "figure": "Gaussian vs Negative-Binomial posterior forecasts",
        "setting": "B: delayed hidden-label assimilation plus current 42-state visible update, then 10-state hidden prediction",
        "representation": "Route B cumulative HiPPO, Mt=32, Ms=32, full_joint_conditional variance",
        "common_plotting_scale": "Z = log1p(weekly admissions per 100,000)",
        "formal_seeds": list(FORMAL_SEEDS),
        "representative_trajectories": representative,
        "calibration_protocol": "K=10 central coverages 0.05, 0.15, ..., 0.95; empirical intervals from the audited S=100 artifacts",
        "calibration": {key: as_python(value) for key, value in calibration.items()},
        "crps_heatmap": heatmap,
        "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in input_paths],
        "outputs": [str(stem.with_suffix(".pdf")), str(stem.with_suffix(".png"))],
    }
    stem.with_suffix(".config.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Gaussian vs Negative-Binomial Posterior Forecasts",
        "",
        "This figure compares only Route B cumulative HiPPO under the formal long-stream Setting B: the previous week's hidden labels are assimilated exactly once, then the 42 currently visible jurisdictions update the posterior before forecasting the 10 hidden jurisdictions.",
        "",
        "All curves and intervals are plotted on the common Z = log1p(weekly admissions per 100,000) scale. The Gaussian 90% interval is analytic; the NB interval is the 5th--95th percentile of 2,048 saved transformed posterior-predictive samples.",
        "",
        "Panel (b) uses the existing formal ECE protocol (seeds 5--9, S=100 empirical samples, central coverages 5%, 15%, ..., 95%). The displayed bands are plus/minus one standard deviation across spatial splits, not IID uncertainty over jurisdiction-week pairs.",
        "",
        "Panel (c) uses one coherent fixed split (seed 5). It displays Delta CRPS = CRPS_NB - CRPS_Gaussian for each hidden-jurisdiction/week pair. Negative values favour NB; the colour scale is centred at zero and is clipped only beyond the 99th percentile of absolute differences.",
        "",
        "The figure does not compare Gaussian and NB NLPD: their native predictive densities are defined on different observation spaces. CRPS supplies the common-scale probabilistic comparison.",
        "",
        f"Mean seed-5 paired CRPS difference: {heatmap['mean_difference']:.5f} (NB minus Gaussian); NB is lower on {heatmap['fraction_nb_better']:.1%} of held-out jurisdiction/week pairs.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--gaussian-root", type=Path, default=DEFAULT_GAUSSIAN_ROOT)
    parser.add_argument("--nb-crps-root", type=Path, default=DEFAULT_NB_CRPS_ROOT)
    parser.add_argument("--gaussian-ece-root", type=Path, default=DEFAULT_GAUSSIAN_ECE_ROOT)
    parser.add_argument("--nb-ece-root", type=Path, default=DEFAULT_NB_ECE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    audit = generate_figure(
        protocol_root=args.protocol_root,
        gaussian_root=args.gaussian_root,
        nb_crps_root=args.nb_crps_root,
        gaussian_ece_root=args.gaussian_ece_root,
        nb_ece_root=args.nb_ece_root,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": audit["status"], "outputs": audit["outputs"], "crps_heatmap": audit["crps_heatmap"]}, indent=2))


if __name__ == "__main__":
    main()
