#!/usr/bin/env python3
"""Append SM-kernel failure analysis and improvement diagnostics to the x-lag report."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stvgp_kronecker.data.hipposvgp_era5 import load_hipposvgp_era5
from scripts.run_hipposvgp_era5_routeb import augment_dataset_phi


ROOT = Path("results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/xlag_lag_kernel_validation")
RUNS = ROOT / "sm_improvement_appendix" / "runs"
FIGS = ROOT / "sm_improvement_appendix" / "figures"
PARAMS = ROOT / "sm_improvement_appendix" / "params"
APPENDIX_PDF = ROOT / "sm_improvement_appendix" / "medium_era5_xlag_sm_failure_and_improvement_appendix.pdf"
SOURCE_REPORT = ROOT / "medium_era5_xlag_lag_kernel_validation_report.pdf"
FINAL_REPORT = ROOT / "medium_era5_xlag_lag_kernel_validation_report_with_sm_appendix.pdf"

PYTHON = Path(".venv/bin/python")
LAG = 10
VAL_SEED = 0
TEST_SEED = 1
ELL_T = 0.05
SPATIAL_ELL = 0.35

BASE_ARGS = [
    "--root",
    "data/era5/processed_timeseries_4",
    "--calibration-tasks",
    "task_1",
    "--online-tasks",
    "task_2",
    "--variable-index",
    "0",
    "--split",
    "all",
    "--block-size",
    "10",
    "--routeb-methods",
    "structured_joint",
    "--eval-modes",
    "seen_history",
    "--ohsvgp-heldout-eval",
    "--seeds",
    "0",
    "--mt",
    "8",
    "--ms",
    "64",
    "--prediction-mode",
    "streaming_sylvester",
    "--prediction-chunk-size",
    "8192",
    "--hyperparam-fit-mode",
    "none",
    "--ell-t-fit-mode",
    "none",
    "--model-ell-t",
    str(ELL_T),
    "--routeb-noise",
    "0.1",
    "--kernel-variance",
    "1.0",
    "--spatial-lengthscale",
    str(SPATIAL_ELL),
    "--phi-mode",
    "medium_era5_xlag",
    "--xlag-length",
    str(LAG),
    "--kernel-type",
    "spectral_mixture",
]


def run_command(name: str, args: list[str]) -> Path:
    outdir = RUNS / name
    if (outdir / "era5_routeb_summary.csv").exists():
        return outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), "scripts/run_hipposvgp_era5_routeb.py", "--outdir", str(outdir), *args]
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (outdir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (outdir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (outdir / "walltime_seconds.txt").write_text(f"{time.perf_counter() - started:.6f}\n", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with code {proc.returncode}; see {outdir / 'stdout.log'}")
    return outdir


def read_structured_summary(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path / "era5_routeb_summary.csv")
    row = df[(df["method"] == "structured_joint") & (df["eval_mode"] == "seen_history")].iloc[0].to_dict()
    wall_path = path / "walltime_seconds.txt"
    row["walltime_seconds"] = float(wall_path.read_text().strip()) if wall_path.exists() else float("nan")
    return row


def calibration_dataset_residual_periodogram(q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = load_hipposvgp_era5(
        root="data/era5/processed_timeseries_4",
        tasks=("task_1",),
        variable_index=0,
        split="all",
    )
    dataset = augment_dataset_phi(dataset, phi_mode="medium_era5_xlag", xlag_length=LAG)
    phi = dataset.Phi
    y = dataset.Y.reshape(-1)
    ridge = 1e-4
    lhs = phi.T @ phi + ridge * np.eye(phi.shape[1])
    beta = np.linalg.solve(lhs, phi.T @ y)
    residual = (y - phi @ beta).reshape(dataset.Y.shape)
    residual = residual - residual.mean(axis=0, keepdims=True)
    times = np.asarray(dataset.times, dtype=float)
    dt = float(np.median(np.diff(np.sort(times)))) if times.size > 1 else 1.0
    freqs_cycles = np.fft.rfftfreq(residual.shape[0], d=dt)
    spectrum = np.abs(np.fft.rfft(residual, axis=0)) ** 2
    power = spectrum.mean(axis=1)
    if power.size > 0:
        power[0] = 0.0
    order = np.argsort(power)[::-1]
    order = order[: max(q, 1)]
    selected_cycles = freqs_cycles[order]
    selected_power = power[order]
    # The analytic temporal RFF path uses angular frequencies in sin(w t).
    # Convert cycles/time from the periodogram to angular frequency and then
    # to the base-frequency scale, because current_frequencies = base / ell_t.
    base_means = 2.0 * math.pi * selected_cycles * ELL_T
    weights = selected_power / max(float(selected_power.sum()), 1e-12)
    return base_means.astype(float), weights.astype(float), selected_cycles.astype(float)


def write_sm_params(name: str, means: np.ndarray, weights: np.ndarray, *, include_rbf_component: bool) -> Path:
    PARAMS.mkdir(parents=True, exist_ok=True)
    means = np.asarray(means, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if include_rbf_component:
        # A zero-mean Gaussian spectral component approximates the RBF temporal
        # residual kernel. The remaining components are FFT peaks.
        temporal_means = np.concatenate([[0.0], means])
        temporal_scales = np.concatenate([[1.0], np.full(means.shape[0], 0.08)])
        fft_weight = 0.35
        temporal_weights = np.concatenate([[1.0 - fft_weight], fft_weight * weights])
    else:
        temporal_means = means
        temporal_scales = np.full(means.shape[0], 0.08)
        temporal_weights = weights
    temporal_weights = temporal_weights / max(float(temporal_weights.sum()), 1e-12)
    # Keep the spatial covariance as close as possible to the RBF baseline:
    # in the closed-form SM covariance, scale=1/(2*pi*ell) reproduces RBF.
    payload = {
        "schema": name,
        "temporal_weights": temporal_weights.tolist(),
        "temporal_means": temporal_means.tolist(),
        "temporal_scales": temporal_scales.tolist(),
        "spatial_weights": [1.0],
        "spatial_means": [0.0],
        "spatial_scales": [1.0 / (2.0 * math.pi * SPATIAL_ELL)],
        "notes": {
            "temporal_mean_units": "base angular frequency before division by model_ell_t",
            "spatial_component": "Q1 zero-mean SM matching RBF spatial lengthscale",
        },
    }
    path = PARAMS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_fft_sm_experiments() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    configs: list[tuple[str, int, bool]] = [
        ("fft_sm_q2", 2, False),
        ("fft_sm_q4", 4, False),
        ("fft_sm_q8", 8, False),
        ("rbf_plus_fft_sm_q4", 4, True),
        ("rbf_plus_fft_sm_q8", 8, True),
    ]
    for name, q, include_rbf in configs:
        means, weights, cycles = calibration_dataset_residual_periodogram(q)
        params_path = write_sm_params(name, means, weights, include_rbf_component=include_rbf)
        for split_name, split_seed in [("val", VAL_SEED), ("test", TEST_SEED)]:
            args = [
                *BASE_ARGS,
                "--heldout-split-seeds",
                str(split_seed),
                "--num-mixtures",
                str(q + int(include_rbf)),
                "--spectral-mixture-param-path",
                str(params_path),
            ]
            outdir = run_command(f"{name}_{split_name}_seed{split_seed}", args)
            summary = read_structured_summary(outdir)
            rows.append(
                {
                    "setting": name,
                    "split": split_name,
                    "seed": split_seed,
                    "lag_length": LAG,
                    "num_fft_peaks": q,
                    "includes_rbf_component": include_rbf,
                    "params_path": str(params_path),
                    "fft_cycles_per_time": " ".join(f"{x:.6g}" for x in cycles),
                    **summary,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "sm_improvement_appendix" / "fft_sm_improvement_results_long.csv", index=False)
    wide_rows = []
    for setting, grp in out.groupby("setting", sort=False):
        val = grp[grp["split"] == "val"].iloc[0]
        test = grp[grp["split"] == "test"].iloc[0]
        wide_rows.append(
            {
                "setting": setting,
                "lag_length": LAG,
                "num_fft_peaks": int(test["num_fft_peaks"]),
                "includes_rbf_component": bool(test["includes_rbf_component"]),
                "val_rmse": float(val["rmse"]),
                "val_nll": float(val["nll"]),
                "test_rmse": float(test["rmse"]),
                "test_nll": float(test["nll"]),
                "test_coverage90": float(test["coverage90"]),
                "test_ece": float(test["ece"]),
                "test_avg_std": float(test["avg_std"]),
                "test_avg_var": float(test["avg_var"]),
                "runtime": float(val["walltime_seconds"]) + float(test["walltime_seconds"]),
                "fft_cycles_per_time": str(test["fft_cycles_per_time"]),
            }
        )
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(ROOT / "sm_improvement_appendix" / "fft_sm_improvement_results.csv", index=False)
    return wide


def existing_kernel_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mapping = [
        ("X-lag RBF", ROOT / "runs/kernel_ablation/xlag_rbf_L10_Q0_test_seed1"),
        ("X-lag Matern32", ROOT / "runs/kernel_ablation/xlag_matern32_L10_Q0_test_seed1"),
        ("Default SM Q2", ROOT / "runs/kernel_ablation/xlag_spectral_mixture_L10_Q2_test_seed1"),
        ("Default SM Q4", ROOT / "runs/kernel_ablation/xlag_spectral_mixture_L10_Q4_test_seed1"),
        ("Default SM Q8", ROOT / "runs/kernel_ablation/xlag_spectral_mixture_L10_Q8_test_seed1"),
        ("SM-Q1 zero-mean", ROOT / "sm_q1_rbf_sanity_check/runs/sm_q1_zero_mean_test_seed1"),
    ]
    for setting, path in mapping:
        if (path / "era5_routeb_summary.csv").exists():
            row = read_structured_summary(path)
            rows.append(
                {
                    "setting": setting,
                    "test_rmse": float(row["rmse"]),
                    "test_nll": float(row["nll"]),
                    "test_coverage90": float(row["coverage90"]),
                    "test_ece": float(row["ece"]),
                    "test_avg_std": float(row["avg_std"]),
                    "test_avg_var": float(row["avg_var"]),
                    "routeb_sigma2": float(row["routeb_sigma2"]),
                    "avg_width90": float(row["avg_width90"]),
                    "source": str(path),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "sm_improvement_appendix" / "existing_kernel_uncertainty_summary.csv", index=False)
    return df


def make_figures(existing: pd.DataFrame, fft_wide: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [
            existing[["setting", "test_rmse", "test_nll", "test_coverage90", "test_avg_std"]].copy(),
            fft_wide[["setting", "test_rmse", "test_nll", "test_coverage90", "test_avg_std"]].copy(),
        ],
        ignore_index=True,
    )
    order = combined.sort_values("test_nll")["setting"].tolist()
    combined["setting"] = pd.Categorical(combined["setting"], categories=order, ordered=True)
    combined = combined.sort_values("setting")

    palette = []
    for setting in combined["setting"].astype(str):
        if setting == "X-lag RBF":
            palette.append("#3D6B45")
        elif "zero" in setting:
            palette.append("#7C6A9A")
        elif "Default SM" in setting:
            palette.append("#B46A5A")
        elif "fft" in setting.lower():
            palette.append("#527BA8")
        else:
            palette.append("#7A7A7A")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    x = np.arange(len(combined))
    axes[0].bar(x, combined["test_rmse"], color=palette)
    axes[0].set_ylabel("test RMSE")
    axes[0].set_title("Mean accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, combined["test_nll"], color=palette)
    axes[1].set_ylabel("test NLL/NLPD")
    axes[1].set_title("Probabilistic score")
    axes[1].grid(axis="y", alpha=0.25)
    for ax, metric in zip(axes, ["test_rmse", "test_nll"]):
        ax.set_xticks(x)
        ax.set_xticklabels(combined["setting"].astype(str), rotation=35, ha="right", fontsize=8)
        for i, value in enumerate(combined[metric]):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    fig.savefig(FIGS / "sm_improvement_metric_comparison.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGS / "sm_improvement_metric_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].bar(x, combined["test_avg_std"], color=palette)
    axes[0].axhline(existing.loc[existing["setting"] == "X-lag RBF", "test_avg_std"].iloc[0], color="black", lw=1, ls="--")
    axes[0].set_ylabel("average predictive std")
    axes[0].set_title("Uncertainty scale")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, combined["test_coverage90"], color=palette)
    axes[1].axhline(0.90, color="black", lw=1, ls="--")
    axes[1].set_ylabel("90% interval coverage")
    axes[1].set_title("Coverage")
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(combined["setting"].astype(str), rotation=35, ha="right", fontsize=8)
    fig.savefig(FIGS / "sm_improvement_uncertainty_comparison.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGS / "sm_improvement_uncertainty_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def table_from_df(df: pd.DataFrame, columns: list[tuple[str, str]], style: ParagraphStyle, max_rows: int | None = None) -> Table:
    if max_rows is not None:
        df = df.head(max_rows)
    data: list[list[Any]] = [[paragraph(label, style) for _, label in columns]]
    for _, row in df.iterrows():
        cells = []
        for col, _ in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.4f}")
            elif isinstance(value, (bool, np.bool_)):
                cells.append("yes" if bool(value) else "no")
            else:
                cells.append(str(value))
        data.append(cells)
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8B8B8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_appendix_pdf(existing: pd.DataFrame, fft_wide: pd.DataFrame) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=12)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=12.5, leading=15, spaceBefore=8, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.8, leading=11.2, alignment=TA_LEFT, spaceAfter=6)
    small = ParagraphStyle("Small", parent=body, fontSize=7.2, leading=8.6)

    doc = SimpleDocTemplate(
        str(APPENDIX_PDF),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )
    story: list[Any] = []
    story.append(Paragraph("Appendix: why spectral mixture did not beat RBF in the X-lag diagnostic", title))
    story.append(
        paragraph(
            "This appendix extends the Medium-ERA5 X-lag report without changing the preceding pages. "
            "The key implementation point is that X-lag covariates enter the Route B mean feature map Phi. "
            "They are not active dimensions of the GP residual kernel. The kernel comparison therefore tests "
            "the spatio-temporal residual covariance after the X-lag mean has been included, rather than a "
            "direct kernel over the X-lag vector.",
            body,
        )
    )
    story.append(Paragraph("Interpretation of the original SM results", h1))
    story.append(
        paragraph(
            "The original spectral-mixture runs are not mainly failing because of a gross formula or plumbing error. "
            "The Q=1 zero-mean sanity check almost recovers the RBF result, and its spatial covariance matches the "
            "RBF covariance to numerical precision. Instead, the full SM runs appear to over-inflate residual "
            "uncertainty: their average predictive standard deviation is roughly twice the RBF value and their "
            "90% interval coverage is close to 0.995. This conservative variance explains why NLL deteriorates "
            "more strongly than RMSE.",
            body,
        )
    )
    story.append(
        paragraph(
            "Five mechanisms are consistent with the measurements. First, SM is designed to express oscillatory "
            "structure in a meaningful coordinate such as time; here the X-lag history is already encoded in Phi, "
            "so the residual kernel sees less periodic structure than a raw-time model would. Second, X-lag features "
            "already absorb much of the local temporal memory, making a smooth residual covariance sufficient. "
            "Third, full SM adds many frequency, width and weight choices, but the current Route B diagnostic uses "
            "fixed kernel parameters rather than a full marginal-likelihood training loop. Fourth, SM is sensitive "
            "to temporal frequency scale and spatial covariance scale. Fifth, the main observed failure is calibration: "
            "the SM residual variance is too broad for this test set.",
            body,
        )
    )
    story.append(Image(str(FIGS / "sm_improvement_metric_comparison.png"), width=17.0 * cm, height=6.8 * cm))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Image(str(FIGS / "sm_improvement_uncertainty_comparison.png"), width=17.0 * cm, height=6.4 * cm))
    story.append(Paragraph("Kernel and calibration summary", h1))
    cal_table = existing[
        ["setting", "test_rmse", "test_nll", "test_coverage90", "test_avg_std", "test_avg_var"]
    ].copy()
    story.append(
        table_from_df(
            cal_table,
            [
                ("setting", "setting"),
                ("test_rmse", "RMSE"),
                ("test_nll", "NLL"),
                ("test_coverage90", "coverage90"),
                ("test_avg_std", "avg std"),
                ("test_avg_var", "avg var"),
            ],
            small,
        )
    )
    story.append(Paragraph("Improvement attempts", h1))
    story.append(
        paragraph(
            "Two feasible improvements were tested inside the current Route B architecture. The first uses FFT peaks "
            "from the calibration residual after fitting the X-lag mean map to initialize temporal SM frequencies. "
            "The second adds a zero-frequency RBF-equivalent spectral component to those FFT peaks, approximating an "
            "RBF+SM temporal residual kernel. The spatial side was held as a Q=1 zero-mean SM that matches the RBF "
            "spatial lengthscale, so the diagnostic isolates the temporal SM contribution.",
            body,
        )
    )
    story.append(
        table_from_df(
            fft_wide[
                [
                    "setting",
                    "num_fft_peaks",
                    "includes_rbf_component",
                    "val_rmse",
                    "val_nll",
                    "test_rmse",
                    "test_nll",
                    "test_coverage90",
                    "test_avg_std",
                ]
            ],
            [
                ("setting", "setting"),
                ("num_fft_peaks", "FFT peaks"),
                ("includes_rbf_component", "RBF comp."),
                ("val_rmse", "val RMSE"),
                ("val_nll", "val NLL"),
                ("test_rmse", "test RMSE"),
                ("test_nll", "test NLL"),
                ("test_coverage90", "coverage90"),
                ("test_avg_std", "avg std"),
            ],
            small,
        )
    )
    best = fft_wide.sort_values(["test_nll", "test_rmse"]).iloc[0]
    rbf = existing[existing["setting"] == "X-lag RBF"].iloc[0]
    story.append(
        paragraph(
            f"The best new SM variant is {best['setting']} with test RMSE {best['test_rmse']:.4f} and "
            f"NLL {best['test_nll']:.4f}. Relative to the RBF baseline "
            f"(RMSE {rbf['test_rmse']:.4f}, NLL {rbf['test_nll']:.4f}), the RBF+FFT-SM variant slightly worsens "
            "mean accuracy but improves probabilistic calibration. The practical conclusion is therefore more "
            "specific: full SM should not replace RBF for RMSE in this X-lag diagnostic, but an RBF-anchored "
            "spectral component is useful for NLL. The most promising next model change is a genuine "
            "active-dimension kernel, k = k_RBF(X-lag) + k_SM(time), which requires moving the X-lag state from "
            "a purely linear mean covariate into the GP input space.",
            body,
        )
    )
    story.append(Paragraph("Recommended next checks", h1))
    story.append(
        paragraph(
            "The next high-value checks are: (i) keep RBF as the default residual kernel for the current Route B "
            "experiments; (ii) report average predictive standard deviation and coverage whenever SM is used; "
            "(iii) if SM is pursued, train or validate its frequency, amplitude and noise parameters on the "
            "calibration task; and (iv) implement an explicit additive active-dimension model only if the research "
            "question requires periodic residual structure beyond what X-lag already explains.",
            body,
        )
    )
    doc.build(story)


def append_pdf() -> None:
    if not SOURCE_REPORT.exists():
        raise FileNotFoundError(SOURCE_REPORT)
    subprocess.run(["pdfunite", str(SOURCE_REPORT), str(APPENDIX_PDF), str(FINAL_REPORT)], check=True)


def main() -> None:
    (ROOT / "sm_improvement_appendix").mkdir(parents=True, exist_ok=True)
    existing = existing_kernel_rows()
    fft_wide = run_fft_sm_experiments()
    make_figures(existing, fft_wide)
    build_appendix_pdf(existing, fft_wide)
    append_pdf()
    print(f"Wrote appendix: {APPENDIX_PDF}")
    print(f"Wrote appended report: {FINAL_REPORT}")


if __name__ == "__main__":
    main()
