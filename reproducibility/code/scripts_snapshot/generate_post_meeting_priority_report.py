#!/usr/bin/env python3
"""Generate the post-meeting P0-P4 LaTeX report from aggregated outputs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def esc(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def pm(mean: Any, sd: Any, digits: int = 4) -> str:
    return f"{float(mean):.{digits}f} $\\pm$ {float(sd):.{digits}f}"


def value(frame: pd.DataFrame, filters: dict[str, Any], column: str) -> float:
    subset = frame
    for key, target in filters.items():
        subset = subset.loc[subset[key] == target]
    if len(subset) != 1:
        raise ValueError(f"Expected one row for {filters}, got {len(subset)}")
    return float(subset.iloc[0][column])


def p0_full_rows(root: Path) -> str:
    p0 = root / "p0b_official_full_era5"
    summary = pd.read_csv(p0 / "summary.csv")
    all_runs = pd.read_csv(p0 / "all_runs.csv")
    model_order = {"st_vgp": 0, "st_svgp": 1, "mf_st_svgp": 2}
    keys = sorted(
        {(str(row.model), int(row.num_spatial_inducing)) for _, row in all_runs.iterrows()},
        key=lambda item: (model_order.get(item[0], 99), item[1]),
    )
    labels = {
        "st_vgp": "Official ST-VGP",
        "st_svgp": "Official ST-SVGP",
        "mf_st_svgp": "Official MF-ST-SVGP",
    }
    rows = []
    for model, ms in keys:
        group = all_runs.loc[
            (all_runs["model"] == model)
            & (pd.to_numeric(all_runs["num_spatial_inducing"], errors="coerce") == ms)
        ]
        completed = group.loc[group["status"] == "completed"]
        spatial = "1000 full" if model == "st_vgp" else f"{ms} ind."
        if not completed.empty:
            aggregate = summary.loc[
                (summary["model"] == model)
                & (pd.to_numeric(summary["num_spatial_inducing"], errors="coerce") == ms)
            ].iloc[0]
            rows.append(
                f"{labels[model]} & {spatial} & {int(aggregate.num_completed_seeds)} & "
                f"{pm(aggregate.rmse_mean, aggregate.rmse_sd)} & {pm(aggregate.nll_mean, aggregate.nll_sd)} & "
                f"{pm(aggregate.coverage90_mean, aggregate.coverage90_sd, 3)} & "
                f"{pm(aggregate.iterations_mean, aggregate.iterations_sd, 1)} & "
                f"{pm(aggregate.wall_seconds_mean / 60.0, aggregate.wall_seconds_sd / 60.0, 1)} & "
                f"{pm(aggregate.peak_rss_mb_mean / 1024.0, aggregate.peak_rss_mb_sd / 1024.0, 1)} \\\\"
            )
        else:
            attempted = group.loc[~group["status"].astype(str).str.startswith("skipped")]
            status = ", ".join(sorted(set(attempted["status"].astype(str)))) or "not run"
            peak = pd.to_numeric(attempted.get("peak_rss_mb"), errors="coerce").max()
            peak_text = f"{peak / 1024.0:.1f}" if math.isfinite(float(peak)) else "--"
            rows.append(
                f"{labels[model]} & {spatial} & 0 & \\multicolumn{{4}}{{c}}{{{esc(status)}}} & -- & {peak_text} \\\\"
            )
    return "\n".join(rows)


def p0_full_interpretation(root: Path) -> str:
    p0 = root / "p0b_official_full_era5"
    summary = pd.read_csv(p0 / "summary.csv")
    all_runs = pd.read_csv(p0 / "all_runs.csv")
    sparse = summary.loc[summary["model"] == "st_svgp"]
    if sparse.empty:
        sparse_text = "No full-protocol ST-SVGP configuration completed within the declared resource limit."
    else:
        best = sparse.loc[sparse["rmse_mean"].idxmin()]
        sparse_text = (
            f"The strongest completed official ST-SVGP configuration used $M_s={int(best.num_spatial_inducing)}$ "
            f"and reached RMSE {best.rmse_mean:.4f} $\\pm$ {best.rmse_sd:.4f} and NLL "
            f"{best.nll_mean:.4f} $\\pm$ {best.nll_sd:.4f}."
        )
    full = all_runs.loc[all_runs["model"] == "st_vgp"]
    full_completed = full.loc[full["status"] == "completed"]
    if full_completed.empty:
        if (full["status"].astype(str) == "resource_exhausted").any():
            full_text = (
                "Full ST-VGP failed during the first full-scale split with an XLA resource-exhausted allocation error; "
                "this is reported as a scalability outcome, not replaced by a subset metric."
            )
        else:
            full_text = (
                "Full ST-VGP did not complete at 1000 spatial states under the declared wall-time/memory envelope; "
                "this is reported as a scalability outcome, not replaced by a subset metric."
            )
    else:
        full_text = "Full ST-VGP completed on all three P1-aligned held-out splits."
    return sparse_text + " " + full_text


def p1_rows(summary: pd.DataFrame) -> str:
    labels = {
        "matched_sparse_stvgp": "Matched sparse Kronecker",
        "structured_joint_hippo_stvgp": "Structured-joint HiPPO-STVGP",
    }
    records = []
    subset = summary.loc[summary["block_size"] == 10].sort_values(["mt", "ms", "architecture", "protocol"])
    for _, row in subset.iterrows():
        records.append(
            f"({int(row.mt)},{int(row.ms)}) & {labels[row.architecture]} & {row.protocol.capitalize()} & "
            f"{pm(row.rmse_mean, row.rmse_sd)} & {pm(row.nll_mean, row.nll_sd)} & "
            f"{pm(row.coverage90_mean, row.coverage90_sd, 3)} & {pm(row.block_incremental_runtime_sec_mean, row.block_incremental_runtime_sec_sd, 2)} & "
            f"{pm(row.peak_rss_mb_mean, row.peak_rss_mb_sd, 1)} \\\\"
        )
    return "\n".join(records)


def block_rows(summary: pd.DataFrame) -> str:
    subset = summary.loc[
        (summary["architecture"] == "structured_joint_hippo_stvgp")
        & (summary["protocol"] == "online")
    ].sort_values(["mt", "ms", "block_size"])
    return "\n".join(
        f"({int(row.mt)},{int(row.ms)}) & {int(row.block_size)} & {pm(row.rmse_mean, row.rmse_sd)} & {pm(row.nll_mean, row.nll_sd)} & {pm(row.block_incremental_runtime_sec_mean, row.block_incremental_runtime_sec_sd, 2)} \\\\"
        for _, row in subset.iterrows()
    )


def block_average_rows(summary: pd.DataFrame) -> str:
    labels = {
        "matched_sparse_stvgp": "Matched sparse Kronecker",
        "structured_joint_hippo_stvgp": "Structured-joint HiPPO-STVGP",
    }
    subset = summary.loc[summary["block_size"] == 10].sort_values(
        ["mt", "ms", "architecture", "protocol"]
    )
    return "\n".join(
        f"({int(row.mt)},{int(row.ms)}) & {labels[row.architecture]} & {row.protocol.capitalize()} & "
        f"{pm(row.rmse_mean, row.rmse_sd)} & {pm(row.nll_mean, row.nll_sd)} & "
        f"{pm(row.coverage90_mean, row.coverage90_sd, 3)} & {pm(row.block_incremental_runtime_sec_mean, row.block_incremental_runtime_sec_sd, 2)} \\\\"
        for _, row in subset.iterrows()
    )


def paired_rows(frame: pd.DataFrame) -> str:
    subset = frame.loc[(frame["block_size"] == 10) & frame["metric"].isin(["rmse", "nll"])].sort_values(["comparison", "mt", "metric"])
    rows = []
    for _, row in subset.iterrows():
        rows.append(
            f"{esc(row.comparison)} & ({int(row.mt)},{int(row.ms)}) & {row.metric.upper()} & "
            f"{float(row.mean_paired_difference):.4f} & [{float(row.ci95_low):.4f}, {float(row.ci95_high):.4f}] & {float(row.paired_t_pvalue):.3f} \\\\"
        )
    return "\n".join(rows)


def p2_rows(summary: pd.DataFrame) -> str:
    order = [
        "Original STVGP direct target",
        "X-lag mean only",
        "STVGP residual + X-lag two-stage",
        "Joint exact STVGP beta-GP",
        "Matched sparse residual (Mt=32, Ms=128)",
        "Online structured-joint HiPPO-STVGP",
    ]
    summary = summary.set_index("method")
    rows = []
    for method in order:
        row = summary.loc[method]
        rows.append(
            f"{esc(method)} & {pm(row.rmse_mean, row.rmse_sd)} & {pm(row.nll_mean, row.nll_sd)} & "
            f"{pm(row.coverage90_mean, row.coverage90_sd, 3)} & {pm(row.avg_std_mean, row.avg_std_sd, 3)} & {pm(row.runtime_sec_mean, row.runtime_sec_sd, 2)} \\\\"
        )
    return "\n".join(rows)


def p3_rows(summary: pd.DataFrame) -> str:
    labels = {
        "analytic_hippo_rff": "Analytic HiPPO--RFF",
        "inducing_points": "Temporal inducing points",
        "ordinary_rff": "Ordinary RFF",
        "full_temporal_kernel": "Full temporal kernel upper bound",
    }
    records = []
    for _, row in summary.sort_values(["mt", "ms", "temporal_representation"]).iterrows():
        records.append(
            f"({int(row.mt)},{int(row.ms)}) & {labels[row.temporal_representation]} & "
            f"{pm(row.rmse_mean, row.rmse_sd)} & {pm(row.nll_mean, row.nll_sd)} & "
            f"{pm(row.coverage90_mean, row.coverage90_sd, 3)} & {pm(row.block_incremental_runtime_sec_mean, row.block_incremental_runtime_sec_sd, 2)} & {pm(row.peak_rss_mb_mean, row.peak_rss_mb_sd, 1)} \\\\"
        )
    return "\n".join(records)


def build_interpretation(p1: pd.DataFrame, p2: pd.DataFrame, p3: pd.DataFrame) -> dict[str, str]:
    cap = {"mt": 32, "ms": 128, "block_size": 10}
    online = value(p1, {**cap, "architecture": "structured_joint_hippo_stvgp", "protocol": "online"}, "rmse_mean")
    batch = value(p1, {**cap, "architecture": "structured_joint_hippo_stvgp", "protocol": "batch"}, "rmse_mean")
    matched_online = value(p1, {**cap, "architecture": "matched_sparse_stvgp", "protocol": "online"}, "rmse_mean")
    matched_batch = value(p1, {**cap, "architecture": "matched_sparse_stvgp", "protocol": "batch"}, "rmse_mean")
    low_cap = {"mt": 8, "ms": 64, "block_size": 10}
    low_online = value(p1, {**low_cap, "architecture": "structured_joint_hippo_stvgp", "protocol": "online"}, "rmse_mean")
    low_batch = value(p1, {**low_cap, "architecture": "structured_joint_hippo_stvgp", "protocol": "batch"}, "rmse_mean")
    low_matched = value(p1, {**low_cap, "architecture": "matched_sparse_stvgp", "protocol": "online"}, "rmse_mean")
    p2i = p2.set_index("method")
    two_stage = float(p2i.loc["STVGP residual + X-lag two-stage", "rmse_mean"])
    joint = float(p2i.loc["Joint exact STVGP beta-GP", "rmse_mean"])
    direct = float(p2i.loc["Original STVGP direct target", "rmse_mean"])
    mean_only = float(p2i.loc["X-lag mean only", "rmse_mean"])
    online_matern = float(p2i.loc["Online structured-joint HiPPO-STVGP", "rmse_mean"])
    online_matern_sd = float(p2i.loc["Online structured-joint HiPPO-STVGP", "rmse_sd"])
    p3_high = p3.loc[(p3.mt == 32) & (p3.ms == 128)].set_index("temporal_representation")
    hippo = float(p3_high.loc["analytic_hippo_rff", "rmse_mean"])
    inducing = float(p3_high.loc["inducing_points", "rmse_mean"])
    ordinary = float(p3_high.loc["ordinary_rff", "rmse_mean"])
    full = float(p3_high.loc["full_temporal_kernel", "rmse_mean"])
    return {
        "p1": (
            f"At $(M_t,M_s)=(32,128)$ and block size 10, the structured-joint final-block RMSE was {online:.4f} online and {batch:.4f} batch "
            f"(relative gap {100.0 * (online - batch) / batch:+.2f}\\%). At $(8,64)$, the corresponding online--batch gap was {low_online - low_batch:+.4f}, while structured online improved on matched online by {low_online - low_matched:+.4f}. "
            f"At $(32,128)$, matched sparse gave {matched_online:.4f} online and {matched_batch:.4f} batch, essentially tying structured joint. Thus joint coupling is most useful under the tighter budget, while the transfer loss shrinks markedly as capacity increases. "
            "Because matched batch and online use identical fixed-basis sufficient statistics, their agreement is an implementation invariant; the non-zero structured-joint gap measures changing HiPPO-basis transfer rather than data reuse."
        ),
        "p2": (
            f"Under the fixed Mat\'ern decomposition protocol, direct-target STVGP reached RMSE {direct:.4f}, two-stage residual STVGP reached {two_stage:.4f}, and joint exact $\\beta$--GP inference reached {joint:.4f}. "
            f"X-lag mean alone was much weaker ({mean_only:.4f}), so the exact residual GP, rather than the covariate mean by itself, supplies most of the predictive accuracy. The joint model further changed RMSE by {joint - two_stage:+.4f} relative to two-stage exact inference. "
            f"In contrast, the Mat\'ern analytic-HiPPO online row was unstable across splits ({online_matern:.4f} $\\pm$ {online_matern_sd:.4f}). This is evidence against treating the current Mat\'ern HiPPO transfer implementation as a mature baseline; it does not invalidate the exact joint coupling result."
        ),
        "p3": (
            f"At $(32,128)$, standard temporal inducing points were the strongest budget-matched representation (RMSE {inducing:.4f}), followed by analytic HiPPO--RFF ({hippo:.4f}) and ordinary RFF ({ordinary:.4f}). The full temporal kernel reached {full:.4f}. "
            "The upper-bound row has effective $M_t=N_t$ and is not budget matched. Together with the P1 capacity trend, these results identify temporal compression and changing-basis transfer as the main remaining bottlenecks, not the structured joint posterior itself."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    agg = root / "aggregated"
    figures = root / "figures"
    p1 = pd.read_csv(agg / "p1_final_block_summary.csv")
    p1_block = pd.read_csv(agg / "p1_block_average_summary.csv")
    paired = pd.read_csv(agg / "p1_paired_seed_comparisons.csv")
    p2 = pd.read_csv(agg / "p2_summary.csv")
    p3 = pd.read_csv(agg / "p3_summary.csv")
    interpretation = build_interpretation(p1, p2, p3)

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=16mm]{{geometry}}
\usepackage{{booktabs,tabularx,array,multirow}}
\usepackage{{graphicx}}
\usepackage{{microtype}}
\usepackage{{xcolor}}
\usepackage{{amsmath,amssymb}}
\usepackage{{hyperref}}
\usepackage{{caption}}
\usepackage{{float}}
\definecolor{{ink}}{{HTML}}{{25313A}}
\definecolor{{accent}}{{HTML}}{{3977A8}}
\definecolor{{soft}}{{HTML}}{{EEF3F6}}
\hypersetup{{colorlinks=true,linkcolor=accent,urlcolor=accent}}
\captionsetup{{font=small,labelfont=bf}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{document}}
\begin{{center}}
{{\LARGE\bfseries Post-meeting validation of structured-joint HiPPO--STVGP}}\\[4pt]
{{\large Literature-faithful baselines, fair batch/streaming controls, residual decomposition and temporal ablations}}\\[6pt]
{{\small ERA5 held-out spatial prediction; updated 15 July 2026}}
\end{{center}}

\begin{{abstract}}
This report executes the experiment priorities agreed after the group meeting. It first restores the distinction between the paper's ST-VGP and ST-SVGP and the project's doubly sparse Kronecker comparator. It then evaluates a complete $2\times2$ batch/online matrix at two inducing budgets, separates direct-target, mean-only, residual and joint inference effects, and tests whether gains are specific to the analytic HiPPO--RFF temporal construction. Final-block RMSE and NLL are the primary endpoints. Block averages are reported separately; P1--P3 summaries use three paired held-out spatial splits and report mean $\pm$ standard deviation, while P0 explicitly reports any official configuration that cannot complete at full ERA5 scale.
\end{{abstract}}

\section{{Questions and decision rules}}
The analysis was pre-structured around five tests. Streaming transfer is supported if online and batch final-block metrics remain close across paired splits. The structured joint posterior is supported if it improves on matched sparse STVGP under both update protocols. The $\beta$--GP contribution is supported if joint exact inference consistently improves on two-stage exact residual fitting. A remaining gap to paper-faithful ST-SVGP points to temporal/spatial representation or optimization rather than to X-lag alone. Direct-target and residual STVGP are compared to determine whether X-lag is an engineering aid or the dominant source of accuracy.

\section{{P0: full-protocol official ST-VGP and ST-SVGP}}
The official AaltoML repository was fixed at commit \texttt{{c5b929e1fc07b14ff9671dd1d66b3b8041e2a2ce}} and run in an isolated Python 3.7.12 environment with JAX 0.2.9, jaxlib 0.1.60, Objax 1.3.1 and Bayes--Newton 1.1. The official requirements are internally inconsistent: jaxlib 0.1.60 requires NumPy below 1.20, whereas the repository pins NumPy 1.21.3. The compatible environment therefore uses NumPy 1.19.5 and records both failed and successful installation logs.

The benchmark now uses the complete P1-aligned protocol: task 2, variable 0, 186 time points, all 1000 spatial locations, 800/200 held-out spatial splits and seeds 0--2. Every official model directly fits scaled $y$ without X-lag or residual decomposition, with product Mat\'ern-$3/2$ spatial and Mat\'ern-$3/2$ temporal kernels. ST-VGP keeps the full spatial state; ST-SVGP and MF-ST-SVGP use spatial k-means inducing locations but retain the full temporal Markov state at every observation time. There is no small $M_t$ in this table.

Training follows the official Bayes--Newton inference plus Objax Adam cycle. Because the paper's air-quality initialization is not scale-compatible with standardized ERA5, all runs use the pre-declared P1-scale initialization $(\ell_t,\ell_s,\sigma^2)=(0.05,0.35,0.01)$. Adam learning rates were selected before held-out evaluation from training-energy stability: 0.1 for ST-SVGP $M_s=30$, 0.05 for ST-SVGP $M_s=64$, 0.02 for ST-SVGP $M_s=128$, 0.1 for MF-ST-SVGP $M_s=30$, and 0.01 for full ST-VGP. Training is capped at 100 iterations and stops only when relative training-energy change remains below $2\times10^{-3}$ for ten consecutive updates after iteration 30. Held-out metrics are never used for stopping or configuration selection.

\begin{{table}}[H]
\centering\scriptsize
\caption{{Literature-faithful full-protocol comparison. All models use the official Bayes--Newton implementation and direct-target training. Runtime is end-to-end wall time per seed; peak RSS is process-level memory. Resource-limited rows are retained rather than replaced by subset results.}}
\label{{tab:p0}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrr}}
\toprule Model & Spatial state & Seeds & RMSE & NLL & Cov$_{{90}}$ & Iter. & Time (min) & Peak GB \\
\midrule
{p0_full_rows(root)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
{p0_full_interpretation(root)} The earlier $20\times(12+4)$ official/local parity experiment is retained only as an implementation sanity check. Its shared learned hyperparameters make it unsuitable as an independent performance comparison, so local-adapter rows no longer appear in the literature-faithful table.

\section{{P1: complete fair batch/streaming matrix}}
All four cells use the same task, held-out split, X-lag mean features ($L=10$), RBF kernel family, noise standard deviation 0.1, spatial lengthscale 0.35 and inducing locations. Matched sparse STVGP uses a fixed temporal inducing basis. Its online implementation accumulates only sufficient statistics and does not retain raw history; under fixed hyperparameters it is algebraically identical to its batch reconstruction. Structured-joint batch recomputes the posterior from the full seen history, whereas structured-joint online transfers the old likelihood summary between changing analytic HiPPO bases. Table~\ref{{tab:p1}} reports the final block for the 19-block (block size 10) protocol.

\begin{{table}}[H]
\centering\scriptsize
\caption{{Final-block fair matrix, mean $\pm$ SD over three paired held-out spatial splits. Runtime is incremental wall time for the final update and prediction; peak RSS is process-level memory.}}
\label{{tab:p1}}
\begin{{tabularx}}{{\textwidth}}{{l>{{\raggedright\arraybackslash}}Xlrrrrr}}
\toprule $(M_t,M_s)$ & Residual architecture & Update & RMSE & NLL & Cov$_{{90}}$ & Time (s) & Peak MB \\
\midrule
{p1_rows(p1)}
\bottomrule
\end{{tabularx}}
\end{{table}}

{interpretation['p1']}

\begin{{table}}[H]
\centering\scriptsize
\caption{{Auxiliary block-average results for block size 10. Each seed is first averaged over its 19 blocks; the table then reports mean $\pm$ SD across seeds.}}
\begin{{tabularx}}{{\textwidth}}{{l>{{\raggedright\arraybackslash}}Xlrrrr}}
\toprule $(M_t,M_s)$ & Residual architecture & Update & RMSE & NLL & Cov$_{{90}}$ & Time (s) \\
\midrule
{block_average_rows(p1_block)}
\bottomrule
\end{{tabularx}}
\end{{table}}

\begin{{table}}[H]
\centering\scriptsize
\caption{{Online structured-joint sensitivity to block size. These are final-block metrics, not block averages.}}
\begin{{tabular}}{{rrrrr}}
\toprule $(M_t,M_s)$ & Block size & RMSE & NLL & Time (s) \\
\midrule
{block_rows(p1)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering\scriptsize
\caption{{Paired seed comparisons at block size 10. Differences are left minus right as named. With only three splits, $p$-values are descriptive and confidence intervals are intentionally retained.}}
\begin{{tabular}}{{llrrrr}}
\toprule Comparison & Capacity & Metric & Mean difference & 95\% CI & Paired $t$ $p$ \\
\midrule
{paired_rows(paired)}
\bottomrule
\end{{tabular}}
\end{{table}}

Block-average summaries are saved separately in \texttt{{aggregated/p1\_block\_average\_summary.csv}}. They are not used as substitutes for final-state accuracy because early blocks contain less history and different test sample counts.

\section{{P2: what does the residual decomposition contribute?}}
The exact rows use the same separable Mat\'ern-3/2 protocol ($\ell_t=0.1$, $\ell_s=1.0$), and the observation variance is fixed at 5\% of the training X-lag residual variance within each split. The sparse rows use the same kernel family and per-split noise, with $(M_t,M_s)=(32,128)$. This table distinguishes direct modelling of $y$, a deterministic mean, two-stage residual kriging and joint posterior coupling.

\begin{{table}}[H]
\centering\scriptsize
\caption{{Residual decomposition, mean $\pm$ SD over three paired held-out splits.}}
\begin{{tabularx}}{{\textwidth}}{{>{{\raggedright\arraybackslash}}Xrrrrr}}
\toprule Method & RMSE & NLL & Cov$_{{90}}$ & Mean std & Runtime (s) \\
\midrule
{p2_rows(p2)}
\bottomrule
\end{{tabularx}}
\end{{table}}

{interpretation['p2']}

\section{{P3: HiPPO temporal representation ablation}}
The mean feature, spatial kernel, spatial inducing set, noise, update protocol and nominal $(M_t,M_s)$ budget are fixed. Analytic HiPPO--RFF is compared with standard temporal inducing points and ordinary RFF. The full temporal-kernel row retains every observed temporal state and is included only as an upper bound; its effective $M_t$ is 186 at the final block.

\begin{{table}}[H]
\centering\scriptsize
\caption{{Final-history temporal representation ablation, mean $\pm$ SD over three held-out splits.}}
\begin{{tabularx}}{{\textwidth}}{{l>{{\raggedright\arraybackslash}}Xrrrrr}}
\toprule Nominal capacity & Temporal representation & RMSE & NLL & Cov$_{{90}}$ & Time (s) & Peak MB \\
\midrule
{p3_rows(p3)}
\bottomrule
\end{{tabularx}}
\end{{table}}

{interpretation['p3']}

\section{{P4: standardized visual diagnostics}}
Figure~\ref{{fig:spatial}} shows the complete spatial protocol. The inducing set is selected before target fitting and contains no held-out labels. Figure~\ref{{fig:maps}} evaluates the final timestamp without selecting a favourable region. Figure~\ref{{fig:cases}} uses a pre-declared location rule: the held-out locations nearest the 10th, 50th and 90th percentiles of online per-location RMSE are labelled success, median and failure. Figure~\ref{{fig:delta}} reports online-minus-batch paired differences through all 19 blocks.

\begin{{figure}}[H]
\centering\includegraphics[width=0.62\linewidth]{{figures/fig_spatial_split_inducing.pdf}}
\caption{{Observed locations, held-out locations and spatial inducing points for split seed 0.}}
\label{{fig:spatial}}
\end{{figure}}

\begin{{figure}}[H]
\centering\includegraphics[width=0.95\linewidth]{{figures/fig_final_time_spatial_maps.pdf}}
\caption{{Final-time spatial truth, online structured-joint prediction and signed error for the primary $(32,128)$ model.}}
\label{{fig:maps}}
\end{{figure}}

\begin{{figure}}[H]
\centering\includegraphics[width=0.98\linewidth]{{figures/fig_success_median_failure_timeseries.pdf}}
\caption{{Algorithmically selected success, median and failure held-out locations. Each column shows the predictive mean and 90\% interval for one update protocol. These locations are held out for the entire timeline; there is no temporal train/test boundary to shade.}}
\label{{fig:cases}}
\end{{figure}}

\begin{{figure}}[H]
\centering\includegraphics[width=0.82\linewidth]{{figures/fig_online_batch_block_differences.pdf}}
\caption{{Paired online-minus-batch RMSE and NLL across the 19 blocks; ribbons show one SD across held-out split seeds.}}
\label{{fig:delta}}
\end{{figure}}

\clearpage
\section{{Conclusions and next decisions}}
The official implementation path is reproducible in an isolated legacy environment and is now evaluated on the complete P1-aligned ERA5 spatial holdout rather than a small parity subset. Literature-faithful ST-SVGP remains separate from the doubly sparse $(M_t,M_s)$ comparison because it is spatially sparse but temporally full/Markov. The fair matrix shows that structured joint inference is beneficial under the tighter inducing budget and that the online penalty becomes small at $(32,128)$. The decomposition shows that X-lag is not carrying the result by itself: exact residual kriging provides a large gain, and joint exact $\beta$--GP coupling provides a further consistent gain.

The main negative result is equally informative. At high capacity, standard temporal inducing points outperform analytic HiPPO--RFF under the same RBF batch protocol, and the current Mat\'ern analytic online path is unstable across held-out splits. The immediate research priority is therefore to audit and improve the analytic temporal basis and cross-basis transfer for non-RBF kernels, then repeat the exact same P1 matrix. Spatial inducing selection and local kriging corrections remain secondary capacity improvements. No conclusion in this report treats the small P0 subset as a full official benchmark, and no claim of broad statistical significance is made from three seeds alone.

\subsection*{{Recommended immediate follow-up}}
\begin{{enumerate}}
\item \textbf{{Temporal basis equivalence audit.}} On a small fixed grid, compare analytic HiPPO--RFF, standard inducing points and the full temporal kernel using the same RBF and Mat\'ern hyperparameters. Record $\|K_{{ff}}-Q_{{ff}}\|_F/\|K_{{ff}}\|_F$, diagonal error, $K_{{uu}}$ condition number and predictive variance error. This separates representation error from posterior inference.
\item \textbf{{Changing-basis transfer reference.}} For two consecutive horizons, materialize the small dense old/new joint Gaussian and compare transferred natural parameters, posterior mean and covariance with the structured transfer. This test should be repeated for RBF and Mat\'ern before another large ERA5 sweep.
\item \textbf{{Repeat P1 with a fixed analytic basis.}} Holding the analytic basis global removes changing-basis transfer while retaining HiPPO features. If fixed-basis HiPPO approaches standard inducing points, transfer is the main defect; if it remains worse, the temporal compression itself is the bottleneck.
\item \textbf{{Only then resume spatial enhancements.}} After the temporal audit, compare linspace, farthest-point and k-means inducing locations, followed by a local nearest-neighbour correction. The primary endpoint should remain final-block paired RMSE/NLL, with coverage monitored to avoid trading calibration for mean accuracy.
\end{{enumerate}}

\section*{{Reproducibility artefacts}}
All commands, environment pins, raw per-block CSVs, pointwise predictions, paired statistics and figure-selection records are stored in this report directory. The key subdirectories are P0b official full ERA5, P1 fair matrix, P2 residual decomposition, P3 HiPPO ablation and aggregated summaries. The earlier P0 parity artefacts remain available for auditability.
\end{{document}}
"""
    tex_path = root / "post_meeting_priority_experiment_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    if shutil.which("pdflatex") is None:
        print(tex_path)
        return
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=root,
        check=True,
        stdout=(root / "pdflatex_pass1.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=root,
        check=True,
        stdout=(root / "pdflatex_pass2.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    print(root / "post_meeting_priority_experiment_report.pdf")


if __name__ == "__main__":
    main()
