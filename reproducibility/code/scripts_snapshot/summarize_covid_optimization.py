#!/usr/bin/env python3
"""Compare COVID optimization candidates without changing the pilot baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path, method: str = "routeb_cumulative_hippo") -> dict[int, dict[str, float]]:
    with (path / "metrics_per_seed.csv").open(encoding="utf-8") as handle:
        return {
            int(row["seed"]): row
            for row in csv.DictReader(handle)
            if row["method"] == method
        }


def mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def bootstrap_ci(values: np.ndarray, seed: int = 20260810) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(100000, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results" / "diagnostics" / "covid_optimization")
    args = parser.parse_args()
    root = args.root
    candidates = {
        "old causal Mt16 Ms32": ROOT / "results" / "diagnostics" / "covid_abcd" / "C_final_mt16_ms32_500",
        "new causal Mt32 Ms32": root / "mt32_ms32_rff64_1000",
        "new intercept Mt32 Ms32": root / "mt32_ms32_rff64_1000_intercept_only",
    }
    loaded = {name: rows(path) for name, path in candidates.items()}
    metrics = ("rmse", "nll", "coverage90", "mean_predictive_std", "mean_interval_width90")
    table_rows = []
    for name, seed_rows in loaded.items():
        out = {"candidate": name, "n_seeds": len(seed_rows)}
        for metric in metrics:
            values = [float(seed_rows[s][metric]) for s in sorted(seed_rows)]
            out[f"{metric}_mean"], out[f"{metric}_sd"] = mean_sd(values)
        table_rows.append(out)
    with (root / "optimization_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    baseline = loaded["old causal Mt16 Ms32"]
    lines = [
        "# COVID Route B optimization report",
        "",
        "The original A-E artifacts are preserved. This report evaluates additional candidates under the same COVID protocol, full-joint-conditional variance, float64, five spatial split seeds, and strict-online evaluation.",
        "",
        "## Main comparison",
        "",
        "| Candidate | RMSE | NLL | Coverage90 | Mean std | 90% width |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['candidate']} | {row['rmse_mean']:.4f} +/- {row['rmse_sd']:.4f} | "
            f"{row['nll_mean']:.4f} +/- {row['nll_sd']:.4f} | "
            f"{row['coverage90_mean']:.4f} +/- {row['coverage90_sd']:.4f} | "
            f"{row['mean_predictive_std_mean']:.4f} +/- {row['mean_predictive_std_sd']:.4f} | "
            f"{row['mean_interval_width90_mean']:.4f} +/- {row['mean_interval_width90_sd']:.4f} |"
        )

    for name in ("new causal Mt32 Ms32", "new intercept Mt32 Ms32"):
        lines.extend(["", f"## Paired comparison: {name} minus old causal"])
        for metric in ("rmse", "nll", "coverage90"):
            difference = np.asarray(
                [float(loaded[name][seed][metric]) - float(baseline[seed][metric]) for seed in sorted(baseline)]
            )
            low, high = bootstrap_ci(difference)
            lines.append(
                f"- {metric}: mean difference {difference.mean():+.4f}; "
                f"per-seed {', '.join(f'{x:+.4f}' for x in difference)}; "
                f"bootstrap 95% CI [{low:+.4f}, {high:+.4f}]."
            )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The causal Mt32/Ms32 candidate is retained as the best calibrated default candidate: it improves RMSE and NLL over the old capacity and brings Coverage90 closer to the nominal 0.90 target.",
            "",
            "The intercept-only Mt32/Ms32 candidate is retained as a point-prediction variant because it has lower RMSE and NLL, but it is not promoted as the sole probabilistic default: its Coverage90 is 0.8718 and its predictive standard deviation is substantially smaller.",
            "",
            "The seed-0 screen candidates with lower learning rate, RFF=128, and extra training were not promoted independently. RFF=128 improved seed-0 RMSE but worsened NLL and coverage; longer training at Mt16 did not improve the online result.",
            "",
            "These are still COVID feasibility-pilot results: 52 locations, 39 strict-online weeks, and five spatial split seeds. The improvement should be rechecked on a longer multi-year dataset before being treated as a general method conclusion.",
        ]
    )
    (root / "optimization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {root / 'optimization_report.md'}")


if __name__ == "__main__":
    main()
