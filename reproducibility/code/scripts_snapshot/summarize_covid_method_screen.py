#!/usr/bin/env python3
"""Apply a predeclared seed-0 gate to COVID method-level candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["overall_current_block"].items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()

    baseline = metrics(args.baseline)
    rows = []
    for name in args.candidates:
        path = args.candidate_root / name / "seed0" / "cumulative_hippo" / "online" / "result.json"
        row = {"candidate": name, "result": str(path), **metrics(path)}
        row["rmse_change"] = row["rmse"] - baseline["rmse"]
        row["nll_change"] = row["nll"] - baseline["nll"]
        row["coverage90_distance_to_nominal"] = abs(row["coverage90"] - 0.90)
        row["passes_seed0_gate"] = (
            row["rmse_change"] < 0.0
            and row["nll_change"] < 0.0
            and row["coverage90_distance_to_nominal"] <= 0.05
        )
        rows.append(row)

    args.candidate_root.mkdir(parents=True, exist_ok=True)
    with (args.candidate_root / "screen_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passing = [row for row in rows if row["passes_seed0_gate"]]
    selected = min(passing, key=lambda row: (row["nll"], row["rmse"])) if passing else None
    report = [
        "# COVID method-level seed-0 screen",
        "",
        "Gate: RMSE and NLL must both improve over the retained causal Mt32/Ms32 Matérn baseline; Coverage90 must remain within 0.05 of 0.90.",
        "",
        "| Candidate | RMSE | Change | NLL | Change | Coverage90 | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['candidate']} | {row['rmse']:.4f} | {row['rmse_change']:+.4f} | "
            f"{row['nll']:.4f} | {row['nll_change']:+.4f} | {row['coverage90']:.4f} | "
            f"{'yes' if row['passes_seed0_gate'] else 'no'} |"
        )
    report.extend(
        [
            "",
            "Selected for five-seed confirmation: " + (selected["candidate"] if selected else "none"),
            "",
            "This is a seed-0 screen only. It does not replace the retained five-seed baseline.",
        ]
    )
    (args.candidate_root / "screen_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"selected": None if selected is None else selected["candidate"], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
