#!/usr/bin/env python3
"""Gate the optional COVID daily appendix on an audited daily CDC snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.source_csv, usecols=["Week Ending Date"])
    dates = pd.DatetimeIndex(pd.to_datetime(frame["Week Ending Date"], errors="coerce").dropna().unique()).sort_values()
    intervals = dates.to_series().diff().dropna().dt.days
    median_days = float(intervals.median()) if not intervals.empty else float("nan")
    payload = {
        "source_csv": str(args.source_csv.resolve()),
        "date_column": "Week Ending Date",
        "unique_endpoints": int(dates.size),
        "minimum_interval_days": int(intervals.min()) if not intervals.empty else None,
        "median_interval_days": median_days,
        "maximum_interval_days": int(intervals.max()) if not intervals.empty else None,
        "status": "pass_daily_completeness" if median_days <= 1.0 else "no_go_weekly_source",
        "reason": (
            "The supplied snapshot has daily endpoints and may be used for a separately named high-frequency appendix."
            if median_days <= 1.0
            else "The audited CDC HRD snapshot is weekly; no daily experiment is run from weekly endpoints."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
