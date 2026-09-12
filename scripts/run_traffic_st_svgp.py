#!/usr/bin/env python3
"""PEMS-BAY entry point for the pinned ST-SVGP causal-refit adapter."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.covid_long_setting_b.adapters.run_st_svgp import main


if __name__ == "__main__":
    if "--protocol-kind" not in sys.argv:
        sys.argv.extend(["--protocol-kind", "traffic"])
    main()
