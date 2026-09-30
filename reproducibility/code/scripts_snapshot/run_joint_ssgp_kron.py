#!/usr/bin/env python
"""Default entry point for the new joint SSGP Kronecker synthetic run."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_joint_ssgp_kron_experiments import main


if __name__ == "__main__":
    main()
