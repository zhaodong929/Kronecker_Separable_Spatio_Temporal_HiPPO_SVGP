#!/usr/bin/env python3
"""Run the unchanged OHSVGP adapter with scoped prediction-basis memoization."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_covid_ohsvgp_own_theta as worker
from baselines.ohsvgp_prediction import cache_prediction_basis
worker.predict=cache_prediction_basis(worker.predict)
if __name__=='__main__':worker.main()
