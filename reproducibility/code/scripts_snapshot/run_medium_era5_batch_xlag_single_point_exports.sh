#!/usr/bin/env bash
set -euo pipefail

cd /home/zd929/projects/stvgp_kronecker

OUT="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/medium_era5_batch_xlag_diagnostic/single_point_exports"
mkdir -p "${OUT}"

COMMON_ARGS=(
  --root data/era5/processed_timeseries_4
  --calibration-tasks task_1
  --online-tasks task_2
  --variable-index 0
  --split all
  --block-size 10
  --routeb-methods structured_joint
  --ohsvgp-heldout-eval
  --heldout-split-seeds 0
  --seeds 0
  --mt 8
  --ms 64
  --prediction-mode streaming_sylvester
  --prediction-chunk-size 8192
  --hyperparam-fit-mode none
  --ell-t-fit-mode none
  --model-ell-t 0.05
  --routeb-noise 0.1
  --kernel-type rbf
  --kernel-variance 1.0
  --save-per-location-predictions
)

.venv/bin/python scripts/run_hipposvgp_era5_routeb.py \
  --outdir "${OUT}/original_medium_safe_lag_seen_history" \
  "${COMMON_ARGS[@]}" \
  --eval-modes seen_history \
  --phi-mode medium_era5

.venv/bin/python scripts/run_hipposvgp_era5_routeb.py \
  --outdir "${OUT}/xlag_seen_history" \
  "${COMMON_ARGS[@]}" \
  --eval-modes seen_history \
  --phi-mode medium_era5_xlag

.venv/bin/python scripts/run_hipposvgp_era5_routeb.py \
  --outdir "${OUT}/safe_recursive_batch" \
  "${COMMON_ARGS[@]}" \
  --eval-modes batch \
  --phi-mode medium_era5

.venv/bin/python scripts/run_hipposvgp_era5_routeb.py \
  --outdir "${OUT}/oracle_ylag_batch" \
  "${COMMON_ARGS[@]}" \
  --eval-modes batch \
  --phi-mode medium_era5_oracle_ylag
