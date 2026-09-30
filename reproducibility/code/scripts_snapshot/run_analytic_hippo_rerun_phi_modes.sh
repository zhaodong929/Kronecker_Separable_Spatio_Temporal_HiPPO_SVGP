#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

base_out="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/analytic_hippo_rff_rerun/phi_modes"
modes=(minimal base engineered medium_era5 rich_era5)

for mode in "${modes[@]}"; do
  echo "=== Running phi mode: ${mode} ==="
  .venv/bin/python scripts/run_hipposvgp_era5_routeb.py \
    --outdir "${base_out}/${mode}" \
    --root data/era5/processed_timeseries_4 \
    --calibration-tasks task_1 \
    --online-tasks task_2 \
    --variable-index 0 \
    --split all \
    --block-size 10 \
    --routeb-methods no_transfer mean_field structured_joint \
    --eval-modes seen_history \
    --phi-mode "${mode}" \
    --ohsvgp-heldout-eval \
    --heldout-split-seeds 0 1 2 \
    --seeds 0 \
    --mt 8 \
    --ms 64 \
    --prediction-mode streaming_sylvester \
    --prediction-chunk-size 8192 \
    --hyperparam-fit-mode none \
    --ell-t-fit-mode none \
    --model-ell-t 0.05 \
    --routeb-noise 0.1 \
    --kernel-type rbf \
    --kernel-variance 1.0 \
    --save-forgetting-block-pairs
  echo "=== Done phi mode: ${mode} ==="
done
