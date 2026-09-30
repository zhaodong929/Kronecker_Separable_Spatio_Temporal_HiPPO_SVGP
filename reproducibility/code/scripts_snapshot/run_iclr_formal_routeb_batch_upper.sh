#!/usr/bin/env bash
set -euo pipefail
cd /home/zd929/projects/stvgp_kronecker

base="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/fairness_diagnostics/routeb_batch_upper"

run_one() {
  local name="$1"
  local mt="$2"
  local ms="$3"
  local sl="$4"
  local out="$base/$name"
  if [[ -f "$out/era5_routeb_summary.csv" ]]; then
    echo "skip $name"
    return
  fi
  echo "running $name"
  .venv/bin/python scripts/run_hipposvgp_era5_routeb.py \
    --outdir "$out" \
    --root data/era5/processed_timeseries_4 \
    --calibration-tasks task_1 \
    --online-tasks task_2 \
    --variable-index 0 \
    --split all \
    --block-size 10 \
    --routeb-methods structured_joint \
    --eval-modes batch \
    --phi-mode medium_era5_xlag \
    --xlag-length 10 \
    --ohsvgp-heldout-eval \
    --heldout-split-seeds 1 \
    --seeds 0 \
    --mt "$mt" \
    --ms "$ms" \
    --prediction-mode streaming_sylvester \
    --prediction-chunk-size 8192 \
    --hyperparam-fit-mode none \
    --ell-t-fit-mode none \
    --model-ell-t 0.05 \
    --routeb-noise 0.1 \
    --kernel-type rbf \
    --kernel-variance 1.0 \
    --spatial-lengthscale "$sl"
}

run_one default_Mt8_Ms64_ls035 8 64 0.35
run_one high_Mt16_Ms256_ls035 16 256 0.35
run_one high_Mt16_Ms256_ls100 16 256 1.0
