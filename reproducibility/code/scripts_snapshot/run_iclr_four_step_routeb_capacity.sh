#!/usr/bin/env bash
set -euo pipefail

cd /home/zd929/projects/stvgp_kronecker
OUT="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/four_step_priority_experiments/step1_routeb_capacity"
mkdir -p "$OUT"

for cfg in 8:64 16:64 16:128 32:128 32:256; do
  mt="${cfg%%:*}"
  ms="${cfg##*:}"
  echo "=== RouteB capacity Mt=${mt} Ms=${ms} ==="
  .venv/bin/python scripts/run_hipposvgp_era5_routeb.py \
    --outdir "$OUT/Mt${mt}_Ms${ms}" \
    --root data/era5/processed_timeseries_4 \
    --calibration-tasks task_1 \
    --online-tasks task_2 \
    --variable-index 0 \
    --split all \
    --block-size 10 \
    --routeb-methods structured_joint \
    --eval-modes seen_history \
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
    --spatial-lengthscale 0.35
done
