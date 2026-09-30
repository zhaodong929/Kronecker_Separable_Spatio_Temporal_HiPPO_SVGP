#!/usr/bin/env bash
set -euo pipefail

cd /home/zd929/projects/stvgp_kronecker
OUT="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/four_step_priority_experiments/step2_spatial_enhancement"
mkdir -p "$OUT"

run_one() {
  local name="$1"
  local selection="$2"
  local kernel="$3"
  local ls="$4"
  echo "=== Spatial enhancement ${name}: selection=${selection}, kernel=${kernel}, ls=${ls} ==="
  .venv/bin/python scripts/run_hipposvgp_era5_routeb.py \
    --outdir "$OUT/$name" \
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
    --mt 16 \
    --ms 128 \
    --prediction-mode streaming_sylvester \
    --prediction-chunk-size 8192 \
    --hyperparam-fit-mode none \
    --ell-t-fit-mode none \
    --model-ell-t 0.05 \
    --routeb-noise 0.1 \
    --kernel-type "$kernel" \
    --kernel-variance 1.0 \
    --spatial-lengthscale "$ls" \
    --spatial-inducing-selection "$selection"
}

run_one "farthest_rbf_ls035" "farthest" "rbf" "0.35"
run_one "kmeans_rbf_ls035" "kmeans" "rbf" "0.35"
run_one "linspace_rbf_ls100" "linspace" "rbf" "1.0"
run_one "farthest_matern32_ls035" "farthest" "matern32" "0.35"
run_one "farthest_matern32_ls100" "farthest" "matern32" "1.0"
