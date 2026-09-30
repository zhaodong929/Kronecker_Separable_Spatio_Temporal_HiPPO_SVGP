#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/kernel_family_fullgp_wide_diagnostic"

ELL_GRID=(0.0125 0.025 0.05 0.075 0.10 0.15 0.20)
NOISE_GRID=(0.05 0.10 0.20 0.30 0.50 0.80)
KVAR_GRID=(0.25 0.50 1.00 1.50)

run_one() {
  local tag="$1"
  local kernel="$2"
  local phi="$3"
  local mt="$4"
  local ms="$5"
  local outdir="${ROOT_DIR}/${tag}"
  if [[ -f "${outdir}/era5_routeb_summary.csv" ]]; then
    echo "SKIP existing ${outdir}"
    return
  fi
  echo "RUN ${tag}: kernel=${kernel} phi=${phi} Mt=${mt} Ms=${ms}"
  uv run --no-sync python scripts/run_hipposvgp_era5_routeb.py \
    --root data/era5/processed_timeseries_4 \
    --calibration-tasks task_1 \
    --online-tasks task_2 \
    --tasks task_2 \
    --split all \
    --seeds 0 \
    --heldout-split-seeds 0 \
    --block-size 10 \
    --heldout-test-fraction 0.2 \
    --mt "${mt}" \
    --ms "${ms}" \
    --kernel-type "${kernel}" \
    --phi-mode "${phi}" \
    --model-ell-t 0.05 \
    --ell-t-fit-mode none \
    --hyperparam-fit-mode initial_task_fullgp_grid \
    --ell-t-grid "${ELL_GRID[@]}" \
    --noise-grid "${NOISE_GRID[@]}" \
    --kernel-variance-grid "${KVAR_GRID[@]}" \
    --hyperparam-fit-max-time 30 \
    --hyperparam-fit-max-locations 30 \
    --routeb-methods structured_joint \
    --eval-modes seen_history \
    --prediction-mode streaming_sylvester \
    --prediction-chunk-size 8192 \
    --save-forgetting-block-pairs \
    --save-per-location-predictions \
    --per-location-indices 99 \
    --outdir "${outdir}"
}

run_one "rbf_base_fullgp_wide" "rbf" "base" 8 64
run_one "matern32_base_fullgp_wide" "matern32" "base" 8 64
run_one "rbf_rich_v3_fullgp_wide" "rbf" "rich_v3" 8 128
run_one "matern32_rich_v3_fullgp_wide" "matern32" "rich_v3" 8 128
run_one "rbf_rich_v4_fullgp_wide" "rbf" "rich_v3" 32 256
run_one "matern32_rich_v4_fullgp_wide" "matern32" "rich_v3" 32 256

