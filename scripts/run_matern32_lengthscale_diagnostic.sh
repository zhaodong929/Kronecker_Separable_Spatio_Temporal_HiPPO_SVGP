#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/matern32_lengthscale_diagnostic"

run_one() {
  local ell="$1"
  local mt="$2"
  local ms="$3"
  local tag="$4"
  local outdir="${ROOT_DIR}/${tag}"
  if [[ -f "${outdir}/era5_routeb_summary.csv" ]]; then
    echo "SKIP existing ${outdir}"
    return
  fi
  echo "RUN ell_t=${ell} mt=${mt} ms=${ms} -> ${outdir}"
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
    --kernel-type matern32 \
    --phi-mode rich_v3 \
    --model-ell-t "${ell}" \
    --ell-t-fit-mode none \
    --routeb-noise 0.09 \
    --kernel-variance 0.5 \
    --routeb-methods structured_joint \
    --eval-modes seen_history \
    --prediction-mode streaming_sylvester \
    --prediction-chunk-size 8192 \
    --save-forgetting-block-pairs \
    --save-per-location-predictions \
    --per-location-indices 99 \
    --outdir "${outdir}"
}

for ell in 0.0125 0.025 0.075 0.10 0.15 0.20; do
  run_one "${ell}" 32 256 "rich_v4_mt32_ms256_ell_${ell}"
done
