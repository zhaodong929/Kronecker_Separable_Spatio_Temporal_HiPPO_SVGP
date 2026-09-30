#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
OUT="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison/phase_d_direct_target"
CAL="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison/phase_b_empirical_bayes"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
mkdir -p "$OUT"

read_param() {
  local path=$1 key=$2
  "$PY" -c "import json; print(json.load(open('$path'))['best']['$key'])"
}

for capacity in "8 64" "32 128"; do
  read -r mt ms <<< "$capacity"
  params="$CAL/Mt${mt}_Ms${ms}/learned_hyperparameters.json"
  ell_t=$(read_param "$params" ell_t)
  ell_s=$(read_param "$params" ell_s)
  kernel_variance=$(read_param "$params" kernel_variance)
  noise=$(read_param "$params" noise_std)

  for split_seed in 0 1 2; do
    batch="$OUT/Mt${mt}_Ms${ms}/seed${split_seed}/batch"
    mkdir -p "$batch"
    if [[ ! -s "$batch/run_metadata.json" ]]; then
      /usr/bin/time -v -o "$batch/resource_usage.txt" \
        "$PY" scripts/run_post_meeting_p1_matrix.py \
          --architecture structured_joint --protocol batch --outdir "$batch" \
          --split-seed "$split_seed" --block-size 10 --mt "$mt" --ms "$ms" \
          --phi-mode direct_y --ell-t "$ell_t" --spatial-lengthscale "$ell_s" \
          --noise "$noise" --kernel-variance "$kernel_variance" --kernel-type matern32 \
          --spatial-inducing-selection kmeans --temporal-representation analytic_hippo_rff \
          --final-block-only > "$batch/stdout.log" 2> "$batch/stderr.log"
    fi

    online="$OUT/Mt${mt}_Ms${ms}/seed${split_seed}/online"
    mkdir -p "$online"
    if [[ ! -s "$online/era5_routeb_report.json" ]]; then
      /usr/bin/time -v -o "$online/resource_usage.txt" \
        "$PY" scripts/run_hipposvgp_era5_routeb.py \
          --outdir "$online" --root data/era5/processed_timeseries_4 \
          --calibration-tasks task_1 --online-tasks task_2 --variable-index 0 --split all \
          --block-size 10 --routeb-methods structured_joint --eval-modes seen_history \
          --phi-mode direct_y --ohsvgp-heldout-eval --heldout-split-seeds "$split_seed" --seeds 0 \
          --mt "$mt" --ms "$ms" --temporal-backend analytic_hippo_rff \
          --temporal-rff-sample-size 256 --temporal-rff-seed 0 \
          --model-ell-t "$ell_t" --ell-t-fit-mode none --routeb-noise "$noise" \
          --kernel-type matern32 --kernel-variance "$kernel_variance" \
          --spatial-lengthscale "$ell_s" --spatial-inducing-selection kmeans \
          --prediction-mode streaming_sylvester --prediction-chunk-size 8192 \
          --skip-heldout-block-pairs --save-per-location-predictions \
          > "$online/stdout.log" 2> "$online/stderr.log"
    fi
  done
done
