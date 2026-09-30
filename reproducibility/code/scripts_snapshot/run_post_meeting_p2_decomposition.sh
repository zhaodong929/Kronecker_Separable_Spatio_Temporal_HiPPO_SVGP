#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
OUT="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/post_meeting_priority_2026-07-14/p2_residual_decomposition"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
mkdir -p "$OUT"

if [[ ! -s "$OUT/exact_decomposition.json" ]]; then
  /usr/bin/time -v -o "$OUT/exact_resource_usage.txt" \
    "$PY" scripts/run_post_meeting_p2_exact_decomposition.py \
      --outdir "$OUT" --split-seeds 0 1 2 --xlag-length 10 \
      --ell-t 0.1 --ell-s 1.0 --noise-scale 0.05 \
      > "$OUT/exact_stdout.log" 2> "$OUT/exact_stderr.log"
fi

for split_seed in 0 1 2; do
  noise_std=$(
    "$PY" - "$OUT/exact_decomposition_metrics.csv" "$split_seed" <<'PY'
import csv, math, sys
path, seed = sys.argv[1], int(sys.argv[2])
with open(path, newline='') as handle:
    rows = list(csv.DictReader(handle))
row = next(r for r in rows if int(r['heldout_split_seed']) == seed and r['method'] == 'STVGP residual + X-lag two-stage')
print(math.sqrt(float(row['noise_variance'])))
PY
  )

  matched="$OUT/seed${split_seed}/matched_sparse_residual_batch"
  mkdir -p "$matched"
  if [[ ! -s "$matched/run_metadata.json" ]]; then
    /usr/bin/time -v -o "$matched/resource_usage.txt" \
      "$PY" scripts/run_post_meeting_p1_matrix.py \
        --architecture matched_sparse_stvgp --protocol batch --outdir "$matched" \
        --split-seed "$split_seed" --block-size 10 --mt 32 --ms 128 \
        --xlag-length 10 --ell-t 0.1 --spatial-lengthscale 1.0 --noise "$noise_std" \
        --kernel-type matern32 --spatial-inducing-selection linspace --final-block-only \
        > "$matched/stdout.log" 2> "$matched/stderr.log"
  fi

  online="$OUT/seed${split_seed}/structured_joint_online"
  mkdir -p "$online"
  if [[ ! -s "$online/era5_routeb_report.json" ]]; then
    /usr/bin/time -v -o "$online/resource_usage.txt" \
      "$PY" scripts/run_hipposvgp_era5_routeb.py \
        --outdir "$online" --root data/era5/processed_timeseries_4 \
        --calibration-tasks task_1 --online-tasks task_2 --variable-index 0 --split all \
        --block-size 10 --routeb-methods structured_joint --eval-modes seen_history \
        --phi-mode medium_era5_xlag --xlag-length 10 \
        --ohsvgp-heldout-eval --heldout-split-seeds "$split_seed" --seeds 0 \
        --mt 32 --ms 128 --temporal-backend analytic_hippo_rff \
        --temporal-rff-sample-size 256 --temporal-rff-seed 0 \
        --model-ell-t 0.1 --ell-t-fit-mode none --routeb-noise "$noise_std" \
        --kernel-type matern32 --kernel-variance 1.0 --spatial-lengthscale 1.0 \
        --spatial-inducing-selection linspace --prediction-mode streaming_sylvester \
        --prediction-chunk-size 8192 --skip-heldout-block-pairs --save-per-location-predictions \
        > "$online/stdout.log" 2> "$online/stderr.log"
  fi
done
