#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
OUT="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/post_meeting_priority_2026-07-14/p3_hippo_ablation"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
mkdir -p "$OUT"

for capacity in "8 64" "32 128"; do
  read -r mt ms <<< "$capacity"
  for split_seed in 0 1 2; do
    for representation in analytic_hippo_rff inducing_points ordinary_rff full_temporal_kernel; do
      run="$OUT/Mt${mt}_Ms${ms}/seed${split_seed}/${representation}"
      mkdir -p "$run"
      if [[ -s "$run/run_metadata.json" ]]; then
        echo "SKIP completed $run"
        continue
      fi
      echo "RUN $run"
      /usr/bin/time -v -o "$run/resource_usage.txt" \
        "$PY" scripts/run_post_meeting_p1_matrix.py \
          --architecture structured_joint --protocol batch --outdir "$run" \
          --split-seed "$split_seed" --block-size 10 --mt "$mt" --ms "$ms" \
          --xlag-length 10 --ell-t 0.05 --spatial-lengthscale 0.35 --noise 0.1 \
          --kernel-type rbf --spatial-inducing-selection linspace \
          --temporal-representation "$representation" --final-block-only \
          > "$run/stdout.log" 2> "$run/stderr.log"
    done
  done
done
