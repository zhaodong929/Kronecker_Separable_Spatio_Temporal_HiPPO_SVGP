#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
OUT="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/post_meeting_priority_2026-07-14/p1_fair_matrix"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
mkdir -p "$OUT"

run_local() {
  local architecture=$1 protocol=$2 mt=$3 ms=$4 block=$5 split_seed=$6
  local run="$OUT/Mt${mt}_Ms${ms}/block${block}/seed${split_seed}/${architecture}_${protocol}"
  mkdir -p "$run"
  if [[ -s "$run/run_metadata.json" ]]; then
    echo "SKIP completed $run"
    return
  fi
  echo "RUN $run"
  /usr/bin/time -v -o "$run/resource_usage.txt" \
    "$PY" scripts/run_post_meeting_p1_matrix.py \
      --architecture "$architecture" --protocol "$protocol" --outdir "$run" \
      --split-seed "$split_seed" --block-size "$block" --mt "$mt" --ms "$ms" \
      --xlag-length 10 --ell-t 0.05 --spatial-lengthscale 0.35 --noise 0.1 \
      --spatial-inducing-selection linspace --temporal-representation analytic_hippo_rff \
      > "$run/stdout.log" 2> "$run/stderr.log"
}

run_structured_online() {
  local mt=$1 ms=$2 block=$3 split_seed=$4
  local run="$OUT/Mt${mt}_Ms${ms}/block${block}/seed${split_seed}/structured_joint_online"
  mkdir -p "$run"
  if [[ -s "$run/era5_routeb_report.json" ]]; then
    echo "SKIP completed $run"
    return
  fi
  echo "RUN $run"
  /usr/bin/time -v -o "$run/resource_usage.txt" \
    "$PY" scripts/run_hipposvgp_era5_routeb.py \
      --outdir "$run" --root data/era5/processed_timeseries_4 \
      --calibration-tasks task_1 --online-tasks task_2 --variable-index 0 --split all \
      --block-size "$block" --routeb-methods structured_joint --eval-modes seen_history \
      --phi-mode medium_era5_xlag --xlag-length 10 \
      --ohsvgp-heldout-eval --heldout-split-seeds "$split_seed" --seeds 0 \
      --mt "$mt" --ms "$ms" --temporal-backend analytic_hippo_rff \
      --temporal-rff-sample-size 256 --temporal-rff-seed 0 \
      --model-ell-t 0.05 --ell-t-fit-mode none --routeb-noise 0.1 \
      --kernel-type rbf --kernel-variance 1.0 --spatial-lengthscale 0.35 \
      --spatial-inducing-selection linspace --prediction-mode streaming_sylvester \
      --prediction-chunk-size 8192 --skip-heldout-block-pairs --save-per-location-predictions \
      > "$run/stdout.log" 2> "$run/stderr.log"
}

for capacity in "8 64" "32 128"; do
  read -r mt ms <<< "$capacity"
  for block in 5 10 20; do
    for split_seed in 0 1 2; do
      run_local matched_sparse_stvgp batch "$mt" "$ms" "$block" "$split_seed"
      run_local matched_sparse_stvgp online "$mt" "$ms" "$block" "$split_seed"
      run_local structured_joint batch "$mt" "$ms" "$block" "$split_seed"
      run_structured_online "$mt" "$ms" "$block" "$split_seed"
    done
  done
done
