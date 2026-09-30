#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
OUT="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison/phase_m_online_temporal_backend_comparison"
PY="$ROOT/.venv/bin/python"

cd "$ROOT"
mkdir -p "$OUT"

run_one() {
  local mt=$1 ms=$2 split_seed=$3 backend_label=$4
  local backend=$backend_label
  local inducing_mode=moving
  if [[ "$backend_label" == "inducing_points_global" ]]; then
    backend=inducing_points
    inducing_mode=global
  fi
  local run="$OUT/Mt${mt}_Ms${ms}/block10/seed${split_seed}/${backend_label}"
  mkdir -p "$run"
  if [[ -s "$run/era5_routeb_report.json" ]]; then
    echo "SKIP completed $run"
    return
  fi

  echo "RUN $run"
  /usr/bin/time -v -o "$run/resource_usage.txt" \
    "$PY" scripts/run_hipposvgp_era5_routeb.py \
      --outdir "$run" \
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
      --heldout-split-seeds "$split_seed" \
      --seeds 0 \
      --mt "$mt" \
      --ms "$ms" \
      --temporal-backend "$backend" \
      --temporal-inducing-mode "$inducing_mode" \
      --temporal-rff-sample-size 256 \
      --temporal-rff-seed 0 \
      --model-ell-t 0.05 \
      --ell-t-fit-mode none \
      --routeb-noise 0.1 \
      --kernel-type rbf \
      --kernel-variance 1.0 \
      --spatial-lengthscale 0.35 \
      --spatial-inducing-selection linspace \
      --prediction-mode streaming_sylvester \
      --prediction-chunk-size 8192 \
      --skip-heldout-block-pairs \
      --save-per-location-predictions \
      > "$run/stdout.log" 2> "$run/stderr.log"
}

if [[ $# -eq 4 ]]; then
  run_one "$1" "$2" "$3" "$4"
  exit 0
fi

for capacity in "8 64" "32 128"; do
  read -r mt ms <<< "$capacity"
  for split_seed in 0 1 2; do
    for backend in analytic_hippo_rff inducing_points inducing_points_global; do
      run_one "$mt" "$ms" "$split_seed" "$backend"
    done
  done
done
