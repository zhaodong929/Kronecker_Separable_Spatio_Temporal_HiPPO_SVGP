#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
BASE="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
OUT="$BASE/phase_n_routeb_compute_and_model_ablation"
PY="$ROOT/.venv/bin/python"

run_one() {
  local mean_mode=$1 seed=$2
  local run="$OUT/$mean_mode/seed$seed"
  local data="$BASE/phase_d_joint_xlag_controlled/seed$seed/era5_xlag_seed$seed.npz"
  mkdir -p "$run"
  if [[ -s "$run/result.json" ]]; then
    echo "SKIP mean_mode=$mean_mode seed=$seed"
    return
  fi
  echo "RUN mean_mode=$mean_mode seed=$seed"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    /usr/bin/time -v -o "$run/resource_usage.txt" \
    "$PY" scripts/run_routeb_batch_empirical_bayes.py \
      --outdir "$run" --controlled-npz "$data" \
      --fit-task task_2 --split-seed "$seed" \
      --representation inducing_points --mt 128 --ms 128 \
      --mean-mode "$mean_mode" --xlag-ridge 1e-3 \
      --iterations 100 --validation-every 5 \
      > "$run/stdout.log" 2> "$run/stderr.log"
  echo "DONE mean_mode=$mean_mode seed=$seed"
}

cd "$ROOT"
active=0
for mean_mode in zero residual_xlag; do
  for seed in 0 1 2; do
    run_one "$mean_mode" "$seed" &
    active=$((active + 1))
    if (( active >= 2 )); then
      wait -n
      active=$((active - 1))
    fi
  done
done
wait

"$PY" scripts/summarize_routeb_compute_and_model_ablation.py

