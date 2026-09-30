#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
BASE="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
OUT="$BASE/phase_o_routeb_vfe_empirical_bayes"
PY="$ROOT/.venv/bin/python"

run_one() {
  local representation=$1
  local seed=$2
  local run="$OUT/task2_vfe/$representation/seed$seed"
  local data="$BASE/phase_d_joint_xlag_controlled/seed$seed/era5_xlag_seed$seed.npz"
  mkdir -p "$run"
  if [[ -s "$run/result.json" ]]; then
    echo "SKIP vfe $representation seed=$seed"
    return
  fi
  echo "RUN vfe $representation seed=$seed"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    /usr/bin/time -v -o "$run/resource_usage.txt" \
    "$PY" scripts/run_routeb_batch_empirical_bayes.py \
      --outdir "$run" --controlled-npz "$data" \
      --fit-task task_2 --split-seed "$seed" \
      --representation "$representation" --mt 128 --ms 128 \
      --iterations 100 --validation-every 5 \
      --training-objective vfe \
      > "$run/stdout.log" 2> "$run/stderr.log"
  echo "DONE vfe $representation seed=$seed"
}

cd "$ROOT"
active=0
for representation in analytic_hippo_rff inducing_points; do
  for seed in 0 1 2; do
    run_one "$representation" "$seed" &
    active=$((active + 1))
    if (( active >= 2 )); then
      wait -n
      active=$((active - 1))
    fi
  done
done
wait

