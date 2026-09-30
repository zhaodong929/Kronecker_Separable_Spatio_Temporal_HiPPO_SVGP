#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
BASE="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
PHASE="$BASE/phase_o_routeb_vfe_empirical_bayes"
OUT="$PHASE/task2_vfe_250"
PY="$ROOT/.venv/bin/python"

for representation in analytic_hippo_rff inducing_points; do
  source_run="$PHASE/convergence_probe_250/$representation/seed0"
  target_run="$OUT/$representation/seed0"
  if [[ ! -s "$target_run/result.json" ]]; then
    mkdir -p "$target_run"
    cp -a "$source_run/." "$target_run/"
  fi
done

run_one() {
  local representation=$1
  local seed=$2
  local run="$OUT/$representation/seed$seed"
  local data="$BASE/phase_d_joint_xlag_controlled/seed$seed/era5_xlag_seed$seed.npz"
  mkdir -p "$run"
  if [[ -s "$run/result.json" ]]; then
    echo "SKIP vfe-250 $representation seed=$seed"
    return
  fi
  echo "RUN vfe-250 $representation seed=$seed"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    /usr/bin/time -v -o "$run/resource_usage.txt" \
    "$PY" scripts/run_routeb_batch_empirical_bayes.py \
      --outdir "$run" --controlled-npz "$data" \
      --fit-task task_2 --split-seed "$seed" \
      --representation "$representation" --mt 128 --ms 128 \
      --iterations 250 --validation-every 5 \
      --training-objective vfe \
      > "$run/stdout.log" 2> "$run/stderr.log"
  echo "DONE vfe-250 $representation seed=$seed"
}

cd "$ROOT"
active=0
for representation in analytic_hippo_rff inducing_points; do
  for seed in 1 2; do
    run_one "$representation" "$seed" &
    active=$((active + 1))
    if (( active >= 2 )); then
      wait -n
      active=$((active - 1))
    fi
  done
done
wait

