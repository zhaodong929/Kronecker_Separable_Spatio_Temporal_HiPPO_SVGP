#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
BASE="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
OUT="$BASE/phase_o_routeb_vfe_empirical_bayes/convergence_probe_250"
DATA="$BASE/phase_d_joint_xlag_controlled/seed0/era5_xlag_seed0.npz"
PY="$ROOT/.venv/bin/python"

run_one() {
  local representation=$1
  local run="$OUT/$representation/seed0"
  mkdir -p "$run"
  "$PY" scripts/run_routeb_batch_empirical_bayes.py \
    --outdir "$run" --controlled-npz "$DATA" \
    --fit-task task_2 --split-seed 0 \
    --representation "$representation" --mt 128 --ms 128 \
    --iterations 250 --validation-every 5 --training-objective vfe \
    > "$run/stdout.log" 2> "$run/stderr.log"
}

cd "$ROOT"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 run_one analytic_hippo_rff &
pid_hippo=$!
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 run_one inducing_points &
pid_inducing=$!
wait "$pid_hippo"
wait "$pid_inducing"

