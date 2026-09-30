#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zd929/projects/stvgp_kronecker}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/kronecker+s2vgp/ICLR Formal experiment/iclr_era5_stage2plus}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/era5/processed_timeseries_4_task1_10_extension}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/diagnostics/routeb_task1_10_vfe_training}"
PY="${PY:-$ROOT/.venv/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

run_calibration() {
  local seed="$1"
  local protocol="$BENCHMARK_ROOT/protocol/task1_10/seed$seed"
  local output="$OUTPUT_ROOT/calibration/seed$seed"
  if [[ -s "$output/result.json" ]]; then
    echo "SKIP calibration seed=$seed"
    return
  fi
  mkdir -p "$output"
  echo "RUN calibration seed=$seed"
  "$PY" "$ROOT/scripts/run_iclr_era5_routeb_batch.py" \
    --protocol-npz "$protocol/protocol.npz" \
    --protocol-json "$protocol/protocol.json" \
    --data-root "$DATA_ROOT" \
    --output-dir "$output" \
    --data-part calibration \
    --target-mode joint_xlag \
    --representation analytic_hippo_rff \
    --mt 128 --ms 128 \
    --iterations 100 --learning-rate 0.02 --validation-every 5 \
    --beta-prior-variance 1000 --rff-sample-size 256 --xlag-length 10 \
    --split-seed "$seed" --model-seed 0 \
    --device cuda --dtype float64 --evaluation-backend torch \
    --warmup-steps 1 --training-objective vfe \
    >"$output/run.log" 2>&1
  echo "DONE calibration seed=$seed"
}

run_online() {
  local seed="$1"
  local protocol="$BENCHMARK_ROOT/protocol/task1_10/seed$seed"
  local calibration="$OUTPUT_ROOT/calibration/seed$seed/result.json"
  local output="$OUTPUT_ROOT/online/seed$seed"
  if [[ -s "$output/result.json" && -s "$output/predictions.npz" ]]; then
    echo "SKIP online seed=$seed"
    return
  fi
  mkdir -p "$output"
  echo "RUN online seed=$seed"
  "$PY" "$ROOT/scripts/run_iclr_era5_routeb_strict_online.py" \
    --protocol-npz "$protocol/protocol.npz" \
    --protocol-json "$protocol/protocol.json" \
    --data-root "$DATA_ROOT" \
    --theta-json "$calibration" \
    --output "$output/result.json" \
    --blockwise-output "$output/blocks.csv" \
    --predictions-output "$output/predictions.npz" \
    --representation analytic_hippo_rff \
    --mt 128 --ms 128 --rff-sample-size 256 \
    --prediction-chunk-size 8192 --beta-prior-variance 1000 \
    --seed "$seed" --solver-backend torch \
    --device cuda --dtype float64 --temporal-factor-device cpu \
    --include-conditional-residual-variance \
    >"$output/run.log" 2>&1
  echo "DONE online seed=$seed"
}

run_parallel() {
  local function_name="$1"
  local active=0
  for seed in 0 1 2 3 4; do
    "$function_name" "$seed" &
    active=$((active + 1))
    if (( active >= MAX_PARALLEL )); then
      wait -n
      active=$((active - 1))
    fi
  done
  wait
}

cd "$ROOT"
run_parallel run_calibration
run_parallel run_online
"$PY" "$ROOT/scripts/summarize_routeb_online_vfe_training.py" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --experiment-root "$OUTPUT_ROOT" \
  --dtc-diagnostic-root "$ROOT/results/diagnostics/routeb_task1_10_variance"
