#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zd929/projects/stvgp_kronecker}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/kronecker+s2vgp/ICLR Formal experiment/iclr_era5_stage2plus}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/era5/processed_timeseries_4_task1_10_extension}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/diagnostics/routeb_task1_10_vfe_budget_comparison}"
PREVIOUS_ROOT="${PREVIOUS_ROOT:-$ROOT/results/diagnostics/routeb_task1_10_vfe_training}"
DTC_DIAGNOSTIC_ROOT="${DTC_DIAGNOSTIC_ROOT:-$ROOT/results/diagnostics/routeb_task1_10_variance}"
PY="${PY:-$ROOT/.venv/bin/python}"

run_calibration() {
  local protocol_name="$1"
  local objective="$2"
  local seed="$3"
  local protocol="$BENCHMARK_ROOT/protocol/task1_10/seed$seed"
  local output="$OUTPUT_ROOT/calibration/$protocol_name/$objective/seed$seed"
  if [[ -s "$output/result.json" ]]; then
    echo "SKIP calibration protocol=$protocol_name objective=$objective seed=$seed"
    return
  fi
  mkdir -p "$output"
  local stopping_args=()
  if [[ "$protocol_name" == "converged" ]]; then
    stopping_args=(
      --iterations 250
      --early-stopping-patience-validations 8
      --early-stopping-min-delta 0.0001
    )
  else
    stopping_args=(--iterations 1000 --max-training-seconds 60)
  fi
  echo "RUN calibration protocol=$protocol_name objective=$objective seed=$seed"
  "$PY" "$ROOT/scripts/run_iclr_era5_routeb_batch.py" \
    --protocol-npz "$protocol/protocol.npz" \
    --protocol-json "$protocol/protocol.json" \
    --data-root "$DATA_ROOT" \
    --output-dir "$output" \
    --data-part calibration \
    --target-mode joint_xlag \
    --representation analytic_hippo_rff \
    --mt 128 --ms 128 \
    --learning-rate 0.02 --validation-every 5 \
    --beta-prior-variance 1000 --rff-sample-size 256 --xlag-length 10 \
    --split-seed "$seed" --model-seed 0 \
    --device cuda --dtype float64 --evaluation-backend torch \
    --warmup-steps 1 --training-objective "$objective" \
    --include-conditional-residual-variance \
    "${stopping_args[@]}" \
    >"$output/run.log" 2>&1
  echo "DONE calibration protocol=$protocol_name objective=$objective seed=$seed"
}

run_online() {
  local protocol_name="$1"
  local objective="$2"
  local seed="$3"
  local protocol="$BENCHMARK_ROOT/protocol/task1_10/seed$seed"
  local calibration="$OUTPUT_ROOT/calibration/$protocol_name/$objective/seed$seed/result.json"
  local output="$OUTPUT_ROOT/online/$protocol_name/$objective/seed$seed"
  if [[ -s "$output/result.json" && -s "$output/predictions.npz" ]]; then
    echo "SKIP online protocol=$protocol_name objective=$objective seed=$seed"
    return
  fi
  mkdir -p "$output"
  echo "RUN online protocol=$protocol_name objective=$objective seed=$seed"
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
  echo "DONE online protocol=$protocol_name objective=$objective seed=$seed"
}

cd "$ROOT"
for protocol_name in converged wallclock_60s; do
  for objective in finite_dtc vfe; do
    for seed in 0 1 2 3 4; do
      run_calibration "$protocol_name" "$objective" "$seed"
    done
    for seed in 0 1 2 3 4; do
      run_online "$protocol_name" "$objective" "$seed"
    done
  done
done

"$PY" -m scripts.summarize_routeb_vfe_budget_comparison \
  --benchmark-root "$BENCHMARK_ROOT" \
  --experiment-root "$OUTPUT_ROOT" \
  --previous-root "$PREVIOUS_ROOT" \
  --dtc-diagnostic-root "$DTC_DIAGNOSTIC_ROOT"
