#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/kronecker+s2vgp/ICLR Formal experiment/iclr_era5_stage2plus}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/diagnostics/routeb_efficiency_optimization/training}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/era5/processed_timeseries_4_task1_10_extension}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-$ROOT/results/diagnostics/routeb_efficiency_optimization/feature_cache}"

run_one() {
  local scope="$1"
  local objective="$2"
  local version="$3"
  local seed="$4"
  local protocol_scope phi_cache optimization cross block output
  if [[ "$scope" == "task2_short" ]]; then
    protocol_scope="task1_2"
    phi_cache="$FEATURE_CACHE_ROOT/task2_stream_phi.npy"
  else
    protocol_scope="task1_10"
    phi_cache="$FEATURE_CACHE_ROOT/tasks2_10_stream_phi.npy"
  fi
  if [[ "$version" == "original" ]]; then
    optimization="E0"
    cross="einsum"
    block=133
  else
    optimization="E1"
    cross="auto"
    block=133
  fi
  output="$OUTPUT_ROOT/$scope/$objective/$version/seed$seed"
  if [[ -s "$output/result.json" ]]; then
    echo "SKIP scope=$scope objective=$objective version=$version seed=$seed"
    return
  fi
  mkdir -p "$output"
  echo "RUN scope=$scope objective=$objective version=$version seed=$seed"
  "$PY" "$ROOT/scripts/run_iclr_era5_routeb_batch.py" \
    --protocol-npz "$BENCHMARK_ROOT/protocol/$protocol_scope/seed$seed/protocol.npz" \
    --protocol-json "$BENCHMARK_ROOT/protocol/$protocol_scope/seed$seed/protocol.json" \
    --data-root "$DATA_ROOT" \
    --joint-phi-npy "$phi_cache" \
    --output-dir "$output" \
    --predictions-output "$output/predictions.npz" \
    --data-part stream \
    --target-mode joint_xlag \
    --representation analytic_hippo_rff \
    --mt 128 --ms 128 --rff-sample-size 256 \
    --iterations 100 --learning-rate 0.02 --validation-every 5 \
    --training-objective "$objective" \
    --include-conditional-residual-variance \
    --beta-prior-variance 1000 \
    --xlag-length 10 \
    --split-seed "$seed" --model-seed 0 \
    --device cuda --dtype float64 --evaluation-backend torch \
    --warmup-steps 10 \
    --objective-optimization-version "$optimization" \
    --cross-contraction "$cross" \
    --feature-block-size "$block" \
    >"$output/run.log" 2>&1
  echo "DONE scope=$scope objective=$objective version=$version seed=$seed"
}

cd "$ROOT"
for scope in task2_short tasks2_10_long; do
  for objective in finite_dtc vfe; do
    for version in original optimized; do
      for seed in 0 1 2 3 4; do
        run_one "$scope" "$objective" "$version" "$seed"
      done
    done
  done
done
