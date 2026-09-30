#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/kronecker+s2vgp/ICLR Formal experiment/iclr_era5_stage2plus}"
DIAG_ROOT="${DIAG_ROOT:-$ROOT/results/diagnostics/routeb_efficiency_optimization}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/era5/processed_timeseries_4_task1_10_extension}"
SEED="${SEED:-0}"

run_batch() {
  local data_part="$1" scope="$2" rank="$3" protocol_scope phi_cache output basis
  basis="$DIAG_ROOT/rank/seed$SEED/basis_rank$rank.npz"
  output="$DIAG_ROOT/rank_ablation/$data_part/$scope/rank$rank/seed$SEED"
  if [[ -s "$output/result.json" ]]; then
    echo "SKIP rank=$rank $data_part $scope"
    return
  fi
  mkdir -p "$output"
  if [[ "$scope" == "task2_short" ]]; then
    protocol_scope="task1_2"
    phi_cache="$DIAG_ROOT/feature_cache/task2_stream_phi.npy"
  else
    protocol_scope="task1_10"
    phi_cache="$DIAG_ROOT/feature_cache/tasks2_10_stream_phi.npy"
  fi
  local -a cache_arg=()
  if [[ "$data_part" == "stream" ]]; then
    cache_arg+=(--joint-phi-npy "$phi_cache")
  fi
  echo "RUN rank=$rank $data_part $scope"
  "$PY" "$ROOT/scripts/run_iclr_era5_routeb_batch.py" \
    --protocol-npz "$BENCHMARK_ROOT/protocol/$protocol_scope/seed$SEED/protocol.npz" \
    --protocol-json "$BENCHMARK_ROOT/protocol/$protocol_scope/seed$SEED/protocol.json" \
    --data-root "$DATA_ROOT" "${cache_arg[@]}" \
    --output-dir "$output" --predictions-output "$output/predictions.npz" \
    --data-part "$data_part" --target-mode joint_xlag \
    --representation analytic_hippo_rff --mt 128 --ms 128 --rff-sample-size 256 \
    --iterations 100 --learning-rate 0.02 --validation-every 5 \
    --training-objective finite_dtc --include-conditional-residual-variance \
    --beta-prior-variance 1000 --xlag-length 10 \
    --split-seed "$SEED" --model-seed 0 --device cuda --dtype float64 \
    --evaluation-backend torch --warmup-steps 10 \
    --objective-optimization-version E1 --cross-contraction auto \
    --feature-block-size "$rank" --feature-projection-npz "$basis" \
    >"$output/run.log" 2>&1
}

run_online() {
  local scope="$1" rank="$2" protocol_scope output basis theta
  basis="$DIAG_ROOT/rank/seed$SEED/basis_rank$rank.npz"
  theta="$DIAG_ROOT/rank_ablation/calibration/task2_short/rank$rank/seed$SEED/result.json"
  output="$DIAG_ROOT/rank_ablation/online/$scope/rank$rank/seed$SEED"
  if [[ -s "$output/result.json" ]]; then
    echo "SKIP rank=$rank online $scope"
    return
  fi
  if [[ "$scope" == "task2_short" ]]; then
    protocol_scope="task1_2"
  else
    protocol_scope="task1_10"
  fi
  mkdir -p "$output"
  echo "RUN rank=$rank online $scope"
  "$PY" "$ROOT/scripts/run_iclr_era5_routeb_strict_online.py" \
    --protocol-npz "$BENCHMARK_ROOT/protocol/$protocol_scope/seed$SEED/protocol.npz" \
    --protocol-json "$BENCHMARK_ROOT/protocol/$protocol_scope/seed$SEED/protocol.json" \
    --data-root "$DATA_ROOT" --theta-json "$theta" \
    --output "$output/result.json" --blockwise-output "$output/blocks.csv" \
    --predictions-output "$output/predictions.npz" \
    --representation analytic_hippo_rff --mt 128 --ms 128 --rff-sample-size 256 \
    --include-conditional-residual-variance --beta-prior-variance 1000 \
    --seed "$SEED" --solver-backend torch --device cuda --dtype float64 \
    --temporal-factor-device auto --feature-projection-npz "$basis" \
    >"$output/run.log" 2>&1
}

run_profile() {
  local scope="$1" rank="$2" protocol_scope output basis theta
  basis="$DIAG_ROOT/rank/seed$SEED/basis_rank$rank.npz"
  theta="$DIAG_ROOT/rank_ablation/stream/$scope/rank$rank/seed$SEED/result.json"
  output="$DIAG_ROOT/rank_ablation/profile/$scope/rank$rank/seed$SEED"
  if [[ -s "$output/batch_ablation.json" ]]; then
    echo "SKIP rank=$rank profile $scope"
    return
  fi
  if [[ "$scope" == "task2_short" ]]; then
    protocol_scope="task1_2"
  else
    protocol_scope="task1_10"
  fi
  mkdir -p "$output"
  echo "RUN rank=$rank profile $scope"
  "$PY" "$ROOT/scripts/benchmark_routeb_batch_objective.py" \
    --protocol-npz "$BENCHMARK_ROOT/protocol/$protocol_scope/seed$SEED/protocol.npz" \
    --protocol-json "$BENCHMARK_ROOT/protocol/$protocol_scope/seed$SEED/protocol.json" \
    --data-root "$DATA_ROOT" --output-dir "$output" --scope "$scope" \
    --seed "$SEED" --objective finite_dtc --versions E1 --theta-json "$theta" \
    --mt 128 --ms 128 --rff-sample-size 256 --warmup 10 --repeats 30 \
    --cross-contraction auto --feature-block-size "$rank" \
    --feature-projection-npz "$basis" \
    >"$output/run.log" 2>&1
}

for rank in 133 73 64 48; do
  # Task-1 is identical in both protocol scopes; store one calibration result.
  run_batch calibration task2_short "$rank"
  for scope in task2_short tasks2_10_long; do
    run_batch stream "$scope" "$rank"
    run_online "$scope" "$rank"
    run_profile "$scope" "$rank"
  done
done

echo "Feature-rank ablation complete under $DIAG_ROOT/rank_ablation"
