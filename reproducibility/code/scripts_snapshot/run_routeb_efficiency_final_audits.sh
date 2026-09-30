#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/kronecker+s2vgp/ICLR Formal experiment/iclr_era5_stage2plus}"
DIAG_ROOT="${DIAG_ROOT:-$ROOT/results/diagnostics/routeb_efficiency_optimization}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/era5/processed_timeseries_4_task1_10_extension}"
CAL_ROOT="${CAL_ROOT:-$ROOT/results/diagnostics/routeb_task1_10_vfe_budget_comparison/calibration/converged}"

completed="$(find "$DIAG_ROOT/training" -name result.json | wc -l)"
if [[ "$completed" -ne 40 ]]; then
  echo "Expected 40 complete training results, found $completed; resume the training matrix first." >&2
  exit 2
fi

"$PY" "$ROOT/scripts/summarize_routeb_efficiency_training.py" \
  --root "$DIAG_ROOT/training" \
  --output "$DIAG_ROOT/summary"

run_batch_profile() {
  local scope="$1" objective="$2" variant="$3" protocol_scope output theta version cross
  if [[ "$scope" == "task2_short" ]]; then
    protocol_scope="task1_2"
  else
    protocol_scope="task1_10"
  fi
  if [[ "$variant" == "original" ]]; then
    version="E0"
    cross="einsum"
  else
    version="E1"
    cross="auto"
  fi
  output="$DIAG_ROOT/batch_final/$scope/seed0/${objective}_${variant}"
  theta="$DIAG_ROOT/training/$scope/$objective/original/seed0/result.json"
  if [[ -s "$output/batch_ablation.json" ]]; then
    echo "SKIP batch profile $scope $objective $variant"
    return
  fi
  mkdir -p "$output"
  echo "RUN batch profile $scope $objective $variant"
  "$PY" "$ROOT/scripts/benchmark_routeb_batch_objective.py" \
    --protocol-npz "$BENCHMARK_ROOT/protocol/$protocol_scope/seed0/protocol.npz" \
    --protocol-json "$BENCHMARK_ROOT/protocol/$protocol_scope/seed0/protocol.json" \
    --data-root "$DATA_ROOT" \
    --output-dir "$output" \
    --scope "$scope" --seed 0 --objective "$objective" \
    --versions "$version" --theta-json "$theta" \
    --mt 128 --ms 128 --rff-sample-size 256 \
    --warmup 10 --repeats 30 \
    --cross-contraction "$cross" --feature-block-size 133 \
    >"$output/run.log" 2>&1
}

for scope in task2_short tasks2_10_long; do
  for objective in finite_dtc vfe; do
    run_batch_profile "$scope" "$objective" original
    run_batch_profile "$scope" "$objective" optimized
  done
done

run_online_audit() {
  local scope="$1" objective="$2" seed="$3" protocol_scope output theta
  local -a profile_option=()
  if [[ "$scope" == "task2_short" ]]; then
    protocol_scope="task1_2"
  else
    protocol_scope="task1_10"
  fi
  output="$DIAG_ROOT/online/$scope/$objective/seed$seed"
  theta="$CAL_ROOT/$objective/seed$seed/result.json"
  if [[ -s "$output/online_efficiency.json" ]]; then
    echo "SKIP online audit $scope $objective seed$seed"
    return
  fi
  # The posterior recursion and tensor shapes do not depend on whether Task-1
  # theta came from DTC or VFE. Profile only DTC seed 0; time every run.
  if [[ "$objective" != "finite_dtc" || "$seed" -ne 0 ]]; then
    profile_option+=(--skip-profile)
  fi
  mkdir -p "$output"
  echo "RUN online audit $scope $objective seed$seed"
  "$PY" "$ROOT/scripts/benchmark_routeb_online_efficiency.py" \
    --protocol-npz "$BENCHMARK_ROOT/protocol/$protocol_scope/seed$seed/protocol.npz" \
    --protocol-json "$BENCHMARK_ROOT/protocol/$protocol_scope/seed$seed/protocol.json" \
    --data-root "$DATA_ROOT" --theta-json "$theta" \
    --output-dir "$output" --scope "$scope" \
    --objective-source "$objective" --seed "$seed" \
    --mt 128 --ms 128 --num-features 133 \
    --rff-sample-size 256 --temporal-factor-device auto \
    "${profile_option[@]}" \
    >"$output/audit.log" 2>&1
}

for scope in task2_short tasks2_10_long; do
  for objective in finite_dtc vfe; do
    for seed in 0 1 2 3 4; do
      run_online_audit "$scope" "$objective" "$seed"
    done
  done
done

echo "Final Route-B efficiency audits complete under $DIAG_ROOT"
