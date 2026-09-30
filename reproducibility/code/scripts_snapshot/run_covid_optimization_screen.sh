#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${ROOT}/data/epidemiology/protocol/covid}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/diagnostics/covid_optimization/screen_seed0}"
DEVICE="${DEVICE:-cuda}"
SEED=0

cd "${ROOT}"

run_candidate() {
  local name="$1"
  local mt="$2"
  local ms="$3"
  local iterations="$4"
  local learning_rate="$5"
  local rff_size="$6"
  local output="${RESULT_ROOT}/${name}/seed${SEED}"
  mkdir -p "${output}"
  "${PY}" scripts/run_epidemiology_pilot.py \
    --protocol-npz "${PROTOCOL_ROOT}/seed${SEED}/protocol.npz" \
    --protocol-json "${PROTOCOL_ROOT}/seed${SEED}/protocol.json" \
    --output-root "${output}" \
    --mt "${mt}" \
    --ms "${ms}" \
    --iterations "${iterations}" \
    --learning-rate "${learning_rate}" \
    --rff-sample-size "${rff_size}" \
    --validation-every 5 \
    --early-stopping-patience-validations 20 \
    --device "${DEVICE}" \
    >"${output}/run.log" 2>&1
}

# Exploratory seed-0 screen. The existing C_final result is untouched.
if [[ "${RUN_EXTRA_ONLY:-0}" != "1" ]]; then
  run_candidate mt16_ms32_1000_lr002 16 32 1000 0.02 64
  run_candidate mt32_ms32_500_lr002 32 32 500 0.02 64
  run_candidate mt16_ms32_1000_lr001 16 32 1000 0.01 64
fi
run_candidate mt32_ms32_1000_lr002 32 32 1000 0.02 64
run_candidate mt32_ms32_500_rff128_lr002 32 32 500 0.02 128

printf 'COVID optimization screen complete under %s\n' "${RESULT_ROOT}"
