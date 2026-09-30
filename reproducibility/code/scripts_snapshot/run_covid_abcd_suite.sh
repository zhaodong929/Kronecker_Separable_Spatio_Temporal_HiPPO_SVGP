#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${ROOT}/data/epidemiology/protocol/covid}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/diagnostics/covid_abcd}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

cd "${ROOT}"

run_pilot() {
  local output_root="$1"
  local mt="$2"
  local ms="$3"
  local iterations="$4"
  local patience="$5"
  mkdir -p "${output_root}/seed${SEED}"
  "${PY}" scripts/run_epidemiology_pilot.py \
    --protocol-npz "${PROTOCOL_ROOT}/seed${SEED}/protocol.npz" \
    --protocol-json "${PROTOCOL_ROOT}/seed${SEED}/protocol.json" \
    --output-root "${output_root}/seed${SEED}" \
    --mt "${mt}" \
    --ms "${ms}" \
    --iterations "${iterations}" \
    --validation-every 5 \
    --early-stopping-patience-validations "${patience}" \
    --device "${DEVICE}" \
    >"${output_root}/seed${SEED}/run.log" 2>&1
}

run_pilot "${RESULT_ROOT}/A_joint_100" 16 8 100 10
run_pilot "${RESULT_ROOT}/A_joint_250" 16 8 250 10
run_pilot "${RESULT_ROOT}/C_joint_mt16_ms8" 16 8 250 10
run_pilot "${RESULT_ROOT}/C_joint_mt16_ms16" 16 16 250 10
run_pilot "${RESULT_ROOT}/C_joint_mt16_ms32" 16 32 250 10
run_pilot "${RESULT_ROOT}/C_joint_mt32_ms16" 32 16 250 10

printf 'COVID A/C seed-%s runs complete under %s\n' "${SEED}" "${RESULT_ROOT}"
