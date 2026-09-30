#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${ROOT}/data/epidemiology/protocol/covid}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/diagnostics/covid_abcd/C_final_mt16_ms32_500}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DEVICE="${DEVICE:-cuda}"

cd "${ROOT}"
for seed in ${SEEDS}; do
  output="${RESULT_ROOT}/seed${seed}"
  mkdir -p "${output}"
  "${PY}" scripts/run_epidemiology_pilot.py \
    --protocol-npz "${PROTOCOL_ROOT}/seed${seed}/protocol.npz" \
    --protocol-json "${PROTOCOL_ROOT}/seed${seed}/protocol.json" \
    --output-root "${output}" \
    --mt 16 \
    --ms 32 \
    --iterations 500 \
    --validation-every 5 \
    --early-stopping-patience-validations 10 \
    --device "${DEVICE}" \
    >"${output}/run.log" 2>&1
done

printf 'COVID final capacity runs complete under %s\n' "${RESULT_ROOT}"
