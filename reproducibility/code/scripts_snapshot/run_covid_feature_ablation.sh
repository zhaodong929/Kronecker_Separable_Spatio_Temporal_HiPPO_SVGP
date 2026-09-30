#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/data/epidemiology/protocol/covid}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${ROOT}/data/epidemiology/protocol/covid_feature_ablation}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/diagnostics/covid_abcd/D_feature_ablation}"
SEEDS="${SEEDS:-0}"
DEVICE="${DEVICE:-cuda}"

cd "${ROOT}"
for mode in zero_mean intercept_only; do
  for seed in ${SEEDS}; do
    protocol_dir="${PROTOCOL_ROOT}/${mode}/seed${seed}"
    output="${RESULT_ROOT}/${mode}/seed${seed}"
    mkdir -p "${protocol_dir}" "${output}"
    "${PY}" scripts/build_covid_feature_ablation_protocol.py \
      --source-npz "${SOURCE_ROOT}/seed${seed}/protocol.npz" \
      --source-json "${SOURCE_ROOT}/seed${seed}/protocol.json" \
      --output "${protocol_dir}/protocol.npz" \
      --mode "${mode}"
    "${PY}" scripts/run_epidemiology_pilot.py \
      --protocol-npz "${protocol_dir}/protocol.npz" \
      --protocol-json "${protocol_dir}/protocol.json" \
      --output-root "${output}" \
      --mt 16 \
      --ms 32 \
      --iterations 500 \
      --validation-every 5 \
      --early-stopping-patience-validations 10 \
      --device "${DEVICE}" \
      >"${output}/run.log" 2>&1
  done
done

printf 'COVID feature ablation complete under %s\n' "${RESULT_ROOT}"
