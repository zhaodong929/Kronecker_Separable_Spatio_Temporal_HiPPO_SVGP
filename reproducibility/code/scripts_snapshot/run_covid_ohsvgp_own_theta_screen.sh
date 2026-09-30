#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${ROOT}/data/epidemiology/protocol/covid}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/diagnostics/covid_ohsvgp_own_theta}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

cd "${ROOT}"
for kernel in rbf spectral_mixture_q2; do
  output="${RESULT_ROOT}/${kernel}/seed${SEED}"
  mkdir -p "${output}"
  command=(
    "${PY}" scripts/run_covid_ohsvgp_own_theta.py
    --protocol-npz "${PROTOCOL_ROOT}/seed${SEED}/protocol.npz"
    --protocol-json "${PROTOCOL_ROOT}/seed${SEED}/protocol.json"
    --output-dir "${output}"
    --kernel "${kernel}"
    --inducing-size 32 --rff-sample-size 64
    --calibration-iterations 1000 --calibration-batch-size 128
    --validation-every 5 --early-stopping-patience-validations 20
    --learning-rate 0.001 --update-steps 1
    --seed "${SEED}" --device "${DEVICE}" --dtype float64
  )
  if [[ "${kernel}" == "spectral_mixture_q2" ]]; then
    command+=(--spectral-mixture-json "${ROOT}/configs/covid_sm_q2.json")
  fi
  "${command[@]}" >"${output}/run.log" 2>&1
done
