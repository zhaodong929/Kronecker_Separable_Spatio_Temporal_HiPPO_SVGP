#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${ROOT}/data/epidemiology/protocol/covid}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/diagnostics/covid_ohsvgp_own_theta/convergence_2500/rbf}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2 3 4}"

cd "${ROOT}"
for seed in ${SEEDS}; do
  output="${RESULT_ROOT}/seed${seed}"
  if [[ -f "${output}/result.json" ]]; then
    echo "SKIP complete seed=${seed}"
    continue
  fi
  mkdir -p "${output}"
  "${PY}" scripts/run_covid_ohsvgp_own_theta.py \
    --protocol-npz "${PROTOCOL_ROOT}/seed${seed}/protocol.npz" \
    --protocol-json "${PROTOCOL_ROOT}/seed${seed}/protocol.json" \
    --output-dir "${output}" \
    --kernel rbf \
    --inducing-size 32 --rff-sample-size 64 \
    --calibration-iterations 2500 --calibration-batch-size 128 \
    --validation-every 5 --early-stopping-patience-validations 20 \
    --learning-rate 0.001 --update-steps 1 \
    --seed "${seed}" --device "${DEVICE}" --dtype float64 \
    >"${output}/run.log" 2>&1
  echo "COMPLETE seed=${seed}"
done
