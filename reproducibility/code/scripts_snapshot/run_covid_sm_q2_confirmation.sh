#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${ROOT}/data/epidemiology/protocol/covid}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/diagnostics/covid_method_optimization/spectral_mixture_q2}"
BASELINE_ROOT="${BASELINE_ROOT:-${ROOT}/results/diagnostics/covid_optimization/mt32_ms32_rff64_1000}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2 3 4}"

cd "${ROOT}"
for seed in ${SEEDS}; do
  output="${RESULT_ROOT}/seed${seed}"
  result="${output}/cumulative_hippo/online/result.json"
  if [[ -f "${result}" ]]; then
    echo "SKIP complete seed${seed}: ${result}"
    continue
  fi
  mkdir -p "${output}"
  "${PY}" scripts/run_epidemiology_pilot.py \
    --protocol-npz "${PROTOCOL_ROOT}/seed${seed}/protocol.npz" \
    --protocol-json "${PROTOCOL_ROOT}/seed${seed}/protocol.json" \
    --output-root "${output}" \
    --representations cumulative_hippo \
    --mt 32 --ms 32 \
    --iterations 1000 --learning-rate 0.02 --rff-sample-size 64 \
    --validation-every 5 --early-stopping-patience-validations 20 \
    --temporal-kernel spectral_mixture \
    --spectral-mixture-json "${ROOT}/configs/covid_sm_q2.json" \
    --device "${DEVICE}" >"${output}/run.log" 2>&1
done

"${PY}" scripts/summarize_covid_method_candidate.py \
  --candidate-root "${RESULT_ROOT}" \
  --baseline-root "${BASELINE_ROOT}" \
  --candidate-name "causal Mt32 Ms32 fixed spectral mixture Q=2" \
  --seeds ${SEEDS}
