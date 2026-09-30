#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
RAW_ROOT="${RAW_ROOT:-${ROOT}/data/epidemiology/raw/covid}"
BASE_PROTOCOL_ROOT="${BASE_PROTOCOL_ROOT:-${ROOT}/data/epidemiology/protocol/covid}"
FEATURE_PROTOCOL_ROOT="${FEATURE_PROTOCOL_ROOT:-${ROOT}/data/epidemiology/protocol/covid_lag_dynamics}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/diagnostics/covid_method_optimization}"
DEVICE="${DEVICE:-cuda}"
SEED=0

cd "${ROOT}"

feature_protocol="${FEATURE_PROTOCOL_ROOT}/seed${SEED}"
mkdir -p "${feature_protocol}"
"${PY}" scripts/build_epidemiology_protocol.py \
  --dataset covid \
  --raw-root "${RAW_ROOT}" \
  --output "${feature_protocol}/protocol.npz" \
  --seed "${SEED}" \
  --calibration-steps 52 \
  --block-size 1 \
  --inducing-sizes 32 \
  --feature-mode lag_dynamics

run_candidate() {
  local name="$1"
  local protocol_root="$2"
  local temporal_kernel="$3"
  local mixture_json="${4:-}"
  local output="${RESULT_ROOT}/${name}/seed${SEED}"
  mkdir -p "${output}"
  local command=(
    "${PY}" scripts/run_epidemiology_pilot.py
    --protocol-npz "${protocol_root}/seed${SEED}/protocol.npz"
    --protocol-json "${protocol_root}/seed${SEED}/protocol.json"
    --output-root "${output}"
    --representations cumulative_hippo
    --mt 32 --ms 32
    --iterations 1000 --learning-rate 0.02 --rff-sample-size 64
    --validation-every 5 --early-stopping-patience-validations 20
    --temporal-kernel "${temporal_kernel}"
    --device "${DEVICE}"
  )
  if [[ -n "${mixture_json}" ]]; then
    command+=(--spectral-mixture-json "${mixture_json}")
  fi
  "${command[@]}" >"${output}/run.log" 2>&1
}

run_candidate lag_dynamics_matern32 "${FEATURE_PROTOCOL_ROOT}" matern32
run_candidate spectral_mixture_q2 "${BASE_PROTOCOL_ROOT}" spectral_mixture "${ROOT}/configs/covid_sm_q2.json"
run_candidate spectral_mixture_q3 "${BASE_PROTOCOL_ROOT}" spectral_mixture "${ROOT}/configs/covid_sm_q3.json"

"${PY}" scripts/summarize_covid_method_screen.py \
  --baseline "${ROOT}/results/diagnostics/covid_optimization/mt32_ms32_rff64_1000/seed0/cumulative_hippo/online/result.json" \
  --candidate-root "${RESULT_ROOT}" \
  --candidates lag_dynamics_matern32 spectral_mixture_q2 spectral_mixture_q3
