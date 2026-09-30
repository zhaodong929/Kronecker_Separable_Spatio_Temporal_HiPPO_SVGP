#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
RAW_ROOT="${RAW_ROOT:-${ROOT}/data/epidemiology/raw}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${ROOT}/data/epidemiology/protocol}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/diagnostics}"
SEEDS="${SEEDS:-0 1 2 3 4}"
read -r -a SEED_ARRAY <<<"${SEEDS}"
DEVICE="${DEVICE:-cuda}"
ITERATIONS="${ITERATIONS:-10}"

cd "${ROOT}"

"${PY}" scripts/download_epidemiology_data.py --output-root "${RAW_ROOT}"
"${PY}" scripts/audit_epidemiology_data.py \
  --raw-root "${RAW_ROOT}" \
  --output "${RESULT_ROOT}/epidemiology_phase0"

for seed in ${SEEDS}; do
  protocol_dir="${PROTOCOL_ROOT}/covid/seed${seed}"
  output_dir="${RESULT_ROOT}/epidemiology_pilot/covid/seed${seed}"
  mkdir -p "${output_dir}"
  "${PY}" scripts/build_epidemiology_protocol.py \
    --dataset covid \
    --raw-root "${RAW_ROOT}/covid" \
    --output "${protocol_dir}/protocol.npz" \
    --seed "${seed}" \
    --calibration-steps 52 \
    --block-size 1 \
    --inducing-sizes 8 16 32
  "${PY}" scripts/run_epidemiology_pilot.py \
    --protocol-npz "${protocol_dir}/protocol.npz" \
    --protocol-json "${protocol_dir}/protocol.json" \
    --output-root "${output_dir}" \
    --mt 16 \
    --ms 8 \
    --iterations "${ITERATIONS}" \
    --device "${DEVICE}" \
    >"${output_dir}/run.log" 2>&1
done

"${PY}" scripts/summarize_epidemiology_pilot.py \
  --input-root "${RESULT_ROOT}/epidemiology_pilot/covid" \
  --dataset covid \
  --expected-seeds "${SEED_ARRAY[@]}"

if [[ -n "${MOSQLIMATE_API_KEY:-}" ]]; then
  dengue_audit="${RESULT_ROOT}/epidemiology_phase0/dengue_audit.json"
  if "${PY}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] == "pass_primary" else 1)' "${dengue_audit}"; then
    for seed in ${SEEDS}; do
      protocol_dir="${PROTOCOL_ROOT}/dengue/seed${seed}"
      output_dir="${RESULT_ROOT}/epidemiology_pilot/dengue/seed${seed}"
      mkdir -p "${output_dir}"
      "${PY}" scripts/build_epidemiology_protocol.py \
        --dataset dengue \
        --raw-root "${RAW_ROOT}/dengue" \
        --output "${protocol_dir}/protocol.npz" \
        --seed "${seed}" \
        --calibration-steps 52 \
        --block-size 1 \
        --inducing-sizes 32 64 128
      "${PY}" scripts/run_epidemiology_pilot.py \
        --protocol-npz "${protocol_dir}/protocol.npz" \
        --protocol-json "${protocol_dir}/protocol.json" \
        --output-root "${output_dir}" \
        --mt 64 \
        --ms 32 \
        --iterations "${ITERATIONS}" \
        --device "${DEVICE}" \
        >"${output_dir}/run.log" 2>&1
    done
    "${PY}" scripts/summarize_epidemiology_pilot.py \
      --input-root "${RESULT_ROOT}/epidemiology_pilot/dengue" \
      --dataset dengue \
      --expected-seeds "${SEED_ARRAY[@]}"
  fi
fi
