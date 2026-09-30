#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zd929/projects/stvgp_kronecker}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/era5_batch_addendum_20260827}"

cd "${ROOT}"
exec "${PYTHON}" scripts/run_era5_batch_addendum.py \
  --output-root "${OUTPUT_ROOT}" \
  --scopes task1_2 task1_10 \
  --seeds 0 1 2 \
  --official-ms 128 \
  --official-iterations "${OFFICIAL_ITERATIONS:-100}" \
  --routeb-iterations "${ROUTEB_ITERATIONS:-100}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-float64}"
