#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${AUTODL_ENV_ROOT:-/root/autodl-tmp/stvgp_envs}/routeb/bin/python"
cd "${ROOT}"

"${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage prepare
"${PYTHON}" cloud/autodl_era5/run_benchmark.py \
  --stage stage2 --scope calibration --seed 0
"${PYTHON}" cloud/autodl_era5/run_benchmark.py \
  --stage stage2 --scope task1_2 --seed 0
"${PYTHON}" cloud/autodl_era5/run_benchmark.py \
  --stage stage3 --scope task1_2 --seed 0
"${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage stage4 --force
"${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage stage5 --force
