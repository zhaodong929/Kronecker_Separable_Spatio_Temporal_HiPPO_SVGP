#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${AUTODL_ENV_ROOT:-/root/autodl-tmp/stvgp_envs}/routeb/bin/python"
cd "${ROOT}"

"${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage prepare "$@"
"${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage stage2 "$@"
"${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage stage3 "$@"
"${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage stage4 "$@"
"${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage stage5 "$@"
