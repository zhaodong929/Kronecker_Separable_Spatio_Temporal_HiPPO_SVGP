#!/usr/bin/env bash
# Select the formal GPflow capacity from seed-0 runnability preflights.
#SBATCH --job-name=era5-gpflow-select
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=era5_gpflow_select_%j.out
#SBATCH --error=era5_gpflow_select_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_ROOT="${ERA5_ENV_ROOT:-${ENV_ROOT:-${HOME}/micromamba/envs}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/results/iclr_era5_a100_shared_online_v2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_ROOT="$2"; shift 2 ;;
    --env) ENV_ROOT="$2"; shift 2 ;;
    --benchmark) BENCHMARK_ROOT="$2"; shift 2 ;;
    --help|-h) exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROUTEB_PY="${ROUTEB_PY:-${ENV_ROOT}/routeb/bin/python}"
exec "${ROUTEB_PY}" "${REPO_ROOT}/scripts/select_era5_gpflow_tier.py" \
  --config "${REPO_ROOT}/cloud/autodl_era5/benchmark_a100_shared_online_v2.json" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --manifest-dir "${BENCHMARK_ROOT}/manifests"
