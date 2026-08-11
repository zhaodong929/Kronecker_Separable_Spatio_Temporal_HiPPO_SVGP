#!/usr/bin/env bash
# Prepare the ERA5 protocol, then build the six canonical JSONL manifests.
# The environment validation job is separate from this real-data preparation.
#SBATCH --job-name=era5-a100-prepare
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=era5_a100_prepare_%j.out
#SBATCH --error=era5_a100_prepare_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO}}"
ENV_ROOT="${ERA5_ENV_ROOT:-${ENV_ROOT:-${HOME}/stvgp_envs}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/iclr_era5_full_benchmark}"

usage() {
  cat >&2 <<'EOF'
Usage: prepare_protocol.sbatch [--repo PATH] [--env PATH] [--benchmark PATH]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_ROOT="$2"; shift 2 ;;
    --env) ENV_ROOT="$2"; shift 2 ;;
    --benchmark) BENCHMARK_ROOT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
BENCHMARK_ROOT="$(mkdir -p "${BENCHMARK_ROOT}" && cd "${BENCHMARK_ROOT}" && pwd)"
ROUTEB_PY="${ROUTEB_PY:-${ENV_ROOT}/routeb/bin/python}"
CONFIG="${REPO_ROOT}/cloud/autodl_era5/benchmark.json"
MANIFEST_CONFIG="${REPO_ROOT}/cloud/autodl_era5/benchmark_a100_shared_online_v2.json"
MANIFEST_DIR="${BENCHMARK_ROOT}/manifests"
BUILDER="${REPO_ROOT}/scripts/build_era5_a100_manifests.py"

if [[ ! -x "${ROUTEB_PY}" ]]; then
  echo "Route B Micromamba Python is missing: ${ROUTEB_PY}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" || ! -f "${MANIFEST_CONFIG}" || ! -f "${BUILDER}" ]]; then
  echo "Protocol preparation inputs are missing under ${REPO_ROOT}" >&2
  exit 1
fi

export AUTODL_ENV_ROOT="${ENV_ROOT}"
export BENCHMARK_ROOT
export PYTHONHASHSEED=0

cd "${REPO_ROOT}"
"${ROUTEB_PY}" "${REPO_ROOT}/cloud/autodl_era5/run_benchmark.py" \
  --config "${CONFIG}" --stage prepare --include-legacy

"${ROUTEB_PY}" "${BUILDER}" \
  --config "${MANIFEST_CONFIG}" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --output-dir "${MANIFEST_DIR}"
