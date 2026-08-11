#!/usr/bin/env bash
# Report generation is CPU-only and is submitted afterany so partial results
# and explicit failure classifications remain visible in the final artifact.
#SBATCH --job-name=era5-a100-report
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=era5_a100_report_%j.out
#SBATCH --error=era5_a100_report_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO}}"
ENV_ROOT="${ERA5_ENV_ROOT:-${ENV_ROOT:-${HOME}/stvgp_envs}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/iclr_era5_full_benchmark}"

usage() {
  cat >&2 <<'EOF'
Usage: generate_report.sbatch [--repo PATH] [--env PATH] [--benchmark PATH]
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
ROUTEB_PY="${ROUTEB_PY:-${ENV_ROOT}/routeb/bin/python}"
CONFIG="${REPO_ROOT}/cloud/autodl_era5/benchmark.json"
AUDIT="${REPO_ROOT}/scripts/audit_era5_a100_shared_online.py"
REPORT="${REPO_ROOT}/scripts/generate_era5_a100_shared_online_report.py"
REPORT_DIR="${BENCHMARK_ROOT}/report"

if [[ ! -x "${ROUTEB_PY}" ]]; then
  echo "Route B Micromamba Python is missing: ${ROUTEB_PY}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" || ! -f "${AUDIT}" || ! -f "${REPORT}" ]]; then
  echo "Report inputs are missing under ${REPO_ROOT}" >&2
  exit 1
fi

mkdir -p "${REPORT_DIR}"
cd "${REPO_ROOT}"

# The audit is useful even when an upstream array task failed. Preserve its
# diagnostics, but do not prevent the report from describing partial output.
if ! "${ROUTEB_PY}" "${AUDIT}" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --output "${BENCHMARK_ROOT}/audit.json" \
  --csv-output "${BENCHMARK_ROOT}/audit.csv"; then
  echo "Benchmark audit reported failures; continuing to report generation" >&2
fi

"${ROUTEB_PY}" "${REPORT}" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --output-dir "${REPORT_DIR}"
