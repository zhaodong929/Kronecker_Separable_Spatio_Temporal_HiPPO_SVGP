#!/usr/bin/env bash
# Generic one-GPU accuracy array worker. The Python runner reads one JSONL
# record using SLURM_ARRAY_TASK_ID and executes its argv directly.
#SBATCH --job-name=era5-a100-array
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=era5_a100_array_%A_%a.out
#SBATCH --error=era5_a100_array_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO}}"
ENV_ROOT="${ERA5_ENV_ROOT:-${ENV_ROOT:-${HOME}/micromamba/envs}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/iclr_era5_full_benchmark}"
MANIFEST=""
MANIFEST_NAME=""
FORCE=0

usage() {
  cat >&2 <<'EOF'
Usage: run_manifest_array.sbatch (--manifest PATH | --manifest-name NAME)
                                [--repo PATH] [--env PATH] [--benchmark PATH] [--force]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --manifest-name) MANIFEST_NAME="$2"; shift 2 ;;
    --repo) REPO_ROOT="$2"; shift 2 ;;
    --env) ENV_ROOT="$2"; shift 2 ;;
    --benchmark) BENCHMARK_ROOT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -n "${MANIFEST_NAME}" ]]; then
  case "${MANIFEST_NAME}" in
    shared_batch_short|official_long_preflight|official_long_full|online_short|online_long|efficiency) ;;
    *) echo "Unknown ERA5 manifest name: ${MANIFEST_NAME}" >&2; exit 2 ;;
  esac
  if [[ -n "${MANIFEST}" ]]; then
    echo "Use only one of --manifest and --manifest-name" >&2
    exit 2
  fi
  MANIFEST="${BENCHMARK_ROOT}/manifests/${MANIFEST_NAME}.jsonl"
fi
if [[ -z "${MANIFEST}" ]]; then
  echo "--manifest is required" >&2
  usage
  exit 2
fi
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
ROUTEB_PY="${ROUTEB_PY:-${ENV_ROOT}/routeb/bin/python}"
RUNNER="${REPO_ROOT}/scripts/run_era5_a100_manifest_job.py"
if [[ ! -x "${ROUTEB_PY}" ]]; then
  echo "Route B Micromamba Python is missing: ${ROUTEB_PY}" >&2
  exit 1
fi
if [[ ! -f "${RUNNER}" ]]; then
  echo "Manifest runner is missing: ${RUNNER}" >&2
  exit 1
fi

args=("${RUNNER}" --manifest "${MANIFEST}" --repo "${REPO_ROOT}")
if [[ "${FORCE}" -eq 1 ]]; then
  args+=(--force)
fi
exec "${ROUTEB_PY}" "${args[@]}"
