#!/usr/bin/env bash
# QOS-safe one-GPU worker. A small Slurm array is submitted and each worker
# processes a deterministic stride of manifest records in sequence.
#SBATCH --job-name=era5-a100-worker
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=era5_a100_worker_%A_%a.out
#SBATCH --error=era5_a100_worker_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO}}"
ENV_ROOT="${ERA5_ENV_ROOT:-${ENV_ROOT:-${HOME}/stvgp_envs}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/results/iclr_era5_a100_shared_online_v2}"
MANIFEST_NAME=""
INDEX_SPEC=""
WORKER_COUNT=""
FORCE=0

usage() {
  cat >&2 <<'EOF'
Usage: run_manifest_worker.sbatch --manifest-name NAME --indices SPEC
                                  --worker-count N [--repo PATH] [--env PATH]
                                  [--benchmark PATH] [--force]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest-name) MANIFEST_NAME="$2"; shift 2 ;;
    --indices) INDEX_SPEC="$2"; shift 2 ;;
    --worker-count) WORKER_COUNT="$2"; shift 2 ;;
    --repo) REPO_ROOT="$2"; shift 2 ;;
    --env) ENV_ROOT="$2"; shift 2 ;;
    --benchmark) BENCHMARK_ROOT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "${MANIFEST_NAME}" in
  shared_batch_short|official_long_preflight|official_long_full|online_short|online_long) ;;
  *) echo "Unknown or missing ERA5 manifest name: ${MANIFEST_NAME}" >&2; exit 2 ;;
esac
if [[ -z "${INDEX_SPEC}" || ! "${WORKER_COUNT}" =~ ^[1-3]$ ]]; then
  echo "--indices and --worker-count 1-3 are required" >&2
  exit 2
fi
if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
  echo "SLURM_ARRAY_TASK_ID is required" >&2
  exit 2
fi

REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
MANIFEST="${BENCHMARK_ROOT}/manifests/${MANIFEST_NAME}.jsonl"
ROUTEB_PY="${ROUTEB_PY:-${ENV_ROOT}/routeb/bin/python}"
RUNNER="${REPO_ROOT}/scripts/run_era5_a100_manifest_job.py"
if [[ ! -x "${ROUTEB_PY}" || ! -f "${RUNNER}" || ! -f "${MANIFEST}" ]]; then
  echo "Worker prerequisite missing: python=${ROUTEB_PY} runner=${RUNNER} manifest=${MANIFEST}" >&2
  exit 1
fi

configure_legacy_cuda() {
  local candidate cuda_root="" ptxas=""
  local candidates=()
  [[ -z "${LEGACY_CUDA_ROOT:-}" ]] || candidates+=("${LEGACY_CUDA_ROOT}")
  candidates+=(/usr/local/cuda-11.8 /usr/local/cuda /usr/local/cuda-*)
  for candidate in "${candidates[@]}"; do
    [[ -x "${candidate}/bin/ptxas" && -d "${candidate}/nvvm/libdevice" ]] || continue
    cuda_root="${candidate}"
    ptxas="${candidate}/bin/ptxas"
    break
  done
  if [[ -z "${cuda_root}" ]]; then
    echo "Legacy JAX CUDA toolkit not found; non-legacy jobs may still run." >&2
    return
  fi
  export PATH="$(dirname "${ptxas}"):${PATH}"
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_cuda_data_dir=${cuda_root}"
  printf 'Legacy JAX CUDA toolkit: %s\n' "${cuda_root}"
}

configure_legacy_cuda

mapfile -t RECORD_INDICES < <(
  "${ROUTEB_PY}" - "${INDEX_SPEC}" "${SLURM_ARRAY_TASK_ID}" "${WORKER_COUNT}" <<'PY'
import sys

spec, worker, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
indices = []
for item in spec.split(","):
    if not item:
        continue
    if "-" in item:
        start, stop = (int(value) for value in item.split("-", 1))
        if stop < start:
            raise SystemExit(f"invalid descending index range: {item}")
        indices.extend(range(start, stop + 1))
    else:
        indices.append(int(item))
if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
    raise SystemExit("manifest indices must be unique non-negative integers")
for index in indices[worker::workers]:
    print(index)
PY
)

echo "Worker ${SLURM_ARRAY_TASK_ID}/${WORKER_COUNT}: ${#RECORD_INDICES[@]} records from ${MANIFEST_NAME}"
failures=0
for index in "${RECORD_INDICES[@]}"; do
  args=("${RUNNER}" --manifest "${MANIFEST}" --index "${index}" --repo "${REPO_ROOT}")
  if [[ "${FORCE}" -eq 1 ]]; then args+=(--force); fi
  if ! "${ROUTEB_PY}" "${args[@]}"; then
    failures=$((failures + 1))
    echo "Manifest record failed but worker will continue: ${MANIFEST_NAME}[${index}]" >&2
  fi
done

echo "Worker complete: manifest=${MANIFEST_NAME} failures=${failures}"
if [[ "${failures}" -ne 0 ]]; then exit 1; fi
