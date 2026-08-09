#!/usr/bin/env bash
# One of three persistent A100 workers. The array is the entire pipeline:
# stages are executed in order and coordinated through the shared result root.
#SBATCH --job-name=era5-a100-pipeline
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=era5_a100_pipeline_%A_%a.out
#SBATCH --error=era5_a100_pipeline_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO}}"
ENV_ROOT="${ERA5_ENV_ROOT:-${ENV_ROOT:-${HOME}/micromamba/envs}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${HOME}/results/iclr_era5_a100_shared_online_v2}"
WORKER_COUNT=3

usage() {
  cat >&2 <<'EOF'
Usage: run_persistent_pipeline_worker.sbatch [--repo PATH] [--env PATH]
                                             [--benchmark PATH]
                                             [--worker-count 3]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_ROOT="$2"; shift 2 ;;
    --env) ENV_ROOT="$2"; shift 2 ;;
    --benchmark) BENCHMARK_ROOT="$2"; shift 2 ;;
    --worker-count) WORKER_COUNT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${WORKER_COUNT}" != "3" ]]; then
  echo "This pipeline requires exactly three persistent workers" >&2
  exit 2
fi
if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-2]$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be 0, 1, or 2" >&2
  exit 2
fi
if [[ ! "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
  echo "SLURM_ARRAY_JOB_ID is required" >&2
  exit 2
fi

WORKER_ID="${SLURM_ARRAY_TASK_ID}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
BENCHMARK_ROOT="$(mkdir -p "${BENCHMARK_ROOT}" && cd "${BENCHMARK_ROOT}" && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm/era5_a100"
ROUTEB_PY="${ROUTEB_PY:-${ENV_ROOT}/routeb/bin/python}"
MANIFEST_DIR="${BENCHMARK_ROOT}/manifests"
CONTROL_DIR="${BENCHMARK_ROOT}/slurm/persistent/${SLURM_ARRAY_JOB_ID}"
WORKER_SCRIPT="${SCRIPT_DIR}/run_manifest_worker.sbatch"
mkdir -p "${CONTROL_DIR}"

if [[ ! -x "${ROUTEB_PY}" ]]; then
  echo "Route B environment is missing: ${ROUTEB_PY}" >&2
  exit 1
fi
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
  echo "Manifest worker is missing: ${WORKER_SCRIPT}" >&2
  exit 1
fi

write_marker() {
  local name="$1" value="$2"
  local target="${CONTROL_DIR}/${name}"
  local temporary="${target}.${WORKER_ID}.$$.tmp"
  printf '%s\n' "${value}" >"${temporary}"
  mv -f "${temporary}" "${target}"
}

wait_marker() {
  local name="$1" target="${CONTROL_DIR}/${name}"
  while [[ ! -f "${target}" ]]; do sleep 10; done
  cat "${target}"
}

run_leader_stage() {
  local name="$1"
  shift
  local marker="leader_${name}.done" rc
  if [[ "${WORKER_ID}" -eq 0 ]]; then
    echo "LEADER START ${name}"
    if "$@"; then rc=0; else rc=$?; fi
    write_marker "${marker}" "${rc}"
    echo "LEADER END ${name} rc=${rc}"
  fi
  rc="$(wait_marker "${marker}")"
  if [[ "${rc}" -ne 0 ]]; then
    echo "Critical leader stage failed: ${name}, rc=${rc}" >&2
    return "${rc}"
  fi
}

manifest_indices() {
  local manifest="$1" mode="$2" value="${3:-}"
  "${ROUTEB_PY}" - "${manifest}" "${mode}" "${value}" <<'PY'
import json
from pathlib import Path
import sys

path, mode, value = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
indices = []
for index, line in enumerate(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()):
    record = json.loads(line)
    kind = str(record.get("kind", ""))
    selected = mode == "all" or (mode == "kind" and kind == value) or (mode == "not_kind" and kind != value)
    if selected:
        indices.append(index)
if not indices:
    raise SystemExit(f"no manifest records selected: path={path}, mode={mode}, value={value}")
print(",".join(str(index) for index in indices))
PY
}

run_parallel_stage() {
  local name="$1" manifest_name="$2" mode="$3" value="${4:-}"
  local manifest="${MANIFEST_DIR}/${manifest_name}.jsonl"
  local indices rc=0 marker
  indices="$(manifest_indices "${manifest}" "${mode}" "${value}")"
  echo "WORKER ${WORKER_ID} START ${name}"
  if SLURM_ARRAY_TASK_ID="${WORKER_ID}" bash "${WORKER_SCRIPT}" \
    --repo "${REPO_ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}" \
    --manifest-name "${manifest_name}" --indices "${indices}" \
    --worker-count "${WORKER_COUNT}"; then
    rc=0
  else
    rc=$?
  fi
  write_marker "phase_${name}.worker_${WORKER_ID}.done" "${rc}"
  echo "WORKER ${WORKER_ID} END ${name} rc=${rc}"

  local worker worker_rc failures=0
  for worker in 0 1 2; do
    marker="phase_${name}.worker_${worker}.done"
    worker_rc="$(wait_marker "${marker}")"
    if [[ "${worker_rc}" -ne 0 ]]; then failures=$((failures + 1)); fi
  done
  if [[ "${WORKER_ID}" -eq 0 ]]; then
    echo "PHASE COMPLETE ${name}: failed_workers=${failures}"
  fi
}

run_leader_stage validation bash "${SCRIPT_DIR}/setup_and_validate.sbatch" \
  --repo "${REPO_ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}"
run_leader_stage prepare bash "${SCRIPT_DIR}/prepare_protocol.sbatch" \
  --repo "${REPO_ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}"

run_parallel_stage gpflow_preflight shared_batch_short kind gpflow_feasibility_preflight
run_leader_stage gpflow_tier bash "${SCRIPT_DIR}/select_gpflow_tier.sbatch" \
  --repo "${REPO_ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}"
run_parallel_stage shared_batch shared_batch_short not_kind gpflow_feasibility_preflight
run_parallel_stage official_long_preflight official_long_preflight all
run_parallel_stage official_long_full official_long_full all
run_parallel_stage online_short online_short kind online
run_parallel_stage online_short_postprocess online_short kind postprocess
run_parallel_stage online_long online_long all

if [[ "${WORKER_ID}" -ne 0 ]]; then
  echo "Persistent worker ${WORKER_ID} completed all parallel accuracy stages"
  exit 0
fi

if bash "${SCRIPT_DIR}/run_efficiency.sbatch" \
  --repo "${REPO_ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}"; then
  efficiency_rc=0
else
  efficiency_rc=$?
  echo "Efficiency stage failed with rc=${efficiency_rc}; report generation will continue" >&2
fi

if bash "${SCRIPT_DIR}/generate_report.sbatch" \
  --repo "${REPO_ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}"; then
  report_rc=0
else
  report_rc=$?
fi

echo "Persistent pipeline complete: efficiency_rc=${efficiency_rc} report_rc=${report_rc}"
if [[ "${report_rc}" -ne 0 ]]; then exit "${report_rc}"; fi
if [[ "${efficiency_rc}" -ne 0 ]]; then exit "${efficiency_rc}"; fi
