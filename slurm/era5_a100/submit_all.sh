#!/usr/bin/env bash
# Submit the entire ERA5 pipeline as exactly three persistent one-GPU array
# tasks. No future stages are pre-submitted as dependency jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO}}"
ENV_ROOT="${ERA5_ENV_ROOT:-${ENV_ROOT:-${HOME}/micromamba/envs}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${HOME}/results/iclr_era5_a100_shared_online_v2}"
PARTITION="${SLURM_PARTITION:-}"
ACCOUNT="${SLURM_ACCOUNT:-}"
MAX_GPUS=3
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: submit_all.sh [options]

Options:
  --repo PATH          Repository checkout
  --env PATH           Environment root containing routeb/gpflow/stvgp_legacy
  --benchmark PATH     Shared benchmark output root
  --partition NAME     Required A100-only Slurm partition
  --account NAME       Optional Slurm account
  --max-gpus 3         Must be 3; retained for command compatibility
  --dry-run            Print the single sbatch submission without running it
  -h, --help           Show this help

The command submits one Slurm array with exactly three persistent tasks. Each
task owns one A100 and processes a deterministic share of every experiment
stage. The complete pipeline therefore consumes three submitted jobs, not one
job per experiment or future stage.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo|--repo-root) REPO_ROOT="$2"; shift 2 ;;
    --env|--env-root) ENV_ROOT="$2"; shift 2 ;;
    --benchmark|--benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --max-gpus) MAX_GPUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
BENCHMARK_ROOT="$(mkdir -p "${BENCHMARK_ROOT}" && cd "${BENCHMARK_ROOT}" && pwd)"
if [[ "${MAX_GPUS}" != "3" ]]; then
  echo "--max-gpus must be exactly 3 for the persistent pipeline" >&2
  exit 2
fi
if [[ -z "${PARTITION}" ]]; then
  echo "--partition is required; use the A100-only partition (for example: a100)" >&2
  exit 2
fi
if [[ "${DRY_RUN}" -eq 0 ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required unless --dry-run is used" >&2
  exit 1
fi

PIPELINE_SCRIPT="${SCRIPT_DIR}/run_persistent_pipeline_worker.sbatch"
JOB_IDS_JSON="${BENCHMARK_ROOT}/slurm_job_ids.json"
if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
  echo "Persistent pipeline script is missing: ${PIPELINE_SCRIPT}" >&2
  exit 1
fi

SBATCH_ARGS=(
  --parsable
  --partition="${PARTITION}"
  --array="0-2%3"
  --chdir="${REPO_ROOT}"
  --time="72:00:00"
)
if [[ -n "${ACCOUNT}" ]]; then SBATCH_ARGS+=(--account="${ACCOUNT}"); fi
SCRIPT_ARGS=(
  --repo "${REPO_ROOT}"
  --env "${ENV_ROOT}"
  --benchmark "${BENCHMARK_ROOT}"
  --worker-count 3
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  JOB_ID="DRY-RUN"
  printf 'DRY-RUN single submission: partition=%s array=0-2%%3 gpus=3 time=72:00:00 script=%s\n' \
    "${PARTITION}" "${PIPELINE_SCRIPT}"
else
  raw="$(sbatch "${SBATCH_ARGS[@]}" "${PIPELINE_SCRIPT}" "${SCRIPT_ARGS[@]}")"
  JOB_ID="${raw##*$'\n'}"
  JOB_ID="${JOB_ID%%;*}"
  JOB_ID="${JOB_ID%%_*}"
  if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse Slurm array job ID: ${raw}" >&2
    exit 1
  fi
fi

python3 - "${JOB_IDS_JSON}" "${JOB_ID}" "${REPO_ROOT}" "${BENCHMARK_ROOT}" \
  "${PARTITION}" "${ACCOUNT}" "${DRY_RUN}" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

output, job_id, repo, benchmark, partition, account, dry_run = sys.argv[1:]
payload = {
    "schema_version": 2,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "submission_mode": "three_persistent_a100_workers",
    "repo_root": repo,
    "benchmark_root": benchmark,
    "partition": partition,
    "account": account or None,
    "worker_count": 3,
    "submitted_job_count": 3,
    "time_limit": "72:00:00",
    "dry_run": bool(int(dry_run)),
    "jobs": {
        "persistent_pipeline": {
            "job_id": job_id,
            "array": "0-2%3",
            "gpu_per_worker": 1,
        }
    },
}
target = Path(output)
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
PY

echo "Submitted one three-worker ERA5 A100 pipeline: ${JOB_ID}_[0-2]"
echo "Job metadata: ${JOB_IDS_JSON}"
