#!/usr/bin/env bash
# Submit the portable ERA5 A100 pipeline. The six manifest names are part of
# the execution contract; long preflight is a real seed-0 ST-SVGP resource
# run, not an environment check.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO}}"
ENV_ROOT="${ERA5_ENV_ROOT:-${ENV_ROOT:-${HOME}/micromamba/envs}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/iclr_era5_full_benchmark}"
PARTITION="${SLURM_PARTITION:-}"
ACCOUNT="${SLURM_ACCOUNT:-}"
MAX_GPUS="${ERA5_MAX_GPUS:-3}"
PYTHON="${PYTHON:-python3}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: submit_all.sh [options]

Options:
  --repo PATH          Repository checkout (default: this checkout)
  --env PATH           Micromamba environment root containing routeb/gpflow/
                       stvgp_legacy environments
  --benchmark PATH     Benchmark output root
  --partition NAME     Optional Slurm partition
  --account NAME       Optional Slurm account
  --max-gpus N         Array concurrency cap, 1-3 (default: 3)
  --dry-run            Print the dependency graph without submitting jobs
  -h, --help           Show this help

The legacy STVGP environment is a prerequisite, not installed here:
STVGP_PY must use a CUDA-enabled JAX/JAXLIB build and jax.devices() must expose
an A100. Markovflow TF2.2 compatibility runs remain CPU-only and are excluded
from A100 timing claims.
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
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "Repository does not exist: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! "${MAX_GPUS}" =~ ^[1-3]$ ]]; then
  echo "--max-gpus must be an integer from 1 through 3" >&2
  exit 2
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python is required to count benchmark jobs: ${PYTHON}" >&2
  exit 1
fi
if [[ "${DRY_RUN}" -eq 0 ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required unless --dry-run is used" >&2
  exit 1
fi

BENCHMARK_ROOT="$(mkdir -p "${BENCHMARK_ROOT}" && cd "${BENCHMARK_ROOT}" && pwd)"
PIPELINE_DIR="${BENCHMARK_ROOT}/slurm"
JOB_IDS_ENV="${PIPELINE_DIR}/job_ids.env"
JOB_IDS_JSON="${BENCHMARK_ROOT}/slurm_job_ids.json"
MANIFEST_DIR="${BENCHMARK_ROOT}/manifests"
MANIFEST_CONFIG="${REPO_ROOT}/cloud/autodl_era5/benchmark_a100_shared_online_v2.json"
mkdir -p "${PIPELINE_DIR}"

count_output="$(AUTODL_ENV_ROOT="${ENV_ROOT}" BENCHMARK_ROOT="${BENCHMARK_ROOT}" \
  "${PYTHON}" - "${REPO_ROOT}" "${BENCHMARK_ROOT}" "${MANIFEST_CONFIG}" "${MANIFEST_DIR}" <<'PY'
from __future__ import annotations

import sys
import json
from pathlib import Path

repo, benchmark, config, output = sys.argv[1:]
repo_path = Path(repo).resolve()
benchmark_path = Path(benchmark).resolve()
config_path = Path(config).resolve()
output_path = Path(output).resolve()
sys.path.insert(0, str(repo_path))
from scripts.build_era5_a100_manifests import build_manifests

outputs = build_manifests(
    spec_path=config_path,
    benchmark_root=benchmark_path,
    output_dir=output_path,
)
names = (
    "shared_batch_short",
    "official_long_preflight",
    "official_long_full",
    "online_short",
    "online_long",
    "efficiency",
)
for name in names:
    expected = output_path / f"{name}.jsonl"
    actual = outputs.get(name)
    if actual is None or actual.resolve() != expected:
        raise SystemExit(f"builder returned an unexpected path for {name}: {actual}")
    count = sum(
        bool(line.strip()) for line in expected.read_text(encoding="utf-8").splitlines()
    )
    if count <= 0:
        raise SystemExit(f"generated manifest is empty: {expected}")
    print(f"{name}={count}")

def emit_indices(key, manifest_name, kind):
    path = output_path / f"{manifest_name}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    indices = [index for index, row in enumerate(rows) if row.get("kind") == kind]
    if not indices:
        raise SystemExit(f"no {kind!r} records in {path}")
    print(f"{key}_count={len(indices)}")
    print(f"{key}_indices={','.join(str(index) for index in indices)}")

emit_indices("shared_preflight", "shared_batch_short", "gpflow_feasibility_preflight")
path = output_path / "shared_batch_short.jsonl"
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
main_indices = [index for index, row in enumerate(rows) if row.get("kind") != "gpflow_feasibility_preflight"]
print(f"shared_main_count={len(main_indices)}")
print(f"shared_main_indices={','.join(str(index) for index in main_indices)}")
emit_indices("online_short_models", "online_short", "online")
emit_indices("online_short_postprocess", "online_short", "postprocess")
PY
)"

declare -A COUNTS=()
while IFS='=' read -r name count; do
  [[ -z "${name}" ]] && continue
  COUNTS["${name}"]="${count}"
done <<<"${count_output}"

MANIFEST_NAMES=(
  shared_batch_short
  official_long_preflight
  official_long_full
  online_short
  online_long
  efficiency
)
for name in "${MANIFEST_NAMES[@]}"; do
  if [[ ! "${COUNTS[${name}]:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid or empty manifest count for ${name}: ${COUNTS[${name}]:-missing}" >&2
    exit 1
  fi
done
for name in shared_preflight shared_main online_short_models online_short_postprocess; do
  if [[ ! "${COUNTS[${name}_count]:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid split count for ${name}: ${COUNTS[${name}_count]:-missing}" >&2
    exit 1
  fi
  if [[ -z "${COUNTS[${name}_indices]:-}" ]]; then
    echo "Missing array index specification for ${name}" >&2
    exit 1
  fi
done

declare -A JOB_IDS=()
declare -A JOB_DEPENDENCIES=()
declare -A JOB_SUBMITTED_AT=()
declare -A JOB_SCRIPTS=()
declare -A JOB_COUNTS=()
declare -A JOB_MANIFESTS=()
JOB_ORDER=()
write_job_ids() {
  local temporary="${JOB_IDS_ENV}.tmp"
  {
    echo "# Slurm IDs for the ERA5 A100 pipeline"
    for name in "${JOB_ORDER[@]}"; do
      printf '%s=%s\n' "${name}" "${JOB_IDS[${name}]}"
    done
  } >"${temporary}"
  mv -f "${temporary}" "${JOB_IDS_ENV}"

  local record_args=()
  local name
  for name in "${JOB_ORDER[@]}"; do
    record_args+=(
      "${name}"
      "${JOB_IDS[${name}]}"
      "${JOB_DEPENDENCIES[${name}]}"
      "${JOB_SUBMITTED_AT[${name}]}"
      "${JOB_SCRIPTS[${name}]}"
      "${JOB_COUNTS[${name}]}"
      "${JOB_MANIFESTS[${name}]}"
    )
  done
  "${PYTHON}" - "${JOB_IDS_JSON}" "${REPO_ROOT}" "${BENCHMARK_ROOT}" \
    "${MANIFEST_DIR}" "${MAX_GPUS}" "${DRY_RUN}" "${record_args[@]}" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

json_path = Path(sys.argv[1])
repo, benchmark, manifest_dir, max_gpus, dry_run = sys.argv[2:7]
fields = sys.argv[7:]
field_count = 7
if len(fields) % field_count:
    raise SystemExit("invalid job metadata record")
jobs = {}
for offset in range(0, len(fields), field_count):
    name, job_id, dependency, submitted_at, script, count, manifest = fields[
        offset : offset + field_count
    ]
    entry = {
        "job_id": job_id,
        "dependency": dependency or None,
        "submitted_at": submitted_at,
        "script": script,
    }
    if count:
        entry["array_count"] = int(count)
    if manifest:
        entry["manifest"] = manifest
    jobs[name] = entry
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "repo_root": repo,
    "benchmark_root": benchmark,
    "manifest_dir": manifest_dir,
    "max_gpus": int(max_gpus),
    "dry_run": bool(int(dry_run)),
    "jobs": jobs,
}
temporary = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
temporary.replace(json_path)
PY
}

SBATCH_COMMON=(--parsable --export=ALL --chdir="${REPO_ROOT}")
if [[ -n "${PARTITION}" ]]; then SBATCH_COMMON+=(--partition="${PARTITION}"); fi
if [[ -n "${ACCOUNT}" ]]; then SBATCH_COMMON+=(--account="${ACCOUNT}"); fi
SCRIPT_ARGS=(--repo "${REPO_ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}")
DRY_ID=90000

submit_single() {
  local name="$1" dependency="$2" script="$3" count="${4:-}" manifest="${5:-}"
  local args=("${SBATCH_COMMON[@]}")
  if [[ -n "${dependency}" ]]; then args+=(--dependency="${dependency}"); fi
  local id
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    id="${DRY_ID}"
    DRY_ID=$((DRY_ID + 1))
    printf 'DRY-RUN %s dependency=%s script=%s\n' "${name}" "${dependency:-none}" "${script}"
  else
    local raw
    raw="$(sbatch "${args[@]}" "${script}" "${SCRIPT_ARGS[@]}")"
    id="${raw##*$'\n'}"
    id="${id%%;*}"
    id="${id%%_*}"
    if [[ ! "${id}" =~ ^[0-9]+$ ]]; then
      echo "Could not parse sbatch job id for ${name}: ${raw}" >&2
      exit 1
    fi
  fi
  JOB_IDS["${name}"]="${id}"
  JOB_DEPENDENCIES["${name}"]="${dependency}"
  JOB_SUBMITTED_AT["${name}"]="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  JOB_SCRIPTS["${name}"]="${script}"
  JOB_COUNTS["${name}"]="${count}"
  JOB_MANIFESTS["${name}"]="${manifest}"
  JOB_ORDER+=("${name}")
  write_job_ids
}

submit_array() {
  local name="$1" dependency="$2" count="$3" script="$4" manifest_name="$5" limit="$6"
  local index_spec="${7:-0-$((count - 1))}"
  local args=("${SBATCH_COMMON[@]}" --array="${index_spec}%${limit}")
  if [[ -n "${dependency}" ]]; then args+=(--dependency="${dependency}"); fi
  local id
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    id="${DRY_ID}"
    DRY_ID=$((DRY_ID + 1))
    printf 'DRY-RUN %s dependency=%s array=%s%%%s manifest=%s\n' \
      "${name}" "${dependency:-none}" "${index_spec}" "${limit}" "${manifest_name}"
  else
    local raw
    raw="$(sbatch "${args[@]}" "${script}" "${SCRIPT_ARGS[@]}" --manifest-name "${manifest_name}")"
    id="${raw##*$'\n'}"
    id="${id%%;*}"
    id="${id%%_*}"
    if [[ ! "${id}" =~ ^[0-9]+$ ]]; then
      echo "Could not parse sbatch array job id for ${name}: ${raw}" >&2
      exit 1
    fi
  fi
  JOB_IDS["${name}"]="${id}"
  JOB_DEPENDENCIES["${name}"]="${dependency}"
  JOB_SUBMITTED_AT["${name}"]="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  JOB_SCRIPTS["${name}"]="${script}"
  JOB_COUNTS["${name}"]="${count}"
  JOB_MANIFESTS["${name}"]="${MANIFEST_DIR}/${manifest_name}.jsonl"
  JOB_ORDER+=("${name}")
  write_job_ids
}

SETUP_SCRIPT="${SCRIPT_DIR}/setup_and_validate.sbatch"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_protocol.sbatch"
ARRAY_SCRIPT="${SCRIPT_DIR}/run_manifest_array.sbatch"
GPFLOW_SELECT_SCRIPT="${SCRIPT_DIR}/select_gpflow_tier.sbatch"
EFFICIENCY_SCRIPT="${SCRIPT_DIR}/run_efficiency.sbatch"
REPORT_SCRIPT="${SCRIPT_DIR}/generate_report.sbatch"

submit_single validation "" "${SETUP_SCRIPT}"
submit_single prepare "afterok:${JOB_IDS[validation]}" "${PREPARE_SCRIPT}"
submit_array shared_batch_gpflow_preflight "afterok:${JOB_IDS[prepare]}" \
  "${COUNTS[shared_preflight_count]}" "${ARRAY_SCRIPT}" shared_batch_short "${MAX_GPUS}" \
  "${COUNTS[shared_preflight_indices]}"
submit_single gpflow_tier_selection \
  "afterany:${JOB_IDS[shared_batch_gpflow_preflight]}" "${GPFLOW_SELECT_SCRIPT}"
submit_array shared_batch_short "afterok:${JOB_IDS[gpflow_tier_selection]}" \
  "${COUNTS[shared_main_count]}" "${ARRAY_SCRIPT}" shared_batch_short "${MAX_GPUS}" \
  "${COUNTS[shared_main_indices]}"
# Slurm's array %N limit is per array, not global. Chaining the accuracy
# arrays is what makes --max-gpus a hard pipeline-wide GPU ceiling.
submit_array official_long_preflight "afterany:${JOB_IDS[shared_batch_short]}" \
  "${COUNTS[official_long_preflight]}" "${ARRAY_SCRIPT}" official_long_preflight "${MAX_GPUS}"
submit_array official_long_full "afterany:${JOB_IDS[official_long_preflight]}" \
  "${COUNTS[official_long_full]}" "${ARRAY_SCRIPT}" official_long_full "${MAX_GPUS}"
submit_array online_short "afterany:${JOB_IDS[official_long_full]}" \
  "${COUNTS[online_short_models_count]}" "${ARRAY_SCRIPT}" online_short "${MAX_GPUS}" \
  "${COUNTS[online_short_models_indices]}"
submit_array online_short_postprocess "afterany:${JOB_IDS[online_short]}" \
  "${COUNTS[online_short_postprocess_count]}" "${ARRAY_SCRIPT}" online_short "${MAX_GPUS}" \
  "${COUNTS[online_short_postprocess_indices]}"
submit_array online_long "afterany:${JOB_IDS[online_short_postprocess]}" \
  "${COUNTS[online_long]}" "${ARRAY_SCRIPT}" online_long "${MAX_GPUS}"
submit_single efficiency "afterany:${JOB_IDS[online_long]}" "${EFFICIENCY_SCRIPT}" \
  "${COUNTS[efficiency]}" "${MANIFEST_DIR}/efficiency.jsonl"
submit_single report \
  "afterany:${JOB_IDS[shared_batch_short]}:${JOB_IDS[official_long_full]}:${JOB_IDS[online_short_postprocess]}:${JOB_IDS[online_long]}:${JOB_IDS[efficiency]}" \
  "${REPORT_SCRIPT}"

echo "Submitted ERA5 A100 pipeline; job IDs saved to ${JOB_IDS_JSON} (env: ${JOB_IDS_ENV})"
