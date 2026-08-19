#!/usr/bin/env bash
# Execute the complete A100 ERA5 protocol serially on one AutoDL GPU.
# The protocol matrix is unchanged; profiling records receive the actual GPU
# label so common-counter FLOPs cannot be mislabeled as A100 measurements.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${AUTODL_ENV_ROOT:-/root/autodl-tmp/stvgp_envs}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/root/autodl-tmp/iclr_era5_a100_protocol_autodl}"
RESULTS_BRANCH="${RESULTS_BRANCH:-codex/autodl-era5-a100-full-results}"
GPU_NAME_REGEX="${GPU_NAME_REGEX:-.*}"
MIN_GPU_MEMORY_GIB="${MIN_GPU_MEMORY_GIB:-24}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
CODE_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
RUN_LABEL="${RUN_LABEL:-a100_protocol_autodl_$(date -u +%Y%m%dT%H%M%SZ)_${CODE_COMMIT:0:7}}"
LOG_DIR="${BENCHMARK_ROOT}/logs"
LOG_PATH="${LOG_DIR}/${RUN_LABEL}.log"
PUBLISH_WORKTREE="${PUBLISH_WORKTREE:-$(dirname "${BENCHMARK_ROOT}")/.autodl_results_publish_${RUN_LABEL}}"
ROUTEB_PY="${ENV_ROOT}/routeb/bin/python"
STVGP_PY="${ENV_ROOT}/stvgp_legacy/bin/python"
TF_PY="${ENV_ROOT}/gpflow/bin/python"
SPEC="${ROOT}/cloud/autodl_era5/benchmark_a100_shared_online_v2.json"
BASE_CONFIG="${ROOT}/cloud/autodl_era5/benchmark.json"
MANIFEST_BUILDER="${ROOT}/scripts/build_era5_a100_manifests.py"
MANIFEST_WORKER="${ROOT}/slurm/era5_a100/run_manifest_worker.sbatch"
VALIDATOR="${ROOT}/slurm/era5_a100/setup_and_validate.sbatch"
GPFLOW_SELECTOR="${ROOT}/scripts/select_era5_gpflow_tier.py"
EFFICIENCY_RUNNER="${ROOT}/slurm/era5_a100/run_efficiency.sbatch"
REPORT_RUNNER="${ROOT}/slurm/era5_a100/generate_report.sbatch"

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

on_error() {
  local exit_code=$?
  printf 'FAILED exit_code=%s utc=%s log=%s\n' "${exit_code}" "$(date -u +%FT%TZ)" "${LOG_PATH}" >&2
  printf 'Results and logs remain at %s. The instance will not be shut down.\n' "${BENCHMARK_ROOT}" >&2
  exit "${exit_code}"
}

configure_legacy_ptxas() {
  local ptxas
  ptxas="$("${STVGP_PY}" - <<'PY'
from pathlib import Path
import site

for root in site.getsitepackages():
    for relative in ("nvidia/cuda_nvcc/bin/ptxas", "nvidia/cuda_nvcc/cu11/bin/ptxas"):
        candidate = Path(root) / relative
        if candidate.is_file():
            print(candidate)
            raise SystemExit(0)
raise SystemExit(1)
PY
)"
  if [[ -z "${ptxas}" || ! -x "${ptxas}" ]]; then
    echo "Legacy CUDA 11.3 ptxas is missing; rerun setup_autodl.sh with INCLUDE_LEGACY=1." >&2
    return 1
  fi
  export PATH="$(dirname "${ptxas}"):${PATH}"
  ptxas --version
}

manifest_indices() {
  local manifest=$1 mode=$2 value=${3:-}
  "${ROUTEB_PY}" - "${manifest}" "${mode}" "${value}" <<'PY'
import json
from pathlib import Path
import sys

path, mode, value = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
indices = []
for index, line in enumerate(item for item in path.read_text(encoding="utf-8").splitlines() if item.strip()):
    kind = str(json.loads(line).get("kind", ""))
    if mode == "all" or (mode == "kind" and kind == value) or (mode == "not_kind" and kind != value):
        indices.append(index)
if not indices:
    raise SystemExit(f"no manifest records selected: {path} mode={mode} value={value}")
print(",".join(str(index) for index in indices))
PY
}

run_manifest_phase() {
  local name=$1 manifest_name=$2 mode=$3 value=${4:-}
  local manifest="${BENCHMARK_ROOT}/manifests/${manifest_name}.jsonl"
  local indices
  indices="$(manifest_indices "${manifest}" "${mode}" "${value}")"
  echo "START phase=${name} manifest=${manifest_name}"
  run env SLURM_ARRAY_TASK_ID=0 bash "${MANIFEST_WORKER}" \
    --repo "${ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}" \
    --manifest-name "${manifest_name}" --indices "${indices}" --worker-count 1
  echo "END phase=${name}"
}

verify_complete_protocol() {
  "${ROUTEB_PY}" - "${BENCHMARK_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = (
    "shared_batch_short.jsonl", "official_long_preflight.jsonl", "official_long_full.jsonl",
    "online_short.jsonl", "online_long.jsonl", "efficiency.jsonl",
)
issues, total, ncu_total = [], 0, 0
for name in names:
    path = root / "manifests" / name
    if not path.is_file():
        issues.append(f"missing_manifest:{path}")
        continue
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        output = Path(row["output_dir"])
        expected = [Path(item) for item in row.get("expected", [])]
        missing = [str(item) for item in expected if not item.is_file() or item.stat().st_size == 0]
        if missing:
            issues.append(f"{name}[{index}] missing_artifacts:{missing}")
        try:
            status = json.loads(Path(row.get("status_path", output / "status.json")).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{name}[{index}] invalid_status:{exc}")
            continue
        if status.get("status") not in {"complete", "skipped"}:
            issues.append(f"{name}[{index}] status:{status.get('status')}")
        ncu = row.get("ncu")
        if isinstance(ncu, dict) and ncu.get("enabled", True):
            ncu_total += 1
            try:
                flops = json.loads((output / "ncu_flops.json").read_text(encoding="utf-8"))
                value = float(flops["nsight_flops_per_unit"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                issues.append(f"{name}[{index}] invalid_ncu_flops:{exc}")
                continue
            if not math.isfinite(value) or value <= 0.0:
                issues.append(f"{name}[{index}] invalid_ncu_value:{value}")
            if flops.get("measurement_backend") != "nsight_compute":
                issues.append(f"{name}[{index}] non_nsight_backend:{flops.get('measurement_backend')}")
            if flops.get("comparison_status") != "common_hardware_counter_complete":
                issues.append(f"{name}[{index}] incomplete_counter:{flops.get('comparison_status')}")
try:
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    issues.append(f"invalid_audit:{exc}")
else:
    for field in ("incomplete_runs", "failed_runs", "missing_runs", "prediction_mismatch_runs"):
        if int(audit.get(field, 0)):
            issues.append(f"audit_{field}:{audit[field]}")
report = root / "report"
if not report.is_dir() or not any(report.iterdir()):
    issues.append("missing_report_artifacts")
if issues:
    raise SystemExit("Full protocol verification failed:\n" + "\n".join(issues))
print(json.dumps({"verification_status": "VERIFIED", "manifest_records": total, "ncu_profiles": ncu_total}, sort_keys=True))
PY
}

copy_without_data_or_predictions() {
  local source=$1 destination=$2
  mkdir -p "${destination}"
  (
    cd "${source}"
    tar --exclude='predictions.npz' --exclude='data.npz' -cf - .
  ) | (
    cd "${destination}"
    tar -xf -
  )
}

publish_verified_results() {
  local bundle_root local_commit remote_commit
  if [[ -e "${PUBLISH_WORKTREE}" ]]; then
    echo "Publish worktree already exists: ${PUBLISH_WORKTREE}" >&2
    return 1
  fi
  run git -C "${ROOT}" fetch origin
  if git -C "${ROOT}" show-ref --verify --quiet "refs/remotes/origin/${RESULTS_BRANCH}"; then
    run git -C "${ROOT}" worktree add -B "${RESULTS_BRANCH}" "${PUBLISH_WORKTREE}" "origin/${RESULTS_BRANCH}"
  else
    run git -C "${ROOT}" worktree add -b "${RESULTS_BRANCH}" "${PUBLISH_WORKTREE}" "${CODE_COMMIT}"
  fi
  bundle_root="${PUBLISH_WORKTREE}/autodl_results/${RUN_LABEL}"
  mkdir -p "${bundle_root}/benchmark" "${bundle_root}/logs"
  copy_without_data_or_predictions "${BENCHMARK_ROOT}" "${bundle_root}/benchmark"
  cp "${LOG_PATH}" "${bundle_root}/logs/"
  git -C "${ROOT}" status --short >"${bundle_root}/code_status.txt"
  git -C "${ROOT}" log -1 --format=fuller >"${bundle_root}/code_commit.txt"
  (
    cd "${BENCHMARK_ROOT}"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
  ) >"${bundle_root}/SHA256SUMS.txt"
  printf '%s\n' \
    '# AutoDL full ERA5 A100-protocol benchmark' \
    "- Run label: \`${RUN_LABEL}\`" \
    "- Code commit: \`${CODE_COMMIT}\`" \
    "- Hardware: \`${GPU_NAME}\`" \
    '- Protocol: complete A100 shared-online manifest, executed serially on one AutoDL GPU.' \
    '- Verification: all declared artifacts and all requested Nsight Compute profiles completed before publication.' \
    '- Excluded: raw protocol data and prediction archives; SHA256SUMS.txt covers the original benchmark root.' \
    >"${bundle_root}/README.md"
  run git -C "${PUBLISH_WORKTREE}" add "autodl_results/${RUN_LABEL}"
  run git -C "${PUBLISH_WORKTREE}" -c user.name="AutoDL Results Publisher" -c user.email="results@local.invalid" commit -m "Add verified AutoDL full A100 protocol ${RUN_LABEL}"
  run git -C "${PUBLISH_WORKTREE}" push --set-upstream origin "${RESULTS_BRANCH}"
  local_commit="$(git -C "${PUBLISH_WORKTREE}" rev-parse HEAD)"
  remote_commit="$(git -C "${PUBLISH_WORKTREE}" ls-remote origin "refs/heads/${RESULTS_BRANCH}" | awk 'NR == 1 {print $1}')"
  [[ "${local_commit}" == "${remote_commit}" ]]
  run git -C "${ROOT}" worktree remove "${PUBLISH_WORKTREE}"
  printf 'Verified GitHub results commit: %s\n' "${local_commit}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf '%s\n' 'Usage: run_a100_protocol_on_autodl.sh'
  exit 0
fi
if [[ $# -ne 0 || ! "${MIN_GPU_MEMORY_GIB}" =~ ^[0-9]+([.][0-9]+)?$ || ( "${AUTO_SHUTDOWN}" != "0" && "${AUTO_SHUTDOWN}" != "1" ) ]]; then
  echo 'Invalid arguments or environment overrides.' >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1
trap on_error ERR
cd "${ROOT}"
for name in protocol calibration runs efficiency report manifests audit.json audit.csv; do
  [[ ! -e "${BENCHMARK_ROOT}/${name}" ]] || { echo "Existing ${name} under ${BENCHMARK_ROOT}; use a fresh root." >&2; exit 1; }
done
[[ -x "${ROUTEB_PY}" && -x "${TF_PY}" && -x "${STVGP_PY}" ]]
[[ -f "${SPEC}" && -f "${BASE_CONFIG}" && -f "${MANIFEST_BUILDER}" && -f "${MANIFEST_WORKER}" ]]
run git ls-remote --exit-code origin HEAD
configure_legacy_ptxas
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
export AUTODL_ENV_ROOT="${ENV_ROOT}" BENCHMARK_ROOT PYTHONHASHSEED=0
export ERA5_GPU_NAME_REGEX="${GPU_NAME_REGEX}" ERA5_MIN_GPU_MEMORY_GIB="${MIN_GPU_MEMORY_GIB}"
run bash "${VALIDATOR}" --repo "${ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}" --gpu-name-regex "${GPU_NAME_REGEX}" --min-gpu-memory-gib "${MIN_GPU_MEMORY_GIB}"
run "${ROUTEB_PY}" "${ROOT}/cloud/autodl_era5/run_benchmark.py" --config "${BASE_CONFIG}" --stage prepare --include-legacy
run "${ROUTEB_PY}" "${MANIFEST_BUILDER}" --config "${SPEC}" --benchmark-root "${BENCHMARK_ROOT}" --output-dir "${BENCHMARK_ROOT}/manifests" --hardware-class "${GPU_NAME}"
run_manifest_phase gpflow_preflight shared_batch_short kind gpflow_feasibility_preflight
run "${ROUTEB_PY}" "${GPFLOW_SELECTOR}" --config "${SPEC}" --benchmark-root "${BENCHMARK_ROOT}" --manifest-dir "${BENCHMARK_ROOT}/manifests" --hardware-class "${GPU_NAME}"
run_manifest_phase shared_batch shared_batch_short not_kind gpflow_feasibility_preflight
run_manifest_phase official_long_preflight official_long_preflight all
run_manifest_phase official_long_full official_long_full all
run_manifest_phase online_short online_short kind online
run_manifest_phase online_short_postprocess online_short kind postprocess
run_manifest_phase online_long online_long all
run bash "${EFFICIENCY_RUNNER}" --repo "${ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}" --gpu-name-regex "${GPU_NAME_REGEX}" --min-gpu-memory-gib "${MIN_GPU_MEMORY_GIB}" --require-ncu
run bash "${REPORT_RUNNER}" --repo "${ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}"
verify_complete_protocol
publish_verified_results
sync
printf 'SUCCESS utc=%s log=%s\n' "$(date -u +%FT%TZ)" "${LOG_PATH}"
if [[ "${AUTO_SHUTDOWN}" == "1" ]]; then
  /usr/bin/shutdown -h now
fi
