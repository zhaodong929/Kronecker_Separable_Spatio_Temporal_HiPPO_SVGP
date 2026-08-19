#!/usr/bin/env bash
# Resume a capacity-ladder ERA5 run after the active worker leaves the GPU.
# It keeps known full ST-VGP RTX 4090 OOMs explicit while requiring every
# non-excluded manifest record before publishing and shutting down.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${AUTODL_ENV_ROOT:-/root/autodl-tmp/stvgp_envs}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/root/autodl-tmp/iclr_era5_a100_protocol_autodl_4090_capacity_ladder_20260820T023000Z}"
RESULTS_BRANCH="${RESULTS_BRANCH:-codex/autodl-era5-a100-full-results}"
ACTIVE_PID="${WAIT_FOR_PID:-}"
ACTIVE_COMMAND_MARKER="${WAIT_FOR_COMMAND_MARKER:-run_a100_protocol_on_autodl.sh}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
RUN_LABEL="${RUN_LABEL:-rtx4090_capacity_ladder_recovery_$(date -u +%Y%m%dT%H%M%SZ)}"
CODE_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
RECOVERY_ROOT="${RECOVERY_ROOT:-${BENCHMARK_ROOT}/recovery_${RUN_LABEL}}"
LOG_DIR="${BENCHMARK_ROOT}/logs"
LOG_PATH="${LOG_DIR}/${RUN_LABEL}.log"
PUBLISH_WORKTREE="${PUBLISH_WORKTREE:-$(dirname "${BENCHMARK_ROOT}")/.autodl_results_publish_${RUN_LABEL}}"
ROUTEB_PY="${ENV_ROOT}/routeb/bin/python"
MANIFEST_TOOL="${ROOT}/scripts/prepare_autodl_era5_recovery.py"
MANIFEST_RUNNER="${ROOT}/scripts/run_era5_a100_manifest_job.py"
EFFICIENCY_RUNNER="${ROOT}/slurm/era5_a100/run_efficiency.sbatch"
REPORT_RUNNER="${ROOT}/slurm/era5_a100/generate_report.sbatch"
PUBLISH_STATUS="${RECOVERY_ROOT}/publish_status.json"

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

on_error() {
  local exit_code=$?
  printf 'FAILED exit_code=%s utc=%s log=%s\n' "${exit_code}" "$(date -u +%FT%TZ)" "${LOG_PATH}" >&2
  printf 'No shutdown: recovery, report, or GitHub publication did not complete.\n' >&2
  exit "${exit_code}"
}

configure_legacy_cuda() {
  local candidate cuda_root="" ptxas=""
  local candidates=(/usr/local/cuda /usr/local/cuda-*)
  for candidate in "${candidates[@]}"; do
    [[ -x "${candidate}/bin/ptxas" && -d "${candidate}/nvvm/libdevice" ]] || continue
    cuda_root="${candidate}"
    ptxas="${candidate}/bin/ptxas"
    break
  done
  [[ -n "${cuda_root}" ]] || { echo 'Legacy JAX CUDA toolkit is unavailable.' >&2; return 1; }
  export PATH="$(dirname "${ptxas}"):${PATH}"
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_cuda_data_dir=${cuda_root}"
  printf 'Legacy JAX CUDA toolkit: %s\n' "${cuda_root}"
}

wait_for_active_runner() {
  [[ -n "${ACTIVE_PID}" ]] || return 0
  while kill -0 "${ACTIVE_PID}" 2>/dev/null; do
    local command
    command="$(ps -p "${ACTIVE_PID}" -o args= 2>/dev/null || true)"
    [[ "${command}" == *"${ACTIVE_COMMAND_MARKER}"* ]] || {
      echo "Refusing to wait on reused PID ${ACTIVE_PID}: ${command}" >&2
      return 1
    }
    printf 'WAITING active_runner_pid=%s utc=%s\n' "${ACTIVE_PID}" "$(date -u +%FT%TZ)"
    sleep 60
  done
  printf 'ACTIVE_RUNNER_EXITED pid=%s utc=%s\n' "${ACTIVE_PID}" "$(date -u +%FT%TZ)"
}

wait_for_gpu_idle() {
  local active
  while :; do
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' || true)"
    [[ -z "${active}" ]] && return 0
    printf 'WAITING_GPU_IDLE pids=%s utc=%s\n' "${active}" "$(date -u +%FT%TZ)"
    sleep 30
  done
}

selected_gpflow_tier() {
  "${ROUTEB_PY}" - "${BENCHMARK_ROOT}/gpflow_feasibility/selected_tier.json" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get("selected_tier")
if value is None:
    raise SystemExit("selected_tier is missing")
print(value)
PY
}

run_manifest_records() {
  local manifest_name=$1 manifest="${RECOVERY_ROOT}/manifests/${manifest_name}.jsonl"
  local index action method failures=0
  while IFS=$'\t' read -r index action method; do
    [[ -n "${index}" ]] || continue
    local args=("${MANIFEST_RUNNER}" --manifest "${manifest}" --index "${index}" --repo "${ROOT}")
    [[ "${action}" != force ]] || args+=(--force)
    if ! "${ROUTEB_PY}" "${args[@]}"; then
      failures=$((failures + 1))
      printf 'RECOVERY_RECORD_FAILED manifest=%s index=%s method=%s\n' "${manifest_name}" "${index}" "${method}" >&2
    fi
  done < <("${ROUTEB_PY}" "${MANIFEST_TOOL}" select --manifest "${manifest}" --repo "${BENCHMARK_ROOT}" --selected-tier "${SELECTED_TIER}")
  return "${failures}"
}

run_routeb_xlag_repair() {
  local manifest="${RECOVERY_ROOT}/manifests/shared_batch_short.jsonl" index failures=0
  for index in $(seq 11 25); do
    if ! "${ROUTEB_PY}" "${MANIFEST_RUNNER}" --manifest "${manifest}" --index "${index}" --repo "${ROOT}" --force; then
      failures=$((failures + 1))
      printf 'ROUTEB_REPAIR_FAILED index=%s\n' "${index}" >&2
    fi
  done
  return "${failures}"
}

copy_without_data_or_predictions() {
  local source=$1 destination=$2
  mkdir -p "${destination}"
  (cd "${source}" && tar --exclude='predictions.npz' --exclude='data.npz' -cf - .) |
    (cd "${destination}" && tar -xf -)
}

publish_partial_results() {
  local bundle_root local_commit remote_commit gpu_name
  [[ ! -e "${PUBLISH_WORKTREE}" ]] || { echo "Publish worktree exists: ${PUBLISH_WORKTREE}" >&2; return 1; }
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
  (cd "${BENCHMARK_ROOT}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) >"${bundle_root}/SHA256SUMS.txt"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  printf '%s\n' \
    '# AutoDL ERA5 capacity-ladder partial RTX 4090 recovery' \
    "- Run label: \`${RUN_LABEL}\`" \
    "- Code commit: \`${CODE_COMMIT}\`" \
    "- Hardware: \`${gpu_name}\`" \
    '- Scope: all non-excluded A100-protocol manifest records were resumed serially after the capacity-ladder runner exited.' \
    '- Explicit exclusion: full ST-VGP records are not reported as successful because they OOM on this RTX 4090; see partial_protocol_verification.json.' \
    '- Nsight Compute: unavailable or permission-blocked on this host, so hardware FLOP claims remain pending and no cross-method FLOP ratio is asserted.' \
    '- Raw protocol data and prediction archives are omitted; SHA256SUMS.txt covers the source benchmark root.' \
    >"${bundle_root}/README.md"
  run git -C "${PUBLISH_WORKTREE}" add "autodl_results/${RUN_LABEL}"
  run git -C "${PUBLISH_WORKTREE}" -c user.name='AutoDL Results Publisher' -c user.email='results@local.invalid' commit -m "Add partial RTX 4090 ERA5 recovery ${RUN_LABEL}"
  run git -C "${PUBLISH_WORKTREE}" push --set-upstream origin "${RESULTS_BRANCH}"
  local_commit="$(git -C "${PUBLISH_WORKTREE}" rev-parse HEAD)"
  remote_commit="$(git -C "${PUBLISH_WORKTREE}" ls-remote origin "refs/heads/${RESULTS_BRANCH}" | awk 'NR == 1 {print $1}')"
  [[ "${local_commit}" == "${remote_commit}" ]]
  "${ROUTEB_PY}" - "${PUBLISH_STATUS}" "${local_commit}" "${remote_commit}" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

path, local_commit, remote_commit = sys.argv[1:]
Path(path).write_text(json.dumps({
    "status": "pushed_and_verified",
    "published_at": datetime.now(timezone.utc).isoformat(),
    "local_commit": local_commit,
    "remote_commit": remote_commit,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  run git -C "${ROOT}" worktree remove "${PUBLISH_WORKTREE}"
  printf 'PUBLISHED_RESULT_COMMIT=%s\n' "${local_commit}"
}

if [[ "${AUTO_SHUTDOWN}" != 0 && "${AUTO_SHUTDOWN}" != 1 ]]; then
  echo 'AUTO_SHUTDOWN must be 0 or 1.' >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "${RECOVERY_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1
trap on_error ERR

wait_for_active_runner
wait_for_gpu_idle
[[ -x "${ROUTEB_PY}" && -f "${MANIFEST_TOOL}" && -f "${MANIFEST_RUNNER}" ]]
configure_legacy_cuda
SELECTED_TIER="$(selected_gpflow_tier)"
printf 'RECOVERY_START commit=%s selected_gpflow_tier=%s utc=%s\n' "${CODE_COMMIT}" "${SELECTED_TIER}" "$(date -u +%FT%TZ)"

run "${ROUTEB_PY}" "${MANIFEST_TOOL}" rewrite \
  --source "${BENCHMARK_ROOT}/manifests" --destination "${RECOVERY_ROOT}/manifests" --repo "${ROOT}"

repair_failures=0
if run_routeb_xlag_repair; then :; else repair_failures=$((repair_failures + 1)); fi
for manifest_name in shared_batch_short official_long_preflight official_long_full online_short online_long; do
  if run_manifest_records "${manifest_name}"; then :; else repair_failures=$((repair_failures + 1)); fi
done
if ! run bash "${EFFICIENCY_RUNNER}" --manifest "${RECOVERY_ROOT}/manifests/efficiency.jsonl" \
  --repo "${ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}" \
  --gpu-name-regex '.*' --min-gpu-memory-gib 24 --disable-ncu; then
  repair_failures=$((repair_failures + 1))
fi
printf 'RECOVERY_COMMAND_FAILURES=%s\n' "${repair_failures}"

if ! run bash "${REPORT_RUNNER}" --repo "${ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}"; then
  echo 'Report generation failed.' >&2
fi
run "${ROUTEB_PY}" "${MANIFEST_TOOL}" verify \
  --manifest-dir "${RECOVERY_ROOT}/manifests" --benchmark "${BENCHMARK_ROOT}" \
  --selected-tier "${SELECTED_TIER}" --require-ncu-audit \
  --output "${BENCHMARK_ROOT}/partial_protocol_verification.json"
publish_partial_results
sync
printf 'SUCCESS utc=%s log=%s\n' "$(date -u +%FT%TZ)" "${LOG_PATH}"
if [[ "${AUTO_SHUTDOWN}" == 1 ]]; then
  /usr/bin/shutdown -h now
fi
