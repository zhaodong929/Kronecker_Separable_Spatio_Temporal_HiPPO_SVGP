#!/usr/bin/env bash
# Complete only the selected GPU rows, then verify, publish, and power down.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${AUTODL_ENV_ROOT:-/root/autodl-tmp/stvgp_envs}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:?BENCHMARK_ROOT is required}"
WAIT_FOR_PID="${WAIT_FOR_PID:?WAIT_FOR_PID is required}"
RESULTS_BRANCH="${RESULTS_BRANCH:-codex/autodl-era5-a100-full-results}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
RUN_LABEL="${RUN_LABEL:-rtx4090_gpu_only_full_matrix_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$(dirname "${BENCHMARK_ROOT}")/era5_gpu_only_full_completion_${RUN_LABEL}}"
ROUTEB_PY="${ENV_ROOT}/routeb/bin/python"
VERIFY="${ROOT}/scripts/verify_autodl_era5_gpu_only.py"
WORKER="${ROOT}/slurm/era5_a100/run_manifest_worker.sbatch"
REPORT_RUNNER="${ROOT}/slurm/era5_a100/generate_report.sbatch"
PUBLISH_WORKTREE="${RUN_ROOT}/results_publish"
LOG_PATH="${RUN_ROOT}/launcher.log"
PUBLISH_STATUS="${RUN_ROOT}/publish_status.json"
CODE_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

on_error() {
  local exit_code=$?
  printf 'FAILED exit_code=%s utc=%s\n' "${exit_code}" "$(date -u +%FT%TZ)" >&2
  printf 'No shutdown: GPU-only execution, verification, or GitHub publication failed.\n' >&2
  exit "${exit_code}"
}

wait_for_worker() {
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    printf 'WAITING initial_gpu_worker_pid=%s utc=%s\n' "${WAIT_FOR_PID}" "$(date -u +%FT%TZ)"
    sleep 60
  done
}

repair_routeb_prediction_contract() {
  "${ROUTEB_PY}" - "${BENCHMARK_ROOT}/manifests/shared_batch_short.jsonl" <<'PY'
import json
import os
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
changed = 0
for row in rows:
    argv = row["argv"]
    if len(argv) < 2 or argv[1] != "scripts/run_iclr_era5_routeb_batch.py":
        continue
    if "--predictions-output" in argv:
        continue
    output = Path(row["output_dir"]) / "predictions.npz"
    argv.extend(["--predictions-output", str(output)])
    changed += 1
temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
temporary.write_text("\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")
os.replace(temporary, manifest)
print(json.dumps({"manifest": str(manifest), "routeb_rows_repaired": changed}, sort_keys=True))
PY
}

pending_indices() {
  local manifest_name=$1
  "${ROUTEB_PY}" - "${BENCHMARK_ROOT}/manifests/${manifest_name}.jsonl" "${manifest_name}" <<'PY'
import json
from pathlib import Path
import sys

from scripts.verify_autodl_era5_gpu_only import GPU_MANIFEST_POLICIES, exclusion_reason

manifest = Path(sys.argv[1])
manifest_name = sys.argv[2]
policy = GPU_MANIFEST_POLICIES[manifest_name]
indices = []
for index, line in enumerate(manifest.read_text(encoding="utf-8").splitlines()):
    if not line:
        continue
    record = json.loads(line)
    if exclusion_reason(record, policy) is not None:
        continue
    output_dir = Path(record["output_dir"])
    status_path = output_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError, TypeError):
        status = None
    artifacts = [Path(path) for path in record.get("expected", [])]
    complete = status == "complete" and all(
        path.is_file() and path.stat().st_size > 0 for path in artifacts
    )
    if not complete:
        indices.append(str(index))
print(",".join(indices))
PY
}

run_pending_manifest() {
  local manifest_name=$1 indices
  indices="$(pending_indices "${manifest_name}")"
  if [[ -z "${indices}" ]]; then
    printf 'NO_PENDING_GPU_RECORDS manifest=%s\n' "${manifest_name}"
    return
  fi
  printf 'RUN_PENDING_GPU_RECORDS manifest=%s indices=%s\n' "${manifest_name}" "${indices}"
  export SLURM_ARRAY_TASK_ID=0
  run bash "${WORKER}" \
    --repo "${ROOT}" \
    --env "${ENV_ROOT}" \
    --benchmark "${BENCHMARK_ROOT}" \
    --manifest-name "${manifest_name}" \
    --indices "${indices}" \
    --worker-count 1 \
    --force
}

copy_without_data_or_predictions() {
  local source=$1 destination=$2
  mkdir -p "${destination}"
  (cd "${source}" && tar --exclude='predictions.npz' --exclude='data.npz' -cf - .) |
    (cd "${destination}" && tar -xf -)
}

publish_results() {
  local bundle_root local_commit remote_commit
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
  git -C "${ROOT}" log -1 --format=fuller >"${bundle_root}/code_commit.txt"
  (cd "${BENCHMARK_ROOT}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) >"${bundle_root}/SHA256SUMS.txt"
  cat >"${bundle_root}/README.md" <<EOF
# AutoDL ERA5 GPU-only RTX 4090 comparison

- Run label: ${RUN_LABEL}
- Code commit: ${CODE_COMMIT}
- Hardware: NVIDIA GeForce RTX 4090
- Verified scope: shared batch, long batch, short online, and long online GPU rows.
- Markovflow is excluded because the official TensorFlow 2.4 stack failed at cusolverDnCreate on this RTX 4090.
- Full ST-VGP rows are explicit RTX 4090 OOM exclusions.
- CPU preparation, CPU X-lag, CPU postprocessing, and GPflow capacity preflights are excluded.
- No CPU-to-GPU or cross-device FLOP ratio is asserted.
EOF
  run git -C "${PUBLISH_WORKTREE}" add "autodl_results/${RUN_LABEL}"
  run git -C "${PUBLISH_WORKTREE}" -c user.name='AutoDL Results Publisher' -c user.email='results@local.invalid' commit -m "Add GPU-only RTX 4090 ERA5 matrix ${RUN_LABEL}"
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
}

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1
trap on_error ERR
cd "${ROOT}"

wait_for_worker
repair_routeb_prediction_contract
for manifest_name in shared_batch_short official_long_preflight official_long_full online_short online_long; do
  run_pending_manifest "${manifest_name}"
done
run "${ROUTEB_PY}" "${VERIFY}" --benchmark-root "${BENCHMARK_ROOT}" \
  --output "${BENCHMARK_ROOT}/gpu_only_protocol_verification.json"
run bash "${REPORT_RUNNER}" --repo "${ROOT}" --env "${ENV_ROOT}" --benchmark "${BENCHMARK_ROOT}"
publish_results
sync
printf 'SUCCESS utc=%s\n' "$(date -u +%FT%TZ)"
if [[ "${AUTO_SHUTDOWN}" == 1 ]]; then
  /usr/bin/shutdown -h now
fi
