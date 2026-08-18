#!/usr/bin/env bash
# Run the short seed-0 pilot followed by the long strict-online seed-0 suite.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${AUTODL_ENV_ROOT:-/root/autodl-tmp/stvgp_envs}/routeb/bin/python"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/root/autodl-tmp/iclr_era5_stage2plus}"
RESULTS_BRANCH="${RESULTS_BRANCH:-codex/autodl-era5-results}"
CODE_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
RUN_LABEL="${RUN_LABEL:-short_then_long_online_seed0_$(date -u +%Y%m%dT%H%M%SZ)_${CODE_COMMIT:0:7}}"
LOG_DIR="${BENCHMARK_ROOT}/logs"
LOG_PATH="${LOG_DIR}/${RUN_LABEL}.log"
SHORT_AUDIT_CONFIG="${BENCHMARK_ROOT}/${RUN_LABEL}_short_audit_config.json"
SHORT_AUDIT_PATH="${BENCHMARK_ROOT}/${RUN_LABEL}_short_audit.json"
LONG_AUDIT_PATH="${BENCHMARK_ROOT}/${RUN_LABEL}_long_online_audit.json"
PUBLISH_WORKTREE="${PUBLISH_WORKTREE:-$(dirname "${BENCHMARK_ROOT}")/.autodl_results_publish_${RUN_LABEL}}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

on_error() {
    local exit_code=$?
    printf 'FAILED exit_code=%s utc=%s\n' "${exit_code}" "$(date -u +%FT%TZ)"
    printf 'Results and logs remain at %s. The instance will not be shut down.\n' "${BENCHMARK_ROOT}"
    exit "${exit_code}"
}
trap on_error ERR

run() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}

copy_without_predictions() {
    local source=$1
    local destination=$2
    mkdir -p "${destination}"
    (
        cd "${source}"
        tar --exclude='predictions.npz' -cf - .
    ) | (
        cd "${destination}"
        tar -xf -
    )
}

require_clean_code() {
    git -C "${ROOT}" diff --quiet
    git -C "${ROOT}" diff --cached --quiet
    if [[ -n "$(git -C "${ROOT}" status --porcelain --untracked-files=no)" ]]; then
        echo "Tracked repository changes are present; refusing to run an unreproducible pilot." >&2
        return 1
    fi
}

require_fresh_benchmark_root() {
    local name
    for name in protocol calibration runs efficiency report manifests; do
        if [[ -e "${BENCHMARK_ROOT}/${name}" ]]; then
            echo "Existing ${name} output found in ${BENCHMARK_ROOT}; choose a fresh BENCHMARK_ROOT to avoid mixing runs." >&2
            return 1
        fi
    done
}

write_short_audit_config() {
    "${PYTHON}" - "${ROOT}/cloud/autodl_era5/benchmark.json" "${SHORT_AUDIT_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
config = json.loads(source.read_text(encoding="utf-8"))
config["split_seeds"] = [0]
config["scopes"] = ["task1_2"]
destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY
}

audit_long_online() {
    "${PYTHON}" - "${ROOT}" "${BENCHMARK_ROOT}" "${LONG_AUDIT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
benchmark = Path(sys.argv[2])
output = Path(sys.argv[3])
sys.path.insert(0, str(root))

from scripts.audit_autodl_era5_benchmark import audit_protocols, audit_run, expected_methods

config = json.loads((root / "cloud/autodl_era5/benchmark.json").read_text(encoding="utf-8"))
config["split_seeds"] = [0]
config["scopes"] = ["task1_10"]
protocols, issues = audit_protocols(benchmark, config)
runs = []
for method in expected_methods(config, "online"):
    result = benchmark / "runs" / "task1_10" / "online" / method / "seed0" / "result.json"
    row = audit_run(
        result,
        scope="task1_10",
        branch="online",
        method=method,
        seed=0,
        expected_times=1674,
        expected_blocks=171,
    )
    runs.append(row)
    issues.extend(f"task1_10/online/{method}/seed0: {issue}" for issue in row["issues"])

payload = {
    "schema_version": 1,
    "verification_status": "VERIFIED" if not issues else "INCOMPLETE",
    "protocols": protocols,
    "runs": runs,
    "issues": issues,
    "boundary": "This audit covers the long strict-online suite only; no long fixed-global-basis batch run was scheduled.",
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"verification_status": payload["verification_status"], "runs": len(runs), "issues": len(issues)}, indent=2))
if issues:
    raise SystemExit("Long strict-online verification failed")
PY
}

verify_audits() {
    "${PYTHON}" - "${SHORT_AUDIT_PATH}" "${LONG_AUDIT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

for value in sys.argv[1:]:
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("verification_status") != "VERIFIED":
        raise SystemExit(f"Verification failed for {path}: {payload.get('issues', [])}")
print("Short and long verification status: VERIFIED")
PY
}

publish_verified_results() {
    local bundle_root
    local remote_commit
    local local_commit

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
    copy_without_predictions "${BENCHMARK_ROOT}" "${bundle_root}/benchmark"
    cp "${SHORT_AUDIT_CONFIG}" "${SHORT_AUDIT_PATH}" "${LONG_AUDIT_PATH}" "${bundle_root}/"
    cp "${LOG_PATH}" "${bundle_root}/logs/"
    git -C "${ROOT}" status --short > "${bundle_root}/code_status.txt"
    git -C "${ROOT}" log -1 --format=fuller > "${bundle_root}/code_commit.txt"

    (
        cd "${BENCHMARK_ROOT}"
        find . -type f -print0 | sort -z | xargs -0 sha256sum
    ) > "${bundle_root}/SHA256SUMS.txt"

    cat > "${bundle_root}/README.md" <<EOF
# AutoDL ERA5 short and long strict-online benchmark

- Run label: `${RUN_LABEL}`
- Code commit: `${CODE_COMMIT}`
- Protocol: Task 1 calibration, Task 1 -> Task 2 short batch/online evaluation,
  then Task 1 -> Task 10 strict-online evaluation, spatial split seed 0.
- Verification: `${SHORT_AUDIT_PATH}` and `${LONG_AUDIT_PATH}` both reported
  `VERIFIED` before this bundle was committed.
- Contents: result metadata, status files, commands, resource logs, block metrics,
  terminal log, protocol metadata, reports, and SHA-256 hashes for every original artifact.
- Deliberately excluded: `predictions.npz`, processed data, and Python environments.
  `SHA256SUMS.txt` covers every file in the original benchmark root, including
  excluded prediction archives. Processed data and environments remain on the instance.
EOF

    run git -C "${PUBLISH_WORKTREE}" add "autodl_results/${RUN_LABEL}"
    run git -C "${PUBLISH_WORKTREE}" -c user.name="AutoDL Results Publisher" -c user.email="results@local.invalid" commit -m "Add verified AutoDL ERA5 short and long run ${RUN_LABEL}"
    run git -C "${PUBLISH_WORKTREE}" push --set-upstream origin "${RESULTS_BRANCH}"

    local_commit="$(git -C "${PUBLISH_WORKTREE}" rev-parse HEAD)"
    remote_commit="$(git -C "${PUBLISH_WORKTREE}" ls-remote origin "refs/heads/${RESULTS_BRANCH}" | awk 'NR == 1 {print $1}')"
    [[ "${local_commit}" == "${remote_commit}" ]]
    printf 'Verified GitHub results commit: %s\n' "${local_commit}"
}

cd "${ROOT}"
require_clean_code
require_fresh_benchmark_root
[[ -x "${PYTHON}" ]]
run git ls-remote --exit-code origin HEAD
run git push --dry-run origin "${CODE_COMMIT}:refs/heads/${RESULTS_BRANCH}-write-check-${RUN_LABEL}"
run bash cloud/autodl_era5/validate_environments.sh

run "${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage prepare
run "${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage stage2 --scope calibration --seed 0
run "${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage stage2 --scope task1_2 --seed 0
run "${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage stage3 --scope task1_2 --seed 0
run "${PYTHON}" cloud/autodl_era5/run_benchmark.py --stage stage3 --scope task1_10 --seed 0

write_short_audit_config
run "${PYTHON}" scripts/audit_autodl_era5_benchmark.py \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --config "${SHORT_AUDIT_CONFIG}" \
    --output "${SHORT_AUDIT_PATH}"
audit_long_online
verify_audits

publish_verified_results
sync
printf 'SUCCESS utc=%s log=%s\n' "$(date -u +%FT%TZ)" "${LOG_PATH}"
/usr/bin/shutdown -h now
