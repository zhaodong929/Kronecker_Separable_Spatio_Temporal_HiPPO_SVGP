#!/usr/bin/env bash
# Finalize only independently gated COVID Setting B baselines on the 4090 node.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
OVC_PYTHON="$ROOT/baselines/.venvs/ovc_svgp/bin/python"
FSDE_PYTHON="${FSDE_PYTHON:-/root/autodl-tmp/stvgp_envs/fsde_svi_py310_cuda/bin/python}"
BASE="$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/gpu_execution_plan"
FACTORIAL_ROOT="$BASE/blocked_development_factorial_v3"
OHSVGP_ROOT="$BASE/blocked_development_ohsvgp_v2"
OVC_ROOT="$BASE/blocked_development_ovc_v2"
ST_ROOT="$BASE/blocked_development_st_svgp_v2"
OHSVGP_GATE="$BASE/ohsvgp_official_gates/gate_status.json"
OVC_AUDIT_ROOT="$BASE/ovc_selected_memory_audit_v2"
OVC_ASSESSMENT="$OVC_AUDIT_ROOT/assessment.json"
ENVIRONMENT_LOCK="$BASE/environment_lock_4090_repaired_v2.json"
FAIRNESS_LOCK="$ROOT/baselines/covid_long_setting_b/BASELINE_FAIRNESS_PROTOCOL_4090_REPAIRED_V3.json"
FORMAL_ROOT="$ROOT/baselines/covid_long_setting_b/results/formal_repaired_4090_v3"
REPORT_ROOT="$ROOT/baselines/covid_long_setting_b/reports/formal_gaussian_repaired_4090_v3"
HARDWARE="$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/cloud_4090_preflight.json"
FROZEN_ARCHIVES="$ROOT/baselines/covid_long_setting_b/reproduction/convergence_repair_v1/frozen_pre_repair_archives.json"
RESULT_BRANCH="${RESULT_BRANCH:-codex/covid-formal-results-4090}"
PID_FILES=(
    "$BASE/gpu_repair_chain_v3.pid"
    "$BASE/ovc_capacity_v2.pid"
    "$BASE/st_svgp_capacity_v2.pid"
)

run() {
    printf '\n[%s]' "$(date -Is)"
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

wait_for_pid_file() {
    local pid_file="$1"
    local pid
    test -f "$pid_file"
    pid="$(<"$pid_file")"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
}

on_failure() {
    local status=$?
    printf '\n[%s] repaired finalization failed with exit status %s; results were not pushed and the node remains on.\n' \
        "$(date -Is)" "$status" >&2
    exit "$status"
}
trap on_failure ERR

cd "$ROOT"
for pid_file in "${PID_FILES[@]}"; do
    wait_for_pid_file "$pid_file"
done

if [[ ! -f "$OVC_ASSESSMENT" ]]; then
    readarray -t OVC_CAPACITY < <("$PYTHON" - "$OVC_ROOT/capacity/capacity_selection.json" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
candidate = record.get("selected", {}).get("ovc", {}).get("candidate", {})
for key in ("temporal_inducing", "spatial_inducing"):
    print(candidate.get(key, ""))
PY
)
    if [[ -n "${OVC_CAPACITY[0]}" && -n "${OVC_CAPACITY[1]}" ]]; then
        run env "PYTHONPATH=$ROOT" "$OVC_PYTHON" baselines/covid_long_setting_b/run_ovc_memory_audit.py \
            --output-dir "${OVC_AUDIT_ROOT#$ROOT/}" --replicate-id 1 \
            --temporal-inducing "${OVC_CAPACITY[0]}" --spatial-inducing "${OVC_CAPACITY[1]}"
        run env "PYTHONPATH=$ROOT" "$OVC_PYTHON" baselines/covid_long_setting_b/run_ovc_memory_audit.py \
            --output-dir "${OVC_AUDIT_ROOT#$ROOT/}" --replicate-id 2 \
            --temporal-inducing "${OVC_CAPACITY[0]}" --spatial-inducing "${OVC_CAPACITY[1]}"
        run "$PYTHON" baselines/covid_long_setting_b/assess_ovc_memory_audits.py \
            --audit-root "${OVC_AUDIT_ROOT#$ROOT/}" --output "${OVC_ASSESSMENT#$ROOT/}"
    else
        mkdir -p "$(dirname "$OVC_ASSESSMENT")"
        printf '%s\n' '{"status": "not_run_no_passing_capacity"}' > "$OVC_ASSESSMENT"
    fi
fi

run "$PYTHON" baselines/covid_long_setting_b/capture_environment_locks.py \
    --environment "ohsvgp=$PYTHON" \
    --environment "ovc=$OVC_PYTHON" \
    --environment "st_svgp=$ROOT/baselines/.venvs/st_svgp/bin/python" \
    --environment "factorial_gpflow=$ROOT/baselines/.venvs/factorial_sde_gpflow/bin/python" \
    --environment "factorial_fsde=$FSDE_PYTHON" \
    --output "${ENVIRONMENT_LOCK#$ROOT/}"
run "$PYTHON" baselines/covid_long_setting_b/generate_baseline_fairness_protocol.py \
    --factorial-development-root "${FACTORIAL_ROOT#$ROOT/}" \
    --ohsvgp-development-root "${OHSVGP_ROOT#$ROOT/}" \
    --ovc-development-root "${OVC_ROOT#$ROOT/}" \
    --st-svgp-development-root "${ST_ROOT#$ROOT/}" \
    --ohsvgp-gate "${OHSVGP_GATE#$ROOT/}" \
    --ovc-memory-assessment "${OVC_ASSESSMENT#$ROOT/}" \
    --environment-lock "${ENVIRONMENT_LOCK#$ROOT/}" \
    --hardware-fingerprint "${HARDWARE#$ROOT/}" \
    --frozen-archives "${FROZEN_ARCHIVES#$ROOT/}" \
    --formal-result-root "${FORMAL_ROOT#$ROOT/}" \
    --output "${FAIRNESS_LOCK#$ROOT/}"
run "$PYTHON" baselines/covid_long_setting_b/run_locked_formal.py \
    --fairness-protocol "${FAIRNESS_LOCK#$ROOT/}" --execute --resume
run "$PYTHON" baselines/covid_long_setting_b/evaluate_formal_gaussian.py \
    --fairness-protocol "${FAIRNESS_LOCK#$ROOT/}" --output-dir "${REPORT_ROOT#$ROOT/}"

git add -f \
    "${FAIRNESS_LOCK#$ROOT/}" \
    "${ENVIRONMENT_LOCK#$ROOT/}" \
    "${OVC_ASSESSMENT#$ROOT/}" \
    "${HARDWARE#$ROOT/}"
while IFS= read -r -d '' artifact; do
    git add -f "${artifact#$ROOT/}"
done < <(find "$FACTORIAL_ROOT" "$OHSVGP_ROOT" "$OVC_ROOT" "$ST_ROOT" "$OVC_AUDIT_ROOT" "$FORMAL_ROOT" "$REPORT_ROOT" \
    -type f \( -name '*.json' -o -name '*.csv' -o -name '*.md' -o -name '*.tex' -o -name 'predictions.npz' \) -print0)
while IFS= read -r -d '' artifact; do
    git add -f "${artifact#$ROOT/}"
done < <(find "$(dirname "$OHSVGP_GATE")" -type f \( -name '*.json' -o -name '*.log' -o -name '*.pass' \) -print0)
run git diff --cached --check
run git commit -m "Add repaired COVID 4090 baseline results"
run git push origin "$RESULT_BRANCH"
local_head="$(git rev-parse HEAD)"
remote_head="$(git ls-remote origin "refs/heads/$RESULT_BRANCH" | cut -f1)"
test "$local_head" = "$remote_head"
sync
/usr/bin/shutdown -h now
