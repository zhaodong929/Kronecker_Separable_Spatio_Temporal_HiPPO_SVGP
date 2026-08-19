#!/usr/bin/env bash
# Run the repaired COVID Setting B benchmark and shut down only after a pushed result branch.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
OVC_PYTHON="$ROOT/baselines/.venvs/ovc_svgp/bin/python"
DEVELOPMENT_PROTOCOLS="$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/cloud_4090_development_protocols"
DEVELOPMENT_MANIFEST="$DEVELOPMENT_PROTOCOLS/development_manifest.json"
DEVELOPMENT_ROOT="$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/blocked_development_4090"
OVC_AUDIT_ROOT="$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/ovc_memory_audit_4090"
OVC_ASSESSMENT="$OVC_AUDIT_ROOT/assessment.json"
ENVIRONMENT_LOCK="$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/environment_lock_4090.json"
FAIRNESS_LOCK="$ROOT/baselines/covid_long_setting_b/BASELINE_FAIRNESS_PROTOCOL.json"
FORMAL_ROOT="$ROOT/baselines/covid_long_setting_b/results/formal_repaired_4090_v1"
REPORT_ROOT="$ROOT/baselines/covid_long_setting_b/reports/formal_gaussian_repaired_4090_v1"
RESULT_BRANCH="${RESULT_BRANCH:-codex/covid-formal-results-4090}"
HARDWARE="$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/cloud_4090_preflight.json"
OHSVGP_GATE="$ROOT/baselines/covid_long_setting_b/reproduction/convergence_repair_v1/ohsvgp/gate_status.json"
OHSVGP_REPRODUCTION_ROOT="$ROOT/baselines/covid_long_setting_b/reproduction/convergence_repair_v1/ohsvgp"
FROZEN_ARCHIVES="$ROOT/baselines/covid_long_setting_b/reproduction/convergence_repair_v1/frozen_pre_repair_archives.json"

run() {
    printf '\n[%s] %q' "$(date -Is)" "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

on_failure() {
    status=$?
    printf '\n[%s] pipeline failed with exit status %s; results were not pushed and the node will remain on.\n' "$(date -Is)" "$status" >&2
    exit "$status"
}
trap on_failure ERR

cd "$ROOT"
if git show-ref --verify --quiet "refs/heads/$RESULT_BRANCH"; then
    run git switch "$RESULT_BRANCH"
else
    run git switch -c "$RESULT_BRANCH"
fi

run "$PYTHON" baselines/covid_long_setting_b/build_blocked_development_protocols.py \
    --output-root "${DEVELOPMENT_PROTOCOLS#$ROOT/}"
run "$PYTHON" baselines/covid_long_setting_b/run_blocked_development.py \
    --development-manifest "${DEVELOPMENT_MANIFEST#$ROOT/}" \
    --output-root "${DEVELOPMENT_ROOT#$ROOT/}" \
    --methods lmc imc fsde ovc st_svgp --phase capacity --execute --resume
run "$PYTHON" baselines/covid_long_setting_b/run_blocked_development.py \
    --development-manifest "${DEVELOPMENT_MANIFEST#$ROOT/}" \
    --output-root "${DEVELOPMENT_ROOT#$ROOT/}" \
    --methods lmc imc fsde --phase online_steps --execute --resume

OVC_TEMPORAL_INDUCING="$($PYTHON -c '
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
candidate = record.get("selected", {}).get("ovc", {}).get("candidate", {})
print(candidate.get("temporal_inducing", ""))
' "$DEVELOPMENT_ROOT/capacity/capacity_selection.json")"
mkdir -p "$OVC_AUDIT_ROOT"
if [[ -n "$OVC_TEMPORAL_INDUCING" ]]; then
    run env "PYTHONPATH=$ROOT" "$OVC_PYTHON" baselines/covid_long_setting_b/run_ovc_memory_audit.py \
        --output-dir "${OVC_AUDIT_ROOT#$ROOT/}" --replicate-id 1 --temporal-inducing "$OVC_TEMPORAL_INDUCING"
    run env "PYTHONPATH=$ROOT" "$OVC_PYTHON" baselines/covid_long_setting_b/run_ovc_memory_audit.py \
        --output-dir "${OVC_AUDIT_ROOT#$ROOT/}" --replicate-id 2 --temporal-inducing "$OVC_TEMPORAL_INDUCING"
    run "$PYTHON" baselines/covid_long_setting_b/assess_ovc_memory_audits.py \
        --audit-root "${OVC_AUDIT_ROOT#$ROOT/}" --output "${OVC_ASSESSMENT#$ROOT/}"
else
    "$PYTHON" -c '
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"status": "not_run_no_passing_capacity"}, indent=2) + "\n", encoding="utf-8")
' "$OVC_ASSESSMENT"
fi

run "$PYTHON" baselines/covid_long_setting_b/capture_environment_locks.py \
    --environment "ohsvgp=$PYTHON" \
    --environment "ovc=$OVC_PYTHON" \
    --environment "st_svgp=$ROOT/baselines/.venvs/st_svgp/bin/python" \
    --environment "factorial_gpflow=$ROOT/baselines/.venvs/factorial_sde_gpflow/bin/python" \
    --environment "factorial_fsde=$ROOT/baselines/.venvs/factorial_sde_fsde39/bin/python" \
    --output "${ENVIRONMENT_LOCK#$ROOT/}"
run "$PYTHON" baselines/covid_long_setting_b/generate_baseline_fairness_protocol.py \
    --development-root "${DEVELOPMENT_ROOT#$ROOT/}" \
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
    "${HARDWARE#$ROOT/}" \
    "${OHSVGP_GATE#$ROOT/}"
while IFS= read -r -d '' artifact; do
    git add -f "${artifact#$ROOT/}"
done < <(find "$OHSVGP_REPRODUCTION_ROOT" -type f \( -name '*.json' -o -name '*.log' -o -name '*.pass' -o -name '*.md' \) -print0)
while IFS= read -r -d '' artifact; do
    git add -f "${artifact#$ROOT/}"
done < <(find "$DEVELOPMENT_ROOT" "$OVC_AUDIT_ROOT" "$FORMAL_ROOT" "$REPORT_ROOT" -type f \( -name '*.json' -o -name '*.csv' -o -name '*.md' -o -name '*.tex' -o -name 'predictions.npz' \) -print0)
for seed in 5 6 7 8 9; do
    git add -f \
        "results/diagnostics/covid_long_stream_2020_2024_mandatory/seed${seed}/deterministic/persistence/predictions.npz" \
        "results/diagnostics/covid_long_stream_2020_2024_mandatory/seed${seed}/deterministic/lag_ridge/predictions.npz" \
        "results/diagnostics/covid_long_stream_2020_2024_mandatory/seed${seed}/routeb_ordinary/online/predictions.npz" \
        "results/diagnostics/covid_long_stream_2020_2024_mandatory/seed${seed}/routeb_cumulative/online/predictions.npz" \
        "baselines/covid_long_setting_b/results/formal_selected_bui_ovc_shared_m8/seed${seed}/bui_controlled/predictions.npz" \
        "baselines/covid_long_setting_b/results/formal_selected_bui_ovc_shared_m8/seed${seed}/bui_adaptive/predictions.npz"
done
run git diff --cached --check
run git commit -m "Add COVID 4090 repaired baseline results"
run git push origin "$RESULT_BRANCH"
run git ls-remote --exit-code --heads origin "$RESULT_BRANCH"
sync
/usr/bin/shutdown -h now
