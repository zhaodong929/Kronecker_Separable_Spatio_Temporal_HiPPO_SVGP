#!/usr/bin/env bash
# Publish the completed exploratory 4090 artifacts from a no-GPU instance.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="${RESULT_BRANCH:-codex/covid-formal-results-4090}"
RESULT_ROOT="$ROOT/baselines/covid_long_setting_b/results/formal_repaired_4090_v5"
LMC_ROOT="$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/gpu_execution_plan/prelock_lmc_gpu_s5_s7_v1"
ST_CAPACITY="$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/gpu_execution_plan/blocked_development_st_svgp_v2/capacity/capacity_selection.json"
OVC_AUDIT="$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/gpu_execution_plan/ovc_selected_memory_audit_accelerated_v3"
REPORT_ROOT="$ROOT/baselines/covid_long_setting_b/reports/exploratory_gpu_4090_s5_s7"

cd "$ROOT"
test "$(git branch --show-current)" = "$BRANCH"

export GIT_INDEX_FILE
GIT_INDEX_FILE="$(mktemp "$ROOT/.git/exploratory-results-index.XXXXXX")"
cleanup() {
    rm -f "$GIT_INDEX_FILE"
}
trap cleanup EXIT
git read-tree HEAD

stage_path() {
    test -e "$1"
    git add -f "$1"
}

stage_path CLOUD_HANDOFF.md
stage_path baselines/covid_long_setting_b/adapters/run_factorial_fsde_svi.py
stage_path baselines/covid_long_setting_b/adapters/run_factorial_lmc_imc.py
stage_path baselines/covid_long_setting_b/adapters/run_ovc_svgp.py
stage_path baselines/covid_long_setting_b/adapters/run_st_svgp.py
stage_path baselines/covid_long_setting_b/evaluate_formal_gaussian.py
stage_path baselines/covid_long_setting_b/finalize_exploratory_gpu_results.py
stage_path baselines/covid_long_setting_b/generate_baseline_fairness_protocol.py
stage_path baselines/covid_long_setting_b/publish_exploratory_no_gpu_results.sh
stage_path baselines/covid_long_setting_b/run_blocked_development.py
stage_path baselines/covid_long_setting_b/run_expedited_gpu_chain_and_publish.sh
stage_path baselines/covid_long_setting_b/run_locked_formal.py
stage_path baselines/covid_long_setting_b/run_ovc_memory_audit.py
stage_path baselines/covid_long_setting_b/run_stsvgp_accelerated_formal_cuda118.sh
stage_path "$REPORT_ROOT"
stage_path "$ST_CAPACITY"
stage_path "$RESULT_ROOT/AUTO_FINALIZATION_STATUS.json"
stage_path "$RESULT_ROOT/st_svgp_accelerated_capacity_unresolved_cuda118_v2/run_manifest.json"
stage_path "$OVC_AUDIT/assessment.json"
stage_path "$OVC_AUDIT/replicate_1/memory_audit.json"
stage_path "$OVC_AUDIT/replicate_2/memory_audit.json"

while IFS= read -r -d '' artifact; do
    git add -f "$artifact"
done < <(find "$LMC_ROOT" \
    "$RESULT_ROOT/imc_accelerated_budget_relaxed_gpu_v1" \
    "$RESULT_ROOT/fsde_accelerated_budget_relaxed_gpu_v1" \
    -type f \( -name predictions.npz -o -name result.json -o -name run_manifest.json \) -print0)

git diff --cached --check
archive_count="$(git diff --cached --name-only | grep -c '/predictions.npz$' || true)"
test "$archive_count" = 9

git commit -m "Archive exploratory COVID 4090 baseline results"
git push origin "HEAD:refs/heads/$BRANCH"

local_head="$(git rev-parse HEAD)"
remote_head="$(git ls-remote origin "refs/heads/$BRANCH" | awk 'NR == 1 {print $1}')"
test "$local_head" = "$remote_head"
printf 'PUSH_VERIFIED branch=%s sha=%s archives=%s\n' "$BRANCH" "$local_head" "$archive_count"
