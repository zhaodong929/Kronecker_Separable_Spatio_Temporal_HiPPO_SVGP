#!/usr/bin/env bash
# Run the remaining budget-relaxed 4090 baselines, validate artifacts, publish, then shut down.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
IMC_PYTHON="${IMC_PYTHON:-/root/autodl-tmp/stvgp_envs/factorial_gpflow_py310/bin/python}"
FSDE_PYTHON="${FSDE_PYTHON:-/root/autodl-tmp/stvgp_envs/fsde_svi_py310_cuda/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/baselines/covid_long_setting_b/results/formal_repaired_4090_v5}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/baselines/covid_long_setting_b/reports/exploratory_gpu_4090_s5_s7}"
LMC_ROOT="${LMC_ROOT:-$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/gpu_execution_plan/prelock_lmc_gpu_s5_s7_v1}"
OVC_AUDIT_ROOT="${OVC_AUDIT_ROOT:-$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/gpu_execution_plan/ovc_selected_memory_audit_accelerated_v3}"
ST_CHECKPOINT_ROOT="${ST_CHECKPOINT_ROOT:-$RESULT_ROOT/st_svgp_accelerated_capacity_unresolved_cuda118_v2}"
ST_OUTPUT_ROOT="${ST_OUTPUT_ROOT:-$RESULT_ROOT/st_svgp_short_budget_2500_from_checkpoint_cuda118_v3}"
ST_PYTHON="${ST_PYTHON:-/root/autodl-tmp/stvgp_envs/st_svgp_py38_cuda/bin/python}"
ST_CUDA_PATH="${ST_CUDA_PATH:-/usr/local/cuda-11.8}"
BRANCH="${RESULT_BRANCH:-codex/covid-formal-results-4090}"
STATUS_PATH="$RESULT_ROOT/AUTO_FINALIZATION_STATUS.json"
ST_PIDS="${ST_PIDS:-955679 955680 955681}"
SEEDS=(5 6 7)

write_status() {
    local state="$1"
    local detail="$2"
    "$PYTHON" - "$STATUS_PATH" "$state" "$detail" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "state": sys.argv[2],
    "detail": sys.argv[3],
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "shutdown_allowed_only_after_remote_sha_match": True,
}, indent=2) + "\n", encoding="utf-8")
PY
}

on_failure() {
    local code=$?
    write_status "failed_before_publish" "pipeline error (exit $code); instance intentionally left running"
    exit "$code"
}
trap on_failure ERR

wait_for_existing_st() {
    for pid in $ST_PIDS; do
        while kill -0 "$pid" 2>/dev/null; do
            sleep 30
        done
    done
}

run_parallel() {
    local method="$1"
    shift
    local pids=()
    local status=0
    for seed in "${SEEDS[@]}"; do
        "$@" "$seed" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    return "$status"
}

run_method_with_serial_fallback() {
    local method="$1"
    shift
    if run_parallel "$method" "$@"; then
        return 0
    fi
    write_status "${method}_parallel_failed_retrying_serial" "parallel GPU attempt failed; retrying incomplete seeds serially"
    local status=0
    for seed in "${SEEDS[@]}"; do
        "$@" "$seed" || status=1
    done
    return "$status"
}

run_imc_seed() {
    local seed="$1"
    local output="$RESULT_ROOT/imc_accelerated_budget_relaxed_gpu_v1/seed$seed/imc_svgp"
    mkdir -p "$output"
    if [[ -f "$output/predictions.npz" && -f "$output/result.json" ]]; then
        return 0
    fi
    env TF_FORCE_GPU_ALLOW_GROWTH=true "$IMC_PYTHON" \
        baselines/covid_long_setting_b/adapters/run_factorial_lmc_imc.py \
        --protocol-npz "data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed$seed/protocol.npz" \
        --protocol-json "data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed$seed/protocol.json" \
        --output-dir "$output" --seed "$seed" --method imc --device gpu \
        --temporal-inducing 32 --latent-rank 8 --batch-size 16 \
        --task1-iterations 7500 --task1-check-interval 250 --task1-min-steps 2500 \
        --task1-plateau-checks 4 --task1-plateau-relative-improvement 0.001 \
        --online-inference-steps 25 > "$output/run.log" 2>&1
}

run_st_checkpoint_seed() {
    local seed="$1"
    local checkpoint="$ST_CHECKPOINT_ROOT/seed$seed/st_svgp/task1_checkpoints/checkpoint_02500.npz"
    local output="$ST_OUTPUT_ROOT/seed$seed/st_svgp"
    if [[ -f "$output/predictions.npz" && -f "$output/status.json" ]]; then
        return 0
    fi
    test -f "$checkpoint"
    mkdir -p "$output"
    env \
        "PATH=$ST_CUDA_PATH/bin:$ST_CUDA_PATH/nvvm/bin:/usr/bin:/bin" \
        "XLA_FLAGS=--xla_gpu_cuda_data_dir=$ST_CUDA_PATH" \
        "JAX_PLATFORM_NAME=gpu" \
        "XLA_PYTHON_CLIENT_PREALLOCATE=false" \
        "$ST_PYTHON" baselines/covid_long_setting_b/adapters/run_st_svgp.py \
        --protocol-npz "data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed$seed/protocol.npz" \
        --protocol-json "data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed$seed/protocol.json" \
        --output-dir "$output" --seed "$seed" --spatial-inducing 52 \
        --online-inference-steps 5 --frozen-task1-state "$checkpoint" > "$output/run.log" 2>&1
}

run_fsde_seed() {
    local seed="$1"
    local output="$RESULT_ROOT/fsde_accelerated_budget_relaxed_gpu_v1/seed$seed/fsde_svi"
    mkdir -p "$output"
    if [[ -f "$output/predictions.npz" && -f "$output/result.json" ]]; then
        return 0
    fi
    env JAX_PLATFORM_NAME=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false "$FSDE_PYTHON" \
        baselines/covid_long_setting_b/adapters/run_factorial_fsde_svi.py \
        --protocol-npz "data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed$seed/protocol.npz" \
        --protocol-json "data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed$seed/protocol.json" \
        --output-dir "$output" --seed "$seed" --device gpu \
        --temporal-inducing 16 --latent-rank 4 --batch-size 16 \
        --task1-iterations 7500 --task1-check-interval 250 --task1-min-steps 2500 \
        --task1-plateau-checks 4 --task1-plateau-relative-improvement 0.001 \
        --online-inference-steps 25 > "$output/run.log" 2>&1
}

stage_artifacts() {
    git add -f \
        CLOUD_HANDOFF.md \
        baselines/covid_long_setting_b/finalize_exploratory_gpu_results.py \
        baselines/covid_long_setting_b/run_expedited_gpu_chain_and_publish.sh \
        baselines/covid_long_setting_b/run_stsvgp_accelerated_formal_cuda118.sh \
        "$STATUS_PATH" \
        "$REPORT_ROOT"
    git add -f "$OVC_AUDIT_ROOT/assessment.json"
    git add -f "$OVC_AUDIT_ROOT/replicate_1/memory_audit.json"
    git add -f "$OVC_AUDIT_ROOT/replicate_2/memory_audit.json"
    while IFS= read -r -d '' artifact; do
        git add -f "${artifact#$ROOT/}"
    done < <(find "$LMC_ROOT" "$RESULT_ROOT" -type f \( \
        -name 'predictions.npz' -o -name 'result.json' -o -name 'status.json' \
        -o -name 'run_manifest.json' -o -name 'staging_validation.json' \
    \) -print0 2>/dev/null)
}

require_complete_report() {
    local validation="$REPORT_ROOT/validation_status.json"
    test -f "$validation"
    "$PYTHON" - "$validation" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {"lmc_svgp", "st_svgp", "imc_svgp", "fsde_svi"}
actual = set(status.get("aggregate_methods", []))
invalid = int(status.get("invalid_or_missing_archives", -1))
if status.get("status") != "complete" or invalid != 0 or actual != expected:
    raise SystemExit(
        "complete-report gate failed: "
        f"status={status.get('status')!r}, invalid={invalid}, methods={sorted(actual)}"
    )
PY
}

publish_and_shutdown() {
    local current_branch
    current_branch="$(git branch --show-current)"
    test "$current_branch" = "$BRANCH"
    export GIT_INDEX_FILE
    GIT_INDEX_FILE="$(mktemp "$ROOT/.git/expedited-results-index.XXXXXX")"
    git read-tree HEAD
    stage_artifacts
    git diff --cached --check
    if ! git diff --cached --quiet; then
        git commit -m "Add accelerated COVID 4090 exploratory results"
    fi
    git push origin "HEAD:refs/heads/$BRANCH"
    local_head="$(git rev-parse HEAD)"
    remote_head="$(git ls-remote origin "refs/heads/$BRANCH" | cut -f1)"
    test "$local_head" = "$remote_head"
    write_status "push_verified_shutdown_pending" "remote branch $BRANCH matches $local_head"
    git add -f "$STATUS_PATH"
    git commit -m "Record verified COVID cloud publication"
    git push origin "HEAD:refs/heads/$BRANCH"
    local_head="$(git rev-parse HEAD)"
    remote_head="$(git ls-remote origin "refs/heads/$BRANCH" | cut -f1)"
    test "$local_head" = "$remote_head"
    sync
    /usr/bin/shutdown -h now
}

cd "$ROOT"
test -x "$PYTHON"
test -x "$IMC_PYTHON"
test -x "$FSDE_PYTHON"
test -x "$ST_PYTHON"
test -f "$ST_CUDA_PATH/bin/ptxas"
write_status "waiting_for_st_svgp" "waiting for the active three-seed ST-SVGP run"
wait_for_existing_st
write_status "running_st_svgp_checkpoint_online" "running the full online stream from each 2,500-update ST-SVGP checkpoint"
if ! run_method_with_serial_fallback st_svgp run_st_checkpoint_seed; then
    write_status "st_svgp_checkpoint_process_failure" "one or more ST-SVGP checkpoint-stream subprocesses exited nonzero; report will retain the failure evidence"
fi
write_status "running_imc_svgp" "starting three GPU IMC-SVGP seeds with the budget-relaxed configuration"
if ! run_method_with_serial_fallback imc run_imc_seed; then
    write_status "imc_process_failure" "one or more IMC-SVGP subprocesses exited nonzero; report will retain the failure evidence"
fi
write_status "running_fsde_svi" "starting three GPU FSDE-SVI seeds with the budget-relaxed configuration"
if ! run_method_with_serial_fallback fsde run_fsde_seed; then
    write_status "fsde_process_failure" "one or more FSDE-SVI subprocesses exited nonzero; report will retain the failure evidence"
fi
"$PYTHON" baselines/covid_long_setting_b/finalize_exploratory_gpu_results.py \
    --output-root "${RESULT_ROOT#$ROOT/}" --lmc-root "${LMC_ROOT#$ROOT/}" \
    --report-dir "${REPORT_ROOT#$ROOT/}" --seeds "${SEEDS[@]}"
require_complete_report
write_status "validation_complete" "exploratory archive and metric report generated; publishing result branch"
publish_and_shutdown
