#!/usr/bin/env bash
# Continue GPU-only development after the active LMC/IMC capacity queue exits.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
FSDE_PYTHON="${FSDE_PYTHON:-/root/autodl-tmp/stvgp_envs/fsde_svi_py310_cuda/bin/python}"
WAIT_PID_FILE="${WAIT_PID_FILE:-$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/gpu_execution_plan/factorial_lmc_imc_capacity_v3.pid}"
DEVELOPMENT_MANIFEST="${DEVELOPMENT_MANIFEST:-$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/cloud_4090_development_protocols/development_manifest.json}"
FACTORIAL_ROOT="${FACTORIAL_ROOT:-$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/gpu_execution_plan/blocked_development_factorial_v3}"
OHSVGP_GATE_ROOT="${OHSVGP_GATE_ROOT:-$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/gpu_execution_plan/ohsvgp_official_gates}"
OHSVGP_DEVELOPMENT_ROOT="${OHSVGP_DEVELOPMENT_ROOT:-$ROOT/baselines/covid_long_setting_b/results/convergence_repair_v1/gpu_execution_plan/blocked_development_ohsvgp_v2}"

wait_for_queue() {
    local pid
    pid="$(<"$WAIT_PID_FILE")"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
}

cd "$ROOT"
test -f "$WAIT_PID_FILE"
wait_for_queue
test -x "$FSDE_PYTHON"

JAX_PLATFORM_NAME=gpu "$FSDE_PYTHON" -c '
import jax
if jax.default_backend() != "gpu":
    raise SystemExit(f"FSDE CUDA gate failed: backend={jax.default_backend()}")
print(jax.devices())
'

"$FSDE_PYTHON" baselines/covid_long_setting_b/run_blocked_development.py \
    --development-manifest "${DEVELOPMENT_MANIFEST#$ROOT/}" \
    --output-root "${FACTORIAL_ROOT#$ROOT/}" \
    --methods lmc imc fsde --phase capacity --execute --resume \
    --factorial-device gpu --fsde-python "$FSDE_PYTHON" --jobs 3

"$FSDE_PYTHON" baselines/covid_long_setting_b/run_blocked_development.py \
    --development-manifest "${DEVELOPMENT_MANIFEST#$ROOT/}" \
    --output-root "${FACTORIAL_ROOT#$ROOT/}" \
    --methods lmc imc fsde --phase online_steps --execute --resume \
    --factorial-device gpu --fsde-python "$FSDE_PYTHON" --jobs 3

"$PYTHON" baselines/covid_long_setting_b/run_ohsvgp_reproduction_gates.py \
    --python "$PYTHON" --output-dir "${OHSVGP_GATE_ROOT#$ROOT/}" --execute

if "$PYTHON" -c '
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("status") == "passed" else 1)
' "$OHSVGP_GATE_ROOT/gate_status.json"; then
    "$PYTHON" baselines/covid_long_setting_b/run_blocked_development.py \
        --development-manifest "${DEVELOPMENT_MANIFEST#$ROOT/}" \
        --output-root "${OHSVGP_DEVELOPMENT_ROOT#$ROOT/}" \
        --methods ohsvgp --phase capacity --execute --ohsvgp-device cuda --jobs 3
else
    printf '%s\n' 'OHSVGP official gate did not pass; no OHSVGP development run was started.'
fi
