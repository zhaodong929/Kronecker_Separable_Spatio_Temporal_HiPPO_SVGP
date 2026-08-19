#!/usr/bin/env bash
# GPU-only five-seed confirmation of the frozen Gaussian Route B configurations.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
RESULT_BRANCH="${RESULT_BRANCH:-codex/covid-formal-results-4090}"
OUTPUT_ROOT="$ROOT/results/diagnostics/covid_long_stream_2020_2024_mandatory_cloud4090_confirmation"
REPORT_ROOT="$ROOT/baselines/covid_long_setting_b/reports/cloud_4090_routeb_confirmation"
RUN_MANIFEST="$REPORT_ROOT/gpu_run_manifest.json"

run() {
    printf '\n[%s]' "$(date -Is)"
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

on_failure() {
    status=$?
    printf '\n[%s] GPU Route B confirmation failed with exit status %s; results were not pushed and the node will remain on.\n' "$(date -Is)" "$status" >&2
    exit "$status"
}
trap on_failure ERR

cd "$ROOT"
if [[ "$(git branch --show-current)" != "$RESULT_BRANCH" ]]; then
    run git switch "$RESULT_BRANCH"
fi

mkdir -p "$REPORT_ROOT"
run nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$REPORT_ROOT/gpu_hardware.csv"
run "$PYTHON" scripts/run_covid_long_stream_suite.py \
    --seeds 5 6 7 8 9 \
    --methods routeb_ordinary routeb_cumulative \
    --device cuda \
    --output-root "${OUTPUT_ROOT#$ROOT/}"
run "$PYTHON" baselines/covid_long_setting_b/evaluate_routeb_gpu_confirmation.py \
    --results-root "${OUTPUT_ROOT#$ROOT/}" \
    --output-dir "${REPORT_ROOT#$ROOT/}"
"$PYTHON" - <<'PY' > "$RUN_MANIFEST"
import json
import subprocess
import torch

print(json.dumps({
    "status": "complete",
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "device": torch.cuda.get_device_name(0),
    "cuda_available": torch.cuda.is_available(),
    "seeds": [5, 6, 7, 8, 9],
    "protocol": "52-week Task-1 plus 143-week strict online Setting B",
    "methods": ["routeb_ordinary", "routeb_cumulative"],
}, indent=2))
PY

git add -f \
    "${OUTPUT_ROOT#$ROOT/}" \
    "${REPORT_ROOT#$ROOT/}"
run git diff --cached --check
run git commit -m "Add RTX 4090 Route B COVID five-seed confirmation"
run git push origin "$RESULT_BRANCH"
run git ls-remote --exit-code --heads origin "$RESULT_BRANCH"
sync
/usr/bin/shutdown -h now
