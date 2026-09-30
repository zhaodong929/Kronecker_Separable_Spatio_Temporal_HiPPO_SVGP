#!/usr/bin/env bash
set -u

ROOT=/home/zd929/projects/stvgp_kronecker
OUT="$ROOT/results/era5_stage2plus_sm_baseline_rerun_20260828_formal"
PROTO="/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/kronecker+s2vgp/ICLR Formal experiment/iclr_era5_stage2plus/protocol"
THETA="$OUT/calibration/external_matern32_control"

run_seed() {
    local seed="$1"
    local output="$OUT/runs/long/online/ohsvgp/seed${seed}"
    mkdir -p "$output"
    printf 'START seed=%s %s\n' "$seed" "$(date -Is)" > "$output/parallel_runner.log"
    started=$(date +%s)
    "$ROOT/.venv_cuda128/bin/python" "$ROOT/scripts/run_official_ohsvgp_era5.py" \
        --protocol-npz "$PROTO/task1_10/seed${seed}/protocol.npz" \
        --theta-json "$THETA/seed${seed}/result.json" \
        --output "$output/result.json" \
        --blockwise-output "$output/blocks.csv" \
        --predictions-output "$output/predictions.npz" \
        --inducing-size 128 \
        --rff-sample-size 256 \
        --microbatch-size 200 \
        --update-steps 1 \
        --seed "$seed" \
        --device cuda \
        --dtype float64 \
        > "$output/run_lazy_float64.log" 2>&1
    local rc=$?
    finished=$(date +%s)
    printf 'END seed=%s rc=%s %s\n' "$seed" "$rc" "$(date -Is)" >> "$output/parallel_runner.log"
    "$ROOT/.venv_cuda128/bin/python" - "$output/status.json" "$output" "$seed" "$rc" "$started" "$finished" <<'PY'
import json
import sys
from pathlib import Path

status_path, output, seed, rc, started, finished = sys.argv[1:]
path = Path(status_path)
record = json.loads(path.read_text()) if path.exists() else {
    "name": f"long/online/ohsvgp/seed{seed}",
    "method": "ohsvgp", "scope": "long", "setting": "online",
    "seed": int(seed), "output": output, "evaluation": True,
}
record.update(
    status="complete" if int(rc) == 0 and Path(output, "predictions.npz").is_file() else "failed",
    returncode=int(rc), elapsed_seconds=int(finished) - int(started),
)
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
    return "$rc"
}

for seed in 0 1 2 3 4; do
    run_seed "$seed" &
done
wait
printf 'OHSVGP_PARALLEL_DONE %s\n' "$(date -Is)"
