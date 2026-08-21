#!/usr/bin/env bash
# GPU-only accelerated exploratory ST-SVGP Setting B run on AutoDL CUDA 11.8.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ST_PYTHON="${ST_PYTHON:-/root/autodl-tmp/stvgp_envs/st_svgp_py38_cuda/bin/python}"
ST_CUDA_PATH="${ST_CUDA_PATH:-/usr/local/cuda-11.8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/baselines/covid_long_setting_b/results/formal_repaired_4090_v5/st_svgp_accelerated_capacity_unresolved_cuda118_v2}"
if [[ "$#" -eq 0 ]]; then
    SEEDS=(5 6 7)
else
    SEEDS=("$@")
fi

test -x "$ST_PYTHON"
test -f "$ST_CUDA_PATH/bin/ptxas"
test -d "$ST_CUDA_PATH/nvvm/libdevice"
mkdir -p "$OUTPUT_ROOT"

run_seed() {
    local seed="$1"
    local output="$OUTPUT_ROOT/seed$seed/st_svgp"
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
        --online-inference-steps 5 --task1-iterations 7500 --task1-check-interval 250 \
        --task1-min-steps 2500 --task1-plateau-checks 4 \
        --task1-plateau-relative-improvement 0.001 > "$output/run.log" 2>&1
}

cd "$ROOT"
printf '{"status":"running","kind":"accelerated_exploratory_capacity_unresolved","spatial_inducing":52,"task1_max_steps":7500,"task1_min_steps":2500,"task1_plateau_checks":4,"online_inference_steps":5,"seeds":[%s]}\n' "$(IFS=,; echo "${SEEDS[*]}")" > "$OUTPUT_ROOT/run_manifest.json"

pids=()
for seed in "${SEEDS[@]}"; do
    run_seed "$seed" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if [[ "$status" -eq 0 ]]; then
    sed -i 's/"running"/"complete"/' "$OUTPUT_ROOT/run_manifest.json"
else
    sed -i 's/"running"/"incomplete_or_failed"/' "$OUTPUT_ROOT/run_manifest.json"
fi
exit "$status"
