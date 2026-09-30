#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd929/projects/stvgp_kronecker"
BENCHMARK="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/iclr_era5_full_benchmark"
PYTHON="${ROOT}/.envs/gpflow_official_py310/bin/python"

cd "${ROOT}"

run_one() {
  local target="$1"
  local seed="$2"
  local name output_dir predictions=()
  if [[ "${target}" == "direct" ]]; then
    name="gpflow_svgp_direct_m512"
  else
    name="gpflow_svgp_shared_residual_m512"
  fi
  output_dir="${BENCHMARK}/runs/task1_10/batch/${name}/seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    echo "skip complete ${name} seed${seed}"
    return
  fi
  mkdir -p "${output_dir}"
  if [[ "${seed}" == "0" ]]; then
    predictions=(--predictions-output "${output_dir}/predictions.npz")
  fi
  "${PYTHON}" scripts/run_official_gpflow_svgp_era5.py \
    --protocol-npz "${BENCHMARK}/protocol/task1_10/seed${seed}/protocol.npz" \
    --output "${output_dir}/result.json" \
    "${predictions[@]}" \
    --target-mode "${target}" \
    --mt 8 \
    --ms 64 \
    --iterations 100 \
    --batch-size 2048 \
    --learning-rate 0.01 \
    --natgrad-gamma 0.1 \
    --validation-every 10 \
    --prediction-chunk-size 8192 \
    --seed "${seed}" \
    >"${output_dir}/run.log" 2>&1
  echo "complete ${name} seed${seed}"
}

if [[ "$#" == 2 ]]; then
  run_one "$1" "$2"
  exit 0
fi

for seed in 0 1 2; do
  run_one direct "${seed}"
  run_one shared_xlag_residual "${seed}"
done
