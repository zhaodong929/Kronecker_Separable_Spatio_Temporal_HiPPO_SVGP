#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd929/projects/stvgp_kronecker"
BENCHMARK="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/iclr_era5_full_benchmark"
PYTHON="${ROOT}/.venv/bin/python"

cd "${ROOT}"

run_one() {
  local target="$1"
  local representation="$2"
  local seed="$3"
  local method_name
  local output_dir
  local pointwise=()

  if [[ "${target}" == "shared_xlag_residual" ]]; then
    method_name="routeb_shared_residual_${representation}"
  else
    method_name="routeb_direct_${representation}"
  fi
  output_dir="${BENCHMARK}/runs/task1_10/batch/${method_name}/seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    echo "skip complete ${method_name} seed${seed}"
    return
  fi
  if [[ "${seed}" == "0" ]]; then
    pointwise=(--save-pointwise)
  fi

  mkdir -p "${output_dir}"
  "${PYTHON}" scripts/run_iclr_era5_routeb_batch.py \
    --protocol-npz "${BENCHMARK}/protocol/task1_10/seed${seed}/protocol.npz" \
    --output-dir "${output_dir}" \
    --target-mode "${target}" \
    --representation "${representation}" \
    --mt 128 \
    --ms 128 \
    --iterations 100 \
    --learning-rate 0.02 \
    --validation-every 5 \
    --rff-sample-size 256 \
    --prediction-chunk-size 8192 \
    --split-seed "${seed}" \
    --model-seed 0 \
    "${pointwise[@]}" \
    >"${output_dir}/run.log" 2>&1
  echo "complete ${method_name} seed${seed}"
}

if [[ "$#" == 3 ]]; then
  run_one "$1" "$2" "$3"
  exit 0
fi

for seed in 0 1 2; do
  for target in direct shared_xlag_residual; do
    for representation in analytic_hippo_rff inducing_points; do
      run_one "${target}" "${representation}" "${seed}"
    done
  done
done
