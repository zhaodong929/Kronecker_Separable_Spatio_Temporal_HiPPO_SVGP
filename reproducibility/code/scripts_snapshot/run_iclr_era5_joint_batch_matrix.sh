#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd929/projects/stvgp_kronecker"
BENCHMARK="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/iclr_era5_full_benchmark"
cd "${ROOT}"

run_one() {
  local scope="$1"
  local representation="$2"
  local seed="$3"
  local output_dir="${BENCHMARK}/runs/${scope}/batch/routeb_joint_xlag_${representation}/seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    echo "skip complete ${scope} joint_xlag ${representation} seed${seed}"
    return
  fi
  mkdir -p "${output_dir}"
  .venv/bin/python scripts/run_iclr_era5_routeb_batch.py \
    --protocol-npz "${BENCHMARK}/protocol/${scope}/seed${seed}/protocol.npz" \
    --protocol-json "${BENCHMARK}/protocol/${scope}/seed${seed}/protocol.json" \
    --output-dir "${output_dir}" \
    --target-mode joint_xlag \
    --representation "${representation}" \
    --mt 128 \
    --ms 128 \
    --iterations 100 \
    --learning-rate 0.02 \
    --validation-every 5 \
    --beta-prior-variance 1000 \
    --rff-sample-size 256 \
    --xlag-length 10 \
    --prediction-chunk-size 8192 \
    --split-seed "${seed}" \
    --model-seed 0 \
    >"${output_dir}/run.log" 2>&1
  echo "complete ${scope} joint_xlag ${representation} seed${seed}"
}

if [[ "$#" == 3 ]]; then
  run_one "$1" "$2" "$3"
  exit 0
fi

for scope in task1_2 task1_10; do
  for representation in analytic_hippo_rff inducing_points; do
    for seed in 0 1 2; do
      run_one "${scope}" "${representation}" "${seed}"
    done
  done
done
