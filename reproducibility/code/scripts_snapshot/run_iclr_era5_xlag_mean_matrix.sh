#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zd929/projects/stvgp_kronecker"
BENCHMARK="results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/iclr_era5_full_benchmark"
cd "${ROOT}"

run_one() {
  local scope="$1"
  local mode="$2"
  local seed="$3"
  local output_dir="${BENCHMARK}/runs/${scope}/online/xlag_mean_${mode}/seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    echo "skip complete ${scope} ${mode} seed${seed}"
    return
  fi
  mkdir -p "${output_dir}"
  .venv/bin/python scripts/run_iclr_era5_xlag_mean_baselines.py \
    --protocol-npz "${BENCHMARK}/protocol/${scope}/seed${seed}/protocol.npz" \
    --protocol-json "${BENCHMARK}/protocol/${scope}/seed${seed}/protocol.json" \
    --output-dir "${output_dir}" \
    --mode "${mode}" \
    --seed "${seed}" \
    >"${output_dir}/run.log" 2>&1
  echo "complete ${scope} ${mode} seed${seed}"
}

if [[ "$#" == 3 ]]; then
  run_one "$1" "$2" "$3"
  exit 0
fi

for scope in task1_2 task1_10; do
  for seed in 0 1 2; do
    for mode in batch_fixed task1_fixed recursive_rls; do
      run_one "${scope}" "${mode}" "${seed}"
    done
  done
done
