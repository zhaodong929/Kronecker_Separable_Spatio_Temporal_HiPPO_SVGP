#!/usr/bin/env bash
set -euo pipefail

cd /home/zd929/projects/stvgp_kronecker
base='results/iclr2027_long_mechanism_20260914/formal'
protocol='/mnt/d/IC Mres AIML/Probabilistic memory states for modern RNNs/kronecker+s2vgp/ICLR Formal experiment/iclr_era5_stage2plus/protocol/task1_10'
theta='results/era5_stage2plus_sm_baseline_rerun_20260828_formal/calibration/routeb_kronhippo_stgp_sm'

for mode in structured_changing structured_fixed zero_cross_changing; do
  for seed in 0 1 2 3 4; do
    out="${base}/${mode}_ms128_mt128_seed${seed}"
    if [[ -f "${out}/result.json" ]]; then
      echo "SKIP ${mode} seed${seed}"
      continue
    fi
    mkdir -p "${out}"
    echo "START ${mode} seed${seed}"
    .venv_cuda128/bin/python scripts/run_iclr_era5_transfer_mechanism.py \
      --protocol-npz "${protocol}/seed${seed}/protocol.npz" \
      --protocol-json "${protocol}/seed${seed}/protocol.json" \
      --data-root data/era5/processed_timeseries_4_task1_10_extension \
      --theta-json "${theta}/seed${seed}/result.json" \
      --spectral-mixture-json configs/era5_sm_q3.json \
      --output-dir "${out}" --mode "${mode}" --seed "${seed}" \
      --ms 128 --mt 128 --rff-sample-size 256 \
      --prediction-chunk-size 8192 --device cuda:0 \
      > "${out}/run.log" 2>&1
    echo "DONE ${mode} seed${seed}"
  done
done

echo ALL_LONG_MECHANISM_JOBS_COMPLETE
