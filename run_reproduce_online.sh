#!/usr/bin/env bash
set -euo pipefail

../.venv/bin/python -m stvgp_kronecker.train_online_joint \
  --dataset era5 \
  --era5-full-task12 \
  --era5-resplit \
  --spatial-inducing-count 64 \
  --temporal-inducing 16 \
  --rff-sample-size 1024 \
  --spatial-kernel matern \
  --block-size 32 \
  --pretrain-steps 300 \
  --pretrain-learning-rate 0.01 \
  --pretrain-log-every 20 \
  --pretrain-early-stopping-patience 20 \
  --pretrain-early-stopping-min-delta 0.001 \
  --era5-covariate-indices \
  --beta-prior-precision 1e-4 \
  --save-era5-maps \
  --map-split test \
  --map-time-indices "0,-1"
