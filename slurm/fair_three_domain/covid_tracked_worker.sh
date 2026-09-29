#!/usr/bin/env bash
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
P="$C/env-routeb/bin/python"
SEED=${1:?seed}
O=${2:?output}
PROTOCOL="$C/protocol/covid-v2/seed${SEED}"
"$P" scripts/run_iclr_era5_routeb_batch.py \
 --protocol-npz "$PROTOCOL/protocol.npz" --protocol-json "$PROTOCOL/protocol.json" \
 --output-dir "$O/calibration" --data-part calibration --target-mode joint_xlag \
 --representation analytic_hippo_rff --mt 32 --ms 32 --rff-sample-size 256 \
 --training-objective vfe --iterations 250 --learning-rate 0.02 \
 --validation-every 5 --early-stopping-patience-validations 8 \
 --split-seed "$SEED" --device cuda --dtype float64 --evaluation-backend torch \
 --objective-optimization-version E3 --max-training-seconds 3600
"$P" scripts/run_iclr_era5_routeb_strict_online.py \
 --protocol-npz "$PROTOCOL/protocol.npz" --protocol-json "$PROTOCOL/protocol.json" \
 --theta-json "$O/calibration/result.json" \
 --representation analytic_hippo_rff --mt 32 --ms 32 --rff-sample-size 256 \
 --seed "$SEED" --device cuda --solver-backend torch --dtype float64 \
 --delayed-observations --task1-posterior-init --temporal-factor-device cpu \
 --temporal-bessel-backend scipy --checkpoint "$O/checkpoint.pt" --checkpoint-every 500 \
 --output "$O/result.json" --blockwise-output "$O/blocks.csv" --predictions-output "$O/predictions.npz"
