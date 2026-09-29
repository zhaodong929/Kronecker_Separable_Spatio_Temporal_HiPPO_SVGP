#!/usr/bin/env bash
#SBATCH --job-name=hippo-fair-online-pilot
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --array=1-3%3
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
SEED=${SLURM_ARRAY_TASK_ID}
O="$C/results/pems/seed${SEED}/online-pilot-${SLURM_ARRAY_JOB_ID}"
mkdir -p "$O"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
cd "$C/source"
trap 'code=$?; printf "{\"exit_code\":%s,\"main_table_admitted\":false}\n" "$code" > "$O/exit.json"' EXIT
sha256sum --quiet -c "$C/source.sha256"
cp "$C/source_commit.txt" "$O/source_commit.txt"
P="$C/env-routeb/bin/python"
"$P" -c 'import torch; assert torch.cuda.is_available()'
"$P" -m pytest -q tests/test_multi_geometry.py tests/test_strict_online_release_boundary.py tests/test_routeb_torch_backend.py > "$O/tests.txt"
"$P" scripts/run_iclr_era5_routeb_strict_online.py \
 --protocol-npz "$C/protocol/pems/seed${SEED}/protocol.npz" \
 --protocol-json "$C/protocol/pems/seed${SEED}/protocol.json" \
 --theta-json "$C/results/pems/seed${SEED}/calibration-294099/result.json" \
 --output "$O/result.json" --blockwise-output "$O/blocks.csv" \
 --predictions-output "$O/predictions.npz" \
 --representation analytic_hippo_rff --mt 128 --ms 32 --rff-sample-size 256 \
 --seed "$SEED" --device cuda --solver-backend torch --dtype float64 \
 --delayed-observations --task1-posterior-init --temporal-factor-device solver \
 --max-blocks 100 > "$O/run.log" 2>&1
