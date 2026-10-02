#!/usr/bin/env bash
#SBATCH --job-name=hippo-final65
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-2%3
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
SOURCE="$C/releases/ea54966e3790bdf33114605048065ac0e98f68c8/source"
DEST=${1:?campaign deployment directory}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=4 TF_NUM_INTRAOP_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export WANDB_API_KEY="$(cat "$C/credentials/wandb-api-key")"
export WANDB_CACHE_DIR="$C/wandb-cache" WANDB_CONFIG_DIR="$C/wandb-config"
export WANDB_DATA_DIR="$C/wandb-staging/final65-20261002/$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
cd "$DEST/operations"
sha256sum --quiet -c operations.sha256
cd "$SOURCE"
exec "$C/env-tracking/bin/python" "$DEST/operations/campaign.py" run --compute-root "$C" \
  --source "$SOURCE" --destination "$DEST" --campaign final65-20261002 --worker "$SLURM_ARRAY_TASK_ID"
