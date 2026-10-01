#!/usr/bin/env bash
#SBATCH --job-name=hippo-gh200-qualify
#SBATCH --account=brics.u6tj
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
C=${1:?deployment root}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export WANDB_API_KEY="$(cat "$C/credentials/wandb-api-key")"
export WANDB_CACHE_DIR="$C/wandb-cache" WANDB_CONFIG_DIR="$C/wandb-config"
export WANDB_DATA_DIR="$C/wandb-staging/$SLURM_JOB_ID"
cd "$C/source"
sha256sum --quiet -c "$C/source.sha256"
srun "$C/env-tracking/bin/python" "$C/operations/qualify.py" "$C"
