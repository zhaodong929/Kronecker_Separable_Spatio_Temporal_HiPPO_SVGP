#!/usr/bin/env bash
#SBATCH --job-name=hippo-mgpvae-era5
#SBATCH --partition=a30
#SBATCH --nice=1000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-4%3
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
RELEASE=${1:?immutable source release}
O="$C/results/fair-three-domain-wandb-20260929/era5/mgpvae/seed$SLURM_ARRAY_TASK_ID/job-$SLURM_ARRAY_JOB_ID"
mkdir -p "$O"
export HIPPO_TEST_DEVICE=cuda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export WANDB_API_KEY="$(cat "$C/credentials/wandb-api-key")"
export WANDB_CACHE_DIR="$C/wandb-cache" WANDB_CONFIG_DIR="$C/wandb-config" WANDB_DATA_DIR="$O/wandb-staging"
cd "$C/releases/$RELEASE/source"
trap 'code=$?; printf "{\"exit_code\":%s,\"main_table_admitted\":false}\n" "$code" > "$O/exit.json"' EXIT
sha256sum --quiet -c "$C/releases/$RELEASE/source.sha256"
"$C/env-tracking/bin/python" scripts/run_mgpvae_final_campaign.py --dataset era5 --seed "$SLURM_ARRAY_TASK_ID" --output "$O" --release "$RELEASE" --compute-root "$C"
