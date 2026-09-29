#!/usr/bin/env bash
#SBATCH --job-name=hippo-baseline-validation
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
RELEASE=${1:?immutable release commit}
CODE="$C/releases/$RELEASE/source"
METHODS=(ohsvgp osgpr st_svgp)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
O="$C/results/fair-three-domain-wandb-20260929/covid/$METHOD/validation-seed5/job-$SLURM_ARRAY_JOB_ID"
mkdir -p "$O"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=4 TF_NUM_INTRAOP_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export WANDB_API_KEY="$(cat "$C/credentials/wandb-api-key")"
export WANDB_CACHE_DIR="$C/wandb-cache" WANDB_CONFIG_DIR="$C/wandb-config"
export WANDB_DATA_DIR="$O/wandb-staging"
cd "$CODE"
trap 'code=$?; printf "{\"exit_code\":%s,\"stage\":\"validation\",\"main_table_admitted\":false}\n" "$code" > "$O/exit.json"' EXIT
sha256sum --quiet -c "$C/releases/$RELEASE/source.sha256"
"$C/env-tracking/bin/python" scripts/run_baseline_validation_campaign.py \
    --method "$METHOD" --output "$O" --release "$RELEASE" --compute-root "$C"
