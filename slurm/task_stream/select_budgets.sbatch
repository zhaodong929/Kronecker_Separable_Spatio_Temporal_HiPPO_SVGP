#!/usr/bin/env bash
#SBATCH --job-name=hippo-budget-validation
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-2%3
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
RELEASE=${1:?immutable release commit}
SOURCE_HASH=${2:?source_identity hash}
MANIFEST=${3:?reviewed validation manifest}
CAMPAIGN=${4:?campaign name}
STATE=${5:?durable worker state directory}
MAX_SECONDS=${6:-165600}
CODE="$C/releases/$RELEASE/source"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=4 TF_NUM_INTRAOP_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export WANDB_API_KEY="$(cat "$C/credentials/wandb-api-key")"
export WANDB_CACHE_DIR="$C/wandb-cache" WANDB_CONFIG_DIR="$C/wandb-config"
export WANDB_DATA_DIR="$C/wandb-staging/$CAMPAIGN/worker-$SLURM_ARRAY_TASK_ID-$SLURM_ARRAY_JOB_ID"
cd "$CODE"
sha256sum --quiet -c "$C/releases/$RELEASE/source.sha256"
[[ "$(cat SOURCE_COMMIT)" == "$RELEASE" ]] || { echo 'Source commit mismatch' >&2; exit 2; }
exec "$C/env-tracking/bin/python" scripts/run_task_budget_campaign.py \
  --manifest "$MANIFEST" --worker-index "$SLURM_ARRAY_TASK_ID" --state-dir "$STATE" \
  --campaign "$CAMPAIGN" --compute-root "$C" --source-commit "$RELEASE" \
  --source-sha256 "$SOURCE_HASH" --max-seconds "$MAX_SECONDS"
