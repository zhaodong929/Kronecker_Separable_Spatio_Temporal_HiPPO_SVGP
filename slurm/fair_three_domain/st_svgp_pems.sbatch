#!/usr/bin/env bash
#SBATCH --job-name=hippo-st-svgp-pems
#SBATCH --partition=a30
#SBATCH --nice=1000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-3%3
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
RELEASE=${1:?immutable source release}
O="$C/results/fair-three-domain-wandb-20260929/pems/st_svgp/seed$SLURM_ARRAY_TASK_ID/job-$SLURM_ARRAY_JOB_ID"
mkdir -p "$O"
EXTRA=()
if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
  EXTRA=(--reuse-validation-root "$C/results/fair-three-domain-wandb-20260929/pems/st_svgp/seed$SLURM_ARRAY_TASK_ID/job-$2")
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_API_KEY="$(cat "$C/credentials/wandb-api-key")"
export WANDB_CACHE_DIR="$C/wandb-cache" WANDB_CONFIG_DIR="$C/wandb-config" WANDB_DATA_DIR="$O/wandb-staging"
cd "$C/releases/$RELEASE/source"
trap 'code=$?; printf "{\"exit_code\":%s,\"main_table_admitted\":false}\n" "$code" > "$O/exit.json"' EXIT
sha256sum --quiet -c "$C/releases/$RELEASE/source.sha256"
"$C/env-tracking/bin/python" scripts/run_st_svgp_pems_campaign.py \
    --seed "$SLURM_ARRAY_TASK_ID" --output "$O" --release "$RELEASE" --compute-root "$C" "${EXTRA[@]}"
