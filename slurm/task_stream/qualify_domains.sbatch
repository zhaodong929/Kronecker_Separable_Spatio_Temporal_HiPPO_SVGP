#!/usr/bin/env bash
#SBATCH --job-name=hippo-task-domains
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%3
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
RELEASE=${1:?immutable source commit}
CONFIGURATIONS=${2:?configuration root with domain subdirectories}
CAMPAIGN=${3:?fresh campaign name}
CODE="$C/releases/$RELEASE/source"
FAILED=0
for pair in covid:5 pems:1 era5:0; do
  DOMAIN=${pair%:*}
  SEED=${pair#*:}
  # Each array worker owns one GPU throughout; domains execute sequentially.
  # At most five queue elements and three active GPUs, including all domains.
  # Keep independent domain diagnostics running after a failure, but return a
  # failing array-worker status if any domain failed. Each child still records
  # its own terminal result and W&B failure; no failed run is resumed here.
  if bash "$CODE/slurm/task_stream/qualify.sbatch" "$RELEASE" \
    "$C/protocol/task-stream-20260930/$DOMAIN/seed$SEED" \
    "$CONFIGURATIONS/$DOMAIN" "$CAMPAIGN-$DOMAIN" shapes; then
    :
  else
    FAILED=1
  fi
done
exit "$FAILED"
