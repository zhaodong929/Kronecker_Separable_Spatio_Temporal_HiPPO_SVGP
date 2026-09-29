#!/usr/bin/env bash
#SBATCH --job-name=hippo-st-continuation
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
RELEASE=${1:?immutable source release}
O="$C/results/fair-three-domain-wandb-20260929/st-svgp-continuation/job-$SLURM_JOB_ID"
mkdir -p "$O/st_svgp"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_API_KEY="$(cat "$C/credentials/wandb-api-key")"
export WANDB_CACHE_DIR="$C/wandb-cache" WANDB_CONFIG_DIR="$C/wandb-config" WANDB_DATA_DIR="$O/wandb-staging"
cd "$C/releases/$RELEASE/source"
trap 'code=$?; printf "{\"exit_code\":%s,\"main_table_admitted\":false}\n" "$code" > "$O/exit.json"' EXIT
sha256sum --quiet -c "$C/releases/$RELEASE/source.sha256"
"$C/env-tracking/bin/python" - "$O/st_svgp" "$RELEASE" "$C" <<'PY'
from pathlib import Path
import json,sys
out,release,c=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
spec=dict(entity='harrisonzhu',project='KronHiPPO-STGP',campaign='fair-three-domain-wandb-20260929',
 dataset='covid',method='st_svgp',split_seed='all-five',training_seed='inherited',stage='qualification',
 worker_python=str(c/'env-jax-gpu/bin/python'),source_commit=release,main_table_admitted=False,
 purpose='Exact continuation equivalence and exclusive A30 timing; no additional independent replicate',
 input_files=[str(c/f'results/fair-three-domain-wandb-20260929/covid/st_svgp/seed{s}/job-294321/{name}') for s in range(5,10) for name in ['task1-frozen.npz','result.json','selection.json']]
 +[str(c/f'protocol/covid-v2/seed{s}/{name}') for s in range(5,10) for name in ['protocol.npz','protocol.json']])
(out/'spec.json').write_text(json.dumps(spec,indent=2))
PY
"$C/env-tracking/bin/python" scripts/run_tracked_experiment.py --spec "$O/st_svgp/spec.json" --output "$O/st_svgp" -- \
 "$C/env-tracking/bin/python" scripts/measure_st_svgp_continuation.py --compute-root "$C" --output "$O/st_svgp"
printf '{"status":"full_reference_equivalence_and_timing_complete","main_table_admitted":false}\n' > "$O/st_svgp/completed.json"
