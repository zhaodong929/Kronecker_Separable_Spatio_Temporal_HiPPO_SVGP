#!/usr/bin/env bash
#SBATCH --job-name=hippo-wandb-covid
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=5-9%3
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
RELEASE=${1:?immutable release commit}
CAMPAIGN=${2:?new campaign name}
ENTITY=${3:?W&B entity}
CODE="$C/releases/$RELEASE/source"
SEED=$SLURM_ARRAY_TASK_ID
O="$C/results/$CAMPAIGN/covid/kronhippo_svgp/seed$SEED/job-$SLURM_ARRAY_JOB_ID"
mkdir -p "$O"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export WANDB_API_KEY="$(cat "$C/credentials/wandb-api-key")"
export WANDB_CACHE_DIR="$C/wandb-cache" WANDB_CONFIG_DIR="$C/wandb-config"
export WANDB_DATA_DIR="$O/wandb-staging"
cd "$CODE"
trap 'code=$?; printf "{\"exit_code\":%s,\"main_table_admitted\":false}\n" "$code" > "$O/exit.json"' EXIT
sha256sum --quiet -c "$C/releases/$RELEASE/source.sha256"
"$C/env-routeb/bin/python" -m pytest -q tests/test_covid_recovered_features.py tests/test_multi_geometry.py tests/test_strict_online_release_boundary.py tests/test_routeb_torch_backend.py tests/test_temporal_analytic_gradients.py > "$O/tests.txt"
"$C/env-tracking/bin/python" - "$O" "$C" "$RELEASE" "$SEED" "$CAMPAIGN" "$ENTITY" <<'PY'
import json,pathlib,sys
o,c,release,seed,campaign,entity=sys.argv[1:];o=pathlib.Path(o);seed=int(seed)
qualification=dict(status='passed',method='kronhippo_svgp',dataset='covid',
 source_commit=release,tests='tests.txt', numerical_scope='multi-geometry adaptation; not single-Sylvester complexity',
 main_table_admitted=False)
(o/'qualification.json').write_text(json.dumps(qualification,indent=2))
spec=dict(entity=entity,project='KronHiPPO-STGP',campaign=campaign,
 dataset='covid',method='kronhippo_svgp',split_seed=seed,training_seed=seed,stage='final',
 source_commit=release,worker_python=c+'/env-routeb/bin/python',
 qualification_record=str(o/'qualification.json'),expected_steps=143,expected_sites=10,
 initial_observed_sites=52,initial_times=52,hidden_delay_steps=1,
 predictive_family='gaussian',metric_scale='standardized log1p admissions per 100k; CDC snapshot 2026-09-29',
 timing_scope='one worker per exclusive A30; includes logging overhead in wall time',
 input_files=[c+f'/protocol/covid-v2/seed{seed}/protocol.npz',c+f'/protocol/covid-v2/seed{seed}/protocol.json'],
 paper_admission_pending='Describe multi-geometry solver adaptation and validate cross-method comparison')
(o/'spec.json').write_text(json.dumps(spec,indent=2))
PY
"$C/env-tracking/bin/python" scripts/run_tracked_experiment.py --spec "$O/spec.json" --output "$O" -- \
 bash slurm/fair_three_domain/covid_tracked_worker.sh "$SEED" "$O"
