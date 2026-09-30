#!/usr/bin/env bash
#SBATCH --job-name=hippo-task-qualify
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-4%3
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
RELEASE=${1:?immutable release commit}
PREPARED=${2:?prepared task-stream directory shared by all methods}
CONFIGURATIONS=${3:?directory containing method-name.json files}
CAMPAIGN=${4:?fresh qualification campaign name}
MODE=${5:-parity}
[[ "$MODE" == parity || "$MODE" == shapes ]] || { echo 'mode must be parity or shapes' >&2; exit 2; }
CODE="$C/releases/$RELEASE/source"
METHODS=(kronhippo_svgp osgpr ohsvgp st_svgp mgpvae)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
case "$METHOD" in
  kronhippo_svgp|ohsvgp) WORKER="$C/env-routeb/bin/python" ;;
  osgpr) WORKER="$C/env-osgpr/bin/python" ;;
  st_svgp|mgpvae) WORKER="$C/env-jax-gpu/bin/python" ;;
esac
O="$C/results/$CAMPAIGN/$METHOD/job-$SLURM_ARRAY_JOB_ID"
mkdir -p "$O"
"$C/env-tracking/bin/python" - "$O/profiling-availability.json" <<'PY'
import json, pathlib, shutil, socket, sys
driver = pathlib.Path('/proc/driver/nvidia/params')
lines = driver.read_text().splitlines() if driver.exists() else []
restriction = next((line.strip() for line in lines if line.startswith('RmProfilingAdminOnly:')), None)
pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(hostname=socket.gethostname(),
    ncu_path=shutil.which('ncu'), driver_counter_restriction=restriction,
    counter_acquisition_verified=False), indent=2))
PY
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=4 TF_NUM_INTRAOP_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export WANDB_API_KEY="$(cat "$C/credentials/wandb-api-key")"
export WANDB_CACHE_DIR="$C/wandb-cache" WANDB_CONFIG_DIR="$C/wandb-config"
export WANDB_DATA_DIR="$C/wandb-staging/$CAMPAIGN/$METHOD-$SLURM_ARRAY_JOB_ID"
cd "$CODE"
sha256sum --quiet -c "$C/releases/$RELEASE/source.sha256"
[[ "$(cat SOURCE_COMMIT)" == "$RELEASE" ]] || { echo 'Source commit mismatch' >&2; exit 2; }
SPEC="$C/results/$CAMPAIGN/spec-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json"
"$C/env-tracking/bin/python" - "$SPEC" "$RELEASE" "$CAMPAIGN" "$METHOD" "$WORKER" "$PREPARED" "$CONFIGURATIONS/$METHOD.json" <<'PY'
import json,pathlib,sys
spec,release,campaign,method,worker,prepared,configuration=sys.argv[1:]
config=json.loads(pathlib.Path(configuration).read_text())
manifest=json.loads((pathlib.Path(prepared)/'manifest.json').read_text())
if config['method'] != method: raise ValueError('Configuration method mismatch')
pathlib.Path(spec).write_text(json.dumps(dict(entity='harrisonzhu',project='KronHiPPO-STGP',
 campaign=campaign,dataset=manifest.get('dataset','task-stream-qualification'),method=method,
 split_seed=manifest.get('split_seed',0),training_seed=config.get('seed',0),
 stage='qualification',source_commit=release,worker_python=worker,
 input_files=[configuration,prepared+'/manifest.json',prepared+'/stream.npz',prepared+'/selection-stream.npz',prepared+'/features.npz'],
 main_table_admitted=False,timing_scope='diagnostic only; one allocated A30 GPU; CPU reference followed by GPU execution'),indent=2))
PY
ARGS=()
if [[ "$MODE" == shapes ]]; then ARGS+=(--shape-only); fi
exec "$C/env-tracking/bin/python" scripts/run_tracked_experiment.py --spec "$SPEC" --output "$O" -- \
  "$WORKER" scripts/qualify_task_stream_device.py --prepared "$PREPARED" \
  --configuration "$CONFIGURATIONS/$METHOD.json" --output "$O" --max-fit-steps 2 "${ARGS[@]}"
