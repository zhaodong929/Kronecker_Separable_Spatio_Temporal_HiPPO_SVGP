#!/usr/bin/env bash
#SBATCH --job-name=hippo-fair-pems-online
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=1-3%3
set -euo pipefail
C=/vol/bitbucket/nk523/hipposvgp-fair-20260929
RELEASE=${HIPPO_RELEASE:?Set an immutable tested release commit}
CODE="$C/releases/$RELEASE/source"
SEED=${SLURM_ARRAY_TASK_ID}
O="$C/results/pems/seed${SEED}/online-${SLURM_ARRAY_JOB_ID}"
mkdir -p "$O"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
cd "$CODE"
trap 'code=$?; printf "{\"exit_code\":%s,\"main_table_admitted\":false}\n" "$code" > "$O/exit.json"' EXIT
sha256sum --quiet -c "$C/releases/$RELEASE/source.sha256"
printf '%s\n' "$RELEASE" > "$O/source_commit.txt"
P="$C/env-routeb/bin/python"
"$P" -c 'import torch; assert torch.cuda.is_available()'
"$P" -m pytest -q tests/test_multi_geometry.py tests/test_strict_online_release_boundary.py tests/test_routeb_torch_backend.py tests/test_temporal_analytic_gradients.py > "$O/tests.txt"
COMMAND=("$P" scripts/run_iclr_era5_routeb_strict_online.py
 --protocol-npz "$C/protocol/pems/seed${SEED}/protocol.npz"
 --protocol-json "$C/protocol/pems/seed${SEED}/protocol.json"
 --theta-json "$C/results/pems/seed${SEED}/calibration-294099/result.json"
 --representation analytic_hippo_rff --mt 128 --ms 32 --rff-sample-size 256
 --seed "$SEED" --device cuda --solver-backend torch --dtype float64
 --delayed-observations --task1-posterior-init --temporal-factor-device cpu
 --temporal-bessel-backend scipy --checkpoint "$O/checkpoint.pt" --checkpoint-every 500)
if [ ! -f "$O/checkpoint.pt" ]; then
 "${COMMAND[@]}" --max-blocks 100 --output "$O/pilot.json" --blockwise-output "$O/pilot-blocks.csv" --predictions-output "$O/pilot-predictions.npz" > "$O/pilot.log" 2>&1
 "$P" - "$O" <<'CHECK'
import json,pathlib,sys,numpy as np
p=pathlib.Path(sys.argv[1]);d=json.loads((p/'pilot.json').read_text())
assert d['num_blocks']==100 and d['delayed_observation_rows']==99*65
assert d['task1_posterior_initialization_rows']==2016*260
with np.load(p/'pilot-predictions.npz') as a:
 assert a['pred_mean'].shape==(100,65)
 assert np.isfinite(a['pred_mean']).all() and np.isfinite(a['pred_var']).all() and (a['pred_var']>0).all()
CHECK
fi
"${COMMAND[@]}" --resume --output "$O/result.json" --blockwise-output "$O/blocks.csv" --predictions-output "$O/predictions.npz" >> "$O/run.log" 2>&1
"$P" - "$O" <<'CHECK'
import json,pathlib,sys,numpy as np
p=pathlib.Path(sys.argv[1]);d=json.loads((p/'result.json').read_text())
assert d['num_blocks']==50100 and d['delayed_observation_rows']==50099*65
with np.load(p/'predictions.npz') as a:
 assert a['pred_mean'].shape==(50100,65)
 assert np.isfinite(a['pred_mean']).all() and np.isfinite(a['pred_var']).all() and (a['pred_var']>0).all()
(p/'completion.json').write_text(json.dumps(dict(status='complete',stream_steps=50100,main_table_admitted=False)))
CHECK
