#!/usr/bin/env bash
#SBATCH --job-name=hippo-fair-pems-cal
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --array=1-3%3
set -euo pipefail
CAMPAIGN=/vol/bitbucket/nk523/hipposvgp-fair-20260929
SOURCE="${CAMPAIGN}/source"
PYTHON="${CAMPAIGN}/env-routeb/bin/python"
SEED="${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONHASHSEED=0
cd "${SOURCE}"
OUTPUT="${CAMPAIGN}/results/pems/seed${SEED}/calibration-${SLURM_ARRAY_JOB_ID}"
mkdir -p "${OUTPUT}"
trap 'printf "%s\n" "{\"stage\":\"task1_calibration\",\"main_table_admitted\":false,\"status\":\"failed\"}" > "${OUTPUT}/campaign_status.json"' ERR
sha256sum --quiet -c "${CAMPAIGN}/source.sha256"
cp "${CAMPAIGN}/source_commit.txt" "${OUTPUT}/source_commit.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${OUTPUT}/hardware.csv"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.__version__, torch.cuda.get_device_name())'
"${PYTHON}" -m pip freeze > "${OUTPUT}/environment.txt" 2>/dev/null || "${CAMPAIGN}/uv" pip freeze --python "${PYTHON}" > "${OUTPUT}/environment.txt"
"${PYTHON}" -m pytest -q tests/test_routeb_torch_backend.py tests/test_temporal_analytic_gradients.py > "${OUTPUT}/backend-tests.txt"
PROTOCOL="${CAMPAIGN}/protocol/pems/seed${SEED}"
if [ ! -f "${PROTOCOL}/raw_provenance.json" ]; then
  "${PYTHON}" scripts/prepare_fair_pems.py --data-root "${CAMPAIGN}/data" --output "${CAMPAIGN}/protocol/pems" --seeds "${SEED}"
fi
"${PYTHON}" - "${PROTOCOL}" <<'CHECK'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); meta=json.loads((p/'protocol.json').read_text())
assert hashlib.sha256((p/'protocol.npz').read_bytes()).hexdigest()==meta['protocol_npz_sha256']
assert hashlib.sha256((p/'split.json').read_bytes()).hexdigest()==meta['split_manifest_sha256']
CHECK
# Development/calibration run only. This does not publish a formal score or
# claim convergence at the iteration budget. Selection uses Task-1 validation.
printf '%s\n' '{"stage":"task1_calibration","main_table_admitted":false,"status":"running"}' > "${OUTPUT}/campaign_status.json"
"${PYTHON}" scripts/run_iclr_era5_routeb_batch.py \
  --protocol-npz "${PROTOCOL}/protocol.npz" --protocol-json "${PROTOCOL}/protocol.json" \
  --output-dir "${OUTPUT}" --data-part calibration --target-mode joint_xlag \
  --representation analytic_hippo_rff --mt 128 --ms 32 --rff-sample-size 256 \
  --training-objective vfe --iterations 250 --learning-rate 0.02 \
  --validation-every 5 --early-stopping-patience-validations 8 \
  --split-seed "${SEED}" --device cuda --dtype float64 --evaluation-backend torch \
  --objective-optimization-version E3 --max-training-seconds 3600
printf '%s\n' '{"stage":"task1_calibration","main_table_admitted":false,"status":"complete"}' > "${OUTPUT}/campaign_status.json"
