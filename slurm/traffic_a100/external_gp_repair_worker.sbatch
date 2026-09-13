#!/usr/bin/env bash
#SBATCH --job-name=pems-gp-repair
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=results/traffic/formal_external_gp_a100_v1/slurm/%x_%A_%a.out
#SBATCH --error=results/traffic/formal_external_gp_a100_v1/slurm/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-${HOME}/stvgp_envs}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${REPO_ROOT}/results/traffic/protocol_n_external_gp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/traffic/formal_external_gp_a100_v1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

cd "${REPO_ROOT}"
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi a100; then
  echo "Allocated GPU is not an A100" >&2
  exit 1
fi

PYTHON="${ENV_ROOT}/routeb/bin/python"
common=(
  scripts/run_traffic_external_gp_three_seed.py
  --env-root "${ENV_ROOT}"
  --protocol-root "${PROTOCOL_ROOT}"
)

if [[ "${TASK_ID}" == "1" ]]; then
  "${PYTHON}" "${common[@]}" --methods maddox_streaming_sgpr --seeds 2 \
    --output "${OUTPUT_ROOT}/repair_canary" --max-stream-steps 3800
  exec "${PYTHON}" "${common[@]}" --methods maddox_streaming_sgpr \
    --seeds 1 2 3 --output "${OUTPUT_ROOT}"
elif [[ "${TASK_ID}" == "3" ]]; then
  CUDA_ROOT="${ENV_ROOT}/stvgp_legacy_cuda111"
  export CUDA_HOME="${CUDA_ROOT}"
  export PATH="${CUDA_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_ROOT}/lib:${LD_LIBRARY_PATH:-}"
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_ROOT} ${XLA_FLAGS:-}"
  "${PYTHON}" "${common[@]}" --methods st_svgp --seeds 1 \
    --output "${OUTPUT_ROOT}/repair_canary" --smoke --max-stream-steps 8
  exec "${PYTHON}" "${common[@]}" --methods st_svgp \
    --seeds 1 2 3 --output "${OUTPUT_ROOT}"
else
  echo "Repair worker only accepts array task 1 (Maddox) or 3 (ST-SVGP)" >&2
  exit 2
fi
