#!/usr/bin/env bash
#SBATCH --job-name=pems-gp
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/traffic/formal_external_gp_a100_v1/slurm/%x_%A_%a.out
#SBATCH --error=results/traffic/formal_external_gp_a100_v1/slurm/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-${HOME}/stvgp_envs}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${REPO_ROOT}/results/traffic/protocol_n_external_gp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/traffic/formal_external_gp_a100_v1}"
MODE="${MODE:-formal}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
METHODS=(ohsvgp maddox_streaming_sgpr bui_osgpr st_svgp)
METHOD="${METHODS[$((TASK_ID / 3))]}"
SEED="$((TASK_ID % 3 + 1))"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}/slurm"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi a100; then
  echo "Allocated GPU is not an A100" >&2
  exit 1
fi

PYTHON="${ENV_ROOT}/routeb/bin/python"
args=(
  scripts/run_traffic_external_gp_three_seed.py
  --methods "${METHOD}"
  --seeds "${SEED}"
  --env-root "${ENV_ROOT}"
  --protocol-root "${PROTOCOL_ROOT}"
)
if [[ "${MODE}" == "smoke" ]]; then
  args+=(--output "${OUTPUT_ROOT}/smoke" --smoke --max-stream-steps 8)
elif [[ "${MODE}" == "formal" ]]; then
  args+=(--output "${OUTPUT_ROOT}")
else
  echo "Unknown MODE=${MODE}" >&2
  exit 2
fi

git rev-parse HEAD >"${OUTPUT_ROOT}/source_commit.txt"
echo "method=${METHOD} seed=${SEED} mode=${MODE} host=$(hostname)" \
  >"${OUTPUT_ROOT}/slurm/task_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.txt"
exec "${PYTHON}" "${args[@]}"
