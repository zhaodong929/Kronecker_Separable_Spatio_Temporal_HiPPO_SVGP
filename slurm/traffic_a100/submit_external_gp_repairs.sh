#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-${HOME}/stvgp_envs}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${REPO_ROOT}/results/traffic/protocol_n_external_gp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/traffic/formal_external_gp_a100_v1}"

mkdir -p "${OUTPUT_ROOT}/slurm" "${OUTPUT_ROOT}/paper_ready/complete_subset"
export REPO_ROOT ENV_ROOT PROTOCOL_ROOT OUTPUT_ROOT

METRICS_JOB=$(sbatch --parsable --partition=long --cpus-per-task=4 --mem=32G \
  --time=02:00:00 --output="${OUTPUT_ROOT}/slurm/partial_metrics_%j.out" \
  --error="${OUTPUT_ROOT}/slurm/partial_metrics_%j.err" \
  --wrap="cd '${REPO_ROOT}' && '${ENV_ROOT}/routeb/bin/python' scripts/summarize_traffic_external_gp_a100.py --input '${OUTPUT_ROOT}' --output '${OUTPUT_ROOT}/paper_ready/complete_subset' --methods ohsvgp bui_osgpr --seeds 1 2 3" | tail -n 1)
MADDOX_JOB=$(sbatch --parsable --array=1 --time=1-00:00:00 --mem=64G \
  --export=ALL "${SCRIPT_DIR}/external_gp_repair_worker.sbatch" | tail -n 1)
ST_JOB=$(sbatch --parsable --array=3 --time=3-00:00:00 --mem=128G \
  --export=ALL "${SCRIPT_DIR}/external_gp_repair_worker.sbatch" | tail -n 1)
SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:${MADDOX_JOB}:${ST_JOB}" \
  --export=ALL "${SCRIPT_DIR}/summarize.sbatch" | tail -n 1)

cat >"${OUTPUT_ROOT}/REPAIR_SUBMISSION.json" <<EOF
{
  "metrics_job": "${METRICS_JOB}",
  "maddox_repair_job": "${MADDOX_JOB}",
  "st_svgp_repair_job": "${ST_JOB}",
  "summary_job": "${SUMMARY_JOB}",
  "maddox_gate": "seed 2 must pass 3,800 blocks before incomplete formal seeds resume",
  "st_svgp_gate": "seed 1 must pass an 8-block A100 smoke with the environment-local CUDA toolkit"
}
EOF
printf 'metrics=%s maddox=%s st_svgp=%s summary=%s\n' \
  "${METRICS_JOB}" "${MADDOX_JOB}" "${ST_JOB}" "${SUMMARY_JOB}"
