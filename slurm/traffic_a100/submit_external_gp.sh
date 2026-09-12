#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-${HOME}/stvgp_envs}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${REPO_ROOT}/results/traffic/protocol_n_external_gp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/traffic/formal_external_gp_a100_v1}"

for seed in 1 2 3; do
  test -s "${PROTOCOL_ROOT}/seed${seed}/protocol.npz"
  test -s "${PROTOCOL_ROOT}/seed${seed}/protocol.json"
  test -s "${REPO_ROOT}/results/traffic/formal_locked_sm_q2_road_context_v1/task1_theta/seed${seed}/theta.json"
done
for env_name in routeb gpflow maddox stvgp_legacy; do
  test -x "${ENV_ROOT}/${env_name}/bin/python"
done
mkdir -p "${OUTPUT_ROOT}/slurm"

export REPO_ROOT ENV_ROOT PROTOCOL_ROOT OUTPUT_ROOT
SMOKE_JOB=$(sbatch --parsable --array=0-11%4 --export=ALL,MODE=smoke "${SCRIPT_DIR}/external_gp_worker.sbatch")
FORMAL_JOB=$(sbatch --parsable --array=0-11%4 --dependency="aftercorr:${SMOKE_JOB}" --export=ALL,MODE=formal "${SCRIPT_DIR}/external_gp_worker.sbatch")
SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:${FORMAL_JOB}" --export=ALL "${SCRIPT_DIR}/summarize.sbatch")

cat >"${OUTPUT_ROOT}/SUBMISSION.json" <<EOF
{
  "smoke_array_job": "${SMOKE_JOB}",
  "formal_array_job": "${FORMAL_JOB}",
  "summary_job": "${SUMMARY_JOB}",
  "array_mapping": "task=floor(id/3) in [OHSVGP,Maddox,Bui,ST-SVGP]; seed=id%3+1",
  "dependency": "each formal task starts after its corresponding smoke task passes"
}
EOF
printf 'smoke=%s formal=%s summary=%s\n' "${SMOKE_JOB}" "${FORMAL_JOB}" "${SUMMARY_JOB}"
