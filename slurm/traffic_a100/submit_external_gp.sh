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
METHOD_JOB=$(sbatch --parsable --array=0-3%4 --export=ALL "${SCRIPT_DIR}/external_gp_worker.sbatch")
SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:${METHOD_JOB}" --export=ALL "${SCRIPT_DIR}/summarize.sbatch")

cat >"${OUTPUT_ROOT}/SUBMISSION.json" <<EOF
{
  "method_array_job": "${METHOD_JOB}",
  "summary_job": "${SUMMARY_JOB}",
  "array_mapping": "task id 0..3 maps to [OHSVGP,Maddox,Bui,ST-SVGP]",
  "execution": "each A100 task smoke-tests seeds 1,2,3 and then runs the same three formal seeds sequentially",
  "dependency": "summary starts only after all four method tasks pass"
}
EOF
printf 'methods=%s summary=%s\n' "${METHOD_JOB}" "${SUMMARY_JOB}"
