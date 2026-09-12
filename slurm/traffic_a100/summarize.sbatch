#!/usr/bin/env bash
#SBATCH --job-name=pems-gp-summary
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/traffic/formal_external_gp_a100_v1/slurm/summary_%j.out
#SBATCH --error=results/traffic/formal_external_gp_a100_v1/slurm/summary_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-${HOME}/stvgp_envs}"
cd "${REPO_ROOT}"
exec "${ENV_ROOT}/routeb/bin/python" scripts/summarize_traffic_external_gp_a100.py
