#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
theta="results/traffic/locked_theta_deterministic_v1/metr_la/seed0/theta.json"
split="results/traffic/protocols/metr_la/metr_la_seed0_spatial_split.json"
root="results/traffic/formal_deterministic_v1/metr_la"

run_one() {
  method="$1"; mode="$2"; output="$3"
  [[ -f "$output/status.json" ]] && return
  args=(.venv/bin/python scripts/run_traffic_routeb.py --dataset metr_la --data-root data/traffic/raw --split-manifest "$split" --output "$output" --protocol nowcast --method "$method" --mechanism-mode "$mode" --task1-steps 2016 --stream-stride 1 --mt 32 --ms 32 --rff 256 --temporal-evaluator scipy_frozen --model-seed 0 --device cpu --dtype float64)
  [[ "$method" == "persistence" || "$method" == "frozen_mean" ]] || args+=(--theta-json "$theta")
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${args[@]}"
}

run_one hippo changing "$root/main/nowcast/kronhippo_stgp/seed0"
run_one ordinary fixed "$root/main/nowcast/kron_stgp/seed0"
run_one persistence changing "$root/main/nowcast/persistence/seed0"
run_one frozen_mean changing "$root/main/nowcast/frozen_mean/seed0"
run_one hippo fixed "$root/mechanism/nowcast/joint_fixed/seed0"
run_one mean_field changing "$root/mechanism/nowcast/decoupled_changing/seed0"
run_one mean_field fixed "$root/mechanism/nowcast/decoupled_fixed/seed0"