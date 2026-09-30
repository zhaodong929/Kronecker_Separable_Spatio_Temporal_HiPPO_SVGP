#!/usr/bin/env bash
set -u -o pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
REPORT_ROOT="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/post_meeting_priority_2026-07-14"
OUT="$REPORT_ROOT/p0b_official_full_era5"
ENV="$ROOT/.envs/stvgp_official_py37"
MAMBA="$ROOT/.tools/micromamba/bin/micromamba"
PY="$ROOT/.venv/bin/python"
LEGACY="$ROOT/scripts/run_official_stvgp_legacy.py"
mkdir -p "$OUT"
cd "$ROOT"

export_data() {
  local seed=$1 data="$OUT/seed${seed}/era5_full_seed${seed}.npz"
  mkdir -p "$OUT/seed${seed}"
  if [[ ! -s "$data" ]]; then
    "$PY" scripts/export_era5_official_stvgp_subset.py \
      --split-seed "$seed" --num-times 186 --num-train-space 800 --num-test-space 200 \
      --output "$data" > "$OUT/seed${seed}/export.log"
  fi
}

run_model() {
  local seed=$1 model=$2 ms=$3 timeout_duration=$4
  local learning_rate=0.1
  if [[ "$model" == st_svgp && $ms -ge 64 ]]; then
    learning_rate=0.05
  fi
  if [[ "$model" == st_svgp && $ms -ge 128 ]]; then
    learning_rate=0.02
  fi
  if [[ "$model" == st_vgp ]]; then
    learning_rate=0.01
  fi
  local run="$OUT/seed${seed}/${model}_Ms${ms}"
  local data="$OUT/seed${seed}/era5_full_seed${seed}.npz"
  mkdir -p "$run"
  if [[ -s "$run/result.json" ]]; then
    printf '{"seed":%s,"model":"%s","num_spatial_inducing":%s,"status":"completed","exit_code":0}\n' \
      "$seed" "$model" "$ms" > "$run/status.json"
    echo "SKIP completed $run"
    return 0
  fi

  echo "RUN $run"
  set +e
  /usr/bin/time -v -o "$run/resource_usage.txt" \
    timeout --signal=TERM --kill-after=60s "$timeout_duration" \
    "$MAMBA" run -p "$ENV" python "$LEGACY" \
      --model "$model" --data-npz "$data" --num-spatial-inducing "$ms" \
      --iterations 100 --temporal-lengthscale 0.05 --spatial-lengthscale 0.35 \
      --likelihood-variance 0.01 --learning-rate "$learning_rate" --newton-rate 1.0 \
      --early-stop-relative-tol 0.002 --early-stop-patience 10 \
      --early-stop-min-iterations 30 --seed "$seed" --jit --log-every 5 \
      --output "$run/result.json" --predictions-output "$run/predictions.npz" \
      > "$run/stdout.log" 2> "$run/stderr.log"
  local code=$?
  set -e
  local status=failed
  if [[ $code -eq 0 && -s "$run/result.json" ]]; then
    status=completed
  elif [[ $code -eq 124 || $code -eq 137 || $code -eq 143 ]]; then
    status=resource_limited
  fi
  printf '{"seed":%s,"model":"%s","num_spatial_inducing":%s,"status":"%s","exit_code":%s}\n' \
    "$seed" "$model" "$ms" "$status" "$code" > "$run/status.json"
  echo "DONE status=$status exit=$code $run"
  return 0
}

set -e
for seed in 0 1 2; do
  export_data "$seed"
done

for ms in 30 64 128; do
  run_model 0 st_svgp "$ms" 60m
  status=$("$PY" -c "import json; print(json.load(open('$OUT/seed0/st_svgp_Ms${ms}/status.json'))['status'])")
  if [[ "$status" == completed ]]; then
    run_model 1 st_svgp "$ms" 60m
    run_model 2 st_svgp "$ms" 60m
  else
    for seed in 1 2; do
      run="$OUT/seed${seed}/st_svgp_Ms${ms}"
      mkdir -p "$run"
      printf '{"seed":%s,"model":"st_svgp","num_spatial_inducing":%s,"status":"skipped_after_seed0_resource_limit","exit_code":null}\n' \
        "$seed" "$ms" > "$run/status.json"
    done
  fi
done

run_model 0 mf_st_svgp 30 60m
status=$("$PY" -c "import json; print(json.load(open('$OUT/seed0/mf_st_svgp_Ms30/status.json'))['status'])")
if [[ "$status" == completed ]]; then
  run_model 1 mf_st_svgp 30 60m
  run_model 2 mf_st_svgp 30 60m
fi

# Full ST-VGP is attempted once per split under a documented resource limit.
run_model 0 st_vgp 1000 20m
status=$("$PY" -c "import json; print(json.load(open('$OUT/seed0/st_vgp_Ms1000/status.json'))['status'])")
if [[ "$status" == completed ]]; then
  run_model 1 st_vgp 1000 20m
  run_model 2 st_vgp 1000 20m
else
  for seed in 1 2; do
    run="$OUT/seed${seed}/st_vgp_Ms1000"
    mkdir -p "$run"
    printf '{"seed":%s,"model":"st_vgp","num_spatial_inducing":1000,"status":"skipped_after_seed0_resource_limit","exit_code":null}\n' \
      "$seed" > "$run/status.json"
  done
fi

"$PY" scripts/aggregate_post_meeting_p0b_official.py --root "$OUT"
