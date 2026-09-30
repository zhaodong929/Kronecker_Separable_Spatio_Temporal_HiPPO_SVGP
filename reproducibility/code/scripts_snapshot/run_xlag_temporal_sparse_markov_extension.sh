#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
BASE="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
SOURCE="$BASE/phase_d_joint_xlag_controlled"
OUT="$BASE/phase_h_xlag_temporal_sparse_markov"
ENV="$ROOT/.envs/stvgp_official_py37"
MAMBA="$ROOT/.tools/micromamba/bin/micromamba"
PY="$ROOT/.venv/bin/python"
LEGACY="$ROOT/scripts/run_official_stvgp_legacy.py"

mkdir -p "$OUT"
cd "$ROOT"

learning_rate_for_ms() {
  if [[ $1 -ge 128 ]]; then
    printf '0.02\n'
  else
    printf '0.05\n'
  fi
}

run_legacy() {
  local seed=$1 model=$2 mt=$3 ms=$4 condition=$5 run=$6
  local data="$SOURCE/seed${seed}/era5_xlag_seed${seed}.npz"
  local learning_rate
  learning_rate=$(learning_rate_for_ms "$ms")
  mkdir -p "$run"
  if [[ -s "$run/result.json" ]]; then
    echo "SKIP completed $run"
    return 0
  fi

  local model_args=(--model "$model" --num-spatial-inducing "$ms")
  if [[ "$model" == "st_dsvgp" ]]; then
    model_args+=(--num-temporal-inducing "$mt")
  fi
  local xlag_args=()
  if [[ "$condition" == "xlag" ]]; then
    xlag_args+=(
      --learn-xlag-mean --xlag-ridge 1e-3 --xlag-update-every 10
      --xlag-update-spatial-count 200 --xlag-update-time-chunk-size 20
      --xlag-update-damping 0.5
    )
  fi

  echo "RUN official model=$model condition=$condition seed=$seed Mt=$mt Ms=$ms"
  /usr/bin/time -v -o "$run/resource_usage.txt" \
    timeout --signal=TERM --kill-after=60s 110m \
    "$MAMBA" run -p "$ENV" python "$LEGACY" \
      "${model_args[@]}" --data-npz "$data" "${xlag_args[@]}" \
      --fixed-spatial-inducing \
      --iterations 100 --temporal-lengthscale 0.05 --spatial-lengthscale 0.35 \
      --likelihood-variance 0.01 --learning-rate "$learning_rate" --newton-rate 1.0 \
      --early-stop-relative-tol 0.002 --early-stop-patience 10 \
      --early-stop-min-iterations 30 --seed "$seed" --jit --log-every 5 \
      --output "$run/result.json" --predictions-output "$run/predictions.npz" \
      > "$run/stdout.log" 2> "$run/stderr.log"
  local status=$?
  printf '{"exit_code": %d, "model": "%s", "condition": "%s", "seed": %d, "mt": %d, "ms": %d}\n' \
    "$status" "$model" "$condition" "$seed" "$mt" "$ms" > "$run/status.json"
  if [[ $status -ne 0 ]]; then
    echo "FAILED official model=$model condition=$condition seed=$seed Mt=$mt Ms=$ms"
  fi
  return "$status"
}

run_routeb() {
  local seed=$1 mt=$2 ms=$3 condition=$4 official_run=$5
  local data="$SOURCE/seed${seed}/era5_xlag_seed${seed}.npz"
  local run="$OUT/seed${seed}/routeb_${condition}_Mt${mt}_Ms${ms}"
  if [[ -s "$run/run_metadata.json" ]]; then
    echo "SKIP completed $run"
    return 0
  fi
  if [[ ! -s "$official_run/result.json" ]]; then
    echo "SKIP Route B: missing parameter source $official_run/result.json"
    return 1
  fi
  mkdir -p "$run"

  local ell_t ell_s0 ell_s1 kernel_variance noise
  read -r ell_t ell_s0 ell_s1 kernel_variance noise < <(
    "$PY" -c "import json,math; r=json.load(open('$official_run/result.json')); print(r['learned_temporal_lengthscale'], *r['learned_spatial_lengthscales'], r['learned_temporal_variance']*r['learned_spatial_variances'][0]*r['learned_spatial_variances'][1], math.sqrt(r['learned_likelihood_variance']))"
  )
  local phi_mode=direct_y
  if [[ "$condition" == "xlag" ]]; then
    phi_mode=medium_era5_xlag
  fi

  echo "RUN Route B condition=$condition seed=$seed Mt=$mt Ms=$ms"
  /usr/bin/time -v -o "$run/resource_usage.txt" \
    "$PY" scripts/run_post_meeting_p1_matrix.py \
      --architecture structured_joint --protocol batch --final-block-only \
      --outdir "$run" --split-seed "$seed" --seed 0 \
      --mt "$mt" --ms "$ms" --phi-mode "$phi_mode" --xlag-length 10 \
      --ell-t "$ell_t" --spatial-lengthscales "$ell_s0" "$ell_s1" \
      --noise "$noise" --kernel-variance "$kernel_variance" \
      --kernel-type matern32 --spatial-kernel-type matern32_separable \
      --spatial-inducing-coords-npz "$data" --beta-prior-variance 1000 \
      --temporal-representation analytic_hippo_rff --temporal-rff-sample-size 256 \
      > "$run/stdout.log" 2> "$run/stderr.log"
  local status=$?
  printf '{"exit_code": %d, "condition": "%s", "seed": %d, "mt": %d, "ms": %d, "parameter_source": "%s"}\n' \
    "$status" "$condition" "$seed" "$mt" "$ms" "$official_run/result.json" > "$run/status.json"
  return "$status"
}

for seed in 0 1 2; do
  for ms in 64 128; do
    run="$OUT/seed${seed}/full_markov_no_xlag_Ms${ms}"
    run_legacy "$seed" st_svgp 0 "$ms" no_xlag "$run" || true
  done
done

for capacity in "8 64" "32 128"; do
  read -r mt ms <<< "$capacity"
  for seed in 0 1 2; do
    for condition in no_xlag xlag; do
      official_run="$OUT/seed${seed}/temporal_sparse_${condition}_Mt${mt}_Ms${ms}"
      if run_legacy "$seed" st_dsvgp "$mt" "$ms" "$condition" "$official_run"; then
        run_routeb "$seed" "$mt" "$ms" "$condition" "$official_run" || true
      fi
    done
  done
done

echo "X-lag and temporal-sparse Markov extension experiments complete: $OUT"
