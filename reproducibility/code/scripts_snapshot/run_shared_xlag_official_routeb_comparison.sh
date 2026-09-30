#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
OUT="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison/phase_d_joint_xlag_controlled"
ENV="$ROOT/.envs/stvgp_official_py37"
MAMBA="$ROOT/.tools/micromamba/bin/micromamba"
PY="$ROOT/.venv/bin/python"
LEGACY="$ROOT/scripts/run_official_stvgp_legacy.py"

mkdir -p "$OUT"
cd "$ROOT"

export_data() {
  local seed=$1
  local data="$OUT/seed${seed}/era5_xlag_seed${seed}.npz"
  mkdir -p "$OUT/seed${seed}"
  if [[ ! -s "$data" ]]; then
    "$PY" scripts/export_era5_official_stvgp_subset.py \
      --split-seed "$seed" --num-times 186 --num-train-space 800 --num-test-space 200 \
      --phi-mode medium_era5_xlag --xlag-length 10 --ridge 1e-3 \
      --spatial-inducing-sizes 64 128 --output "$data" \
      > "$OUT/seed${seed}/export.log"
  fi
}

run_official() {
  local seed=$1 ms=$2
  local learning_rate=0.05
  if [[ $ms -ge 128 ]]; then
    learning_rate=0.02
  fi
  local data="$OUT/seed${seed}/era5_xlag_seed${seed}.npz"
  local run="$OUT/seed${seed}/official_st_svgp_Ms${ms}"
  mkdir -p "$run"
  if [[ -s "$run/result.json" ]]; then
    echo "SKIP completed $run"
    return
  fi
  /usr/bin/time -v -o "$run/resource_usage.txt" \
    timeout --signal=TERM --kill-after=60s 110m \
    "$MAMBA" run -p "$ENV" python "$LEGACY" \
      --model st_svgp --data-npz "$data" --num-spatial-inducing "$ms" \
      --learn-xlag-mean --xlag-ridge 1e-3 --xlag-update-every 10 \
      --xlag-update-spatial-count 200 --xlag-update-time-chunk-size 20 \
      --xlag-update-damping 0.5 \
      --fixed-spatial-inducing \
      --iterations 100 --temporal-lengthscale 0.05 --spatial-lengthscale 0.35 \
      --likelihood-variance 0.01 --learning-rate "$learning_rate" --newton-rate 1.0 \
      --early-stop-relative-tol 0.002 --early-stop-patience 10 \
      --early-stop-min-iterations 30 --seed "$seed" --jit --log-every 5 \
      --output "$run/result.json" --predictions-output "$run/predictions.npz" \
      > "$run/stdout.log" 2> "$run/stderr.log"
}

run_routeb() {
  local seed=$1 mt=$2 ms=$3
  local data="$OUT/seed${seed}/era5_xlag_seed${seed}.npz"
  local official="$OUT/seed${seed}/official_st_svgp_Ms${ms}/result.json"
  local run="$OUT/seed${seed}/routeb_Mt${mt}_Ms${ms}"
  if [[ -s "$run/run_metadata.json" ]]; then
    echo "SKIP completed $run"
    return
  fi
  read -r ell_t ell_s0 ell_s1 kernel_variance noise < <(
    "$PY" -c "import json,math; r=json.load(open('$official')); print(r['learned_temporal_lengthscale'], *r['learned_spatial_lengthscales'], r['learned_temporal_variance']*r['learned_spatial_variances'][0]*r['learned_spatial_variances'][1], math.sqrt(r['learned_likelihood_variance']))"
  )
  "$PY" scripts/run_post_meeting_p1_matrix.py \
    --architecture structured_joint --protocol batch --final-block-only \
    --outdir "$run" --split-seed "$seed" --seed 0 \
    --mt "$mt" --ms "$ms" --phi-mode medium_era5_xlag --xlag-length 10 \
    --ell-t "$ell_t" --spatial-lengthscales "$ell_s0" "$ell_s1" \
    --noise "$noise" --kernel-variance "$kernel_variance" \
      --kernel-type matern32 --spatial-kernel-type matern32_separable \
      --spatial-inducing-coords-npz "$data" \
      --beta-prior-variance 1000 \
      --temporal-representation analytic_hippo_rff --temporal-rff-sample-size 256 \
    > "$run.stdout.log" 2> "$run.stderr.log"
}

for seed in 0 1 2; do
  export_data "$seed"
done

for capacity in "8 64" "32 128"; do
  read -r mt ms <<< "$capacity"
  for seed in 0 1 2; do
    if run_official "$seed" "$ms"; then
      run_routeb "$seed" "$mt" "$ms"
    else
      echo "RESOURCE/EXECUTION FAILURE seed=$seed Ms=$ms; continuing"
    fi
  done
done

echo "Shared-X-lag controlled comparison complete: $OUT"
