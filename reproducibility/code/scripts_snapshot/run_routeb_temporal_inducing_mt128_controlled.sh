#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
BASE="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
DATA_SOURCE="$BASE/phase_d_joint_xlag_controlled"
OUT="$BASE/phase_k_routeb_temporal_representation"
PY="$ROOT/.venv/bin/python"

mkdir -p "$OUT"
cd "$ROOT"

for seed in 0 1 2; do
  data="$DATA_SOURCE/seed${seed}/era5_xlag_seed${seed}.npz"
  official="$DATA_SOURCE/seed${seed}/official_st_svgp_Ms128/result.json"
  run="$OUT/seed${seed}/routeb_inducing_Mt128_Ms128"

  if [[ ! -s "$data" || ! -s "$official" ]]; then
    echo "Missing controlled input for seed $seed" >&2
    exit 1
  fi
  if [[ -s "$run/run_metadata.json" ]]; then
    echo "SKIP completed $run"
    continue
  fi

  mkdir -p "$run"
  read -r ell_t ell_s0 ell_s1 kernel_variance noise < <(
    "$PY" -c "import json,math; r=json.load(open('$official')); print(r['learned_temporal_lengthscale'], *r['learned_spatial_lengthscales'], r['learned_temporal_variance']*r['learned_spatial_variances'][0]*r['learned_spatial_variances'][1], math.sqrt(r['learned_likelihood_variance']))"
  )

  echo "RUN Route B ordinary temporal inducing points seed=$seed Mt=128 Ms=128"
  /usr/bin/time -v -o "$run/resource_usage.txt" \
    "$PY" scripts/run_post_meeting_p1_matrix.py \
      --architecture structured_joint --protocol batch --final-block-only \
      --outdir "$run" --split-seed "$seed" --seed 0 \
      --mt 128 --ms 128 --phi-mode medium_era5_xlag --xlag-length 10 \
      --ell-t "$ell_t" --spatial-lengthscales "$ell_s0" "$ell_s1" \
      --noise "$noise" --kernel-variance "$kernel_variance" \
      --kernel-type matern32 --spatial-kernel-type matern32_separable \
      --spatial-inducing-coords-npz "$data" --beta-prior-variance 1000 \
      --temporal-representation inducing_points --temporal-rff-sample-size 256 \
      > "$run/stdout.log" 2> "$run/stderr.log"

  printf '{"exit_code": 0, "temporal_representation": "inducing_points", "seed": %d, "mt": 128, "ms": 128, "parameter_source": "%s"}\n' \
    "$seed" "$official" > "$run/status.json"
done

"$PY" scripts/summarize_routeb_temporal_representation.py --outdir "$OUT"

echo "Controlled temporal representation comparison complete: $OUT"
