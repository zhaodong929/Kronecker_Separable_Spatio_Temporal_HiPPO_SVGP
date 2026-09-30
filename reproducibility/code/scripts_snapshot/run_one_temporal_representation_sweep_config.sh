#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 MT MS REPRESENTATION" >&2
  exit 2
fi

MT=$1
MS=$2
REP=$3
ROOT=/home/zd929/projects/stvgp_kronecker
BASE="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
OUT="$BASE/phase_l_temporal_representation_sweep/Ms${MS}/Mt${MT}/${REP}"
PY="$ROOT/.venv/bin/python"

cd "$ROOT"
mkdir -p "$OUT"

for seed in 0 1 2; do
  data="$BASE/phase_d_joint_xlag_controlled/seed${seed}/era5_xlag_seed${seed}.npz"
  official="$BASE/phase_d_joint_xlag_controlled/seed${seed}/official_st_svgp_Ms${MS}/result.json"
  run="$OUT/seed${seed}"
  mkdir -p "$run"
  if [[ -s "$run/run_metadata.json" ]]; then
    echo "SKIP completed $run"
    continue
  fi
  if [[ ! -s "$data" || ! -s "$official" ]]; then
    echo "Missing controlled input for seed=$seed Ms=$MS" >&2
    exit 1
  fi

  read -r ell_t ell_s0 ell_s1 kernel_variance noise < <(
    "$PY" -c "import json,math; r=json.load(open('$official')); print(r['learned_temporal_lengthscale'], *r['learned_spatial_lengthscales'], r['learned_temporal_variance']*r['learned_spatial_variances'][0]*r['learned_spatial_variances'][1], math.sqrt(r['learned_likelihood_variance']))"
  )

  echo "RUN Ms=$MS Mt=$MT representation=$REP seed=$seed"
  /usr/bin/time -v -o "$run/resource_usage.txt" \
    "$PY" scripts/run_post_meeting_p1_matrix.py \
      --architecture structured_joint --protocol batch --final-block-only \
      --outdir "$run" --split-seed "$seed" --seed 0 \
      --mt "$MT" --ms "$MS" --phi-mode medium_era5_xlag --xlag-length 10 \
      --ell-t "$ell_t" --spatial-lengthscales "$ell_s0" "$ell_s1" \
      --noise "$noise" --kernel-variance "$kernel_variance" \
      --kernel-type matern32 --spatial-kernel-type matern32_separable \
      --spatial-inducing-coords-npz "$data" --beta-prior-variance 1000 \
      --temporal-representation "$REP" --temporal-rff-sample-size 256 \
      > "$run/stdout.log" 2> "$run/stderr.log"
done
