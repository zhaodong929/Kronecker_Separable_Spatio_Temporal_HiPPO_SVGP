#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zd929/projects/stvgp_kronecker
BASE="$ROOT/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/unified_stvgp_routeb_comparison"
DATA_SOURCE="$BASE/phase_d_joint_xlag_controlled"
OFFICIAL_SOURCE="$BASE/phase_h_xlag_temporal_sparse_markov"
OUT="$BASE/phase_j_no_xlag_mt128_controlled"
PY="$ROOT/.venv/bin/python"

mkdir -p "$OUT"
cd "$ROOT"

for seed in 0 1 2; do
  data="$DATA_SOURCE/seed${seed}/era5_xlag_seed${seed}.npz"
  official="$OFFICIAL_SOURCE/seed${seed}/full_markov_no_xlag_Ms128/result.json"
  run="$OUT/seed${seed}/routeb_no_xlag_Mt128_Ms128"

  if [[ ! -s "$data" ]]; then
    echo "Missing controlled data file: $data" >&2
    exit 1
  fi
  if [[ ! -s "$official" ]]; then
    echo "Missing no-X-lag official parameter source: $official" >&2
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

  echo "RUN no-X-lag Route B seed=$seed Mt=128 Ms=128"
  /usr/bin/time -v -o "$run/resource_usage.txt" \
    "$PY" scripts/run_post_meeting_p1_matrix.py \
      --architecture structured_joint --protocol batch --final-block-only \
      --outdir "$run" --split-seed "$seed" --seed 0 \
      --mt 128 --ms 128 --phi-mode direct_y --xlag-length 10 \
      --ell-t "$ell_t" --spatial-lengthscales "$ell_s0" "$ell_s1" \
      --noise "$noise" --kernel-variance "$kernel_variance" \
      --kernel-type matern32 --spatial-kernel-type matern32_separable \
      --spatial-inducing-coords-npz "$data" --beta-prior-variance 1000 \
      --temporal-representation analytic_hippo_rff --temporal-rff-sample-size 256 \
      > "$run/stdout.log" 2> "$run/stderr.log"

  printf '{"exit_code": 0, "condition": "no_xlag", "seed": %d, "mt": 128, "ms": 128, "official_parameter_source": "%s"}\n' \
    "$seed" "$official" > "$run/status.json"
done

"$PY" scripts/summarize_no_xlag_mt128_comparison.py --base "$BASE"
echo "No-X-lag controlled comparison complete: $OUT"
