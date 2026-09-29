#!/usr/bin/env bash
# Run on a DoC compute workstation, not a submission/login host.
set -euo pipefail
CAMPAIGN=/vol/bitbucket/nk523/hipposvgp-fair-20260929
UV="${CAMPAIGN}/uv"
"${UV}" venv --python 3.11 "${CAMPAIGN}/env-mgpvae"
"${UV}" pip install --python "${CAMPAIGN}/env-mgpvae/bin/python" -r "${CAMPAIGN}/source/environments/mgpvae-cpu.txt" pandas h5py
OFFICIAL="${CAMPAIGN}/official/MGPVAE"
mkdir -p "${CAMPAIGN}/official"
git clone --filter=blob:none --no-checkout https://github.com/harrisonzhu508/MGPVAE.git "${OFFICIAL}"
git -C "${OFFICIAL}" sparse-checkout set mgpvae
git -C "${OFFICIAL}" checkout --detach c9a80a05fca66b2911c8e7cb0deccb39d261a5b8
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1
export MGPVAE_SOURCE="${OFFICIAL}"
cd "${CAMPAIGN}/source"
"${CAMPAIGN}/env-mgpvae/bin/python" -m pytest -q tests/test_mgpvae_official_filter.py tests/test_mgpvae_mixture_metrics.py
exec "${CAMPAIGN}/env-mgpvae/bin/python" scripts/qualify_mgpvae_traffic.py \
  --h5 "${CAMPAIGN}/data/pems_bay/PEMS-BAY.h5" \
  --coordinates "${CAMPAIGN}/data/pems_bay/graph_sensor_locations_bay.csv" \
  --official-source "${OFFICIAL}" --output "${CAMPAIGN}/results/mgpvae-pems-qualification" --steps 20
