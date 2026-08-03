#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${AUTODL_ENV_ROOT:-/root/autodl-tmp/stvgp_envs}"
TOOLS_ROOT="${AUTODL_TOOLS_ROOT:-/root/autodl-tmp/stvgp_tools}"
INCLUDE_LEGACY="${INCLUDE_LEGACY:-0}"
INSTALL_TEX="${INSTALL_TEX:-0}"
mkdir -p "${ENV_ROOT}" "${TOOLS_ROOT}"

if command -v micromamba >/dev/null 2>&1; then
  MAMBA="$(command -v micromamba)"
else
  MAMBA="${TOOLS_ROOT}/micromamba"
  if [[ ! -x "${MAMBA}" ]]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C "${TOOLS_ROOT}" --strip-components=1 bin/micromamba
  fi
fi

export MAMBA_ROOT_PREFIX="${TOOLS_ROOT}/mamba-root"

create_base_env() {
  local name="$1"
  local python_version="$2"
  if [[ ! -x "${ENV_ROOT}/${name}/bin/python" ]]; then
    "${MAMBA}" create -y -p "${ENV_ROOT}/${name}" \
      -c conda-forge "python=${python_version}" pip
  fi
}

bash "${ROOT}/cloud/autodl_era5/clone_official_baselines.sh"

create_base_env routeb 3.11
ROUTEB_PY="${ENV_ROOT}/routeb/bin/python"
"${ROUTEB_PY}" -m pip install --upgrade "pip<25.3" wheel setuptools
"${ROUTEB_PY}" -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
"${ROUTEB_PY}" -m pip install \
  numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 \
  pandas==2.2.3 matplotlib==3.9.2 gpytorch==1.14 \
  pytest==8.3.4 psutil==6.1.1 pyyaml==6.0.2
"${ROUTEB_PY}" -m pip install -e "${ROOT}" --no-deps

create_base_env gpflow 3.10
GPFLOW_PY="${ENV_ROOT}/gpflow/bin/python"
"${GPFLOW_PY}" -m pip install --upgrade "pip<25.3" wheel "setuptools<81"
"${GPFLOW_PY}" -m pip install \
  "tensorflow[and-cuda]==2.15.1" tensorflow-probability==0.23.0 \
  gpflow==2.9.2 numpy==1.26.4 scipy==1.12.0 \
  pandas==2.2.3 matplotlib==3.9.2 psutil==6.1.1
"${GPFLOW_PY}" -m pip install -e "${ROOT}" --no-deps

# GPyTorch 1.9 retains the lazy-tensor API used by the official Maddox commit,
# while PyTorch 2.0.1/cu118 supports Ada GPUs such as RTX 4090.
create_base_env maddox 3.10
MADDOX_PY="${ENV_ROOT}/maddox/bin/python"
"${MADDOX_PY}" -m pip install --upgrade "pip<25.3" wheel setuptools
"${MADDOX_PY}" -m pip install \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
"${MADDOX_PY}" -m pip install \
  gpytorch==1.9.1 botorch==0.8.1 \
  numpy==1.24.4 scipy==1.10.1 psutil==6.1.1
"${MADDOX_PY}" -m pip install -e "${ROOT}" --no-deps

if [[ "${INCLUDE_LEGACY}" == "1" ]]; then
  create_base_env stvgp_legacy 3.7
  STVGP_PY="${ENV_ROOT}/stvgp_legacy/bin/python"
  "${STVGP_PY}" -m pip install --upgrade "pip<24.1" "setuptools<68" wheel
  "${STVGP_PY}" -m pip install \
    numpy==1.19.5 scipy==1.7.1 scikit-learn==1.0.2 \
    jax==0.2.9 jaxlib==0.1.60 \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html
  "${STVGP_PY}" -m pip install \
    objax==1.3.1 bayesnewton==1.1 numba==0.54.1 \
    matplotlib==3.4.3 pandas==1.3.4 protobuf==3.20.3

  create_base_env markovflow 3.7
  MARKOV_PY="${ENV_ROOT}/markovflow/bin/python"
  "${MARKOV_PY}" -m pip install --upgrade "pip<24.1" "setuptools<68" wheel
  "${MARKOV_PY}" -m pip install \
    numpy==1.18.5 scipy==1.4.1 tensorflow==2.2.1 \
    tensorflow-probability==0.11.0 gpflow==2.1.3 \
    banded-matrices==0.0.6 protobuf==3.20.3
  "${MARKOV_PY}" -m pip install \
    -e "${ROOT}/baselines/external/secondmind_labs_markovflow_v0.0.13" \
    --no-deps
fi

if [[ "${INSTALL_TEX}" == "1" ]]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-extra texlive-fonts-recommended latexmk
fi

echo
echo "Environment installation complete: ${ENV_ROOT}"
echo "Run validation with: cloud/autodl_era5/validate_environments.sh"
