#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${ROOT}/baselines/external"
mkdir -p "${DEST}"

checkout_repo() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local directory="${DEST}/${name}"
  if [[ ! -d "${directory}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${directory}"
  fi
  if [[ -n "$(git -C "${directory}" status --porcelain)" ]]; then
    echo "Refusing to change dirty external repository: ${directory}" >&2
    return 1
  fi
  git -C "${directory}" fetch --quiet origin "${commit}"
  git -C "${directory}" checkout --quiet --detach "${commit}"
  local actual
  actual="$(git -C "${directory}" rev-parse HEAD)"
  if [[ "${actual}" != "${commit}" ]]; then
    echo "Commit verification failed for ${name}: ${actual}" >&2
    return 1
  fi
  printf '%-42s %s\n' "${name}" "${actual}"
}

checkout_repo \
  thangbui_streaming_sparse_gp \
  https://github.com/thangbui/streaming_sparse_gp.git \
  d95081b8a67316981088172124bc44e3c2235e49
checkout_repo \
  wjmaddox_online_gp \
  https://github.com/wjmaddox/online_gp.git \
  3bff4c347263a9b8b1f0aa801a986f4aaa019a66
checkout_repo \
  harrisonzhu508_HIPPOSVGP \
  https://github.com/harrisonzhu508/HIPPOSVGP.git \
  a1bff1bc81629316af92c97bde53d2792f4d8025
checkout_repo \
  aaltoml_spatio_temporal_gps \
  https://github.com/AaltoML/spatio-temporal-GPs.git \
  c5b929e1fc07b14ff9671dd1d66b3b8041e2a2ce
checkout_repo \
  secondmind_labs_markovflow_v0.0.13 \
  https://github.com/secondmind-labs/markovflow.git \
  06f21e08f379f167861c89ed75da08e0e34a9ed3

echo "Official baseline sources are pinned and verified."
