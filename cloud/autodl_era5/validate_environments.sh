#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${AUTODL_ENV_ROOT:-/root/autodl-tmp/stvgp_envs}"

"${ENV_ROOT}/routeb/bin/python" - <<'PY'
import torch
import stvgp_kronecker
print("routeb", torch.__version__, torch.version.cuda, torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Route B environment cannot see CUDA")
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

(
  cd "${ROOT}"
  "${ENV_ROOT}/routeb/bin/python" -m pytest -q tests/test_routeb_torch_backend.py
)

"${ENV_ROOT}/gpflow/bin/python" - <<'PY'
import gpflow
import tensorflow as tf
print("gpflow", gpflow.__version__, "tensorflow", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs", gpus)
if not gpus:
    raise SystemExit("GPflow environment cannot see CUDA")
PY

"${ENV_ROOT}/maddox/bin/python" - <<PY
import sys
import torch
import gpytorch
sys.path.insert(0, "${ROOT}/baselines/external/wjmaddox_online_gp")
from online_gp.models.streaming_sgpr import StreamingSGPR
print("maddox", torch.__version__, gpytorch.__version__, torch.cuda.is_available(), StreamingSGPR)
if not torch.cuda.is_available():
    raise SystemExit("Maddox environment cannot see CUDA")
PY

if [[ -x "${ENV_ROOT}/stvgp_legacy/bin/python" ]]; then
  LEGACY_CUDA_ROOT="${LEGACY_CUDA_ROOT:-/usr/local/cuda-11.8}"
  [[ -x "${LEGACY_CUDA_ROOT}/bin/ptxas" && -d "${LEGACY_CUDA_ROOT}/nvvm/libdevice" ]]
  PATH="${LEGACY_CUDA_ROOT}/bin:${PATH}" \
    XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_cuda_data_dir=${LEGACY_CUDA_ROOT}" \
    "${ENV_ROOT}/stvgp_legacy/bin/python" - <<'PY'
import bayesnewton, jax, jaxlib, jax.numpy as jnp, objax
print("legacy ST-SVGP", jax.__version__, jaxlib.__version__, objax.__version__)
if jax.lib.xla_bridge.get_backend().platform != "gpu":
    raise SystemExit("Legacy ST-SVGP environment cannot execute JAX on CUDA")
print("legacy JAX smoke", jnp.arange(8).reshape(2, 4).sum(axis=1))
PY
fi

if [[ -x "${ENV_ROOT}/markovflow/bin/python" ]]; then
  PYTHONPATH="${ROOT}/baselines/external/secondmind_labs_markovflow_v0.0.13" \
    "${ENV_ROOT}/markovflow/bin/python" - <<'PY'
import gpflow, markovflow, tensorflow as tf
print("legacy Markovflow", gpflow.__version__, tf.__version__)
print("GPU support is not expected for the TensorFlow 2.2.1 compatibility stack.")
PY
fi
