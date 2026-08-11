#!/usr/bin/env bash
# This job validates pre-created environments. It intentionally does not run
# setup_autodl.sh and never installs JAX: a plain jaxlib wheel can be CPU-only.
# STVGP_PY must point to a pre-created Micromamba environment containing a
# CUDA-enabled legacy jax/jaxlib build. Validation hard-fails unless
# jax.devices() contains an A100 GPU. The old Markovflow TF2.2 stack is an
# optional CPU-only compatibility path and is excluded from A100 timing.
#
# The GRES request is deliberately generic for cluster portability. The checks
# below enforce an A100 with at least 75 GiB visible memory.
#SBATCH --job-name=era5-a100-validate
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=era5_a100_validate_%j.out
#SBATCH --error=era5_a100_validate_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO}}"
ENV_ROOT="${ERA5_ENV_ROOT:-${ENV_ROOT:-${HOME}/stvgp_envs}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/iclr_era5_full_benchmark}"
STVGP_PY="${STVGP_PY:-${ENV_ROOT}/stvgp_legacy/bin/python}"
ROUTEB_PY="${ROUTEB_PY:-${ENV_ROOT}/routeb/bin/python}"
TF_PY="${TF_PY:-${ENV_ROOT}/gpflow/bin/python}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${ENV_ROOT}/micromamba-root}"

usage() {
  cat >&2 <<'EOF'
Usage: setup_and_validate.sbatch [--repo PATH] [--env PATH] [--benchmark PATH]
                                [--stvgp-python PATH]

The --stvgp-python path must already contain a CUDA-enabled legacy JAX/JAXLIB
installation. This job never installs or replaces JAX. jax.devices() must
expose an A100 device; CPU-only legacy environments are rejected.
Markovflow/TF2.2 compatibility environments are labeled CPU-only and excluded
from A100 timing claims.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_ROOT="$2"; shift 2 ;;
    --env) ENV_ROOT="$2"; shift 2 ;;
    --benchmark) BENCHMARK_ROOT="$2"; shift 2 ;;
    --stvgp-python) STVGP_PY="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
BENCHMARK_ROOT="$(mkdir -p "${BENCHMARK_ROOT}" && cd "${BENCHMARK_ROOT}" && pwd)"
mkdir -p "${BENCHMARK_ROOT}/slurm/validation"
VALIDATION_DIR="${BENCHMARK_ROOT}/slurm/validation"
TMP_DIR="$(mktemp -d "${VALIDATION_DIR}/.validate.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

CHECK_FAILED=0
GPU_CSV="${TMP_DIR}/nvidia_smi.csv"
GPU_LIST="${TMP_DIR}/nvidia_smi_L.txt"
GPU_QUERY="${TMP_DIR}/nvidia_smi_q.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi --query-gpu=index,name,memory.total,driver_version,uuid --format=csv,noheader,nounits >"${GPU_CSV}"; then
    echo "nvidia-smi could not query the allocated GPU" >&2
    CHECK_FAILED=1
  fi
  nvidia-smi -L >"${GPU_LIST}" || CHECK_FAILED=1
  nvidia-smi -q >"${GPU_QUERY}" || CHECK_FAILED=1
else
  echo "nvidia-smi is required for A100 validation" >&2
  CHECK_FAILED=1
  : >"${GPU_CSV}"
  : >"${GPU_LIST}"
  : >"${GPU_QUERY}"
fi

if ! python3 - "${GPU_CSV}" <<'PY'
import csv
import sys

path = sys.argv[1]
rows = []
try:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
except OSError:
    pass
valid = []
for row in rows:
    if len(row) < 3:
        continue
    try:
        memory_mib = float(row[2].strip())
    except ValueError:
        continue
    if "a100" in row[1].lower() and memory_mib >= 75 * 1024:
        valid.append((row[1].strip(), memory_mib))
if not valid:
    raise SystemExit("No visible A100 with at least 75 GiB was found")
for name, memory_mib in valid:
    print(f"validated {name} with {memory_mib:.0f} MiB")
PY
then
  CHECK_FAILED=1
fi

for required in "${ROUTEB_PY}" "${TF_PY}" "${STVGP_PY}"; do
  if [[ ! -x "${required}" ]]; then
    echo "Required pre-created environment Python is missing or not executable: ${required}" >&2
    CHECK_FAILED=1
  fi
done

git -C "${REPO_ROOT}" rev-parse HEAD >"${TMP_DIR}/git_commit.txt" 2>"${TMP_DIR}/git_commit.log" || CHECK_FAILED=1
for item in "routeb:${ROUTEB_PY}" "gpflow:${TF_PY}" "stvgp_legacy:${STVGP_PY}"; do
  name="${item%%:*}"
  python_path="${item#*:}"
  if [[ -x "${python_path}" ]]; then
    "${python_path}" -m pip freeze >"${TMP_DIR}/${name}_freeze.txt" 2>"${TMP_DIR}/${name}_freeze.log" || CHECK_FAILED=1
  fi
done

check_torch() {
  local output="${TMP_DIR}/torch.json"
  if "${ROUTEB_PY}" - "${output}" >"${TMP_DIR}/torch.log" 2>&1 <<'PY'
import json
import sys
import torch

output = sys.argv[1]
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("Torch cannot see CUDA; CPU fallback is forbidden")
torch.cuda.set_device(0)
props = torch.cuda.get_device_properties(0)
name = str(props.name)
memory_bytes = int(props.total_memory)
if "a100" not in name.lower() or memory_bytes < 75 * 1024**3:
    raise SystemExit(f"Torch device is not an A100-80GB: {name}, {memory_bytes} bytes")
torch.ones(1, device="cuda").sum().item()
torch.cuda.synchronize()
payload = {
    "status": "passed",
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()),
    "device_name": name,
    "device_memory_bytes": memory_bytes,
    "device_capability": list(torch.cuda.get_device_capability(0)),
    "cpu_fallback": False,
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY
  then
    echo "Torch CUDA validation passed"
  else
    echo "Torch CUDA validation failed; see ${TMP_DIR}/torch.log" >&2
    CHECK_FAILED=1
  fi
}

check_reporting() {
  local output="${TMP_DIR}/reporting.json"
  if "${ROUTEB_PY}" - "${output}" >"${TMP_DIR}/reporting.log" 2>&1 <<'PY'
import json
import sys
import matplotlib
import reportlab

output = sys.argv[1]
with open(output, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "status": "passed",
            "matplotlib": matplotlib.__version__,
            "reportlab": getattr(reportlab, "Version", "unknown"),
        },
        handle,
        indent=2,
    )
PY
  then
    echo "Report dependencies validation passed"
  else
    echo "Report dependencies validation failed; see ${TMP_DIR}/reporting.log" >&2
    CHECK_FAILED=1
  fi
}

check_tensorflow() {
  local output="${TMP_DIR}/tensorflow.json"
  if "${TF_PY}" - "${output}" >"${TMP_DIR}/tensorflow.log" 2>&1 <<'PY'
import json
import sys
import tensorflow as tf

output = sys.argv[1]
physical = tf.config.list_physical_devices("GPU")
if not physical:
    raise SystemExit("TensorFlow cannot see a GPU; CPU fallback is forbidden")
tf.config.set_soft_device_placement(False)
with tf.device("/GPU:0"):
    value = tf.constant([1.0]) + tf.constant([1.0])
value.numpy()
if "GPU" not in str(value.device).upper():
    raise SystemExit(f"TensorFlow placed the validation op on CPU: {value.device}")
payload = {
    "status": "passed",
    "tensorflow_version": tf.__version__,
    "physical_gpus": [device.name for device in physical],
    "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
    "probe_device": str(value.device),
    "cpu_fallback": False,
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY
  then
    echo "TensorFlow GPU validation passed"
  else
    echo "TensorFlow GPU validation failed; see ${TMP_DIR}/tensorflow.log" >&2
    CHECK_FAILED=1
  fi
}

check_legacy_jax() {
  local output="${TMP_DIR}/legacy_jax.json"
  if JAX_PLATFORM_NAME=gpu "${STVGP_PY}" - "${output}" >"${TMP_DIR}/legacy_jax.log" 2>&1 <<'PY'
import json
import platform
import sys
import jax
import jax.numpy as jnp
import jaxlib
import bayesnewton
import objax

output = sys.argv[1]
if sys.version_info[:2] != (3, 7):
    raise SystemExit(f"Legacy ST-VGP environment must use Python 3.7, got {platform.python_version()}")
devices = list(jax.devices())
gpu_devices = [
    device
    for device in devices
    if str(getattr(device, "platform", "")).lower() in {"gpu", "cuda"}
]
a100_devices = [
    device
    for device in gpu_devices
    if "a100" in (
        str(getattr(device, "device_kind", "")) + " " + str(device)
    ).lower()
]
if not a100_devices:
    raise SystemExit(
        "Legacy JAX must expose an A100 through jax.devices(); CPU-only jaxlib is rejected"
    )
if str(jax.default_backend()).lower() not in {"gpu", "cuda"}:
    raise SystemExit(f"Legacy JAX default backend is CPU: {jax.default_backend()}")
probe = jax.device_put(jnp.ones((1,), dtype=jnp.float32), a100_devices[0])
if hasattr(probe, "block_until_ready"):
    probe.block_until_ready()
payload = {
    "status": "passed",
    "python_version": platform.python_version(),
    "jax_version": getattr(jax, "__version__", None),
    "jaxlib_version": getattr(jaxlib, "__version__", None),
    "bayesnewton_version": getattr(bayesnewton, "__version__", "unknown"),
    "objax_version": getattr(objax, "__version__", "unknown"),
    "backend_platform_version": str(getattr(jax.lib.xla_bridge.get_backend(), "platform_version", "unknown")),
    "default_backend": str(jax.default_backend()),
    "devices": [str(device) for device in devices],
    "gpu_devices": [str(device) for device in gpu_devices],
    "a100_devices": [str(device) for device in a100_devices],
    "cpu_fallback": False,
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY
  then
    echo "Legacy JAX A100 validation passed"
  else
    echo "Legacy JAX A100 validation failed; see ${TMP_DIR}/legacy_jax.log" >&2
    CHECK_FAILED=1
  fi
}

check_stsvgp_smoke() {
  local output="${TMP_DIR}/legacy_stsvgp_smoke.json"
  if JAX_PLATFORM_NAME=gpu "${STVGP_PY}" - "${REPO_ROOT}" "${output}" >"${TMP_DIR}/legacy_stsvgp_smoke.log" 2>&1 <<'PY'
import argparse
import importlib.util
import json
from pathlib import Path
import sys

repo = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location(
    "official_stvgp_smoke", repo / "scripts/run_official_stvgp_legacy.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load the official ST-SVGP runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.import_backend()
times, train_coords, _, y_train, _ = runner.synthetic_data(0)
values = (times, train_coords, train_coords[:2], y_train, y_train[:, :2])
splits, _ = runner.prepare_splits(values, {})
phase = runner.make_phase_data("st_svgp", times, splits["fit"])
args = argparse.Namespace(
    model="st_svgp",
    num_spatial_inducing=3,
    seed=0,
    fixed_spatial_inducing=True,
    temporal_lengthscale=0.2,
    spatial_lengthscale=0.8,
    likelihood_variance=0.05,
    num_temporal_inducing=None,
    parallel=False,
    jit=False,
    newton_rate=1.0,
    learning_rate=0.01,
    early_stop_relative_tol=None,
    early_stop_min_iterations=1,
    early_stop_patience=1,
    validation_every=1,
    log_every=1,
    trajectory_every=0,
)
built = runner._build_phase(args, times, phase, {})
forward_mean, forward_var, _, _ = runner._predict_record(
    built["model"], built["phase"], times, splits["fit"], "fit"
)
if forward_mean.shape != splits["fit"]["y"].shape:
    raise RuntimeError(f"unexpected ST-SVGP forward shape: {forward_mean.shape}")
if not (bool((forward_mean == forward_mean).all()) and bool((forward_var == forward_var).all())):
    raise RuntimeError("ST-SVGP forward produced non-finite values")
training = runner._train_phase(args, built, times, 1)
loss = float(training["losses"][0])
if not (loss == loss and abs(loss) != float("inf")):
    raise RuntimeError("ST-SVGP update produced a non-finite loss")
payload = {
    "status": "passed",
    "model": "st_svgp",
    "data": "official runner synthetic_data(seed=0)",
    "forward": {
        "mean_shape": list(forward_mean.shape),
        "variance_shape": list(forward_var.shape),
    },
    "update": {
        "iterations": int(training["iterations"]),
        "loss": loss,
    },
    "jax_backend": str(runner.jax.default_backend()),
    "jax_devices": [str(device) for device in runner.jax.devices()],
    "cpu_fallback": False,
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  then
    echo "Official ST-SVGP synthetic forward/update smoke passed"
  else
    echo "Official ST-SVGP synthetic forward/update smoke failed; see ${TMP_DIR}/legacy_stsvgp_smoke.log" >&2
    CHECK_FAILED=1
  fi
}

if [[ -x "${ROUTEB_PY}" ]]; then check_torch; fi
if [[ -x "${ROUTEB_PY}" ]]; then check_reporting; fi
if [[ -x "${TF_PY}" ]]; then check_tensorflow; fi
if [[ -x "${STVGP_PY}" ]]; then
  check_legacy_jax
  check_stsvgp_smoke
else
  printf '%s\n' '{"status":"failed","detail":"STVGP_PY is missing; official ST-SVGP smoke was not runnable"}' \
    >"${TMP_DIR}/legacy_stsvgp_smoke.json"
fi

python3 - "${GPU_CSV}" "${GPU_LIST}" "${GPU_QUERY}" "${TMP_DIR}" "${VALIDATION_DIR}/environment_metadata.json" "${REPO_ROOT}" "${ENV_ROOT}" "${MAMBA_ROOT_PREFIX}" "${ROUTEB_PY}" "${TF_PY}" "${STVGP_PY}" "${CHECK_FAILED}" <<'PY'
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import sys

gpu_csv, gpu_list, gpu_query, checks_dir, output, repo, env_root, mamba_root, routeb, tf, stvgp, failed = sys.argv[1:]
gpu_rows = []
with open(gpu_csv, newline="", encoding="utf-8") as handle:
    for row in csv.reader(handle):
        if len(row) >= 5:
            gpu_rows.append(
                {
                    "index": row[0].strip(),
                    "name": row[1].strip(),
                    "memory_total_mib": row[2].strip(),
                    "driver_version": row[3].strip(),
                    "uuid": row[4].strip(),
                }
            )


def read_json(name):
    path = Path(checks_dir) / name
    if not path.is_file():
        result = {"status": "failed", "detail": f"no result written: {path}"}
        log_path = path.with_suffix(".log")
        if log_path.is_file():
            result["log_tail"] = log_path.read_text(
                encoding="utf-8", errors="ignore"
            )[-4000:]
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "failed", "detail": str(exc)}


payload = {
    "schema_version": 1,
    "status": "failed" if failed != "0" else "passed",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "platform": platform.platform(),
    "repo": str(Path(repo).resolve()),
    "git_commit": Path(checks_dir, "git_commit.txt").read_text(encoding="utf-8", errors="ignore").strip(),
    "micromamba": {
        "environment_root": str(Path(env_root).resolve()),
        "mamba_root_prefix": str(Path(mamba_root).resolve()),
        "stvgp_python": str(Path(stvgp).resolve()),
        "routeb_python": str(Path(routeb).resolve()),
        "tensorflow_python": str(Path(tf).resolve()),
    },
    "slurm": {
        key: os.environ.get(key)
        for key in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_PARTITION",
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "CUDA_VISIBLE_DEVICES",
        )
        if os.environ.get(key) is not None
    },
    "gpu": {
        "requirement": "NVIDIA A100 with at least 75 GiB visible memory",
        "nvidia_smi": gpu_rows,
        "nvidia_smi_l": Path(gpu_list)
        .read_text(encoding="utf-8", errors="ignore")
        .splitlines(),
        "nvidia_smi_q": Path(gpu_query).read_text(encoding="utf-8", errors="ignore"),
    },
    "checks": {
        "torch": read_json("torch.json"),
        "reporting": read_json("reporting.json"),
        "tensorflow": read_json("tensorflow.json"),
        "legacy_jax": read_json("legacy_jax.json"),
        "legacy_stsvgp_smoke": read_json("legacy_stsvgp_smoke.json"),
    },
    "timing_policy": {
        "accuracy": "one allocated GPU, 16 CPUs, 128G, 24 hours",
        "legacy_jax": "must be a CUDA-enabled pre-created environment; CPU fallback is forbidden",
        "markovflow": "TF2.2 compatibility stack may remain CPU-only and is excluded from A100 timing claims",
    },
    "prerequisites": [
        "Do not replace STVGP_PY with the plain jaxlib==0.1.60 CPU wheel.",
        "STVGP_PY must expose an A100 in jax.devices().",
    ],
    "environment_freezes": {
        name: Path(checks_dir, f"{name}_freeze.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
        if Path(checks_dir, f"{name}_freeze.txt").is_file()
        else []
        for name in ("routeb", "gpflow", "stvgp_legacy")
    },
}
Path(output).parent.mkdir(parents=True, exist_ok=True)
temporary = Path(output).with_name(f".{Path(output).name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary, output)
print(json.dumps({"status": payload["status"], "output": output}, indent=2))
PY

if [[ "${CHECK_FAILED}" -ne 0 ]]; then
  echo "A100 environment validation failed; inspect ${VALIDATION_DIR}/environment_metadata.json" >&2
  exit 1
fi
echo "A100 environment validation passed: ${VALIDATION_DIR}/environment_metadata.json"
