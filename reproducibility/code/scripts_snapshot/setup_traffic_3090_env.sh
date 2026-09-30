#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
VENV="${VENV:-$ROOT/.venv_3090}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

"$PYTHON_BOOTSTRAP" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install --index-url "$TORCH_INDEX_URL" "torch==2.7.1"
"$VENV/bin/python" -m pip install -r requirements/traffic-3090.txt
"$VENV/bin/python" -m pip install --no-deps -e .

"$VENV/bin/python" - <<'PY'
import torch
print({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
if not torch.cuda.is_available():
    raise SystemExit("Environment installed, but CUDA is unavailable")
print({"device": torch.cuda.get_device_name(0), "memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30})
PY

echo "PASS: traffic GPU environment created at $VENV"
