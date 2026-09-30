#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv_3090/bin/python" ]]; then
    PYTHON="$ROOT/.venv_3090/bin/python"
  else
    PYTHON="$ROOT/.venv_cuda128/bin/python"
  fi
fi
OUTPUT="${OUTPUT:-$ROOT/results/traffic/formal_future_locked_sm_q2_road_context_v1}"
SMOKE="$OUTPUT/smoke_seed1_3090"
LOG_DIR="$OUTPUT/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run_pems_future_3090.log") 2>&1

git rev-parse HEAD > "$OUTPUT/source_commit.txt"
"$PYTHON" -m pip freeze > "$OUTPUT/environment_freeze.txt"

echo "[1/9] GPU preflight"
nvidia-smi
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("FAIL: CUDA is unavailable")
properties = torch.cuda.get_device_properties(0)
print({"device": properties.name, "memory_gib": properties.total_memory / 2**30, "torch": torch.__version__})
PY

echo "[2/9] Protocol and feature-policy tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -q \
  tests/test_traffic_protocol.py \
  tests/test_protocol_f_access_guard.py \
  tests/test_traffic_runner_temporal_cache.py \
  tests/test_traffic_spatial_kernels.py \
  tests/test_traffic_graph_future_baseline.py

echo "[3/9] Seed-1 smoke: persistence, Kron-STGP and KronHiPPO-STGP"
"$PYTHON" scripts/run_traffic_locked_future_three_seed.py \
  --seeds 1 \
  --methods persistence kron_stgp kronhippo_stgp \
  --output "$SMOKE" \
  --chunk-steps 144 \
  --max-stream-steps 288 \
  --device cuda

echo "[4/9] Seed-1 smoke: Graph WaveNet and DCRNN"
"$PYTHON" scripts/run_traffic_graph_future_three_seed.py \
  --seeds 1 \
  --methods graph_wavenet dcrnn \
  --output "$SMOKE" \
  --batch-size 32 \
  --max-epochs 2 \
  --patience 2 \
  --max-stream-origins 288 \
  --device cuda

echo "[5/9] Joint smoke archive audit"
"$PYTHON" scripts/audit_traffic_future_results.py \
  --root "$SMOKE" \
  --seeds 1 \
  --methods persistence kron_stgp kronhippo_stgp graph_wavenet dcrnn \
  --expected-steps 288

echo "[6/9] Formal GP/control seeds 1--3; rerun this script to resume"
"$PYTHON" scripts/run_traffic_locked_future_three_seed.py \
  --seeds 1 2 3 \
  --methods persistence kron_stgp kronhippo_stgp \
  --output "$OUTPUT" \
  --chunk-steps 2000 \
  --device cuda

echo "[7/9] Formal Graph WaveNet seeds 1--3"
"$PYTHON" scripts/run_traffic_graph_future_three_seed.py \
  --seeds 1 2 3 \
  --methods graph_wavenet \
  --output "$OUTPUT" \
  --batch-size 32 \
  --max-epochs 100 \
  --patience 10 \
  --device cuda

echo "[8/9] Formal DCRNN seeds 1--3"
"$PYTHON" scripts/run_traffic_graph_future_three_seed.py \
  --seeds 1 2 3 \
  --methods dcrnn \
  --output "$OUTPUT" \
  --batch-size 32 \
  --max-epochs 100 \
  --patience 10 \
  --device cuda

echo "[9/9] Joint formal archive audit and summary"
"$PYTHON" scripts/audit_traffic_future_results.py \
  --root "$OUTPUT" \
  --seeds 1 2 3 \
  --methods persistence kron_stgp kronhippo_stgp graph_wavenet dcrnn \
  --expected-steps 50088
"$PYTHON" scripts/summarize_traffic_locked_future.py \
  --root "$OUTPUT" \
  --seeds 1 2 3 \
  --methods persistence kron_stgp kronhippo_stgp graph_wavenet dcrnn

echo "PASS: all five PEMS-BAY Protocol-F methods completed on seeds 1--3, audited and summarised."
