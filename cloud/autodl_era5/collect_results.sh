#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="${1:-${BENCHMARK_ROOT:-/root/autodl-tmp/iclr_era5_stage2plus}}"
OUTPUT="${2:-/root/autodl-tmp/iclr_era5_stage2plus_results.tar.zst}"
mkdir -p "$(dirname "${OUTPUT}")"
OUTPUT_DIR="$(cd "$(dirname "${OUTPUT}")" && pwd)"
OUTPUT_NAME="$(basename "${OUTPUT}")"
OUTPUT="${OUTPUT_DIR}/${OUTPUT_NAME}"

if [[ ! -d "${BENCHMARK_ROOT}" ]]; then
  echo "Benchmark directory does not exist: ${BENCHMARK_ROOT}" >&2
  exit 1
fi

PARENT="$(dirname "${BENCHMARK_ROOT}")"
NAME="$(basename "${BENCHMARK_ROOT}")"
tar --zstd -cf "${OUTPUT}" -C "${PARENT}" "${NAME}"
(cd "${OUTPUT_DIR}" && sha256sum "${OUTPUT_NAME}") | tee "${OUTPUT}.sha256"
du -h "${OUTPUT}"
