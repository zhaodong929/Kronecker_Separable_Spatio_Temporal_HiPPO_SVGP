#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="${1:-${BENCHMARK_ROOT:-/root/autodl-tmp/iclr_era5_stage2plus}}"
OUTPUT="${2:-/root/autodl-tmp/iclr_era5_stage2plus_results.tar.zst}"

if [[ ! -d "${BENCHMARK_ROOT}" ]]; then
  echo "Benchmark directory does not exist: ${BENCHMARK_ROOT}" >&2
  exit 1
fi

PARENT="$(dirname "${BENCHMARK_ROOT}")"
NAME="$(basename "${BENCHMARK_ROOT}")"
tar --zstd -cf "${OUTPUT}" -C "${PARENT}" "${NAME}"
sha256sum "${OUTPUT}" | tee "${OUTPUT}.sha256"
du -h "${OUTPUT}"
