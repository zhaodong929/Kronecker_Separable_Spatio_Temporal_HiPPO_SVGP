#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-${ROOT}/era5_task1_10_processed.tar.zst}"
cd "${ROOT}"

for path in \
  data/era5/processed_timeseries_4 \
  data/era5/processed_timeseries_4_task1_10_extension; do
  if [[ ! -d "${path}" ]]; then
    echo "Missing processed dataset: ${path}" >&2
    exit 1
  fi
done

tar --zstd -cf "${OUTPUT}" \
  data/era5/processed_timeseries_4 \
  data/era5/processed_timeseries_4_task1_10_extension
sha256sum "${OUTPUT}" | tee "${OUTPUT}.sha256"
du -h "${OUTPUT}"
