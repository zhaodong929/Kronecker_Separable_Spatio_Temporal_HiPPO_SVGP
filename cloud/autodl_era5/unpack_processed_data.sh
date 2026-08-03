#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE="${1:?Usage: unpack_processed_data.sh ARCHIVE [SHA256_FILE]}"
CHECKSUM="${2:-${ARCHIVE}.sha256}"

if [[ -f "${CHECKSUM}" ]]; then
  (cd "$(dirname "${ARCHIVE}")" && sha256sum -c "$(basename "${CHECKSUM}")")
else
  echo "Warning: no checksum file found at ${CHECKSUM}" >&2
fi
mkdir -p "${ROOT}/data/era5"
tar --zstd -xf "${ARCHIVE}" -C "${ROOT}"

test -f "${ROOT}/data/era5/processed_timeseries_4_task1_10_extension/verification_report.json"
echo "Processed ERA5 data installed under ${ROOT}/data/era5"
