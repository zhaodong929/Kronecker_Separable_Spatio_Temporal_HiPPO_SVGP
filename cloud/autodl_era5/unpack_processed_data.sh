#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE="${1:?Usage: unpack_processed_data.sh ARCHIVE [SHA256_FILE]}"
CHECKSUM="${2:-${ARCHIVE}.sha256}"

if [[ -f "${CHECKSUM}" ]]; then
  EXPECTED_SHA256="$(awk 'NF {print $1; exit}' "${CHECKSUM}")"
  ACTUAL_SHA256="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
  if [[ -z "${EXPECTED_SHA256}" || "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "SHA-256 mismatch for ${ARCHIVE}" >&2
    echo "expected: ${EXPECTED_SHA256:-missing}" >&2
    echo "actual:   ${ACTUAL_SHA256}" >&2
    exit 1
  fi
  echo "SHA-256 verified: ${ACTUAL_SHA256}"
else
  echo "Warning: no checksum file found at ${CHECKSUM}" >&2
fi
mkdir -p "${ROOT}/data/era5"
tar --zstd -xf "${ARCHIVE}" -C "${ROOT}"

test -f "${ROOT}/data/era5/processed_timeseries_4_task1_10_extension/verification_report.json"
echo "Processed ERA5 data installed under ${ROOT}/data/era5"
