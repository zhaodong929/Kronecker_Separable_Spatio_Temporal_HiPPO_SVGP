#!/usr/bin/env python3
"""Download and fingerprint the CDC NHSN mandatory-period COVID history."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_URL = (
    "https://data.cdc.gov/api/views/ua7e-t2fy/rows.csv?accessType=DOWNLOAD"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "cdc_hrd_weekly_hospital_respiratory.csv"
    manifest_path = output_root / "download_manifest.json"
    started = datetime.now(timezone.utc).isoformat()
    attempts: list[dict[str, object]] = []

    for attempt in range(1, int(args.retries) + 1):
        request = Request(args.source_url, headers={"User-Agent": "routeb-covid-long-stream/1"})
        temp_path: Path | None = None
        try:
            with urlopen(request, timeout=int(args.timeout_seconds)) as response:
                headers = {str(key): str(value) for key, value in response.headers.items()}
                with tempfile.NamedTemporaryFile(dir=output_root, delete=False) as handle:
                    temp_path = Path(handle.name)
                    shutil.copyfileobj(response, handle)
            if temp_path.stat().st_size == 0:
                raise ValueError("CDC response body was empty")
            temp_path.replace(target)
            manifest = {
                "status": "complete",
                "source_url": args.source_url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "started_at": started,
                "attempt": attempt,
                "http_headers": headers,
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "mandatory_window": {"start": "2020-08-01", "end": "2024-04-30"},
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(manifest, indent=2))
            return
        except (OSError, URLError, ValueError) as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            attempts.append(
                {
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            if attempt < int(args.retries):
                time.sleep(min(30, 2**attempt))

    manifest = {
        "status": "download_error",
        "source_url": args.source_url,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "attempts": attempts,
        "manual_download_url": "https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
