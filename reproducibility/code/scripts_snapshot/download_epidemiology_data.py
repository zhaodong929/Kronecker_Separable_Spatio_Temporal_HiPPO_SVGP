#!/usr/bin/env python3
"""Download versioned public epidemiology inputs for the Route B benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


COVID_FILES = {
    "covid_hospital_admissions.csv": (
        "https://raw.githubusercontent.com/CDCgov/covid19-forecast-hub/main/"
        "target-data/covid-hospital-admissions.csv"
    ),
    "time-series.parquet": (
        "https://raw.githubusercontent.com/CDCgov/covid19-forecast-hub/main/"
        "target-data/time-series.parquet"
    ),
    "locations.csv": (
        "https://raw.githubusercontent.com/CDCgov/covid19-forecast-hub/main/"
        "auxiliary-data/locations.csv"
    ),
    "target-data-README.md": (
        "https://raw.githubusercontent.com/CDCgov/covid19-forecast-hub/main/"
        "target-data/README.md"
    ),
    "LICENSE": (
        "https://raw.githubusercontent.com/CDCgov/covid19-forecast-hub/main/LICENSE"
    ),
    "2025_Gaz_state_national.zip": (
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
        "2025_Gazetteer/2025_Gaz_state_national.zip"
    ),
}

MALARIA_FILES = {
    "printables_catalog.json": (
        "https://data.malariaatlas.org/map-platform-app-backend/api/v1/printables"
    ),
    "portal.html": "https://data.malariaatlas.org/",
}

DENGUE_COORDINATES_URL = (
    "https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/main/"
    "csv/municipios.csv"
)
DENGUE_API_URL = "https://api.mosqlimate.org/api/datastore/infodengue/"
DENGUE_REGISTRATION_URL = "https://mosqlimate.org/profile/auth"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, force: bool) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return {
            "status": "existing",
            "url": url,
            "path": str(destination.resolve()),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    request = Request(url, headers={"User-Agent": "routeb-epidemiology-benchmark/1"})
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            with urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, temp)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    temp_path.replace(destination)
    return {
        "status": "downloaded",
        "url": url,
        "path": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def download_static_group(
    files: dict[str, str], destination: Path, *, force: bool
) -> dict[str, Any]:
    records = []
    status = "complete"
    for name, url in files.items():
        try:
            records.append(download_file(url, destination / name, force=force))
        except Exception as exc:  # network failures belong in the provenance record
            status = "download_error"
            records.append(
                {"status": "download_error", "url": url, "error": f"{type(exc).__name__}: {exc}"}
            )
    return {"status": status, "files": records}


def _request_json(url: str, api_key: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": "routeb-epidemiology-benchmark/1",
            "X-UID-Key": api_key,
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def download_dengue(
    destination: Path,
    *,
    force: bool,
    start: str,
    end: str,
    uf: str,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    coordinate_record = download_file(
        DENGUE_COORDINATES_URL,
        destination / "municipality_coordinates.csv",
        force=force,
    )
    api_key = os.environ.get("MOSQLIMATE_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "blocked_by_auth",
            "registration_url": DENGUE_REGISTRATION_URL,
            "required_environment_variable": "MOSQLIMATE_API_KEY",
            "coordinates": coordinate_record,
            "query": {"disease": "dengue", "start": start, "end": end, "uf": uf},
        }

    output = destination / "infodengue.jsonl"
    if output.exists() and not force:
        return {
            "status": "complete",
            "data": {
                "status": "existing",
                "path": str(output.resolve()),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            },
            "coordinates": coordinate_record,
            "query": {"disease": "dengue", "start": start, "end": end, "uf": uf},
        }

    query = {
        "disease": "dengue",
        "start": start,
        "end": end,
        "uf": uf,
        "page": 1,
        "per_page": 300,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination, delete=False
    ) as temp:
        temp_path = Path(temp.name)
        page = 1
        total_pages = 1
        rows = 0
        try:
            while page <= total_pages:
                query["page"] = page
                payload = _request_json(f"{DENGUE_API_URL}?{urlencode(query)}", api_key)
                items = payload.get("items", [])
                for item in items:
                    temp.write(json.dumps(item, ensure_ascii=True, allow_nan=False) + "\n")
                rows += len(items)
                pagination = payload.get("pagination", {})
                total_pages = int(pagination.get("total_pages", total_pages))
                page += 1
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    temp_path.replace(output)
    return {
        "status": "complete",
        "data": {
            "status": "downloaded",
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "rows": rows,
            "pages": total_pages,
        },
        "coordinates": coordinate_record,
        "query": {"disease": "dengue", "start": start, "end": end, "uf": uf},
        "revision_warning": (
            "Infodengue notified cases are retrospectively updated; this download is a fixed retrospective snapshot."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/epidemiology/raw"))
    parser.add_argument(
        "--datasets", nargs="+", choices=["covid", "dengue", "malaria"], default=["covid", "dengue", "malaria"]
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dengue-start", default="2015-01-01")
    parser.add_argument("--dengue-end", default="2025-12-31")
    parser.add_argument("--dengue-uf", default="SP")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    results: dict[str, Any] = {}
    if "covid" in args.datasets:
        results["covid"] = download_static_group(
            COVID_FILES, args.output_root / "covid", force=args.force
        )
    if "malaria" in args.datasets:
        results["malaria"] = download_static_group(
            MALARIA_FILES, args.output_root / "malaria", force=args.force
        )
    if "dengue" in args.datasets:
        try:
            results["dengue"] = download_dengue(
                args.output_root / "dengue",
                force=args.force,
                start=args.dengue_start,
                end=args.dengue_end,
                uf=args.dengue_uf,
            )
        except Exception as exc:
            results["dengue"] = {
                "status": "download_error",
                "error": f"{type(exc).__name__}: {exc}",
                "registration_url": DENGUE_REGISTRATION_URL,
            }

    manifest = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
