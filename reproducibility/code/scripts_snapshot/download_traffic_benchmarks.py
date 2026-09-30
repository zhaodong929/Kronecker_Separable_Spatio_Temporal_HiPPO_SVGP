#!/usr/bin/env python3
"""Fetch the canonical-form PEMS-BAY and METR-LA HDF files with provenance."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stvgp_kronecker.data.traffic import DATASET_SPECS


SOURCES = {
    "pems_bay": {
        "h5": "https://huggingface.co/datasets/jimmygao3218/PEMSBAY/resolve/main/PEMS-BAY.h5?download=true",
        "coordinates": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations_bay.csv",
        "source_revision": "c858397fb381d82c3d55ad0a81beb51a9bb9bdaa",
    },
    "metr_la": {
        "h5": "https://huggingface.co/datasets/jimmygao3218/METRLA/resolve/main/metr-la.h5?download=true",
        "coordinates": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv",
        "source_revision": "ec9772c0e8dd48b3470a6381dff1888257a9ffe4",
    },
}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, *, overwrite: bool) -> dict[str, object]:
    if destination.exists() and not overwrite:
        return {
            "path": str(destination),
            "downloaded": False,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "stvgp-kronecker-traffic-adapter/1.0"})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    temporary.replace(destination)
    return {
        "path": str(destination),
        "downloaded": True,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/traffic/raw"))
    parser.add_argument("--dataset", choices=["all", *DATASET_SPECS], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    datasets = sorted(DATASET_SPECS) if args.dataset == "all" else [args.dataset]
    manifest: dict[str, object] = {"schema_version": 1, "datasets": {}}
    for name in datasets:
        spec = DATASET_SPECS[name]
        source = SOURCES[name]
        root = args.data_root / name
        h5 = download(source["h5"], root / str(spec["h5_name"]), overwrite=args.overwrite)
        coordinates = download(
            source["coordinates"],
            root / str(spec["coordinates_name"]),
            overwrite=args.overwrite,
        )
        manifest["datasets"][name] = {
            "source_revision": source["source_revision"],
            "h5_url": source["h5"],
            "coordinates_url": source["coordinates"],
            "h5": h5,
            "coordinates": coordinates,
        }
        print(json.dumps({name: manifest["datasets"][name]}, sort_keys=True), flush=True)
    args.data_root.mkdir(parents=True, exist_ok=True)
    (args.data_root / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
