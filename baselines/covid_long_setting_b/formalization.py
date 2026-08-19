"""Small shared checks for the repaired Setting B formalization workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPAIRED_METHOD_IDS = (
    "ohsvgp_rbf",
    "ovc_svgp",
    "st_svgp",
    "lmc_svgp",
    "imc_svgp",
    "fsde_svi",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def formal_catalog_methods(catalog: dict[str, Any]) -> set[str]:
    """Return only methods whose catalog status allows a current main-table row."""

    return {
        str(entry["id"])
        for entry in catalog["methods"]
        if entry.get("setting_b_status") == "formal_result_available"
    }


def validate_prediction_archive(path: Path, *, weeks: int = 143, locations: int = 10) -> None:
    """Raise when an archive cannot be admitted to a formal Gaussian table."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        fields = {name: np.asarray(arrays[name], dtype=np.float64) for name in ("y_true", "pred_mean", "pred_var")}
    expected = (int(weeks), int(locations))
    if any(value.shape != expected for value in fields.values()):
        raise ValueError(f"{path} must contain {expected} y_true/pred_mean/pred_var arrays")
    if not all(np.isfinite(value).all() for value in fields.values()):
        raise ValueError(f"{path} contains non-finite formal predictions")
    if np.any(fields["pred_var"] <= 0.0):
        raise ValueError(f"{path} contains non-positive predictive variance")


def snapshot_archives(paths: Iterable[Path]) -> list[dict[str, str]]:
    """Record immutable archive identities without writing to their result roots."""

    records: list[dict[str, str]] = []
    for path in sorted({Path(item).resolve() for item in paths}):
        if not path.is_file():
            continue
        records.append({"path": str(path), "sha256": sha256_file(path)})
    return records


def verify_snapshot(records: Iterable[dict[str, str]]) -> list[str]:
    """Return human-readable mismatches instead of silently accepting drift."""

    mismatches: list[str] = []
    for record in records:
        path = Path(record["path"])
        if not path.is_file():
            mismatches.append(f"missing: {path}")
        elif sha256_file(path) != record["sha256"]:
            mismatches.append(f"changed: {path}")
    return mismatches
