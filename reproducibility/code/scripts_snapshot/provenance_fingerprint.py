"""Capture reproducible source fingerprints for experiment run manifests."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root)


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    return _sha256_bytes(p.read_bytes()) if p.is_file() else ""


def capture_source_fingerprint(
    root: str | Path,
    *,
    config_paths: Iterable[str | Path] = (),
    split_paths: Iterable[str | Path] = (),
    data_manifest_paths: Iterable[str | Path] = (),
) -> dict[str, object]:
    root = Path(root)
    def joined_hash(paths: Iterable[str | Path]) -> str:
        parts = []
        for path in sorted((str(Path(p)) for p in paths)):
            parts.append(path + ":" + file_sha256(path))
        return _sha256_bytes("\n".join(parts).encode())
    return {
        "git_commit": _git(root, "rev-parse", "HEAD").decode().strip(),
        "git_status_sha256": _sha256_bytes(_git(root, "status", "--short")),
        "git_diff_sha256": _sha256_bytes(_git(root, "diff", "--no-ext-diff")),
        "untracked_manifest_sha256": _sha256_bytes(_git(root, "ls-files", "--others", "--exclude-standard")),
        "config_sha256": joined_hash(config_paths),
        "split_sha256": joined_hash(split_paths),
        "data_manifest_sha256": joined_hash(data_manifest_paths),
    }


def source_mutated(before: dict[str, object], after: dict[str, object]) -> bool:
    keys = (
        "git_commit", "git_status_sha256", "git_diff_sha256",
        "untracked_manifest_sha256", "config_sha256", "split_sha256",
        "data_manifest_sha256",
    )
    return any(before.get(k) != after.get(k) for k in keys)
