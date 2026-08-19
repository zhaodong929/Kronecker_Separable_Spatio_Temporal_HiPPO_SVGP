"""Validation helpers for fixed analytic temporal-kernel configurations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_spectral_mixture_config(path: Path | None) -> dict[str, tuple[float, ...]] | None:
    """Load one fixed one-dimensional spectral-mixture configuration."""

    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, tuple[float, ...]] = {}
    for key in ("weights", "means", "scales"):
        if key not in payload:
            raise ValueError(f"Spectral-mixture configuration is missing '{key}'")
        values = np.asarray(payload[key], dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Spectral-mixture '{key}' must be finite and non-empty")
        result[key] = tuple(float(value) for value in values)
    if len({len(values) for values in result.values()}) != 1:
        raise ValueError("Spectral-mixture weights, means, and scales must have equal length")
    if any(value < 0.0 for value in result["weights"]):
        raise ValueError("Spectral-mixture weights must be non-negative")
    if sum(result["weights"]) <= 0.0:
        raise ValueError("Spectral-mixture weights must contain a positive value")
    if any(value <= 0.0 for value in result["scales"]):
        raise ValueError("Spectral-mixture scales must be positive")
    return result


def temporal_kernel_metadata(
    kernel_type: str, spectral_mixture: dict[str, tuple[float, ...]] | None
) -> dict[str, object]:
    """Return JSON-safe metadata sufficient to reconstruct a fixed kernel."""

    if kernel_type != "spectral_mixture":
        return {"family": kernel_type, "spectral_mixture": None}
    if spectral_mixture is None:
        raise ValueError("Spectral-mixture kernel requires a configuration")
    return {
        "family": kernel_type,
        "spectral_mixture": {
            key: list(values) for key, values in spectral_mixture.items()
        },
    }
