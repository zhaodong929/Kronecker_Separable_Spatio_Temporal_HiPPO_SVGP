"""Online baselines for ERA5 continual-learning comparisons."""

from .online_baselines import (
    ClimatologyBaseline,
    GPyTorchSGPRBaseline,
    GPyTorchSVGPBaseline,
    IndependentTemporalGPBaseline,
    OnlineBaseline,
    PersistenceBaseline,
    PredictionResult,
    RidgeBaseline,
    make_baseline,
)

__all__ = [
    "ClimatologyBaseline",
    "GPyTorchSGPRBaseline",
    "GPyTorchSVGPBaseline",
    "IndependentTemporalGPBaseline",
    "OnlineBaseline",
    "PersistenceBaseline",
    "PredictionResult",
    "RidgeBaseline",
    "make_baseline",
]
