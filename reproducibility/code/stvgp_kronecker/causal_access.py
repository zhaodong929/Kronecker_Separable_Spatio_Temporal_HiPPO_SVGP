"""Feature-access taxonomy and causal guard for Protocol F.

The guard is deliberately small and framework-independent so future-target and
future-exogenous reads can be tested before wiring them into a runner.
"""
from __future__ import annotations

from enum import Enum


class FeatureAccess(str, Enum):
    PAST_OBSERVED = "PAST_OBSERVED"
    KNOWN_FUTURE = "KNOWN_FUTURE"
    UNKNOWN_FUTURE_EXOGENOUS = "UNKNOWN_FUTURE_EXOGENOUS"
    TARGET = "TARGET"


class ProtocolFAccessGuard:
    """Enforce the strict-causal Protocol F information boundary.

    Past observations and deterministic/calendar features known at forecast time
    are always readable. Unknown future exogenous features require the explicit
    oracle opt-in. Targets are readable only after prediction has completed,
    which models the separate reveal/evaluation step.
    """

    def __init__(self, *, oracle_future_exogenous: bool = False) -> None:
        self.oracle_future_exogenous = bool(oracle_future_exogenous)
        self.prediction_complete = False
        self.read_log: list[FeatureAccess] = []

    def mark_prediction_complete(self) -> None:
        self.prediction_complete = True

    def read(self, access: FeatureAccess | str) -> None:
        kind = FeatureAccess(access)
        if kind is FeatureAccess.TARGET and not self.prediction_complete:
            raise RuntimeError("target read before prediction/reveal")
        if (
            kind is FeatureAccess.UNKNOWN_FUTURE_EXOGENOUS
            and not self.oracle_future_exogenous
        ):
            raise RuntimeError(
                "unknown future exogenous read requires --oracle-future-exogenous"
            )
        self.read_log.append(kind)
