"""Information-boundary guards for paired spatial traffic experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from stvgp_kronecker.causal_access import FeatureAccess, ProtocolFAccessGuard


@dataclass
class StrictOnlineNowcastingGuard:
    """Audit the ``delayed hidden -> current visible -> current hidden`` order."""

    active_time: int | None = None
    prediction_complete: bool = False
    absorbed_times: set[int] = field(default_factory=set)
    current_visible_reads: int = 0
    current_hidden_reads_before_prediction: int = 0
    current_hidden_reveals: int = 0

    def begin(self, time_index: int) -> None:
        if self.active_time is not None:
            raise RuntimeError("A strict-online step is already active")
        self.active_time = int(time_index)
        self.prediction_complete = False

    def absorb_delayed_hidden(self, time_index: int) -> None:
        if self.active_time is None:
            raise RuntimeError("Cannot absorb a delayed label outside an online step")
        if time_index >= self.active_time:
            raise RuntimeError("Only a strictly earlier held-out label may be absorbed")
        if time_index in self.absorbed_times:
            raise RuntimeError(f"Held-out label at t={time_index} was absorbed more than once")
        self.absorbed_times.add(int(time_index))

    def read_current_visible(self, time_index: int) -> None:
        if self.active_time != int(time_index):
            raise RuntimeError("Current visible readings may only be read for the active time")
        self.current_visible_reads += 1

    def mark_prediction_complete(self) -> None:
        if self.active_time is None:
            raise RuntimeError("No active strict-online step to complete")
        self.prediction_complete = True

    def reveal_current_hidden(self, time_index: int) -> None:
        if self.active_time != int(time_index):
            raise RuntimeError("Current held-out labels may only be revealed for the active time")
        if not self.prediction_complete:
            self.current_hidden_reads_before_prediction += 1
            raise RuntimeError("Current held-out labels cannot be read before prediction")
        self.current_hidden_reveals += 1
        self.active_time = None

    def report(self) -> dict[str, int]:
        return {
            "current_visible_reads": self.current_visible_reads,
            "current_hidden_reads_before_prediction": self.current_hidden_reads_before_prediction,
            "current_hidden_reveals": self.current_hidden_reveals,
            "unique_delayed_hidden_absorptions": len(self.absorbed_times),
        }


class TrafficForecastGuard(ProtocolFAccessGuard):
    """Protocol-F guard with a compact report for the traffic forecast runner."""

    def read_current_visible(self) -> None:
        self.read(FeatureAccess.PAST_OBSERVED)

    def read_known_future_calendar(self) -> None:
        self.read(FeatureAccess.KNOWN_FUTURE)

    def reveal_future_target(self) -> None:
        self.read(FeatureAccess.TARGET)

    def report(self) -> dict[str, int | bool]:
        return {
            "oracle_future_exogenous": self.oracle_future_exogenous,
            "past_observed_reads": sum(item is FeatureAccess.PAST_OBSERVED for item in self.read_log),
            "known_future_reads": sum(item is FeatureAccess.KNOWN_FUTURE for item in self.read_log),
            "unknown_future_exogenous_reads": sum(
                item is FeatureAccess.UNKNOWN_FUTURE_EXOGENOUS for item in self.read_log
            ),
            "target_reads": sum(item is FeatureAccess.TARGET for item in self.read_log),
        }
