"""Read-only access to the audited COVID long-stream Setting B protocol.

External baseline adapters receive only the observations legally available at
an online week.  The current hidden targets remain accessible solely to the
common archive writer after prediction, never through :meth:`week`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


@dataclass(frozen=True)
class KnownObservation:
    """Labels that a baseline is permitted to condition on."""

    kind: str
    stream_week: Optional[int]
    global_week: int
    time: float
    locations: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class HiddenQuery:
    """Current hidden locations, deliberately without their labels."""

    stream_week: int
    global_week: int
    time: float
    locations: np.ndarray


@dataclass(frozen=True)
class WeekInformation:
    """The complete legal information set for one Setting B online week."""

    delayed_hidden: KnownObservation | None
    current_visible: KnownObservation
    hidden_query: HiddenQuery


class COVIDSettingBProtocol:
    """Audited Setting B protocol, including explicitly marked development folds."""

    def __init__(self, npz_path: Path, metadata_path: Optional[Path] = None) -> None:
        self.npz_path = Path(npz_path)
        self.metadata_path = (
            self.npz_path.with_suffix(".json") if metadata_path is None else Path(metadata_path)
        )
        if not self.npz_path.is_file() or not self.metadata_path.is_file():
            raise FileNotFoundError(
                "Setting B requires matching protocol.npz and protocol.json: "
                f"{self.npz_path}, {self.metadata_path}"
            )
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        with np.load(self.npz_path, allow_pickle=False) as archive:
            required = (
                "calibration_y",
                "stream_y",
                "train_indices",
                "fit_indices",
                "validation_indices",
                "test_indices",
                "calibration_times",
                "stream_times",
                "coordinates",
            )
            missing = [name for name in required if name not in archive.files]
            if missing:
                raise ValueError(f"Protocol archive misses required fields: {missing}")
            self._calibration_y = np.asarray(archive["calibration_y"], dtype=np.float64).copy()
            self._stream_y = np.asarray(archive["stream_y"], dtype=np.float64).copy()
            self._visible = np.asarray(archive["train_indices"], dtype=np.int64).copy()
            self._fit = np.asarray(archive["fit_indices"], dtype=np.int64).copy()
            self._validation = np.asarray(archive["validation_indices"], dtype=np.int64).copy()
            self._hidden = np.asarray(archive["test_indices"], dtype=np.int64).copy()
            self._calibration_times = np.asarray(archive["calibration_times"], dtype=np.float64).copy()
            self._stream_times = np.asarray(archive["stream_times"], dtype=np.float64).copy()
            self.coordinates = np.asarray(archive["coordinates"], dtype=np.float64).copy()
            self._spatial_inducing = {
                int(name[len("inducing_coords_ms") :]): np.asarray(
                    archive[name], dtype=np.float64
                ).copy()
                for name in archive.files
                if name.startswith("inducing_coords_ms")
            }
        step = float(np.median(np.diff(self._calibration_times)))
        self._chronological_stream_times = (
            self._calibration_times[-1] + step * np.arange(1, self._stream_times.size + 1)
        )
        self._validate()

    @property
    def calibration_weeks(self) -> int:
        return int(self._calibration_y.shape[0])

    @property
    def online_weeks(self) -> int:
        return int(self._stream_y.shape[0])

    @property
    def locations(self) -> int:
        return int(self._calibration_y.shape[1])

    @property
    def visible_locations(self) -> np.ndarray:
        return self._visible.copy()

    @property
    def hidden_locations(self) -> np.ndarray:
        return self._hidden.copy()

    @property
    def fit_locations(self) -> np.ndarray:
        """Return the 38 Task-1 fitting jurisdictions for capacity selection."""

        return self._fit.copy()

    @property
    def validation_locations(self) -> np.ndarray:
        """Return the four Task-1 validation jurisdictions for capacity selection."""

        return self._validation.copy()

    @property
    def stream_times(self) -> np.ndarray:
        """Return the legacy local online-grid coordinates used by existing archives."""

        return self._stream_times.copy()

    @property
    def calibration_times(self) -> np.ndarray:
        return self._calibration_times.copy()

    def calibration_targets(self, locations: np.ndarray) -> np.ndarray:
        """Return Task-1 labels for scoring a predeclared spatial subset only."""

        locations = np.asarray(locations, dtype=np.int64)
        if locations.ndim != 1 or np.setdiff1d(locations, np.arange(self.locations)).size:
            raise ValueError("Task-1 locations must be a one-dimensional subset of the 52 jurisdictions")
        return self._calibration_y[:, locations].copy()

    def spatial_inducing_locations(self, count: int) -> np.ndarray:
        """Return the audited Route-B spatial inducing grid with the requested size."""

        count = int(count)
        if count not in self._spatial_inducing:
            raise KeyError(
                f"Protocol has no inducing_coords_ms{count}; available={sorted(self._spatial_inducing)}"
            )
        return self._spatial_inducing[count].copy()

    @property
    def chronological_stream_times(self) -> np.ndarray:
        """Return the continuous Task-1-to-online time axis for model fitting."""

        return self._chronological_stream_times.copy()

    def task1(self) -> KnownObservation:
        """Return all Task-1 labels, which initialize every baseline."""

        return KnownObservation(
            kind="task1",
            stream_week=None,
            global_week=0,
            time=float(self._calibration_times[0]),
            locations=np.arange(self.locations, dtype=np.int64),
            targets=self._calibration_y.copy(),
        )

    def week(self, stream_week: int) -> WeekInformation:
        """Return delayed hidden labels, current visible labels, and a blind query."""

        if not 0 <= int(stream_week) < self.online_weeks:
            raise IndexError(f"stream_week must be in [0, {self.online_weeks}), got {stream_week}")
        stream_week = int(stream_week)
        global_week = self.calibration_weeks + stream_week
        delayed = None
        if stream_week > 0:
            source_week = stream_week - 1
            delayed = KnownObservation(
                kind="delayed_hidden",
                stream_week=source_week,
                global_week=self.calibration_weeks + source_week,
                time=float(self._chronological_stream_times[source_week]),
                locations=self.hidden_locations,
                targets=self._stream_y[source_week, self._hidden].copy(),
            )
        visible = KnownObservation(
            kind="current_visible",
            stream_week=stream_week,
            global_week=global_week,
            time=float(self._chronological_stream_times[stream_week]),
            locations=self.visible_locations,
            targets=self._stream_y[stream_week, self._visible].copy(),
        )
        query = HiddenQuery(
            stream_week=stream_week,
            global_week=global_week,
            time=float(self._chronological_stream_times[stream_week]),
            locations=self.hidden_locations,
        )
        return WeekInformation(delayed_hidden=delayed, current_visible=visible, hidden_query=query)

    def evaluation_targets(self) -> np.ndarray:
        """Return hidden labels only for common scoring/archive generation."""

        return self._stream_y[:, self._hidden].copy()

    def make_audit(self) -> "ProtocolAudit":
        return ProtocolAudit(self)

    def _validate(self) -> None:
        if self._calibration_y.ndim != 2 or self._stream_y.ndim != 2:
            raise ValueError("calibration_y and stream_y must have shape [week, location]")
        if self._calibration_y.shape[1] != self._stream_y.shape[1]:
            raise ValueError("Task-1 and online streams use different location counts")
        development = bool(self.metadata.get("development_protocol", False))
        if self.locations != 52:
            raise ValueError("Setting B requires exactly 52 jurisdictions")
        if development:
            if self.calibration_weeks < 1 or self.online_weeks < 1:
                raise ValueError("A development fold requires non-empty history and validation stream")
        elif self.calibration_weeks != 52 or self.online_weeks != 143:
            raise ValueError("The formal protocol is restricted to the audited 52/52/143 horizon")
        if self._visible.size != 42 or self._hidden.size != 10:
            raise ValueError("Setting B requires 42 current visible and 10 current hidden locations")
        if set(self._visible) & set(self._hidden) or set(self._visible) | set(self._hidden) != set(range(52)):
            raise ValueError("Visible/hidden split must be disjoint and cover all 52 locations")
        if self._fit.size != 38 or self._validation.size != 4:
            raise ValueError("Task-1 selection requires 38 fit and 4 validation jurisdictions")
        if set(self._fit) & set(self._validation) or set(self._fit) | set(self._validation) != set(self._visible):
            raise ValueError("Task-1 fit and validation jurisdictions must partition the 42 visible locations")
        if self._calibration_times.shape != (self.calibration_weeks,) or self._stream_times.shape != (
            self.online_weeks,
        ):
            raise ValueError("Protocol time arrays do not match the target arrays")
        if not np.all(np.diff(self._calibration_times) > 0.0):
            raise ValueError("Task-1 times must be strictly increasing")
        if not np.all(np.diff(self._chronological_stream_times) > 0.0):
            raise ValueError("Online chronological times must be strictly increasing")
        if self._chronological_stream_times[0] <= self._calibration_times[-1]:
            raise ValueError("Online chronology must begin after Task 1")
        if self.coordinates.shape != (52, 2):
            raise ValueError("Setting B requires one latitude/longitude pair for each location")
        if int(self.metadata.get("xlag", {}).get("delay_weeks", -1)) != 1:
            raise ValueError("External baseline comparison is frozen to the one-week delayed protocol")
        fit_scope = self.metadata.get("target_standardization", {}).get("fit_scope")
        expected_scope = (
            "development training prefix visible locations only"
            if development
            else "Task-1 visible locations only"
        )
        if fit_scope != expected_scope:
            raise ValueError(f"Protocol normalisation must be fit on {expected_scope}")


class ProtocolAudit:
    """Records one legal update/prediction sequence per online week."""

    def __init__(self, protocol: COVIDSettingBProtocol) -> None:
        self.protocol = protocol
        self._steps: set[int] = set()
        self._delayed_labels = 0
        self._visible_labels = 0
        self._predictions = 0

    def record_step(
        self,
        information: WeekInformation,
        predictive_mean: np.ndarray,
        predictive_variance: np.ndarray,
    ) -> None:
        query = information.hidden_query
        step = query.stream_week
        if step in self._steps:
            raise ValueError(f"Online week {step} was updated or predicted more than once")
        mean = np.asarray(predictive_mean, dtype=np.float64).reshape(-1)
        variance = np.asarray(predictive_variance, dtype=np.float64).reshape(-1)
        if mean.shape != (self.protocol.hidden_locations.size,) or variance.shape != mean.shape:
            raise ValueError("Each Setting B prediction must contain exactly the 10 hidden locations")
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance < 0.0).any():
            raise ValueError("Predictive mean/variance must be finite and have non-negative variance")
        if information.current_visible.stream_week != step:
            raise ValueError("Current visible observation does not match the prediction week")
        if information.delayed_hidden is not None:
            if information.delayed_hidden.stream_week != step - 1:
                raise ValueError("A delayed hidden update must contain exactly the preceding week")
            self._delayed_labels += int(information.delayed_hidden.targets.size)
        elif step != 0:
            raise ValueError("Only the first online week may have no delayed hidden labels")
        self._visible_labels += int(information.current_visible.targets.size)
        self._predictions += int(mean.size)
        self._steps.add(step)

    def summary(self, *, require_complete: bool = True) -> Dict[str, Union[int, bool]]:
        expected_steps = self.protocol.online_weeks
        completed_steps = len(self._steps)
        if self._steps != set(range(completed_steps)):
            raise ValueError("Online trace must contain one contiguous prefix of weeks")
        if require_complete and completed_steps != expected_steps:
            missing = sorted(set(range(expected_steps)) - self._steps)
            raise ValueError(f"Online trace is incomplete; missing weeks: {missing[:10]}")
        return {
            "online_steps_completed": completed_steps,
            "delayed_hidden_labels": self._delayed_labels,
            "expected_delayed_hidden_labels": (expected_steps - 1) * self.protocol.hidden_locations.size,
            "current_visible_labels": self._visible_labels,
            "expected_current_visible_labels": expected_steps * self.protocol.visible_locations.size,
            "current_hidden_labels_read": 0,
            "hidden_predictions": self._predictions,
            "expected_hidden_predictions": expected_steps * self.protocol.hidden_locations.size,
            "passed": (
                completed_steps == expected_steps
                and self._delayed_labels == (expected_steps - 1) * self.protocol.hidden_locations.size
                and self._visible_labels == expected_steps * self.protocol.visible_locations.size
                and self._predictions == expected_steps * self.protocol.hidden_locations.size
            ),
        }
