"""Read-only PEMS-BAY Protocol-N access for external GP baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from baselines.covid_long_setting_b.protocol import (
    COVIDSettingBProtocol,
    KnownObservation,
    ProtocolAudit,
)


class TrafficProtocolN(COVIDSettingBProtocol):
    """Traffic counterpart of the audited delayed-hidden spatial protocol."""

    protocol_id = "pems_bay_protocol_n"

    def task1(self) -> KnownObservation:
        """Task 1 exposes only the 260 sensors visible in the formal split."""

        locations = self.visible_locations
        return KnownObservation(
            kind="task1",
            stream_week=None,
            global_week=0,
            time=float(self._calibration_times[0]),
            locations=locations,
            targets=self._calibration_y[:, locations].copy(),
        )

    def spatial_inducing_locations(self, count: int) -> np.ndarray:
        """Preserve stored grids; derive missing sizes from input geometry only."""
        count = int(count)
        if count in self._spatial_inducing:
            return self._spatial_inducing[count].copy()
        if not 1 <= count <= self.visible_locations.size:
            raise ValueError('Spatial inducing count exceeds initial visible sites')
        from stvgp_kronecker.joint_ssgp_kron.synthetic import select_spatial_inducing_indices
        coordinates = self.coordinates[self.visible_locations]
        indices = select_spatial_inducing_indices(coordinates,count,method='farthest')
        return coordinates[indices].copy()

    def make_audit(self) -> ProtocolAudit:
        return ProtocolAudit(self)

    def _validate(self) -> None:
        if self.metadata.get("protocol_id") != self.protocol_id:
            raise ValueError("Traffic archive is not the locked PEMS-BAY Protocol N")
        if self._calibration_y.ndim != 2 or self._stream_y.ndim != 2:
            raise ValueError("calibration_y and stream_y must have shape [time, sensor]")
        if self._calibration_y.shape[1] != self._stream_y.shape[1]:
            raise ValueError("Task 1 and stream sensor counts differ")
        if self.locations != 325 or self.calibration_weeks != 2016 or self.online_weeks != 50100:
            raise ValueError("Formal PEMS-BAY Protocol N requires 325 sensors and a 2016/50100 split")
        if self._visible.size != 260 or self._hidden.size != 65:
            raise ValueError("Formal PEMS-BAY Protocol N requires 260 visible and 65 held-out sensors")
        if self._fit.size != 234 or self._validation.size != 26:
            raise ValueError("Task-1 visible selection requires 234 fit and 26 validation sensors")
        all_locations = set(range(self.locations))
        if set(self._visible) & set(self._hidden) or set(self._visible) | set(self._hidden) != all_locations:
            raise ValueError("Visible and held-out sensors must form a partition")
        if set(self._fit) & set(self._validation) or set(self._fit) | set(self._validation) != set(self._visible):
            raise ValueError("Task-1 fit and validation sensors must partition visible sensors")
        if self.coordinates.shape != (325, 2):
            raise ValueError("PEMS-BAY requires one latitude/longitude pair per sensor")
        if self._calibration_times.shape != (self.calibration_weeks,):
            raise ValueError("Task-1 times do not match Task-1 targets")
        if self._stream_times.shape != (self.online_weeks,):
            raise ValueError("Stream times do not match stream targets")
        if not np.all(np.diff(self._calibration_times) > 0.0):
            raise ValueError("Task-1 times must be strictly increasing")
        if not np.all(np.diff(self._chronological_stream_times) > 0.0):
            raise ValueError("Stream times must be strictly increasing")
        if int(self.metadata.get("delayed_target_steps", -1)) != 1:
            raise ValueError("Protocol N requires exactly one-step delayed held-out labels")
        if int(self.metadata.get("split_seed", -1)) not in (1, 2, 3):
            raise ValueError("This formal archive is restricted to split seeds 1, 2 and 3")


def load_protocol(
    npz_path: Path,
    metadata_path: Path | None,
    *,
    protocol_kind: str,
) -> COVIDSettingBProtocol:
    if protocol_kind == "traffic":
        return TrafficProtocolN(npz_path, metadata_path)
    if protocol_kind == "covid":
        return COVIDSettingBProtocol(npz_path, metadata_path)
    raise ValueError(f"Unknown protocol kind: {protocol_kind}")
