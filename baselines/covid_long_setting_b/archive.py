"""Common prediction archive writer for Setting B external baseline adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .protocol import COVIDSettingBProtocol, ProtocolAudit, WeekInformation


class PredictionArchive:
    """Accumulate validated predictions and write the common evaluator schema."""

    def __init__(self, protocol: COVIDSettingBProtocol, *, method: str, seed: int) -> None:
        self.protocol = protocol
        self.method = str(method)
        self.seed = int(seed)
        self.audit: ProtocolAudit = protocol.make_audit()
        shape = (protocol.online_weeks, protocol.hidden_locations.size)
        self._mean = np.full(shape, np.nan, dtype=np.float64)
        self._variance = np.full(shape, np.nan, dtype=np.float64)

    def append(
        self,
        information: WeekInformation,
        predictive_mean: np.ndarray,
        predictive_variance: np.ndarray,
    ) -> None:
        self.audit.record_step(information, predictive_mean, predictive_variance)
        week = information.hidden_query.stream_week
        self._mean[week] = np.asarray(predictive_mean, dtype=np.float64).reshape(-1)
        self._variance[week] = np.asarray(predictive_variance, dtype=np.float64).reshape(-1)

    def write(
        self,
        path: Path,
        *,
        extra_metadata: Optional[Dict[str, object]] = None,
        require_complete: bool = True,
    ) -> Dict[str, Union[int, bool]]:
        audit = self.audit.summary(require_complete=require_complete)
        completed = int(audit["online_steps_completed"])
        metadata = {"method": self.method, "seed": self.seed, "protocol": "covid_long_setting_b"}
        if extra_metadata:
            metadata.update(extra_metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            y_true=self.protocol.evaluation_targets()[:completed],
            pred_mean=self._mean[:completed],
            pred_var=self._variance[:completed],
            test_indices=self.protocol.hidden_locations,
            times=self.protocol.stream_times[:completed],
            metadata_json=np.asarray(
                json.dumps(
                    {
                        **metadata,
                        "fitting_time_convention": "continuous_task1_to_online",
                        "archive_time_convention": "legacy_local_online_grid",
                    },
                    sort_keys=True,
                )
            ),
        )
        return audit
