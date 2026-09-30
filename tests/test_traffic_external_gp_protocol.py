from __future__ import annotations

import numpy as np

from baselines.traffic_protocol_n import TrafficProtocolN


def synthetic_protocol() -> TrafficProtocolN:
    protocol = TrafficProtocolN.__new__(TrafficProtocolN)
    protocol._calibration_y = np.arange(9, dtype=np.float64).reshape(3, 3)
    protocol._stream_y = np.arange(12, dtype=np.float64).reshape(4, 3)
    protocol._calibration_times = np.arange(3, dtype=np.float64)
    protocol._chronological_stream_times = np.arange(3, 7, dtype=np.float64)
    protocol._visible = np.asarray([0, 1], dtype=np.int64)
    protocol._hidden = np.asarray([2], dtype=np.int64)
    protocol._chronological = True
    protocol._delayed_target_steps = 1
    protocol.metadata = {"protocol_id": "pems_bay_protocol_n"}
    return protocol


def test_task1_exposes_only_visible_sensors() -> None:
    task1 = synthetic_protocol().task1()
    assert np.array_equal(task1.locations, np.asarray([0, 1]))
    assert task1.targets.shape == (3, 2)


def test_week_orders_delayed_hidden_before_current_visible() -> None:
    protocol = synthetic_protocol()
    first = protocol.week(0)
    second = protocol.week(1)
    assert first.delayed_hidden is None
    assert np.array_equal(first.current_visible.locations, np.asarray([0, 1]))
    assert second.delayed_hidden is not None
    assert second.delayed_hidden.stream_week == 0
    assert np.array_equal(second.delayed_hidden.locations, np.asarray([2]))
    assert second.hidden_query.stream_week == 1
