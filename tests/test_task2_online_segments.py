from __future__ import annotations

import numpy as np
import pytest

from scripts.summarize_task2_online_segments import metrics, segment_slices


def test_task2_ten_segments_have_requested_lengths() -> None:
    slices = segment_slices(186, 10)
    assert [stop - start for start, stop in slices] == [19] * 6 + [18] * 4
    assert slices[0][0] == 0
    assert slices[-1][1] == 186


def test_segments_reject_invalid_counts() -> None:
    with pytest.raises(ValueError):
        segment_slices(0, 10)
    with pytest.raises(ValueError):
        segment_slices(10, 11)


def test_coverage_is_nominal_interval_coverage_not_ece() -> None:
    y = np.zeros((2, 1))
    mean = np.zeros((2, 1))
    variance = np.ones((2, 1))
    result = metrics(y, mean, variance)
    assert result["coverage50"] == 1.0
    assert result["coverage90"] == 1.0
    assert result["coverage95"] == 1.0
    assert result["mean_interval_width90"] > 0.0


def test_metrics_reject_nonpositive_variance() -> None:
    with pytest.raises(FloatingPointError):
        metrics(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
