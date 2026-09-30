import pytest
from benchmarks.task_stream.training_budget import FitClock
from benchmarks.task_stream.factory import Configuration


class Timer:
    def __init__(self): self.now = 0.; self.syncs = 0
    def monotonic(self): return self.now
    def synchronize(self): self.syncs += 1


def test_wall_clock_covers_optimizer_setup_and_one_step_overshoot(monkeypatch):
    import benchmarks.task_stream.training_budget as module
    events = []; monkeypatch.setattr(module, 'emit', lambda *args: events.append(args))
    timer = Timer(); clock = FitClock(60., timer.synchronize, monotonic=timer.monotonic)
    timer.now = 100.  # earlier feature/ridge work is outside the optimizer scope
    clock.start()
    timer.now += 45.  # optimizer setup + compilation
    timer.now += 10.  # first completed update
    assert not clock.should_stop(1)
    timer.now += 10.
    assert clock.should_stop(2)
    record = clock.finish(2)
    assert record['elapsed_seconds'] == 65.
    assert record['overshoot_seconds'] == 5.
    assert record['stop_reason'] == 'wall_time_budget_reached'
    assert record['completed_steps'] == 2 and record['convergence_claimed'] is False
    assert timer.syncs == 4
    timer.now += 100.  # subsequent posterior initialization is excluded
    assert clock.finish(2) == record
    assert len(events) == 1 and events[0][0] == 'initial_fit_budget'


def test_at_least_one_complete_step_and_explicit_iteration_cap():
    timer = Timer(); clock = FitClock(60, timer.synchronize, monotonic=timer.monotonic)
    clock.start(); timer.now = 95.
    with pytest.raises(ValueError): clock.should_stop(0)
    assert clock.should_stop(1)
    assert clock.finish(1)['overshoot_seconds'] == 35.
    timer = Timer(); capped = FitClock(60, monotonic=timer.monotonic)
    capped.start(); timer.now = 1.
    assert capped.finish(2)['stop_reason'] == 'iteration_limit_reached'


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, '60'])
def test_time_policy_validation(value):
    with pytest.raises(ValueError): FitClock(value)
    with pytest.raises(ValueError): Configuration('ohsvgp', 1000000, .001, initial_max_seconds=value)


def test_time_policy_is_common_and_mutually_exclusive_with_oh_exposures():
    for method in ('kronhippo_svgp', 'osgpr', 'ohsvgp', 'st_svgp', 'mgpvae'):
        config = Configuration(method, 1000000, .001, initial_max_seconds=60.)
        budget = config.resolve_initial_budget(99)
        assert budget['max_seconds'] == 60.
        assert budget['effective_iterations'] == 1000000
        assert budget['iteration_count_role'] == 'safety_cap'
        assert budget['expected_row_exposures'] is None
    with pytest.raises(ValueError, match='mutually exclusive'):
        Configuration('ohsvgp', 1000000, .001, initial_max_seconds=60., initial_expected_passes=2.)
