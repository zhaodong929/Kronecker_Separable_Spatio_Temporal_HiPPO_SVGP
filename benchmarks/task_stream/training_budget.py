"""Common synchronized wall-time cap for initial optimizer work only.

A clock begins immediately before optimizer construction, including its setup,
compilation and completed updates. Ridge fitting, feature preparation, posterior
initialization after fitting and online inference are outside this scope. At
least one completed update is allowed; the last update can overshoot the cap.
This is a compute-budget stopping rule and makes no convergence claim.
"""
import math
import numbers
import time
from benchmarks.three_domain.tracking import emit


class FitClock:
    def __init__(self, max_seconds, synchronize=lambda: None, *, monotonic=None):
        if (isinstance(max_seconds, bool) or not isinstance(max_seconds, numbers.Real)
                or not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError('Initial optimizer time budget must be finite and positive')
        self.max_seconds = float(max_seconds)
        self.synchronize = synchronize
        self._now = monotonic or time.monotonic
        self._start = None
        self._last_steps = 0
        self.record = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Initial fit clock may only start once')
        # Drain earlier model preparation before starting the optimizer scope.
        self.synchronize()
        self._start = self._now()

    def _elapsed(self):
        if self._start is None:
            raise RuntimeError('Initial fit clock has not started')
        self.synchronize()
        elapsed = self._now()-self._start
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError('Invalid monotonic fit duration')
        return elapsed

    def _steps(self, completed_steps):
        if (isinstance(completed_steps, bool) or not isinstance(completed_steps, numbers.Integral)
                or completed_steps < 1 or completed_steps < self._last_steps):
            raise ValueError('Completed fit steps must be positive and nondecreasing')
        return int(completed_steps)

    def should_stop(self, completed_steps):
        if self.record is not None:
            raise RuntimeError('Initial fit clock has already finished')
        steps = self._steps(completed_steps)
        elapsed = self._elapsed()
        self._last_steps = steps
        return elapsed >= self.max_seconds

    def finish(self, completed_steps):
        steps = self._steps(completed_steps)
        if self.record is not None:
            if steps != self.record['completed_steps']:
                raise ValueError('Finished fit step count cannot change')
            return dict(self.record)
        elapsed = self._elapsed()
        self._last_steps = steps
        self.record = dict(scope='initial_optimizer_setup_compilation_and_completed_updates',
            policy='synchronized_wall_time_v1', max_seconds=self.max_seconds,
            elapsed_seconds=elapsed, completed_steps=steps,
            stop_reason='wall_time_budget_reached' if elapsed >= self.max_seconds else 'iteration_limit_reached',
            overshoot_seconds=max(0., elapsed-self.max_seconds), minimum_completed_steps=1,
            convergence_claimed=False)
        emit('initial_fit_budget', steps, self.record)
        return dict(self.record)
