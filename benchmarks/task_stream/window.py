"""Bounded task replay around a method's unchanged observation-wise update.

No smoothing approximation is prescribed here. The method supplies its own
update and task-end prediction. Only the preceding task can be released late.
"""
from dataclasses import dataclass
import numpy as np
from .protocol import Observations


@dataclass(frozen=True)
class FilterTrace:
    times: np.ndarray
    means: tuple
    covariances: tuple
    query_start: int


class TaskWindow:
    def __init__(self, step, mean, covariance):
        self.step = step
        self.mean, self.covariance = mean, covariance
        self.time = None
        self.previous = None
        self.before_previous = None
        self.next_task = 0

    def _filter(self, batch):
        means, covariances = [], []
        for time, values in zip(batch.times, batch.values):
            if self.time is not None and time <= self.time:
                raise ValueError('Observations must advance model time')
            dt = 0. if self.time is None else float(time - self.time)
            self.mean, self.covariance = self.step(self.mean, self.covariance, dt, batch.sites, values)
            self.time = float(time)
            means.append(self.mean)
            covariances.append(self.covariance)
        return means, covariances

    def initialize(self, initial):
        if self.time is not None:
            raise ValueError('Initialization is allowed once')
        self._filter(initial)

    def advance(self, task):
        if task.index != self.next_task or self.time is None:
            raise ValueError('Tasks must follow initialization in order')
        if task.times[0] <= self.time:
            raise ValueError('Current task must follow the previous task')
        times, means, covariances = [], [], []
        if task.delayed is not None:
            if self.previous is None or not np.array_equal(task.delayed.times, self.previous.times):
                raise ValueError('Only the immediately preceding task can be released')
            if np.intersect1d(task.delayed.sites, self.previous.sites).size:
                raise ValueError('Duplicate delayed labels')
            # Recompute each old time slice once with its full released site set.
            combined = Observations(self.previous.times,
                np.concatenate([self.previous.sites, task.delayed.sites]),
                np.concatenate([self.previous.values, task.delayed.values], axis=1))
            self.mean, self.covariance, self.time = self.before_previous
            times.append(self.time); means.append(self.mean); covariances.append(self.covariance)
            m, p = self._filter(combined)
            times.extend(combined.times); means.extend(m); covariances.extend(p)
        else:
            times.append(self.time); means.append(self.mean); covariances.append(self.covariance)
        self.before_previous = self.mean, self.covariance, self.time
        start = len(times)
        m, p = self._filter(task.visible)
        times.extend(task.times); means.extend(m); covariances.extend(p)
        self.previous = task.visible
        self.next_task += 1
        return FilterTrace(np.asarray(times), tuple(means), tuple(covariances), start)
