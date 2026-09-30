"""One information contract; observation-wise inference remains method specific.

Adapters receive released labels and query coordinates only. Evaluation targets
are retrieved by the evaluator after prediction. Times are the actual canonical
model coordinates, never reconstructed independently by an adapter.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def readonly(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def indices(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iu' or raw.ndim != 1:
        raise ValueError('Site indices must be a one-dimensional integer array')
    return readonly(raw, int)


def positive_integer(value, name):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in 'iu' or int(raw) < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(raw)


@dataclass(frozen=True)
class Observations:
    times: np.ndarray
    sites: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        times, sites, values = (readonly(self.times), indices(self.sites), readonly(self.values))
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or sites.ndim != 1 or not len(sites)
                or len(np.unique(sites)) != len(sites) or np.any(sites < 0)
                or values.shape != (len(times), len(sites)) or not np.isfinite(values).all()):
            raise ValueError("Invalid observation grid")
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'sites', sites)
        object.__setattr__(self, 'values', values)


@dataclass(frozen=True)
class Task:
    index: int
    start: int
    stop: int
    visible: Observations
    delayed: Observations | None
    query_sites: np.ndarray

    @property
    def times(self):
        return self.visible.times

    @property
    def query_shape(self):
        return len(self.times), len(self.query_sites)


class TaskStream:
    def __init__(self, *, times, targets, coordinates, visible, hidden, initial_sites,
                 initial_steps, task_steps, release_previous, metadata=None):
        self.times = readonly(times)
        self._targets = readonly(targets)
        self.coordinates = readonly(coordinates)
        self.visible = indices(visible)
        self.hidden = indices(hidden)
        self.initial_sites = indices(initial_sites)
        self.initial_steps = positive_integer(initial_steps, 'initial_steps')
        self.task_steps = positive_integer(task_steps, 'task_steps')
        self.release_previous = bool(release_previous)
        self.metadata = dict(metadata or {})
        n = len(self.coordinates)
        if (self.times.ndim != 1 or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0) or self._targets.shape != (len(self.times), n)
                or not np.isfinite(self._targets).all() or self.coordinates.shape != (n, 2)
                or not np.isfinite(self.coordinates).all()
                or not 1 <= self.initial_steps < len(self.times) or self.task_steps < 1):
            raise ValueError('Invalid stream dimensions, times, or values')
        for sites in [self.visible, self.hidden, self.initial_sites]:
            if (sites.ndim != 1 or not len(sites) or len(np.unique(sites)) != len(sites)
                    or np.any(sites < 0) or np.any(sites >= n)):
                raise ValueError('Invalid sites')
        if (np.intersect1d(self.visible, self.hidden).size
                or len(self.visible) + len(self.hidden) != n
                or not set(self.visible) <= set(self.initial_sites)):
            raise ValueError('Invalid spatial partition or initialization')
        if not self.release_previous and set(self.initial_sites) != set(self.visible):
            raise ValueError('Permanent holdout cannot be observed during initialization')
        self.bounds = tuple((start, min(start + self.task_steps, len(self.times)))
                            for start in range(self.initial_steps, len(self.times), self.task_steps))

    def initial(self):
        return Observations(self.times[:self.initial_steps], self.initial_sites,
                            self._targets[:self.initial_steps, self.initial_sites])

    def task(self, index):
        if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)) or not 0 <= index < len(self.bounds):
            raise IndexError(index)
        start, stop = self.bounds[index]
        delayed = None
        if index and self.release_previous:
            previous_start, previous_stop = self.bounds[index - 1]
            delayed = Observations(self.times[previous_start:previous_stop], self.hidden,
                                   self._targets[previous_start:previous_stop, self.hidden])
        return Task(index, start, stop,
                    Observations(self.times[start:stop], self.visible,
                                 self._targets[start:stop, self.visible]), delayed, self.hidden)

    def truth(self, task):
        if self.bounds[task.index] != (task.start, task.stop):
            raise ValueError('Task does not belong to this stream')
        return self._targets[task.start:task.stop, self.hidden].copy()

    def identity(self):
        h = hashlib.sha256()
        for value in [self.times, self._targets, self.coordinates, self.visible,
                      self.hidden, self.initial_sites]:
            h.update(str((value.dtype.str, value.shape)).encode())
            h.update(value.tobytes())
        h.update(json.dumps(dict(initial_steps=self.initial_steps, task_steps=self.task_steps,
                                 release_previous=self.release_previous, metadata=self.metadata),
                            sort_keys=True, allow_nan=False).encode())
        return h.hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, times=self.times, targets=self._targets, coordinates=self.coordinates,
                            visible=self.visible, hidden=self.hidden, initial_sites=self.initial_sites,
                            initial_steps=self.initial_steps, task_steps=self.task_steps,
                            release_previous=self.release_previous,
                            metadata_json=json.dumps(self.metadata, sort_keys=True, allow_nan=False))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            args = {key: data[key].copy() for key in ['times', 'targets', 'coordinates', 'visible',
                    'hidden', 'initial_sites', 'initial_steps', 'task_steps', 'release_previous']}
            args['metadata'] = json.loads(str(data['metadata_json']))
        return cls(**args)
