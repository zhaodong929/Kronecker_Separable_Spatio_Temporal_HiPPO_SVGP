"""Comparable synchronized latency and explicitly scoped arithmetic counters."""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import time
import numpy as np


ONLINE_SCOPE = 'release_to_observation_distribution_ready'


class Measurements:
    def __init__(self, synchronize=lambda: None, clock=time.perf_counter):
        self.synchronize, self.clock = synchronize, clock
        self.rows = []
        self.active = False

    @contextmanager
    def phase(self, name, *, task=None, queries=0):
        if name not in {'initial_state', 'initial_refit', 'online', 'scoring', 'serialization', 'selection', 'profiling'}:
            raise ValueError('Unknown measurement phase')
        if self.active:
            raise ValueError('Overlapping phases would double-count wall time')
        self.active = True
        try:
            self.synchronize()
        except BaseException:
            self.active = False
            raise
        start = self.clock()
        complete = False
        try:
            yield
            self.synchronize()
            complete = True
        finally:
            seconds = self.clock() - start
            self.active = False
            self.rows.append(dict(phase=name, task=task, queries=int(queries), seconds=seconds,
                status='complete' if complete else 'failed',
                scope=ONLINE_SCOPE if name == 'online' else name))


@dataclass(frozen=True)
class ArithmeticCount:
    """Not a claim of total model FLOPs: omitted operations remain explicit."""
    status: str
    fp64_add_mul_fma: int | None
    kernel_count: int
    scope: str = ONLINE_SCOPE
    provenance: str = ''
    excluded: tuple = ('CPU arithmetic', 'FP32 and lower precision', 'tensor-core operations',
                       'division, square root and transcendental instructions')

    @classmethod
    def from_kernels(cls, kernels, *, provenance):
        rows = list(kernels)
        if not rows:
            return cls('unavailable', None, 0, provenance=provenance)
        seen, count, incomplete = set(), 0, False
        for row in rows:
            identity = row['id']
            if identity in seen:
                raise ValueError('Duplicate profiler kernel; possible replay double-counting')
            seen.add(identity)
            values = [row.get(key) for key in ('dadd', 'dmul', 'dfma')]
            incomplete = incomplete or any(v is None for v in values)
            if any(v is not None and (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0) for v in values):
                raise ValueError('Instruction counters must be nonnegative integers')
            if all(v is not None for v in values):
                count += int(values[0] + values[1] + 2 * values[2])
        if incomplete:
            return cls('incomplete', None, len(rows), provenance=provenance)
        return cls('measured_subset', count, len(rows), provenance=provenance)

    def as_dict(self):
        return asdict(self)


def timing_summary(rows):
    online = [r for r in rows if r['phase'] == 'online']
    if not online or any(r['status'] != 'complete' for r in online):
        raise ValueError('A complete online trace is required')
    if any(r['scope'] != ONLINE_SCOPE or r['queries'] <= 0 for r in online):
        raise ValueError('Incompatible timing scope or empty query task')
    if len({r['task'] for r in online}) != len(online):
        raise ValueError('Duplicate task timing; possible replay double-counting')
    seconds = np.asarray([r['seconds'] for r in online])
    if not np.isfinite(seconds).all() or np.any(seconds < 0):
        raise ValueError('Invalid elapsed time')
    return dict(online_seconds=float(seconds.sum()), tasks=len(online),
        queries=sum(r['queries'] for r in online),
        seconds_per_query=float(seconds.sum() / sum(r['queries'] for r in online)),
        task_seconds_median=float(np.median(seconds)), task_seconds_p95=float(np.quantile(seconds, .95)),
        phase_seconds={name: float(sum(r['seconds'] for r in rows if r['phase'] == name))
                       for name in sorted({r['phase'] for r in rows})},
        compilation_policy='first_use_included; warmed profiling is a separate execution')


def profile_record(kernels, *, provenance, profiled_task_ids, expected_task_ids,
                   replay_policy, profiler_seconds, timing_run_id, profile_run_id):
    """Attach counter coverage without treating profiler replay as benchmark latency.

    Counters must describe executed FP64 scalar instructions (FMA counts twice),
    not an uncalibrated hardware metric such as warp instructions or utilization.
    Kernel ids must identify original launches, with replay already reconciled.
    """
    if timing_run_id == profile_run_id or not timing_run_id or not profile_run_id:
        raise ValueError('Profiling must be a separately identified execution')
    if replay_policy not in {'no_replay', 'original_launch_ids_deduplicated'}:
        raise ValueError('An explicit replay accounting policy is required')
    observed, expected = list(profiled_task_ids), list(expected_task_ids)
    if (not expected or len(set(expected)) != len(expected) or
            len(set(observed)) != len(observed) or not set(observed).issubset(expected)):
        raise ValueError('Invalid task coverage')
    if not np.isfinite(profiler_seconds) or profiler_seconds < 0:
        raise ValueError('Invalid profiling duration')
    count = ArithmeticCount.from_kernels(kernels, provenance=provenance).as_dict()
    count.update(profiled_task_ids=observed, expected_task_ids=expected,
        task_coverage=len(observed) / len(expected), replay_policy=replay_policy,
        counter_unit='executed scalar FP64 instructions; FMA weighted by 2',
        profiler_seconds=float(profiler_seconds), timing_run_id=timing_run_id,
        profile_run_id=profile_run_id, extrapolated_to_total=False)
    return count
