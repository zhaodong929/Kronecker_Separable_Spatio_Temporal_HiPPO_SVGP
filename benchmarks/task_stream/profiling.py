"""Optional explicit NVTX scope and strict Nsight scalar FP64 counter ingestion."""
from contextlib import contextmanager
import csv
import ctypes
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import time

METRICS = {name: f'smsp__sass_thread_inst_executed_op_{name}_pred_on.sum'
           for name in ('dadd', 'dmul', 'dfma')}
RANGE_NAME = 'hippo_task_online'


def selected_tasks(text, total):
    if text is None:
        return ()
    pieces = text.split(',')
    if not pieces or any(not part.strip().isdigit() for part in pieces):
        raise ValueError('HIPPO_PROFILE_TASKS requires an explicit comma-separated list of task indices')
    tasks = tuple(int(part) for part in pieces)
    if len(set(tasks)) != len(tasks) or any(task >= total for task in tasks):
        raise ValueError('Profile task indices must be unique and in range')
    return tasks


def nvtx_library():
    for name in ('libnvToolsExt.so.1', 'libnvToolsExt.so'):
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        library.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        library.nvtxRangePushA.restype = ctypes.c_int
        library.nvtxRangePop.argtypes = []
        library.nvtxRangePop.restype = ctypes.c_int
        return library
    raise RuntimeError('Profiling requested but the NVTX shared library is unavailable')


class TaskProfiler:
    def __init__(self, total_tasks):
        self.tasks = selected_tasks(os.environ.get('HIPPO_PROFILE_TASKS'), total_tasks)
        self.enabled = bool(self.tasks)
        self.library = nvtx_library() if self.enabled else None
        self.completed = set()
        self.journal = os.environ.get('HIPPO_PROFILE_JOURNAL')
        if self.enabled and not self.journal:
            raise ValueError('Profiling requires HIPPO_PROFILE_JOURNAL for range coverage evidence')

    @contextmanager
    def range(self, index, synchronize=lambda: None):
        if index not in self.tasks:
            yield
            return
        if index in self.completed:
            raise ValueError('A profiled task may not execute twice in one process')
        synchronize()
        if self.library.nvtxRangePushA(RANGE_NAME.encode()) < 0:
            raise RuntimeError('NVTX range push failed')
        complete = False
        try:
            yield
            synchronize()
            complete = True
        finally:
            popped = self.library.nvtxRangePop()
            complete = complete and popped >= 0
            row = dict(task=index, status='complete' if complete else 'failed', range=RANGE_NAME,
                       time_unix=time.time(), pid=os.getpid(), ordinary_latency_valid=False)
            fd = os.open(self.journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                payload = (json.dumps(row)+'\n').encode()
                if os.write(fd, payload) != len(payload):
                    raise OSError('Incomplete profiling journal write')
                os.fsync(fd)
            finally:
                os.close(fd)
            if complete:
                self.completed.add(index)
            if popped < 0:
                raise RuntimeError('NVTX range pop failed')


def parse_ncu_csv(path):
    """Long-form raw CSV, original launch IDs, separate rows for distinct metrics.

    NCU kernel replay is already reconciled in its exported metric value. A
    second row for the same launch+metric is ambiguous and is rejected, never
    silently summed or deduplicated. Warp instructions are not accepted.
    """
    inverse = {value: key for key, value in METRICS.items()}
    launches = {}
    header = None
    with Path(path).open(newline='') as handle:
        for cells in csv.reader(handle):
            if not cells or cells[0].startswith('==PROF=='):
                continue
            if {'ID', 'Metric Name', 'Metric Unit', 'Metric Value'}.issubset(cells):
                if header is not None:
                    raise ValueError('Repeated CSV header; concatenated profiler executions are not supported')
                header = cells
                continue
            if header is None:
                continue
            if len(cells) != len(header):
                raise ValueError('Malformed Nsight metric row')
            row = dict(zip(header, cells))
            metric = inverse.get(row['Metric Name'])
            if metric is None:
                continue
            if not row['ID'].strip():
                raise ValueError('Missing original kernel launch ID')
            identity = '/'.join(row.get(key, '') for key in ('Process ID', 'Context', 'Stream', 'ID'))
            launch = launches.setdefault(identity, dict(id=identity, kernel=row.get('Kernel Name', '')))
            if metric in launch:
                raise ValueError('Duplicate original kernel metric; possible replay or concatenated export')
            value = row['Metric Value'].strip().replace(',', '')
            if value.lower() in {'n/a', 'nan', '', 'none'}:
                launch[metric] = None
                continue
            unit = row['Metric Unit'].strip()
            # Scaled display values can be rounded: require raw base units.
            if unit != 'inst':
                raise ValueError(f'Unsupported counter unit: {unit}')
            try:
                count = Decimal(value)
            except InvalidOperation as error:
                raise ValueError('Invalid numeric instruction counter') from error
            if not count.is_finite() or count < 0 or count != count.to_integral_value():
                raise ValueError('Instruction counters must be exact nonnegative integers')
            launch[metric] = int(count)
    if header is None:
        raise ValueError('Expected Nsight long-form raw CSV header with launch ID and units')
    return list(launches.values())


def ncu_command(command, csv_path):
    if not command:
        raise ValueError('A separate profiling worker command is required')
    return ['ncu', '--target-processes', 'all', '--nvtx', '--nvtx-include', RANGE_NAME+'/',
            '--replay-mode', 'kernel', '--metrics', ','.join(METRICS.values()),
            '--csv', '--page', 'raw', '--print-units', 'base', '--print-metric-name', 'name',
            '--log-file', str(csv_path), *command]
