"""Shared event loop with durable artifacts; adapters never receive evaluation targets.

The existing external W&B supervisor consumes HIPPO_EVENT_PATH and recursively
uploads the output directory. No network access occurs here. Checkpoints are
inspection/recovery material, never implicitly loaded or resumed.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
import time
import traceback
import numpy as np
from benchmarks.three_domain.evaluation import gaussian_scores, LEVELS
from benchmarks.three_domain.metrics import gaussian_mixture_metrics, gaussian_mixture_calibration
from benchmarks.three_domain.tracking import emit
from .measurement import Measurements, timing_summary
from .profiling import TaskProfiler


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unsupported record value: {type(value).__name__}')


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(path, value):
    data = json.dumps(value, indent=2, allow_nan=False, default=_json_value).encode()
    _atomic(path, lambda handle: handle.write(data))


def _npz(path, values):
    arrays = {str(key): np.asarray(value) for key, value in values.items()}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise TypeError('Object arrays are prohibited in experiment artifacts')
    _atomic(path, lambda handle: np.savez_compressed(handle, **arrays))


def _resources():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux ru_maxrss is KiB; it is a process lifetime high-water mark, not GPU memory.
    return dict(process_peak_rss_bytes=int(usage.ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)),
                process_user_seconds=usage.ru_utime, process_system_seconds=usage.ru_stime,
                memory_scope='process_lifetime_high_water; GPU memory unavailable unless adapter reports it')


def _runtime():
    versions = {}
    for name in ('numpy', 'scipy', 'torch', 'jax', 'jaxlib', 'tensorflow', 'gpflow', 'bayesnewton', 'wandb'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), platform=platform.platform(),
                hostname=platform.node(), packages=versions,
                thread_environment={key: os.environ[key] for key in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')
                    if key in os.environ})


def _checkpoint(adapter, directory, index):
    callback = getattr(adapter, 'checkpoint_state', None)
    if callback is None:
        return dict(status='unavailable', reason='adapter has no explicit serializable checkpoint interface')
    state = callback()
    if not isinstance(state, dict):
        raise TypeError('checkpoint_state must return a dictionary')
    arrays, metadata = {}, {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise TypeError('Checkpoint keys must be strings')
        if isinstance(value, np.ndarray) or hasattr(value, '__array__'):
            arrays[key] = np.asarray(value)
        else:
            metadata[key] = value
    _npz(directory / 'checkpoint.npz', arrays)
    _json(directory / 'checkpoint.json', dict(task=index, state=metadata, array_keys=list(arrays),
        adapter=f'{type(adapter).__module__}.{type(adapter).__qualname__}', auto_resume=False))
    return dict(status='saved', auto_resume=False)


def native_target_transform(metadata):
    """The inverse affine target transform; never undo COVID's log1p transform."""
    transform = metadata.get('target_standardization')
    if transform is None:
        return 0., 1., 'standardized (native target transform unavailable)'
    center, scale = float(transform['mean']), float(transform['scale'])
    if not np.isfinite([center, scale]).all() or scale <= 0:
        raise ValueError('Target standardization requires finite center and positive scalar scale')
    units = metadata.get('target_unit') or {
        'era5': 'kelvin', 'covid': 'log1p admissions per 100000 population',
        'pems': 'miles per hour'}.get(metadata.get('dataset'), 'pre-standardization target units')
    return center, scale, units


def native_scores(scores, scale):
    """Exact affine score conversion for Gaussian and mixture distributions."""
    result = dict(scores)
    for key in ('rmse', 'crps'):
        result[key] = float(scores[key] * scale)
    result['nlpd'] = float(scores['nlpd'] + np.log(scale))
    return result


def run(stream, adapter, *, output=None, synchronize=lambda: None, score=None,
        provenance=None, configuration=None, checkpoints=True):
    output = None if output is None else Path(output)
    previous_journal = os.environ.get('HIPPO_EVENT_PATH')
    measurement = Measurements(synchronize)
    means, variances, truths, times, scores = [], [], [], [], []
    completed_tasks = 0
    index = None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'result.json').exists() or (output / 'configuration.json').exists():
            raise FileExistsError('Use a fresh output directory for each attempt; implicit resume is prohibited')
        if previous_journal is None:
            os.environ['HIPPO_EVENT_PATH'] = str(output / 'events.jsonl')
    def event(phase, step, data):
        emit(phase, step, data)
        journal = os.environ.get('HIPPO_EVENT_PATH')
        if journal:
            fd = os.open(journal, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    try:
        protocol_identity = stream.identity()
        profiler = TaskProfiler(len(stream.bounds))
        target_center, target_scale, native_units = native_target_transform(stream.metadata)
        transform_record = dict(mean=target_center, scale=target_scale, native_units=native_units)
        config = dict(protocol_sha256=protocol_identity, tasks=len(stream.bounds),
            initial_steps=stream.initial_steps, task_steps=stream.task_steps,
            release_previous=stream.release_previous, metadata=stream.metadata,
            target_standardization=transform_record,
            adapter=f'{type(adapter).__module__}.{type(adapter).__qualname__}',
            predictive_family=getattr(adapter, 'predictive_family', 'gaussian'),
            ordinary_latency_valid=not profiler.enabled, profiled_task_ids=list(profiler.tasks),
            runtime=_runtime(), provenance=provenance or {}, configuration=configuration or {},
            checkpoint_policy='explicit adapter serialization; no implicit resume',
            evaluation_policy='task-end hidden-site observation distribution; scoring excluded from online timer')
        if output is not None:
            _json(output / 'configuration.json', config)
        event('configuration', 0, config)
        with measurement.phase(getattr(adapter, 'initialization_phase', 'initial_state')):
            adapter.initialize(stream.initial())
        fit_budget = getattr(adapter, 'fit_budget_record', None)
        if output is not None and fit_budget is not None:
            _json(output / 'fit-budget.json', fit_budget)
        if output is not None and checkpoints:
            with measurement.phase('serialization'):
                _json(output / 'initial-checkpoint-status.json', _checkpoint(adapter, output / 'initial', -1))
        for index in range(len(stream.bounds)):
            start, stop = stream.bounds[index]
            with measurement.phase('online', task=index, queries=(stop-start)*len(stream.hidden)), profiler.range(index, synchronize):
                task = stream.task(index)
                mean, variance = adapter.predict_task(task)
                mean, variance = np.asarray(mean), np.asarray(variance)
                if (mean.shape != task.query_shape or variance.shape != task.query_shape
                        or not np.isfinite(mean).all() or not np.isfinite(variance).all()
                        or np.any(variance <= 0)):
                    raise FloatingPointError('Invalid observation-space task prediction')
                mixture = None
                if getattr(adapter, 'predictive_family', 'gaussian') == 'gaussian_mixture':
                    components, noise = adapter.components
                    components, noise = np.asarray(components), float(noise)
                    if components.ndim != 3 or components.shape[1:] != task.query_shape or not components.shape[0]:
                        raise ValueError('Mixture components must have shape (samples, task times, query sites)')
                    if (not np.isfinite(components).all() or not np.isfinite(noise) or noise <= 0 or
                            not np.allclose(mean, components.mean(axis=0), rtol=1e-5, atol=1e-7) or
                            not np.allclose(variance, components.var(axis=0) + noise, rtol=1e-5, atol=1e-7)):
                        raise ValueError('Reported moments do not match the recorded observation mixture')
                    mixture = (components, noise)
            with measurement.phase('scoring', task=index):
                truth = stream.truth(task)
                if score is not None:
                    task_score = score(task, truth, adapter)
                elif mixture is not None:
                    components, noise = mixture
                    components = components.reshape(components.shape[0], -1)
                    task_score = gaussian_mixture_metrics(truth, components, noise)
                    task_score.update(gaussian_mixture_calibration(truth, components, noise))
                    task_score['coverage90'] = gaussian_mixture_calibration(
                        truth, components, noise, levels=np.array([.9]))['coverage'][0]
                else:
                    task_score = gaussian_scores(truth, mean, variance)
                if not np.allclose(task_score['levels'], LEVELS, rtol=0, atol=1e-14):
                    raise ValueError('Incompatible calibration levels')
                scores.append(dict(queries=truth.size, **task_score))
            task_record = dict(task=index, start=task.start, stop=task.stop,
                time_start=float(task.times[0]), time_stop=float(task.times[-1]),
                current_labels=task.visible.values.size,
                delayed_labels=0 if task.delayed is None else task.delayed.values.size,
                scores=task_score, metrics_native=native_scores(task_score, target_scale),
                metric_scale=native_units, timing=measurement.rows[-2:].copy(), resources=_resources())
            diagnostic = getattr(adapter, 'diagnostics', None)
            if callable(diagnostic):
                task_record['adapter_diagnostics'] = diagnostic()
            if output is not None:
                directory = output / 'tasks' / f'{index:06d}'
                with measurement.phase('serialization', task=index):
                    values = dict(times=task.times, query_sites=task.query_sites,
                        y_true=truth, pred_mean=mean, pred_var=variance,
                        target_center=target_center, target_scale=target_scale, metric_scale=native_units)
                    if mixture is not None:
                        values.update(component_means=mixture[0], noise_variance=mixture[1])
                    latent = getattr(adapter, 'latent', None)
                    if latent is not None:
                        values.update(latent_mean=np.asarray(latent[0]), latent_var=np.asarray(latent[1]))
                    _npz(directory / 'predictions.npz', values)
                    task_record['checkpoint'] = (_checkpoint(adapter, directory, index) if checkpoints
                        else dict(status='disabled'))
                    _json(directory / 'record.json', task_record)
                    _json(output / 'progress.json', dict(status='running', completed_tasks=index + 1,
                        protocol_sha256=protocol_identity, main_table_admitted=False))
            event('online', index + 1, task_record)
            completed_tasks += 1
            means.append(mean); variances.append(variance); truths.append(truth); times.append(task.times)
        mean, variance, truth = map(np.concatenate, [means, variances, truths])
        result = dict(status='completed', protocol_sha256=protocol_identity,
            ordinary_latency_valid=not profiler.enabled, profiled_task_ids=list(profiler.tasks),
            main_table_admitted=False, timing=timing_summary(measurement.rows),
            predictive_family=config['predictive_family'], provenance=provenance or {})
        if fit_budget is not None:
            result['fit_budget'] = fit_budget
        weights = np.asarray([s['queries'] for s in scores], dtype=float)
        weights /= weights.sum()
        coverage = sum(w * np.asarray(s['coverage']) for w, s in zip(weights, scores))
        result['metrics'] = {key: float(sum(w * s[key] for w, s in zip(weights, scores)))
                            for key in ['nlpd', 'crps', 'coverage90']}
        result['metrics'].update(rmse=float(np.sqrt(np.mean((truth-mean)**2))),
            coverage=coverage.tolist(), levels=LEVELS.tolist(), ece=float(np.mean(abs(coverage-LEVELS))))
        result['metrics_native'] = native_scores(result['metrics'], target_scale)
        result['metric_scale'] = native_units
        result['standardized_metric_scale'] = 'fit-only standardized target'
        result['target_standardization'] = transform_record
        if output is not None:
            with measurement.phase('serialization'):
                _npz(output / 'predictions.npz', dict(times=np.concatenate(times), query_sites=stream.hidden,
                    y_true=truth, pred_mean=mean, pred_var=variance,
                    target_center=target_center, target_scale=target_scale, metric_scale=native_units))
                _json(output / 'task-metrics.json', scores)
                _json(output / 'task-metrics-native.json', [native_scores(row, target_scale) for row in scores])
            result['timing'] = timing_summary(measurement.rows)
            _json(output / 'timing.json', measurement.rows)
            files = {}
            for path in sorted(output.rglob('*')):
                if path.is_file() and not path.is_symlink() and path.suffix in {'.json', '.npz'}:
                    digest = hashlib.sha256()
                    with path.open('rb') as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b''):
                            digest.update(block)
                    files[str(path.relative_to(output))] = dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)
            _json(output / 'artifacts.json', files)
            _json(output / 'result.json', result)
        event('complete', len(scores), result)
        return result
    except BaseException as error:
        failure = dict(status='failed', completed_tasks=completed_tasks, failed_task=index,
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            time_unix=time.time(), main_table_admitted=False, resources=_resources())
        if output is not None:
            _json(output / 'timing.json', measurement.rows)
            _json(output / 'failure.json', failure)
            _json(output / 'result.json', failure)
        event('failed', completed_tasks, failure)
        raise
    finally:
        if previous_journal is None:
            os.environ.pop('HIPPO_EVENT_PATH', None)
