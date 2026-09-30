"""Task preparation with locked fit-only transforms and explicit release features.

Targets are accessible only in the preparation/evaluation object. Dynamic label
features are materialized through observations actually released by TaskStream.
Historical regressors stay frozen when delayed targets subsequently arrive.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .protocol import Observations, TaskStream

SETTINGS = {'era5': (336, 168, 24, 1860), 'covid': (52, 40, 1, 195),
            'pems': (4032, 2016, 12, 12096)}
SEEDS = {'era5': tuple(range(5)), 'covid': tuple(range(5, 10)), 'pems': (1, 2, 3)}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class LegalFeatureProvider:
    """No target array is accepted; only explicitly observed values enter lags."""
    def __init__(self, times, coordinates, *, period=52.1775, context_sites=None):
        self.times = np.asarray(times)
        self.coordinates = np.asarray(coordinates)
        self.lookup = {float(t): i for i, t in enumerate(times)}
        self.values = np.full((len(times), len(coordinates)), np.nan)
        self.context = np.asarray(context_sites if context_sites is not None else
                                  np.arange(len(coordinates)), dtype=int)
        self.period = period

    def observe(self, observations):
        rows = np.array([self.lookup[float(t)] for t in observations.times])
        old = self.values[np.ix_(rows, observations.sites)]
        if np.any(np.isfinite(old) & (old != observations.values)):
            raise ValueError('A previously released target changed')
        self.values[np.ix_(rows, observations.sites)] = observations.values

    def __call__(self, times, sites):
        rows = np.array([self.lookup[float(t)] for t in times])
        sites = np.asarray(sites)
        shape = (len(rows), len(sites))
        broadcast = lambda x: np.broadcast_to(x[:, None], shape)
        lagged = []
        for lag in range(1, 5):
            values = self.values[np.ix_(np.maximum(rows-lag, 0), sites)].copy()
            values[rows < lag] = np.nan
            lagged.append(values)
        stacked = np.stack(lagged)
        count = np.isfinite(stacked).sum(axis=0)
        rolling = np.nansum(stacked, axis=0) / np.maximum(count, 1)
        rolling[count == 0] = np.nan
        context = self.values[np.ix_(np.maximum(rows-1, 0), self.context)]
        count_context = np.isfinite(context).sum(axis=1)
        aggregate = np.nansum(context, axis=1) / np.maximum(count_context, 1)
        aggregate[(rows == 0) | (count_context == 0)] = np.nan
        phase = 2*np.pi*rows/self.period
        return np.stack([broadcast(rows / 51.), broadcast(np.sin(phase)),
            broadcast(np.cos(phase)), *lagged, rolling, lagged[0]-lagged[1],
            broadcast(aggregate), np.broadcast_to(self.coordinates[sites, 0], shape),
            np.broadcast_to(self.coordinates[sites, 1], shape)], axis=-1)


def label_features(stream, context):
    provider = LegalFeatureProvider(stream.times, stream.coordinates, context_sites=context)
    result = np.empty((*stream._targets.shape, 12))
    initial = stream.initial()
    provider.observe(initial)
    result[:stream.initial_steps] = provider(initial.times, np.arange(len(stream.coordinates)))
    for index, (start, stop) in enumerate(stream.bounds):
        task = stream.task(index)
        if task.delayed is not None:
            provider.observe(task.delayed)
        provider.observe(task.visible)
        result[start:stop] = provider(task.times, np.arange(len(stream.coordinates)))
    return result


def normalize_features(features, fit_steps, fit_sites, stats=None):
    if stats is None:
        fit = features[:fit_steps, fit_sites]
        valid = np.isfinite(fit)
        count = valid.sum(axis=(0, 1))
        center = np.where(valid, fit, 0.).sum(axis=(0, 1)) / np.maximum(count, 1)
        variance = np.where(valid, (fit-center)**2, 0.).sum(axis=(0, 1))/np.maximum(count, 1)
        scale = np.maximum(np.sqrt(variance), 1e-8)
        stats = {'mean': center.tolist(), 'scale': scale.tolist()}
    center, scale = np.asarray(stats['mean']), np.asarray(stats['scale'])
    values = (np.where(np.isfinite(features), features, center)-center)/scale
    return np.concatenate([np.ones((*values.shape[:2], 1)), values], axis=-1).astype(np.float32), stats


def road_features(values, coordinates, context, heat, calendar):
    """Original Road-context L10: self excluded; hidden labels never inputs."""
    weights = np.maximum(np.asarray(heat)[:, context], 0.).copy()
    for position, site in enumerate(context):
        weights[site, position] = 0.
    empty = weights.sum(axis=1) <= 1e-12
    if empty.any():
        distance = np.linalg.norm(coordinates[empty, None]-coordinates[context][None], axis=-1)
        fallback = 1./np.maximum(distance, 1e-6)
        for row, site in enumerate(np.flatnonzero(empty)):
            fallback[row, np.asarray(context) == site] = 0.
        weights[empty] = fallback
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    current = values[:, context] @ weights.T
    lags = [current[np.maximum(np.arange(len(values))-lag, 0)] for lag in range(1, 11)]
    dynamic = np.stack([current, *lags, *[current-lag for lag in lags]], axis=-1)
    return np.concatenate([calendar[..., 1:], dynamic], axis=-1)


@dataclass
class PreparedData:
    stream: TaskStream
    features: np.ndarray
    fit_sites: np.ndarray
    validation_sites: np.ndarray
    selection_stream: TaskStream
    selection_features: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with np.load(directory/'features.npz', allow_pickle=False) as arrays:
            return cls(TaskStream.load(directory/'stream.npz'), arrays['features'],
                arrays['fit_sites'], arrays['validation_sites'],
                TaskStream.load(directory/'selection-stream.npz'), arrays['selection_features'],
                json.loads(str(arrays['metadata_json'])))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.stream.save(directory/'stream.npz')
        self.selection_stream.save(directory/'selection-stream.npz')
        np.savez_compressed(directory/'features.npz', features=self.features,
            selection_features=self.selection_features, fit_sites=self.fit_sites,
            validation_sites=self.validation_sites,
            metadata_json=json.dumps(self.metadata, sort_keys=True, allow_nan=False))
        artifacts = {name: dict(size_bytes=(directory/name).stat().st_size,
                               sha256=file_hash(directory/name))
                     for name in ('stream.npz', 'selection-stream.npz', 'features.npz')}
        manifest = dict(self.metadata, artifacts=artifacts, stream_identity=self.stream.identity(),
                        selection_stream_identity=self.selection_stream.identity())
        (directory/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')


def prepare(dataset, targets, coordinates, visible, hidden, fit, validation, *,
            seed, weather=None, calendar=None, road_heat=None, source=None):
    if dataset not in SETTINGS or seed not in SEEDS[dataset]:
        raise ValueError('Unregistered dataset/split seed')
    initial, fit_steps, task_steps, total = SETTINGS[dataset]
    targets, coordinates = np.asarray(targets, dtype=float)[:total], np.asarray(coordinates, dtype=float)
    visible, hidden, fit, validation = [np.asarray(x, dtype=int) for x in (visible, hidden, fit, validation)]
    if len(targets) != total or not np.isfinite(targets).all():
        raise ValueError('Incomplete or non-finite canonical targets')
    if set(fit) & set(validation) or set(fit) | set(validation) != set(visible):
        raise ValueError('Internal fit/validation must partition final visible sites')
    center = float(targets[:fit_steps, fit].mean())
    scale = float(targets[:fit_steps, fit].std())
    if scale <= 0:
        raise ValueError('Degenerate fit-only target scale')
    y = (targets-center)/scale
    # One canonical elapsed grid, affine-scaled by the initial duration.
    times = np.arange(total, dtype=float)/(initial-1)
    initial_sites = np.arange(len(coordinates)) if dataset == 'covid' else visible
    metadata = dict(schema_version=1, dataset=dataset, split_seed=int(seed),
        target_standardization=dict(mean=center, scale=scale, fit_steps=fit_steps,
                                    fit_sites=fit.tolist()),
        initial_steps=initial, task_steps=task_steps, total_steps=total,
        time_unit={'era5': 'hour', 'covid': 'week', 'pems': 'five_minutes'}[dataset],
        time_divisor=initial-1, historical_feature_policy='frozen_at_original_task_release',
        source=source or {}, main_table_admitted=False)
    stream = TaskStream(times=times, targets=y, coordinates=coordinates, visible=visible,
        hidden=hidden, initial_sites=initial_sites, initial_steps=initial, task_steps=task_steps,
        release_previous=dataset != 'era5', metadata=metadata)
    inverse = {int(site): index for index, site in enumerate(visible)}
    inner_fit = np.array([inverse[int(site)] for site in fit])
    inner_query = np.array([inverse[int(site)] for site in validation])
    selection = TaskStream(times=times[:initial], targets=y[:initial, visible],
        coordinates=coordinates[visible], visible=inner_fit, hidden=inner_query,
        initial_sites=np.arange(len(visible)) if dataset == 'covid' else inner_fit,
        initial_steps=fit_steps, task_steps=task_steps, release_previous=dataset != 'era5',
        metadata=dict(metadata, original_site_indices=visible.tolist(), role='selection'))
    if dataset == 'era5':
        if weather is None or np.asarray(weather).shape != (total, len(coordinates), 6):
            raise ValueError('ERA5 requires aligned six-variable contemporaneous weather')
        from benchmarks.three_domain.era5_features import features
        phi, feature_stats = features(weather, coordinates, fit, initial=fit_steps)
        inner_phi = phi[:initial, visible].copy()
        metadata['feature_family'] = 'official_133_weather_L10_with_prefix_time_phase'
    elif dataset == 'covid':
        # Selection scalers must not see extra initially observed COVID sites.
        inner_raw = label_features(selection, inner_fit)
        inner_phi, feature_stats = normalize_features(inner_raw, fit_steps, inner_fit)
        phi, _ = normalize_features(label_features(stream, fit), fit_steps, fit, feature_stats)
        metadata['feature_family'] = '13_calendar_spatial_released_L4'
    else:
        if road_heat is None or calendar is None:
            raise ValueError('PEMS requires verified road heat kernel and timestamp calendar')
        raw = road_features(y, coordinates, fit, road_heat, calendar[:total])
        phi, feature_stats = normalize_features(raw, fit_steps, fit)
        # Context remains the calibration-fit sensors throughout selection/refit.
        inner_phi = phi[:initial, visible].copy()
        metadata['feature_family'] = '28_calendar_spatial_road_context_L10'
    metadata['feature_normalization'] = feature_stats
    metadata['selection_original_site_indices'] = visible.tolist()
    metadata['selection_fit_indices'] = inner_fit.tolist()
    metadata['selection_query_indices'] = inner_query.tolist()
    stream.metadata = dict(metadata)
    selection.metadata.update(feature_family=metadata['feature_family'])
    return PreparedData(stream, phi, fit, validation, selection, inner_phi, metadata)
