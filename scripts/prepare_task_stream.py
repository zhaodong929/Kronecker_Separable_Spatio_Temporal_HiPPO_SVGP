#!/usr/bin/env python3
"""Prepare locked task datasets from verified local sources, without training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.task_stream.data import prepare, file_hash, SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=SEEDS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--era-root', type=Path, default=ROOT/'outputs/era5-recovered-fair-20260930')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/fair-three-domain-20260929')
    parser.add_argument('--pems-reference', type=Path, help='Paired legacy protocol directory for requested seed')
    parser.add_argument('--mask-source', type=Path, help='Small NPZ containing the five original geometry/mask arrays')
    parser.add_argument('--pems-raw-hdf', type=Path, default=ROOT/'data/task-stream-sources/pems/PEMS-BAY.h5')
    parser.add_argument('--road-csv', type=Path, default=Path('/data/nk523/projects/dcrnn-data-source-20260929/data/sensor_graph/distances_bay_2017.csv'))
    args = parser.parse_args()
    if args.seed not in SEEDS[args.dataset]:
        parser.error('Seed outside preregistered paired splits')
    extra = {}
    if args.dataset == 'era5':
        reference = args.era_root/f'seed{args.seed}'
        raw_path = args.era_root/'raw-recovered.npz'
        with np.load(raw_path) as raw:
            values = raw['values']
            dates = raw['times']
        expected = np.datetime64('2020-01-01T00', 'h')+np.arange(1860).astype('timedelta64[h]')
        np.testing.assert_array_equal(dates, expected)
        targets = values[..., 0]
        extra['weather'] = values[..., 1:]
        source = dict(raw_sha256=file_hash(raw_path), raw_size_bytes=raw_path.stat().st_size, start=str(dates[0]),
            last_inclusive=str(dates[-1]), end_exclusive='2020-03-18T12:00:00',
            source_verification=json.loads((args.era_root/'source-verification.json').read_text()))
    elif args.dataset == 'covid':
        reference = args.data_root/'covid-v2'/f'seed{args.seed}'
        raw_path = args.data_root/'covid-source/recovered_targets.npz'
        with np.load(raw_path) as raw:
            targets = raw['log1p_per100k']
            dates = raw['dates']
        source = dict(raw_sha256=file_hash(raw_path), raw_size_bytes=raw_path.stat().st_size,
                      start=str(dates[0]), last_inclusive=str(dates[-1]))
    else:
        reference = args.pems_reference or args.data_root/'pems-v2'/f'seed{args.seed}'
        old_meta = json.loads((reference/'protocol.json').read_text())
        normalization = old_meta['target_standardisation']
        if not args.pems_raw_hdf.exists():
            with np.load(reference/'protocol.npz') as old:
                targets = np.concatenate([old['calibration_y'], old['stream_y'][:10080]])
                elapsed_hours = np.concatenate([old['calibration_times'], old['stream_times'][:10080]])
            np.testing.assert_allclose(elapsed_hours-elapsed_hours[0], np.arange(12096)/12.,
                                       rtol=0., atol=1e-8)
            targets = targets.astype(float)*normalization['scale']+normalization['mean']
        source = dict(recovered_from_standardized_archive=True,
            reconstruction_precision='float32 standardized archive; raw speed rounding uncertainty retained',
            calendar_start='2017-01-01T00:00:00', calendar_clock='source naive local; initial42days precede DST',
            road_csv_sha256=file_hash(args.road_csv))
    reference_meta = json.loads((reference/'protocol.json').read_text())
    if int(reference_meta['split_seed']) != args.seed:
        raise ValueError('Reference mask belongs to a different paired split seed')
    mask_source = args.mask_source or reference/'protocol.npz'
    with np.load(mask_source) as old:
        coordinates, visible, hidden, fit, validation = [old[key] for key in
            ('coordinates', 'train_indices', 'test_indices', 'fit_indices', 'validation_indices')]
    source['paired_mask_reference_sha256'] = file_hash(mask_source)
    source['original_protocol_sha256'] = reference_meta.get('protocol_npz_sha256')
    if args.dataset == 'pems':
        import pandas as pd
        from stvgp_kronecker.data.traffic import build_calendar_spatial_features
        from stvgp_kronecker.traffic_spatial_kernels import load_road_laplacian
        split = json.loads((reference/'split.json').read_text())['split']
        sensor_ids = [None]*len(coordinates)
        for label in ('visible', 'heldout'):
            for index, sensor in zip(split[label+'_indices'], split[label+'_sensor_ids']):
                sensor_ids[index] = sensor
        if any(x is None for x in sensor_ids):
            raise ValueError('Incomplete sensor identity map')
        if args.pems_raw_hdf.exists():
            from stvgp_kronecker.data.traffic import _read_traffic_hdf, _normalise_sensor_id
            frame = _read_traffic_hdf(args.pems_raw_hdf)
            if frame.shape != (52116, 325):
                raise ValueError('Unexpected raw PEMS shape')
            actual_ids = tuple(_normalise_sensor_id(x) for x in frame.columns)
            if actual_ids != tuple(sensor_ids):
                raise ValueError('Raw HDF sensor order differs from paired mask source')
            frame = frame.iloc[:12096]
            dates = frame.index
            expected_dates = pd.date_range('2017-01-01', periods=12096, freq='5min')
            np.testing.assert_array_equal(dates.to_numpy(), expected_dates.to_numpy())
            targets = frame.to_numpy(dtype=float)
            source = dict(raw_sha256=file_hash(args.pems_raw_hdf),
                raw_size_bytes=args.pems_raw_hdf.stat().st_size,
                paired_mask_reference_sha256=file_hash(mask_source),
                original_protocol_sha256=reference_meta.get('protocol_npz_sha256'),
                recovered_from_standardized_archive=False, start=str(dates[0]),
                last_inclusive=str(dates[-1]), calendar_clock='source naive local; 42days precede DST',
                sensor_order_verified=True, road_csv_sha256=file_hash(args.road_csv))
        laplacian, graph_metadata = load_road_laplacian(args.road_csv, tuple(sensor_ids))
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        # Match the existing official-adaptation graph diffusion parameter.
        diffusion = old_meta['task1_mean_metadata']['graph_diffusion']
        heat = (eigenvectors*np.exp(-float(diffusion)*np.maximum(eigenvalues, 0.)))@eigenvectors.T
        extra['road_heat'] = np.maximum(.5*(heat+heat.T), 0.)
        dates = pd.date_range('2017-01-01', periods=12096, freq='5min')
        extra['calendar'] = build_calendar_spatial_features(dates, coordinates)
        source.update(graph_diffusion=diffusion, graph_metadata=graph_metadata)
    prepared = prepare(args.dataset, targets, coordinates, visible, hidden, fit, validation,
                       seed=args.seed, source=source, **extra)
    prepared.save(args.output)
    print(json.dumps(dict(path=str(args.output), identity=prepared.stream.identity(),
        tasks=len(prepared.stream.bounds), features=prepared.features.shape[-1],
        selection_tasks=len(prepared.selection_stream.bounds))))


if __name__ == '__main__':
    main()
