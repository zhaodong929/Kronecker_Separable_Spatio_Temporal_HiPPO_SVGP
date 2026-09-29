#!/usr/bin/env python3
"""Rebuild the 195-week admissions protocol from a checksummed CDC snapshot.

This is a new, explicitly versioned download, not a claim of bitwise recovery
of the author's historical snapshot. Location order/coordinates and split rules
are recovered from the archived development protocol; no test score is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.covid_long_setting_b.development import _ridge, _spatial_inducing
from scripts.iclr_era5_full_benchmark_protocol import inner_spatial_split

JURISDICTIONS = 'AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR'.split()


def features(target, coordinates, fit, initial=52):
    """Use only fit-site history during selection, legal one-step lags online."""
    count, sites = target.shape
    index = np.arange(count, dtype=float)
    lags = np.full((4, count, sites), np.nan)
    unavailable = np.setdiff1d(np.arange(sites), fit)
    for k, lag in enumerate(range(1, 5)):
        lags[k, lag:] = target[:-lag]
        # Validation and future hidden sites are not encoder/feature context
        # during Task-1 spatial selection. After initialization all Task-1
        # labels are legally public, and each online hidden label arrives at +1.
        lags[k, :initial, unavailable] = np.nan
    n = np.isfinite(lags).sum(axis=0)
    rolling = np.nansum(lags, axis=0)/np.maximum(n, 1)
    rolling[n == 0] = np.nan
    aggregate = np.full(count, np.nan)
    aggregate[1:] = target[:-1, fit].mean(axis=1)
    broadcast = lambda v: np.broadcast_to(v[:, None], (count, sites))
    dynamic = np.stack([broadcast(index/51), broadcast(np.sin(2*np.pi*index/52.1775)),
        broadcast(np.cos(2*np.pi*index/52.1775)), *lags, rolling, lags[0]-lags[1],
        broadcast(aggregate), np.broadcast_to(coordinates[None,:,0], (count,sites)),
        np.broadcast_to(coordinates[None,:,1], (count,sites))], axis=-1)
    reference = dynamic[:initial, fit].reshape(-1, dynamic.shape[-1])
    mean, scale = np.nanmean(reference, axis=0), np.maximum(np.nanstd(reference, axis=0), 1e-12)
    phi = np.concatenate([np.ones((count,sites,1)),
        (np.where(np.isfinite(dynamic), dynamic, mean)-mean)/scale], axis=-1)
    return phi, dict(mean=mean.tolist(), scale=scale.tolist(), context_indices=fit.tolist(),
        fit_scope='Task-1 calibration-fit sites only', initial_unobserved_lags='training-feature mean imputation')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-json', type=Path, required=True)
    p.add_argument('--reference-npz', type=Path, required=True)
    p.add_argument('--reference-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[5,6,7,8,9])
    a = p.parse_args()
    frame = pd.DataFrame(json.loads(a.raw_json.read_text()))
    frame['date'] = frame.weekendingdate.str[:10]
    frame['value'] = pd.to_numeric(frame.totalconfc19newadmper100k)
    wide = frame.pivot(index='date', columns='jurisdiction', values='value').loc[:,JURISDICTIONS]
    dates = pd.date_range('2020-08-08','2024-04-27',freq='7D').strftime('%Y-%m-%d').to_numpy()
    if len(dates) != 195 or list(wide.index) != list(dates):
        raise ValueError('Expected exactly the 195 consecutive mandatory-period weeks')
    rates = wide.to_numpy(dtype=float)
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError('Missing or invalid admissions rates; do not silently impute targets')
    raw = np.log1p(rates)
    with np.load(a.reference_npz) as r:
        coordinates = r['coordinates']
        old = np.concatenate([r['calibration_y'],r['stream_y']])
    old_meta = json.loads(a.reference_json.read_text())['target_standardization']
    old = old*old_meta['scale']+old_meta['mean']
    correlations = [float(np.corrcoef(old[:,i], raw[:len(old),i])[0,1]) for i in range(52)]
    if min(correlations) < .98:
        raise ValueError('Archived location alignment requires review')
    hashfile = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    source = dict(url='https://data.cdc.gov/resource/ua7e-t2fy.json',
        column='totalconfc19newadmper100k', raw_sha256=hashfile(a.raw_json),
        reference_sha256=hashfile(a.reference_npz), downloaded_snapshot=a.raw_json.name,
        historical_snapshot_bitwise_identical=False,
        max_initial_log1p_difference=float(np.max(np.abs(old-raw[:len(old)]))),
        minimum_location_correlation=min(correlations), jurisdictions=JURISDICTIONS)
    for seed in a.seeds:
        if seed not in range(5,10):
            raise ValueError('Formal COVID split seeds are 5 through 9')
        permutation = np.random.default_rng(seed).permutation(52)
        hidden, visible = np.sort(permutation[:10]), np.sort(permutation[10:])
        fit, validation = inner_spatial_split(visible, split_seed=seed)
        mean, scale = float(raw[:52,fit].mean()), float(raw[:52,fit].std())
        y = (raw-mean)/scale
        phi, feature_meta = features(y, coordinates, fit)
        beta = _ridge(phi[:52], y[:52], fit)
        payload = dict(train_indices=visible, test_indices=hidden, fit_indices=fit,
            validation_indices=validation, calibration_y=y[:52], stream_y=y[52:],
            calibration_phi=phi[:52], stream_phi=phi[52:], coordinates=coordinates,
            calibration_times=np.arange(52)/51., stream_times=np.arange(52,195)/51.,
            calibration_week_dates=dates[:52], stream_week_dates=dates[52:],
            task1_ridge_beta=beta, task1_calibration_mean=np.einsum('tsp,p->ts',phi[:52],beta),
            task1_stream_mean=np.einsum('tsp,p->ts',phi[52:],beta),
            block_start=np.arange(143), block_stop=np.arange(1,144))
        payload.update(_spatial_inducing(coordinates, fit, (16,32)))
        out = a.output/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(out/'protocol.npz',**payload)
        meta = dict(schema_version=2,dataset='cdc_covid_nhsn_mandatory_long_stream',split_seed=seed,
            target='log1p weekly confirmed COVID admissions per 100,000 population',
            target_standardization=dict(mean=mean,scale=scale,fit_scope='Task-1 calibration-fit locations only',
                fit_indices=fit.tolist(),metric_scale='log1p(per-100k)'),
            xlag=dict(delay_weeks=1,features=13,normalization=feature_meta),
            task1_observed_indices=list(range(52)),num_calibration_times=52,num_stream_times=143,
            num_locations=52,source=source,protocol_npz_sha256=hashfile(out/'protocol.npz'))
        (out/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(json.dumps(dict(seed=seed,path=str(out),source=source)))


if __name__ == '__main__':
    main()
