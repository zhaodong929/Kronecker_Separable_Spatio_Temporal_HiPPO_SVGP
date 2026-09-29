"""Chronological budget selection using only the formal initial window."""
import hashlib
import json
from pathlib import Path
import numpy as np


def build_fold(source, output, initial=40):
    from scripts.prepare_fair_covid import features
    from baselines.covid_long_setting_b.development import _ridge
    source,output=Path(source),Path(output)
    meta=json.loads(source.with_suffix('.json').read_text())
    with np.load(source,allow_pickle=False) as data:
        # Deliberately never load stream targets, features or dates here.
        keys=['calibration_y','calibration_times','coordinates','train_indices',
              'test_indices','fit_indices','validation_indices']
        arrays={key:data[key].copy() for key in keys}
        arrays.update({key:data[key].copy() for key in data.files if key.startswith('inducing_coords_ms')})
    target=arrays.pop('calibration_y')
    times=arrays.pop('calibration_times')
    if target.shape!=(52,52) or not 4<=initial<52:
        raise ValueError('Expected a full COVID initial window and nonempty internal stream')
    raw=target*meta['target_standardization']['scale']+meta['target_standardization']['mean']
    fit=arrays['fit_indices']
    center=float(raw[:initial,fit].mean());scale=float(raw[:initial,fit].std())
    if scale<=0:raise ValueError('Degenerate prefix target scale')
    y=(raw-center)/scale
    phi,stats=features(y,arrays['coordinates'],fit,initial=initial)
    beta=_ridge(phi[:initial],y[:initial],fit)
    n=len(y)-initial
    arrays.update(calibration_y=y[:initial],stream_y=y[initial:],
        calibration_phi=phi[:initial],stream_phi=phi[initial:],
        calibration_times=times[:initial],stream_times=times[initial:],
        task1_ridge_beta=beta,task1_calibration_mean=np.einsum('tsp,p->ts',phi[:initial],beta),
        task1_stream_mean=np.einsum('tsp,p->ts',phi[initial:],beta),
        block_start=np.arange(n),block_stop=np.arange(1,n+1))
    metadata=dict(schema_version=2,development_protocol=True,split_seed=meta['split_seed'],
        dataset=meta['dataset'],task1_observed_indices=list(range(52)),
        source_protocol_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        purpose='Online update budget selection using formal Task-1 labels only',
        formal_stream_used=False,initial_weeks=initial,validation_weeks=n,
        target_standardization=dict(mean=center,scale=scale,
            fit_scope='development training prefix calibration-fit locations only'),
        xlag=dict(delay_weeks=1,normalization=stats))
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'protocol.npz',**arrays)
    (output/'protocol.json').write_text(json.dumps(metadata,indent=2))
    return output/'protocol.npz'
