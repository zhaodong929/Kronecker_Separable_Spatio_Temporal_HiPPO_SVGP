import json
import numpy as np
from benchmarks.three_domain.online_validation import build_fold
from baselines.covid_long_setting_b.protocol import COVIDSettingBProtocol


def test_budget_fold_never_uses_formal_stream_or_future_normalization(tmp_path):
    rng=np.random.default_rng(49)
    payload=dict(calibration_y=rng.normal(size=(52,52)),stream_y=rng.normal(size=(143,52)),
        calibration_times=np.arange(52)/51.,coordinates=rng.normal(size=(52,2)),
        train_indices=np.arange(42),fit_indices=np.arange(38),validation_indices=np.arange(38,42),
        test_indices=np.arange(42,52))
    meta=dict(split_seed=5,dataset='test',target_standardization=dict(mean=2.,scale=.8))
    def make(name,payload):
        source=tmp_path/(name+'.npz');np.savez(source,**payload)
        source.with_suffix('.json').write_text(json.dumps(meta))
        result=build_fold(source,tmp_path/name)
        COVIDSettingBProtocol(result)
        with np.load(result) as a:return {key:a[key].copy() for key in a.files}
    base=make('base',payload)
    changed=make('stream_changed',{**payload,'stream_y':payload['stream_y']+1000})
    for key in base:np.testing.assert_array_equal(base[key],changed[key])
    y=payload['calibration_y'].copy();y[40:]+=1000
    changed=make('future_changed',{**payload,'calibration_y':y})
    for key in ['calibration_y','calibration_phi','task1_ridge_beta','task1_calibration_mean']:
        np.testing.assert_array_equal(base[key],changed[key])
    assert np.all(base['stream_times']>base['calibration_times'][-1])
