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


def test_pems_fold_excludes_original_heldout_targets_and_formal_stream(tmp_path):
    from benchmarks.three_domain.online_validation import build_pems_fold
    from baselines.traffic_protocol_n import TrafficProtocolN
    rng=np.random.default_rng(9)
    payload=dict(calibration_y=rng.normal(size=(2016,325)),stream_y=np.zeros((2,325)),
        calibration_phi=rng.normal(size=(2016,325,3)),calibration_times=np.arange(2016)/12.,
        coordinates=rng.normal(size=(325,2)),train_indices=np.arange(260),
        fit_indices=np.arange(234),validation_indices=np.arange(234,260))
    payload['calibration_phi'][:,:,0]=1.
    def make(name,values):
        p=tmp_path/(name+'.npz');np.savez(p,**values)
        p.with_suffix('.json').write_text(json.dumps(dict(split_seed=1,task1_mean_metadata=dict(base_feature_count=1))))
        out=build_pems_fold(p,tmp_path/name)
        protocol=TrafficProtocolN(out)
        assert len(protocol.task1().locations)==260
        assert len(protocol.week(0).current_visible.locations)==234
        assert len(protocol.week(1).delayed_hidden.locations)==26
        with np.load(out) as a:return {key:a[key].copy() for key in a.files}
    base=make('base_pems',payload)
    changed_y=payload['calibration_y'].copy();changed_y[:,260:]+=10000
    changed_phi=payload['calibration_phi'].copy();changed_phi[:,260:]+=10000
    changed=make('heldout_pems',{**payload,'calibration_y':changed_y,'calibration_phi':changed_phi,'stream_y':payload['stream_y']+200})
    for key in base:np.testing.assert_array_equal(base[key],changed[key])
    changed_y=payload['calibration_y'].copy();changed_y[1728:,:260]+=1000
    changed_phi=payload['calibration_phi'].copy();changed_phi[1728:,:,1:]+=1000
    future=make('future_pems',{**payload,'calibration_y':changed_y,'calibration_phi':changed_phi})
    for key in ['calibration_y','calibration_phi','task1_ridge_beta']:
        np.testing.assert_array_equal(base[key],future[key])
