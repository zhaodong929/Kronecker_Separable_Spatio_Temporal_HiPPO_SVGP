import json
import numpy as np
from baselines.era5_protocol import ERA5Protocol
from benchmarks.three_domain.era5_features import features


def test_hourly_hidden_labels_never_released(tmp_path):
    rng=np.random.default_rng(8);coordinates=rng.normal(size=(800,2));initial=5;steps=3
    payload=dict(calibration_y=rng.normal(size=(initial,800)),stream_y=rng.normal(size=(steps,800)),train_indices=np.arange(720),fit_indices=np.arange(720),validation_indices=np.arange(720,800),test_indices=np.arange(720,800),coordinates=coordinates,calibration_times=np.arange(initial),stream_times=np.arange(initial,initial+steps))
    meta=dict(protocol_id='era5_land_hourly_causal',development_protocol=True,hidden_label_policy='never released',task1_observed_indices=list(range(720)))
    def load(name):
        f=tmp_path/(name+'.npz');np.savez(f,**payload);f.with_suffix('.json').write_text(json.dumps(meta));return ERA5Protocol(f)
    ref=load('ref');payload['calibration_y'][:,720:]+=1e6;payload['stream_y'][:,720:]-=1e6;other=load('changed')
    np.testing.assert_array_equal(ref.task1().targets,other.task1().targets)
    audit=ref.make_audit()
    for step in range(steps):
        a,b=ref.week(step),other.week(step);assert a.delayed_hidden is None and b.delayed_hidden is None
        np.testing.assert_array_equal(a.current_visible.targets,b.current_visible.targets)
        audit.record_step(a,np.zeros(80),np.ones(80))
    report=audit.summary();assert report['passed'] and report['delayed_hidden_labels']==0


def test_weather_feature_statistics_use_only_initial_fit_sites():
    rng=np.random.default_rng(19);weather=rng.normal(size=(24,8,6));coords=rng.normal(size=(8,2));fit=np.arange(5)
    before,stats=features(weather,coords,fit,initial=12)
    changed=weather.copy();changed[12:]+=10000;changed[:12,5:]-=10000
    after,other=features(changed,coords,fit,initial=12)
    assert stats==other
    np.testing.assert_array_equal(before[:12,:5],after[:12,:5])
    assert before.shape==(24,8,133)
