import numpy as np
import pytest
from benchmarks.task_stream.factory import Configuration, FeatureTable, FittedTaskAdapter
from benchmarks.task_stream.protocol import TaskStream
from benchmarks.task_stream.pipeline import run


def tiny():
    t=np.arange(7.)/10
    coords=np.array([[0.,0.],[.2,.3],[.6,.1],[.8,.4]])
    s=TaskStream(times=t,targets=np.sin(t[:,None]+coords[:,0]),coordinates=coords,
        visible=[0,1,2],hidden=[3],initial_sites=[0,1,2],initial_steps=3,task_steps=2,release_previous=True)
    features=FeatureTable(t,np.stack([np.ones((7,4)),np.broadcast_to(t[:,None],(7,4))],axis=-1))
    return s,features


@pytest.mark.parametrize('method',['kronhippo_svgp','ohsvgp'])
def test_torch_initial_fit_and_stream_are_reproducible(method,tmp_path):
    import torch
    s,f=tiny()
    c=Configuration(method,initial_iterations=2,learning_rate=.001,spatial_inducing=2,
                    temporal_inducing=3,inducing_size=3,rff=16,grid_rows=4,batch_rows=4,online_iterations=1)
    predictions=[]
    for seed in [99,2]:
        torch.manual_seed(seed)
        a=FittedTaskAdapter(c,s.coordinates,s.visible,f,initial_step=.1,release_previous=True)
        out=tmp_path/str(seed)
        result=run(s,a,output=out)
        assert result['metrics']['rmse']>=0
        predictions.append(np.load(out/'predictions.npz')['pred_mean'])
    np.testing.assert_array_equal(*predictions)


def test_feature_time_lookup_and_configuration_reject_silent_changes():
    _,f=tiny()
    with pytest.raises(ValueError,match='clocks'): f([99.],[0])
    with pytest.raises(ValueError,match='excluded'): Configuration('gpvae',1,.01)
    with pytest.raises(ValueError,match='integer'): Configuration('mgpvae',1.5,.01)


def test_contribution_arms_reuse_actual_fit_and_change_only_declared_update():
    from benchmarks.task_stream.ablations import ARMS
    s, features = tiny()
    config = Configuration('kronhippo_svgp', initial_iterations=1, learning_rate=.001,
        spatial_inducing=2, temporal_inducing=3, rff=16)
    fits, predictions = [], []
    for arm in ARMS:
        adapter = FittedTaskAdapter(config, s.coordinates, s.visible, features,
            initial_step=.1, release_previous=True, arm=arm)
        adapter.initialize(s.initial())
        fits.append(adapter.fit_sha256)
        predictions.append(adapter.predict_task(s.task(0))[0])
    assert len(set(fits)) == 1
    assert not np.allclose(predictions[0], predictions[1])
    assert not np.allclose(predictions[0], predictions[2])
