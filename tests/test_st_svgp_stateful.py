import numpy as np


def test_state_continuation_matches_official_sparse_gaussian_replay():
    from baselines.covid_long_setting_b.adapters.run_st_svgp import make_model,assign_frozen_hyperparameters,frozen_hyperparameters
    from baselines.st_svgp_filter import GaussianSTFilter
    import bayesnewton
    coords=np.array([[0.,0.],[.1,.4],[.5,.2],[.7,.6]])
    z=coords[[0,2]]
    times=np.array([0.,.1,.25,.4,.8])
    targets=np.random.default_rng(48).normal(size=(5,4))
    base=make_model(times[:2,None],np.repeat(coords[None],2,axis=0),targets[:2],z,trainable_inducing=False)
    kv,lv=frozen_hyperparameters(base)
    state=GaussianSTFilter(base.kernel,base.likelihood,coords)
    x=[];y=[]
    def record(t,sites,values):
        x.extend(np.column_stack([np.full(len(sites),t),coords[sites]]));y.extend(values)
    for i in range(2):
        state.advance(times[i],np.arange(4),targets[i]);record(times[i],np.arange(4),targets[i])
    for i in range(2,5):
        delayed={}
        if i>2:
            delayed=dict(delayed_time=times[i-1],delayed_sites=np.array([2,3]),delayed_values=targets[i-1,2:])
            record(times[i-1],np.array([2,3]),targets[i-1,2:])
        state.advance(times[i],np.array([0,1]),targets[i,:2],**delayed)
        record(times[i],np.array([0,1]),targets[i,:2])
        t,r,observed=bayesnewton.utils.create_spatiotemporal_grid(np.array(x),np.array(y)[:,None])
        reference=make_model(t,r,observed,z,trainable_inducing=False)
        assign_frozen_hyperparameters(reference,kv,lv)
        reference.inference(lr=1.)
        m,v=reference.predict_y(X=np.array([[times[i]],[times[i]]]),R=np.repeat(coords[None],2,axis=0))
        actual=state.predict(np.array([2,3]))
        np.testing.assert_allclose(actual[0],np.asarray(m)[-1,2:],rtol=2e-6,atol=2e-7)
        np.testing.assert_allclose(actual[1],np.asarray(v)[-1,2:],rtol=2e-6,atol=2e-7)
