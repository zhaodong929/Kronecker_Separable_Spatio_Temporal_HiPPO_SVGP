"""Same official SGPR/OSGPR objective and Adam updates in eager/graph execution."""
import os
import numpy as np
import pytest


@pytest.mark.parametrize('streaming',[False,True])
def test_graph_initial_optimizer_matches_eager_official_model(streaming):
    from scripts.run_official_bui_osgpr_era5 import adapt_model,make_kernel,OSGPR_VFE,configure_tensorflow
    import tensorflow as tf
    import gpflow
    configure_tensorflow(tf,device=os.environ.get('HIPPO_TEST_DEVICE','cpu'),dtype='float64')
    gpflow.config.set_default_float(np.float64)
    rng=np.random.default_rng(20);x=rng.normal(size=(12,3));y=rng.normal(size=(12,1));z=x[:4].copy()
    theta=dict(ell_t=.5,ell_s=[1.,1.],kernel_variance=.8,noise_std=.2)
    def make():
        kernel=make_kernel(theta,frozen=False)
        if streaming:
            old=np.asarray(kernel(z))+1e-6*np.eye(4)
            model=OSGPR_VFE(data=(x,y),kernel=kernel,mu_old=np.zeros((4,1)),
                Su_old=.5*old,Kaa_old=old,Z_old=z,Z=z)
            model.likelihood.variance.assign(.04)
            return model
        return gpflow.models.SGPR(data=(x,y),kernel=kernel,inducing_variable=z,noise_variance=.04)
    reference,graph=make(),make()
    adapt_model(reference,steps=8,learning_rate=.001,execution='eager')
    adapt_model(graph,steps=8,learning_rate=.001,execution='graph')
    for left,right in zip(reference.trainable_variables,graph.trainable_variables):
        np.testing.assert_allclose(left.numpy(),right.numpy(),rtol=1e-9,atol=1e-10)
    for left,right in zip(reference.predict_f(x),graph.predict_f(x)):
        np.testing.assert_allclose(left.numpy(),right.numpy(),rtol=1e-9,atol=1e-10)


def test_reused_graph_matches_fresh_official_updates_and_resets_adam():
    from scripts.run_official_bui_osgpr_era5 import adapt_model,make_kernel,OSGPR_VFE,configure_tensorflow
    import tensorflow as tf
    import gpflow
    configure_tensorflow(tf,device=os.environ.get('HIPPO_TEST_DEVICE','cpu'),dtype='float64')
    gpflow.config.set_default_float(np.float64)
    rng=np.random.default_rng(72);cache={}
    for iteration,rows in enumerate([12,7,12,7]):
        x=rng.normal(size=(rows,3));y=rng.normal(size=(rows,1));z=rng.normal(size=(4,3))
        old_z=z+.05;mu=rng.normal(size=(4,1))*.1
        theta=dict(ell_t=.5+.1*iteration,ell_s=[1.,1.2],kernel_variance=.8,noise_std=.2)
        old=np.asarray(make_kernel(theta,frozen=False)(old_z))+1e-4*np.eye(4)
        def make():
            model=OSGPR_VFE(data=(x,y),kernel=make_kernel(theta,frozen=False),mu_old=mu,
                Su_old=.7*old,Kaa_old=old,Z_old=old_z,Z=z)
            model.likelihood.variance.assign(.04)
            return model
        reference,actual=make(),make()
        adapt_model(reference,steps=5,learning_rate=.001,execution='eager')
        adapt_model(actual,steps=5,learning_rate=.001,execution='graph',graph_cache=cache)
        for expected,value in zip(reference.trainable_variables,actual.trainable_variables):
            np.testing.assert_allclose(value.numpy(),expected.numpy(),rtol=1e-9,atol=1e-10)
        for expected,value in zip(reference.predict_f(x),actual.predict_f(x)):
            np.testing.assert_allclose(value.numpy(),expected.numpy(),rtol=1e-9,atol=1e-10)
    assert len(cache)==2
    assert all(entry[-1].experimental_get_tracing_count()<=2 for entry in cache.values())
