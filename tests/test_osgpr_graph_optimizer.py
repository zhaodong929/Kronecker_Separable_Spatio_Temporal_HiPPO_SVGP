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
