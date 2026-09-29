"""Official Gaussian ST-SVGP with all spatial inducing points matches dense GP."""

def test_official_gaussian_posterior_matches_independent_dense_covariance():
    from baselines.covid_long_setting_b.adapters.run_st_svgp import make_model
    import numpy as np
    s=np.array([[0.,0.],[.3,.5]])
    t=np.array([[0.],[.15],[.4]])
    y=np.array([[.5,-.2],[.1,.8],[-.3,.2]])
    r=np.repeat(s[None],3,axis=0)
    m=make_model(t,r,y,s,trainable_inducing=False)
    m.inference(lr=1.)
    mu,var=m.predict_y(X=t,R=r)
    x=np.column_stack([np.repeat(t[:,0],2),np.tile(s,(3,1))])
    k=np.ones((6,6))
    for i,ell in enumerate([.2,1.,1.]):
     d=np.sqrt(3)*np.abs(x[:,None,i]-x[None,:,i])/ell
     k*= (1+d)*np.exp(-d)
    a=np.linalg.solve(k+.1*np.eye(6),y.ravel())
    expected=k@a
    v=np.diag(k-k@np.linalg.solve(k+.1*np.eye(6),k))+.1
    np.testing.assert_allclose(np.asarray(mu).ravel(),expected,rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(np.asarray(var).ravel(),v,rtol=2e-6,atol=2e-7)
