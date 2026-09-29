"""Frozen Gaussian ST-SVGP state continuation, qualified against BayesNewton.

Use the official kernel's transitions, stationary covariance and spatial
conditional. For the Gaussian likelihood, unit natural-gradient VI gives
pseudo precision B.T B / noise and natural mean B.T y / noise. A Kalman
update with measurement B H and noise*I is the same posterior operation.
Spatial residual C enters prediction, not the variational site precision.
"""
import numpy as np


class GaussianSTFilter:
    def __init__(self, kernel, likelihood, coordinates):
        import jax
        import jax.numpy as jnp
        self.kernel=kernel
        self.noise=float(likelihood.variance)
        self.coordinates=np.asarray(coordinates)
        b,c=kernel.spatial_conditional(jnp.array([[0.]]),jnp.asarray(coordinates)[None])
        self.h=b[0]@kernel.measurement_model()
        self.conditional=jnp.diag(c[0])
        self.pinf=kernel.stationary_covariance()
        self.mean=jnp.zeros((len(self.pinf),1))
        self.covariance=self.pinf
        self.time=None
        self.previous=None
        self.before_previous=None
        self.previous_dt=0.
        def step(mean,covariance,dt,sites,values):
            a=kernel.state_transition(dt)
            q=self.pinf-a@self.pinf@a.T
            mean=a@mean
            covariance=a@covariance@a.T+q
            h=self.h[sites]
            hp=h@covariance
            innovation=hp@h.T+self.noise*jnp.eye(len(sites))
            gain=jnp.linalg.solve(innovation,hp).T
            mean=mean+gain@(values[:,None]-h@mean)
            covariance=covariance-gain@hp
            return mean,(covariance+covariance.T)*.5
        self.step=jax.jit(step)

    def advance(self,time,sites,values,*,delayed_time=None,delayed_sites=(),delayed_values=()):
        import jax.numpy as jnp
        time=float(time)
        if not np.isfinite(time) or (self.time is not None and time<=self.time):
            raise ValueError('Strictly increasing finite times required')
        def checked(s,v):
            s=np.asarray(s);v=np.asarray(v,dtype=float).reshape(-1)
            if (s.ndim!=1 or not np.issubdtype(s.dtype,np.integer) or len(s)!=len(v)
                or len(np.unique(s))!=len(s) or np.any(s<0) or np.any(s>=len(self.coordinates))
                or not np.isfinite(v).all()):raise ValueError('Invalid released observation')
            return s,v
        sites,values=checked(sites,values)
        if delayed_time is not None:
            if self.time is None or float(delayed_time)!=self.time:
                raise ValueError('Only immediately previous-step releases are supported')
            ds,dv=checked(delayed_sites,delayed_values)
            ps,pv=self.previous
            if np.intersect1d(ds,ps).size:raise ValueError('Duplicate delayed observation')
            self.mean,self.covariance=self.step(*self.before_previous,self.previous_dt,
                jnp.asarray(np.concatenate([ps,ds])),jnp.asarray(np.concatenate([pv,dv])))
        elif len(delayed_sites) or len(delayed_values):raise ValueError('Missing delayed time')
        self.before_previous=(self.mean,self.covariance)
        self.previous_dt=0. if self.time is None else time-self.time
        self.mean,self.covariance=self.step(self.mean,self.covariance,self.previous_dt,
            jnp.asarray(sites),jnp.asarray(values))
        self.time=time;self.previous=(sites.copy(),values.copy())

    def predict(self,sites):
        import jax.numpy as jnp
        h=self.h[np.asarray(sites,dtype=int)]
        mean=h@self.mean
        variance=jnp.sum((h@self.covariance)*h,axis=1)+self.conditional[sites]+self.noise
        return np.asarray(mean).reshape(-1),np.asarray(variance)
