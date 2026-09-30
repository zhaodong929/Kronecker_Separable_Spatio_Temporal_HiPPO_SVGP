"""Prefix-normalized meteorological covariates shared by every ERA5 method."""
import numpy as np

def features(weather,coordinates,fit,initial=186):
    """Same 7+126 feature family, continuous time and prefix-only scaling."""
    weather=np.asarray(weather,dtype=float)
    lagged=[weather[np.maximum(np.arange(len(weather))-lag,0)] for lag in range(1,11)]
    dynamic=np.concatenate([weather,*lagged,*[weather-lag for lag in lagged]],axis=-1)
    center=dynamic[:initial,fit].mean((0,1));scale=np.maximum(dynamic[:initial,fit].std((0,1)),1e-8)
    dynamic=(dynamic-center)/scale
    times=np.arange(len(weather))/(initial-1);phase=2*np.pi*times
    static=np.broadcast_to(coordinates,(len(times),*coordinates.shape))
    temporal=np.stack([np.ones(len(times)),np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)],axis=-1)
    phi=np.concatenate([np.broadcast_to(temporal[:,None,:],(*weather.shape[:2],5)),static,dynamic],axis=-1).astype(np.float32)
    return phi,dict(mean=center.tolist(),scale=scale.tolist(),lag_hours=10,fit_scope='initial fitting sites only',time_scale_hours=initial-1,initial_lag_rule='repeat first available exogenous value')
