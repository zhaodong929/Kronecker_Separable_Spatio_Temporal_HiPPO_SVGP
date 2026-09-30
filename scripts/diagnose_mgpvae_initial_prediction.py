#!/usr/bin/env python3
"""Capture eager/compiled initial-prediction intermediates; never admits a run."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.mgpvae.official import make_model
from baselines.traffic_protocol_n import load_protocol
from baselines.causal_mean import get_mean,select_initial_targets
from baselines.mgpvae.initial import make_initial_predictor

p=argparse.ArgumentParser()
p.add_argument('--compute-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--require-gpu',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
protocol=load_protocol(a.compute_root/'protocol/era5/seed0/protocol.npz',a.compute_root/'protocol/era5/seed0/protocol.json',protocol_kind='era5')
mean=get_mean(protocol);sites=protocol.fit_locations;times=protocol.calibration_times
residual=select_initial_targets(protocol.task1(),sites)-np.stack([mean.at(t,sites) for t in times])
model=make_model(a.compute_root/'official/MGPVAE',protocol.coordinates[sites],seed=0,latent=4,
    correct_spatial_covariance=True,compact_spatial_marginals=True,sitewise_training_filter=True)
import jax
import jax.numpy as jnp
from baselines.mgpvae.memory import enable_scan_rematerialization
enable_scan_rematerialization()
if a.require_gpu:assert jax.default_backend()=='gpu'
t=jnp.asarray(((times-times[0])/(times[-1]-times[0]))[:3,None])
y=jnp.asarray(residual.T[:,:3,None]);q=jnp.asarray(protocol.coordinates[protocol.validation_locations])
# Instrument an isolated copy of the inspected function, leaving production untouched.
source=inspect.getsource(make_initial_predictor)
source=source.replace('return latent_mean,latent_variance','return latent_mean,latent_variance,fm,fp,mean,variance,mixing,residual,forward,backward,pinf')
namespace={'np':np};exec(source,namespace)
compiled=namespace['make_initial_predictor'](model,t,y,q)
namespace_eager={'np':np};exec(source.replace('return objax.Jit(predict)','return predict'),namespace_eager)
eager=namespace_eager['make_initial_predictor'](model,t,y,q)
keys=['mean','variance','filtered_mean','filtered_cov','bridge_mean','bridge_var','mixing','residual','forward','backward','pinf']
arrays={};report={'backend':jax.default_backend(),'main_table_admitted':False,'comparisons':{}}
for label,fn in [('compiled',compiled),('eager',eager),('official',lambda:model.predict(t,t,y,q))]:
    values=fn()
    for key,value in zip(keys,values):arrays[label+'_'+key]=np.asarray(value)
    np.savez_compressed(a.output/'intermediates.npz',**arrays)
    print(json.dumps({'completed':label,'first_mean':arrays[label+'_mean'].reshape(-1)[:4].tolist()}),flush=True)
for key in keys:
    x=arrays['compiled_'+key];z=arrays['eager_'+key]
    report['comparisons'][key]={'max_absolute_error':float(np.max(abs(x-z))),
        'allclose':bool(np.allclose(x,z,rtol=1e-6,atol=1e-7))}
for key in keys[:2]:
    x=arrays['compiled_'+key];z=arrays['official_'+key].reshape(x.shape)
    report['comparisons']['official_'+key]={'max_absolute_error':float(np.max(abs(x-z))),
        'allclose':bool(np.allclose(x,z,rtol=1e-6,atol=1e-7))}
(a.output/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
