#!/usr/bin/env python3
"""Bounded real-PEMS Task-1-only qualification; never a main-table result."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from baselines.mgpvae.official import make_model, COMMIT
from baselines.mgpvae.filtering import OfficialVisibleFilter
from benchmarks.three_domain.metrics import gaussian_mixture_metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--h5',type=Path,required=True)
    ap.add_argument('--coordinates',type=Path,required=True)
    ap.add_argument('--official-source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=20)
    ap.add_argument('--seed',type=int,default=1)
    args=ap.parse_args()
    if args.steps < 1 or args.steps > 200:
        ap.error('qualification steps must be in 1..200')
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    status=dict(stage='qualification',main_table_admitted=False,status='running',official_commit=COMMIT)
    (args.output/'status.json').write_text(json.dumps(status,indent=2))
    import h5py
    import pandas as pd
    with h5py.File(args.h5) as f:
        group=next(v for v in f.values() if hasattr(v,'keys') and 'block0_values' in v)
        raw=np.asarray(group['block0_values'][:64],dtype=float)
        ids=[str(x.decode() if isinstance(x,bytes) else x) for x in group['axis0'][:]]
    coords=pd.read_csv(args.coordinates,header=None)
    lookup={str(int(row[0])):np.asarray(row[1:3],dtype=float) for row in coords.to_numpy()}
    r=np.asarray([lookup[k] for k in ids])
    r=(r-r.mean(axis=0))/r.std(axis=0)
    rng=np.random.default_rng(args.seed)
    hidden=np.sort(rng.choice(325,65,replace=False))
    visible=np.setdiff1d(np.arange(325),hidden)
    validation=np.sort(rng.choice(visible,26,replace=False))
    fit=np.setdiff1d(visible,validation)[:8]
    query=validation[:2]
    offset=raw[:48,fit].mean(); scale=raw[:48,fit].std()
    y=(raw-offset)/scale
    model=make_model(args.official_source,r[fit],seed=args.seed)
    import jax
    import jax.numpy as jnp
    import objax
    training=jnp.asarray(y[:48,fit].T[...,None])
    t=jnp.arange(48,dtype=float)[:,None]/12.
    optimizer=objax.optimizer.Adam(model.vars())
    grad=objax.GradValues(model.energy,model.vars())
    @objax.Function.with_vars(model.vars()+optimizer.vars())
    def update(i):
        gradients, losses=grad(training,jax.random.PRNGKey(i),t=t,num_samples=1)
        optimizer(.001,gradients)
        return losses
    update=objax.Jit(update)
    trace=[]
    for i in range(args.steps):
        losses=[float(v) for v in update(i)]
        if not np.isfinite(losses).all():raise RuntimeError('Nonfinite official training objective')
        trace.append(dict(step=i+1,negative_elbo=losses[0],elapsed_seconds=time.monotonic()-start))
        print(json.dumps(trace[-1]),flush=True)
    adapter=OfficialVisibleFilter(model)
    for i in range(48):adapter.observe(i/12.,y[i,fit])
    scores=[];pred=[]
    for i in range(48,64):
        adapter.observe(i/12.,y[i,fit])
        cm,cv=adapter.gaussian_components(r[query],seed=100000+args.seed*100+i)
        scores.append(gaussian_mixture_metrics(y[i,query],cm,cv));pred.append(cm)
    np.savez_compressed(args.output/'predictions.npz',component_means=np.stack(pred),
        noise_variance=cv,y_true=y[48:64,query],test_indices=query,times=np.arange(48,64)/12.)
    result=dict(status,status='complete',purpose='real-data integration and timing only',
        elapsed_seconds=time.monotonic()-start,backend=jax.default_backend(),steps=args.steps,
        train_sites=fit.tolist(),validation_sites=query.tolist(),train_times=48,validation_times=16,
        full_task1_steps=2016,formal_stream_used=False,convergence_claim=False,
        metrics={k:float(np.mean([s[k] for s in scores])) for k in ('nlpd','crps')},
        normalization=dict(mean=float(offset),scale=float(scale),fit='first 48 times and 8 training sites'),
        data_sha256=hashlib.sha256(args.h5.read_bytes()).hexdigest(),trace=trace)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'status.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)

if __name__=='__main__':main()
