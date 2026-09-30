#!/usr/bin/env python3
"""Full Task-1 real-data loss/gradient parity before enabling sitewise filtering."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.mgpvae.official import make_model,import_official
from baselines.traffic_protocol_n import load_protocol
from baselines.causal_mean import get_mean,select_initial_targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--protocol-kind',choices=['traffic','era5'],required=True)
    p.add_argument('--official-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-gpu',action='store_true')
    p.add_argument('--max-sites',type=int,default=0)
    p.add_argument('--max-times',type=int,default=0)
    p.add_argument('--cpu-reference',action='store_true')
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.require_gpu and not a.cpu_reference:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--protocol',str(a.protocol),
            '--protocol-kind',a.protocol_kind,'--official-source',str(a.official_source),
            '--output',str(a.output.with_name('cpu-reference.json')),'--cpu-reference',
            '--max-sites',str(a.max_sites),'--max-times',str(a.max_times)],check=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu',
                 'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    protocol=load_protocol(a.protocol/'protocol.npz',a.protocol/'protocol.json',protocol_kind=a.protocol_kind)
    mean=get_mean(protocol);sites=protocol.fit_locations
    if a.max_sites:sites=sites[:a.max_sites]
    times=protocol.calibration_times
    residual=select_initial_targets(protocol.task1(),sites)-np.stack([mean.at(t,sites) for t in times])
    scaled=(times-times[0])/(times[-1]-times[0])
    if a.max_times:scaled=scaled[:a.max_times];residual=residual[:a.max_times]
    import_official(a.official_source)
    import jax
    import jax.numpy as jnp
    import objax
    from baselines.mgpvae.memory import enable_scan_rematerialization
    enable_scan_rematerialization()
    if a.require_gpu:assert jax.default_backend()=='gpu'
    if a.cpu_reference:assert jax.default_backend()=='cpu'
    rows=[]
    for latent in [2,4]:
        captured=[]
        for sitewise in ([True] if a.cpu_reference else [False,True]):
            model=make_model(a.official_source,protocol.coordinates[sites],seed=17,latent=latent,
                correct_spatial_covariance=True,compact_spatial_marginals=True,sitewise_training_filter=sitewise)
            y=jnp.asarray(residual.T[...,None]);t=jnp.asarray(scaled[:,None])
            derivative=objax.GradValues(model.energy,model.vars())
            @objax.Function.with_vars(model.vars())
            def evaluate():return derivative(y,jax.random.PRNGKey(12),t=t,num_samples=4)
            compiled=objax.Jit(evaluate)
            started=time.perf_counter()
            grads,loss=compiled();grads=[np.asarray(g) for g in grads];loss=np.asarray(loss)
            compile_and_first=time.perf_counter()-started
            started=time.perf_counter()
            warm_grads,warm_loss=compiled()
            warm_grads=[np.asarray(g) for g in warm_grads];warm_loss=np.asarray(warm_loss)
            warm_seconds=time.perf_counter()-started
            np.testing.assert_allclose(warm_loss,loss,rtol=1e-12,atol=1e-12)
            captured.append((grads,loss,dict(sitewise=sitewise,compile_and_first_seconds=compile_and_first,warm_seconds=warm_seconds)))
            del compiled,evaluate,derivative,model,y,t,warm_grads,warm_loss
            jax.clear_caches();gc.collect()
        if a.cpu_reference:
            grads,loss,timing=captured[0]
            np.savez_compressed(a.output.with_name(f'cpu-reference-latent{latent}.npz'),loss=loss,
                **{f'gradient_{i}':g for i,g in enumerate(grads)})
            rows.append(dict(latent=latent,timing=timing));continue
        rg,rl,rt=captured[0];fg,fl,ft=captured[1]
        np.testing.assert_allclose(fl,rl,rtol=1e-10,atol=1e-10)
        assert len(rg)==len(fg)
        for x,y in zip(fg,rg):np.testing.assert_allclose(x,y,rtol=1e-9,atol=1e-10)
        row=dict(latent=latent,loss_max_absolute_error=float(np.max(np.abs(fl-rl))),
            gradient_max_absolute_error=max(float(np.max(np.abs(x-y))) for x,y in zip(fg,rg)),
            reference_timing=rt,sitewise_timing=ft)
        if a.require_gpu:
            with np.load(a.output.with_name(f'cpu-reference-latent{latent}.npz')) as cpu:
                np.testing.assert_allclose(fl,cpu['loss'],rtol=1e-10,atol=1e-10)
                assert len(cpu.files)==len(fg)+1
                for i,g in enumerate(fg):
                    np.testing.assert_allclose(g,cpu[f'gradient_{i}'],rtol=1e-9,atol=1e-10)
                row['cpu_reference_loss_max_absolute_error']=float(np.max(abs(fl-cpu['loss'])))
                row['cpu_reference_gradient_max_absolute_error']=max(float(np.max(abs(g-cpu[f'gradient_{i}']))) for i,g in enumerate(fg))
        rows.append(row);print(json.dumps(row),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(status='passed',backend=jax.default_backend(),sites=len(sites),
        times=len(scaled),full_task1=not(a.max_sites or a.max_times),comparisons=rows,main_table_admitted=False),indent=2))


if __name__=='__main__':main()
