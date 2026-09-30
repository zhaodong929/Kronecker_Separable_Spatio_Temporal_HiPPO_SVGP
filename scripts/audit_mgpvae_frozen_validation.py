#!/usr/bin/env python3
"""Recompute selected candidate scores on CPU from saved model variables."""
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.mgpvae.official import make_model
from baselines.mgpvae.initial import make_initial_predictor
from baselines.traffic_protocol_n import load_protocol
from baselines.causal_mean import get_mean,select_initial_targets
from benchmarks.three_domain.metrics import gaussian_mixture_metrics
p=argparse.ArgumentParser()
p.add_argument('--compute-root',type=Path,required=True)
p.add_argument('--dataset',choices=['pems','era5'],required=True)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();selection=json.loads((a.run/'selection.json').read_text())
first=selection['candidates'][0]['result'];seed=first['split_seed']
protocol_root=a.compute_root/f'protocol/{a.dataset}/seed{seed}'
protocol=load_protocol(protocol_root/'protocol.npz',protocol_root/'protocol.json',
    protocol_kind={'pems':'traffic','era5':'era5'}[a.dataset])
mean=get_mean(protocol);sites=protocol.fit_locations;times=protocol.calibration_times
residual=select_initial_targets(protocol.task1(),sites)-np.stack([mean.at(t,sites) for t in times])
offsets=np.stack([mean.at(t,protocol.validation_locations) for t in times])
truth=protocol.calibration_targets(protocol.validation_locations)
rows=[]
for candidate in selection['candidates']:
    result=candidate['result'];latent=result['latent'];started=time.perf_counter()
    model=make_model(a.compute_root/'official/MGPVAE',protocol.coordinates[sites],seed=seed,latent=latent,
        correct_spatial_covariance=True,compact_spatial_marginals=True,sitewise_training_filter=True)
    import jax
    import jax.numpy as jnp
    assert jax.default_backend()=='cpu'
    from baselines.mgpvae.memory import enable_scan_rematerialization
    enable_scan_rematerialization()
    checkpoint=Path(candidate['path']).parent/'selected_fit_model.npz'
    with np.load(checkpoint,allow_pickle=False) as saved:
        assert list(saved['variable_names'])==list(model.vars().keys())
        assert int(saved['selected_iteration'])==result['selected']['step']
        model.vars().assign([jnp.asarray(saved[f'variable_{i}']) for i in range(len(model.vars()))])
    t=jnp.asarray(((times-times[0])/(times[-1]-times[0]))[:,None]);y=jnp.asarray(residual.T[...,None])
    lm,lv=make_initial_predictor(model,t,y,jnp.asarray(protocol.coordinates[protocol.validation_locations]))()
    assert np.isfinite(lv).all() and np.all(np.asarray(lv)>0)
    z=lm[None]+jnp.sqrt(lv)[None]*jax.random.normal(jax.random.PRNGKey(seed+100000),(512,*lm.shape))
    components=np.concatenate([np.asarray(model.likelihood.decoder(z[i:i+16])[...,0]).transpose(0,2,1) for i in range(0,512,16)])
    components=(components+offsets[None]).reshape(512,-1)
    metrics=gaussian_mixture_metrics(truth.ravel(),components,float(model.likelihood.variance),compute_crps=False)
    reference=result['selected']['validation'];errors={k:abs(metrics[k]-reference[k]) for k in ['rmse','nlpd']}
    passed=all(np.isclose(metrics[k],reference[k],rtol=1e-6,atol=1e-7) for k in errors)
    rows.append(dict(latent=latent,selected_iteration=result['selected']['step'],checkpoint=str(checkpoint),
        cpu_scores={k:metrics[k] for k in errors},recorded_gpu_scores={k:reference[k] for k in errors},
        absolute_errors=errors,passed=passed,elapsed_seconds=time.perf_counter()-started))
    report=dict(status='passed' if all(r['passed'] for r in rows) else 'requires_review',completed_candidates=len(rows),
        expected_candidates=len(selection['candidates']),comparisons=rows,rtol=1e-6,atol=1e-7,main_table_admitted=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(rows[-1]),flush=True)
    del model,z,components,lm,lv,y,t
    jax.clear_caches();gc.collect()
if not all(r['passed'] for r in rows):raise SystemExit(1)
