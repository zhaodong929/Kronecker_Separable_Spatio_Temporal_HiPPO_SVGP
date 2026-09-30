#!/usr/bin/env python3
"""Qualify exact continuation on real PEMS before its full final comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['pems','era5'],default='pems')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--release',required=True)
    p.add_argument('--compute-root',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    worker=str(a.compute_root/'env-jax-gpu/bin/python')
    protocol=a.compute_root/f'protocol/{a.dataset}/seed{a.seed}'
    inputs=[str(protocol/'protocol.npz'),str(protocol/'protocol.json')]
    common=['--protocol-npz',inputs[0],'--protocol-json',inputs[1],'--seed',str(a.seed),
            '--protocol-kind','era5' if a.dataset=='era5' else 'traffic']
    tests=['tests/test_st_svgp_dense.py','tests/test_st_svgp_release_boundary.py',
           'tests/test_st_svgp_stateful.py','tests/test_causal_mean.py']
    if a.dataset=='era5':tests.append('tests/test_era5_information_boundary.py')
    with (a.output/'tests.txt').open('w') as f:
        subprocess.run([worker,'-m','pytest','-q',*tests],stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([worker,'-c',"import jax; assert jax.default_backend()=='gpu'"],check=True)
    def run(output,stage,capacity,iterations,backend='stateful',validation=False,extra=()):
        output.mkdir(parents=True,exist_ok=True)
        spec=dict(entity='harrisonzhu',project='KronHiPPO-STGP',campaign='fair-three-domain-wandb-20260929',
            dataset=a.dataset,method='st_svgp',split_seed=a.seed,training_seed=a.seed,stage=stage,
            worker_python=worker,source_commit=a.release,input_files=inputs,
            spatial_inducing=capacity,max_iterations=iterations,online_backend=backend,
            history_window=0,main_table_admitted=False)
        if stage=='final':
            spec.update(qualification_record=str(a.output/'qualification.json'),expected_steps=1674 if a.dataset=='era5' else 50100,
                expected_sites=200 if a.dataset=='era5' else 65,hidden_delay_steps=None if a.dataset=='era5' else 1,initial_observed_sites=800 if a.dataset=='era5' else 260,predictive_family='gaussian')
        elif stage=='qualification':
            spec.update(expected_steps=3,expected_sites=200 if a.dataset=='era5' else 65,hidden_delay_steps=None if a.dataset=='era5' else 1)
        (output/'spec.json').write_text(json.dumps(spec,indent=2))
        command=[worker,'baselines/covid_long_setting_b/adapters/run_st_svgp.py',*common,
            '--output-dir',str(output),'--spatial-inducing',str(capacity),'--task1-iterations',str(iterations),
            '--task1-min-steps',str(iterations),'--task1-check-interval','5',
            '--online-inference-steps','1','--history-window','0','--online-backend',backend,*extra]
        if validation:command.append('--task1-validation-only')
        subprocess.run([sys.executable,'scripts/run_tracked_experiment.py','--spec',str(output/'spec.json'),
            '--output',str(output),'--',*command],check=True)
    reference=a.output/'qualification-replay'
    frozen=reference/'task1-frozen.npz'
    run(reference,'qualification',16,1,'replay',extra=['--max-weeks','3','--write-frozen-task1-state',str(frozen)])
    continuation=a.output/'qualification-stateful'
    run(continuation,'qualification',16,1,extra=['--max-weeks','3','--frozen-task1-state',str(frozen)])
    errors={}
    with np.load(reference/'predictions.npz') as ref,np.load(continuation/'predictions.npz') as actual:
        for key in ['pred_mean','pred_var']:
            np.testing.assert_allclose(actual[key],ref[key],rtol=1e-5,atol=1e-6)
            errors[key]=float(np.max(np.abs(actual[key]-ref[key])))
        for key in ['y_true','test_indices','times']:
            np.testing.assert_array_equal(actual[key],ref[key])
    (a.output/'continuation-equivalence.json').write_text(json.dumps(errors,indent=2))
    candidates=[]
    for capacity in [16,32]:
        budget=500
        output=a.output/f'calibration-ms{capacity}-b{budget}'
        run(output,'validation',capacity,budget,validation=True)
        result=json.loads((output/'task1_validation.json').read_text())
        if result['selected_iteration']>=budget:
            budget=1000;output=a.output/f'calibration-ms{capacity}-b{budget}'
            run(output,'validation',capacity,budget,validation=True)
            result=json.loads((output/'task1_validation.json').read_text())
        result['requires_larger_budget']=result['selected_iteration']>=budget
        candidates.append(result)
    selected=min(candidates,key=lambda c:c['metrics']['gaussian_nlpd'])
    (a.output/'selection.json').write_text(json.dumps(dict(candidates=candidates,selected=selected),indent=2))
    if selected['requires_larger_budget']:
        raise RuntimeError('Selected PEMS initial fit improves at budget boundary; extend qualification')
    qualification=dict(status='passed',method='st_svgp',dataset=a.dataset,source_commit=a.release,
        tests=tests,real_protocol_prefix_max_absolute_error=errors,
        online_budget='One conjugate Gaussian update, equivalent to official legal-prefix replay',
        main_table_admitted=False)
    (a.output/'qualification.json').write_text(json.dumps(qualification,indent=2))
    run(a.output,'final',selected['capacity']['spatial_inducing'],selected['selected_iteration'],
        extra=['--write-frozen-task1-state',str(a.output/'task1-frozen.npz')])


if __name__=='__main__':main()
