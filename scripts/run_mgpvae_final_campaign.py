#!/usr/bin/env python3
"""Per-split Task-1 selection followed by full causal COVID MGPVAE evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--dataset',choices=['covid','pems','era5'],default='covid')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--release',required=True)
    p.add_argument('--compute-root',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    worker=str(a.compute_root/'env-jax-gpu/bin/python')
    os.environ['MGPVAE_SOURCE']=str(a.compute_root/'official/MGPVAE')
    protocol=a.compute_root/f"protocol/{'covid-v2' if a.dataset=='covid' else a.dataset}/seed{a.seed}"
    expected_steps,expected_sites,initial_sites={'covid':(143,10,52),'pems':(50100,65,260),'era5':(1674,200,800)}[a.dataset]
    inputs=[str(protocol/'protocol.npz'),str(protocol/'protocol.json')]
    common=['--protocol-npz',inputs[0],'--protocol-json',inputs[1],'--seed',str(a.seed),
        '--official-source',os.environ['MGPVAE_SOURCE'],'--prediction-samples','512',
        '--protocol-kind',{'covid':'covid','pems':'traffic','era5':'era5'}[a.dataset],
        '--metric-backend','numpy' if a.dataset=='covid' else 'jax']
    if a.dataset!='covid':common += ['--rematerialize-scans','--initial-marginals','direct']
    tests=['tests/test_mgpvae_spatial_moments.py','tests/test_mgpvae_selected.py',
        'tests/test_mgpvae_partial.py','tests/test_mgpvae_official_filter.py','tests/test_mgpvae_mixture_metrics.py',
        'tests/test_mgpvae_gpu_metrics.py','tests/test_mgpvae_memory.py',
        'tests/test_mgpvae_initial_marginals.py']
    if a.dataset=='era5':tests.append('tests/test_era5_information_boundary.py')
    with (a.output/'tests.txt').open('w') as log:
        subprocess.run([worker,'-m','pytest','-q',*tests],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([worker,'-c',"import jax; assert jax.default_backend() == 'gpu'"],check=True)
    def run(output,stage,capacity,budget,selection=None,qualification=False):
        output.mkdir(parents=True,exist_ok=True)
        spec=dict(entity='harrisonzhu',project='KronHiPPO-STGP',campaign='fair-three-domain-wandb-20260929',
            dataset=a.dataset,method='mgpvae',split_seed=a.seed,training_seed=a.seed,stage=stage,
            worker_python=worker,source_commit=a.release,input_files=inputs,latent=capacity,width=16,
            max_iterations=budget,decoder_samples=512,metric_backend='numpy' if a.dataset=='covid' else 'jax',predictive_family='Gaussian decoder mixture',
            covariance_pushforward_corrected=True,main_table_admitted=False)
        spec['scan_rematerialization']=a.dataset!='covid'
        spec['initial_marginals']='reference' if a.dataset=='covid' else 'direct'
        if stage=='final':
            spec.update(qualification_record=str(a.output/'qualification.json'),expected_steps=expected_steps,
                expected_sites=expected_sites,hidden_delay_steps=None if a.dataset=='era5' else 1,initial_observed_sites=initial_sites)
        elif qualification:
            spec.update(expected_steps=3,expected_sites=expected_sites,hidden_delay_steps=None if a.dataset=='era5' else 1)
        (output/'spec.json').write_text(json.dumps(spec,indent=2))
        command=[worker,'scripts/run_mgpvae_causal.py',*common,'--output-dir',str(output),
            '--latent',str(capacity),'--iterations',str(budget),'--check-every','25']
        if qualification:command += ['--max-blocks','3','--check-initial-marginal-prefix']
        else:command += ['--validation-only'] if selection is None else ['--selection-json',str(selection)]
        subprocess.run([sys.executable,'scripts/run_tracked_experiment.py','--spec',str(output/'spec.json'),
            '--output',str(output),'--',*command],check=True)
    if a.dataset!='covid':
        for capacity in [2,4]:
            run(a.output/f'full-initial-resource-pilot-latent{capacity}',
                'qualification',capacity,1,qualification=True)
    candidates=[]
    for capacity in [2,4]:
        for budget in [500,1000]:
            output=a.output/f'calibration-latent{capacity}-budget{budget}'
            run(output,'validation',capacity,budget)
            result=json.loads((output/'calibration.json').read_text())
            if result['selected']['step'] < budget: break
        if result['selected']['step'] >= budget:
            raise RuntimeError('Validation optimum remains at budget boundary; do not certify split')
        candidates.append(dict(result=result,path=str(output/'calibration.json')))
    selected=min(candidates,key=lambda r:r['result']['selected']['validation']['nlpd'])
    (a.output/'selection.json').write_text(json.dumps(dict(candidates=candidates,selected=selected),indent=2))
    (a.output/'qualification.json').write_text(json.dumps(dict(status='passed',method='mgpvae',dataset=a.dataset,
        source_commit=a.release,tests=tests,online_parameters='frozen',
        covariance_pushforward_corrected=True,main_table_admitted=False),indent=2))
    run(a.output,'final',selected['result']['latent'],selected['result']['selected']['step'],selected['path'])


if __name__=='__main__':main()
