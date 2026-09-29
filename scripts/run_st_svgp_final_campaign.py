#!/usr/bin/env python3
"""Per-split validation, frozen selection, refit and complete COVID comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--release',required=True)
    p.add_argument('--compute-root',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    worker=str(a.compute_root/'env-jax-gpu/bin/python')
    protocol=a.compute_root/f'protocol/covid-v2/seed{a.seed}'
    inputs=[str(protocol/'protocol.npz'),str(protocol/'protocol.json')]
    common=['--protocol-npz',inputs[0],'--protocol-json',inputs[1],'--seed',str(a.seed)]
    tests=['tests/test_st_svgp_dense.py','tests/test_st_svgp_release_boundary.py','tests/test_causal_mean.py']
    with (a.output/'tests.txt').open('w') as log:
        subprocess.run([worker,'-m','pytest','-q',*tests],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([worker,'-c',"import jax; assert jax.default_backend() == 'gpu'"],check=True)
    qualification=dict(status='passed',method='st_svgp',dataset='covid',source_commit=a.release,
        tests=tests,online_budget='One unit natural update: Gaussian conjugate posterior, checked against dense GP',
        main_table_admitted=False)
    (a.output/'qualification.json').write_text(json.dumps(qualification,indent=2))
    def run(output,stage,capacity,iterations,validation):
        output.mkdir(parents=True,exist_ok=True)
        spec=dict(entity='harrisonzhu',project='KronHiPPO-STGP',campaign='fair-three-domain-wandb-20260929',
            dataset='covid',method='st_svgp',split_seed=a.seed,training_seed=a.seed,stage=stage,
            worker_python=worker,source_commit=a.release,input_files=inputs,
            spatial_inducing=capacity,max_iterations=iterations,history_window=0,
            timing_scope='Exclusive allocated A30; complete arrived-history replay including cache release',
            main_table_admitted=False)
        if stage=='final':
            spec.update(qualification_record=str(a.output/'qualification.json'),expected_steps=143,
                expected_sites=10,hidden_delay_steps=1,initial_observed_sites=52,predictive_family='gaussian')
        (output/'spec.json').write_text(json.dumps(spec,indent=2))
        command=[worker,'baselines/covid_long_setting_b/adapters/run_st_svgp.py',*common,
            '--output-dir',str(output),'--spatial-inducing',str(capacity),'--task1-iterations',str(iterations),
            '--task1-min-steps',str(iterations),'--task1-check-interval','5',
            '--online-inference-steps','1','--history-window','0']
        if validation: command.append('--task1-validation-only')
        else: command += ['--write-frozen-task1-state',str(output/'task1-frozen.npz')]
        subprocess.run([sys.executable,'scripts/run_tracked_experiment.py','--spec',str(output/'spec.json'),
            '--output',str(output),'--',*command],check=True)
    candidates=[]
    for capacity in [16,32]:
        budget=500
        output=a.output/f'calibration-ms{capacity}-budget{budget}'
        run(output,'validation',capacity,budget,True)
        result=json.loads((output/'task1_validation.json').read_text())
        if result['selected_iteration'] >= budget:
            budget=1000
            output=a.output/f'calibration-ms{capacity}-budget{budget}'
            run(output,'validation',capacity,budget,True)
            result=json.loads((output/'task1_validation.json').read_text())
        if result['selected_iteration'] >= budget:
            raise RuntimeError('Validation optimum is still at the maximum budget; do not certify this split')
        candidates.append(result)
    selected=min(candidates,key=lambda x:x['metrics']['gaussian_nlpd'])
    (a.output/'selection.json').write_text(json.dumps(dict(selection='Minimum Task-1 validation NLPD only',
        candidates=candidates,selected=selected),indent=2))
    run(a.output,'final',selected['capacity']['spatial_inducing'],selected['selected_iteration'],False)


if __name__=='__main__':main()
