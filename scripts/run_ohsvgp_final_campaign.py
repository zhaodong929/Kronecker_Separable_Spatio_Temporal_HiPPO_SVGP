#!/usr/bin/env python3
"""Own spatial validation and Task-1 chronological online-budget selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from benchmarks.three_domain.convergence import initial_budget_resolved
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--dataset',choices=['covid','pems','era5'],default='covid')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--release',required=True)
    p.add_argument('--compute-root',type=Path,required=True)
    p.add_argument('--reuse-validation-root',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    os.environ['HIPPO_TEST_DEVICE']='cuda'
    worker=str(a.compute_root/'env-routeb/bin/python')
    source=a.compute_root/f"protocol/{'covid-v2' if a.dataset=='covid' else a.dataset}/seed{a.seed}"
    expected_steps,expected_sites,initial_sites={'covid':(143,10,52),'pems':(50100,65,260),'era5':(1674,200,800)}[a.dataset]
    fold_function={'covid':'build_fold','pems':'build_pems_fold','era5':'build_era5_fold'}[a.dataset]
    tests=['tests/test_ohsvgp_release_boundary.py','tests/test_ohsvgp_lengthscale_learning.py',
           'tests/test_online_budget_fold.py','tests/test_initial_budget_gate.py']
    if a.dataset!='covid':tests.append('tests/test_ohsvgp_prediction_cache.py')
    if a.dataset=='era5':tests += ['tests/test_era5_information_boundary.py','tests/test_era5_adapter_boundary.py::test_actual_era5_no_release_adapter[ohsvgp]']
    with (a.output/'tests.txt').open('w') as f:
        subprocess.run([worker,'-m','pytest','-q',*tests],stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([worker,'-c','import torch; assert torch.cuda.is_available()'],check=True)
    fold=a.output/'online-budget-protocol'
    # Use the numerical environment, which has the protocol builder dependencies.
    subprocess.run([worker,'-c',
        f'from benchmarks.three_domain.online_validation import {fold_function}; import sys; {fold_function}(sys.argv[1],sys.argv[2])',
        str(source/'protocol.npz'),str(fold)],check=True)
    def run(output,stage,capacity,budget,updates,protocol,calibration_only=False,online_lr=None):
        output.mkdir(parents=True,exist_ok=True)
        spec=dict(entity='harrisonzhu',project='KronHiPPO-STGP',campaign='fair-three-domain-wandb-20260929',
            dataset=a.dataset,method='ohsvgp',split_seed=a.seed,training_seed=a.seed,stage=stage,
            source_commit=a.release,worker_python=worker,
            input_files=[str(protocol/'protocol.npz'),str(protocol/'protocol.json')],
            inducing_size=capacity,max_iterations=budget,online_update_steps=updates,
            main_table_admitted=False)
        if a.dataset!='covid':spec.update(online_learning_rate=online_lr or .001,online_batch_size=1024)
        if a.reuse_validation_root is not None and stage=='validation':
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
            from benchmarks.three_domain.reuse_validation import reuse_ohsvgp
            cached=reuse_ohsvgp(a.reuse_validation_root/output.name,output,spec,a.compute_root,
                Path(__file__).resolve().parents[1],'calibration.json' if calibration_only else 'result.json')
            if cached is not None:return cached
        if stage=='final':
            spec.update(qualification_record=str(a.output/'qualification.json'),expected_steps=expected_steps,
                expected_sites=expected_sites,hidden_delay_steps=None if a.dataset=='era5' else 1,initial_observed_sites=initial_sites,predictive_family='gaussian')
        (output/'spec.json').write_text(json.dumps(spec,indent=2))
        entry='scripts/run_covid_ohsvgp_own_theta.py' if a.dataset=='covid' else 'scripts/run_ohsvgp_cached_prediction.py'
        command=[worker,entry,'--protocol-npz',str(protocol/'protocol.npz'),
            '--protocol-json',str(protocol/'protocol.json'),'--output-dir',str(output),'--seed',str(a.seed),
            '--protocol-kind',{'covid':'covid','pems':'traffic','era5':'era5'}[a.dataset],
            '--kernel','rbf','--inducing-size',str(capacity),'--rff-sample-size','256',
            '--basis-grid-size','1024','--calibration-iterations',str(budget),
            '--task1-min-steps',str(budget),'--task1-check-interval','5',
            '--calibration-batch-size','256','--update-steps',str(updates),'--device','cuda']
        if a.dataset!='covid':command += ['--online-learning-rate',str(online_lr or .001),'--online-batch-size','1024']
        if a.dataset!='era5':command.append('--delayed-observations')
        if calibration_only:command.append('--calibration-only')
        subprocess.run([sys.executable,'scripts/run_tracked_experiment.py','--spec',str(output/'spec.json'),
            '--output',str(output),'--',*command],check=True)
        return json.loads((output/('calibration.json' if calibration_only else 'result.json')).read_text())
    candidates=[]
    for capacity in [32,64]:
        for budget in [500,1000,2000,4000,8000,16000,32000]:
            result=run(a.output/f'calibration-m{capacity}-b{budget}','validation',capacity,budget,1,source,True)
            resolved=initial_budget_resolved(result['best_validation_iteration'],budget,result['convergence_status'])
            if resolved:break
        if not resolved:
            raise RuntimeError('Initial validation best checkpoint remains at budget boundary')
        candidates.append(dict(capacity=capacity,budget=budget,result=result))
    selected=min(candidates,key=lambda c:c['result']['best_validation_nll'])
    capacity,budget=selected['capacity'],selected['budget']
    online=[]
    rates=[.001] if a.dataset=='covid' else [.001,.01,.05]
    def trial(updates,rate):
        label=f'online-budget-u{updates}' if a.dataset=='covid' else f'online-lr{rate:g}-u{updates}'
        result=run(a.output/label,'validation',capacity,budget,updates,fold,online_lr=rate)
        row=dict(updates=updates,learning_rate=rate,nlpd=result['overall_current_block']['nll'],update_seconds=result['timing']['mean_block_update_seconds'])
        online.append(row);return row
    for rate in rates:
        for updates in [1,5,20]:trial(updates,rate)
        if a.dataset!='covid':trial(80,rate)
    best=min(online,key=lambda c:c['nlpd'])
    for updates in ([80,320,1280] if a.dataset=='covid' else [320,1280]):
        at_rate=[c for c in online if c['learning_rate']==best['learning_rate']]
        if best['updates'] != max(c['updates'] for c in at_rate):break
        previous_best=best
        current=trial(updates,best['learning_rate'])
        best=min(online,key=lambda c:c['nlpd'])
        if current['nlpd'] >= previous_best['nlpd']-0.01:break
        if updates==1280:
            (a.output/'online-budget-incomplete.json').write_text(json.dumps(online,indent=2))
            raise RuntimeError('Online budget still improves at maximum; extend qualification')
    selection=dict(initial_candidates=candidates,selected_initial=selected,
        online_candidates=online,selected_online=best,
        selection_scope='Formal Task-1 only: spatial capacity plus chronological online-budget fold; '+a.dataset)
    selection['projected_stream_update_seconds']=best['update_seconds']*expected_steps
    (a.output/'selection.json').write_text(json.dumps(selection,indent=2))
    qualification=dict(status='passed',method='ohsvgp',dataset=a.dataset,source_commit=a.release,
        tests=tests,qualification_test_device='cuda',selection=selection,
        scoped_deterministic_prediction_basis_cache=a.dataset!='covid',main_table_admitted=False)
    (a.output/'qualification.json').write_text(json.dumps(qualification,indent=2))
    run(a.output,'final',capacity,budget,best['updates'],source,online_lr=best['learning_rate'])


if __name__=='__main__':main()
