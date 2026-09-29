#!/usr/bin/env python3
"""Own spatial validation and Task-1 chronological online-budget selection."""
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
    worker=str(a.compute_root/'env-routeb/bin/python')
    source=a.compute_root/f'protocol/covid-v2/seed{a.seed}'
    tests=['tests/test_ohsvgp_release_boundary.py','tests/test_ohsvgp_lengthscale_learning.py',
           'tests/test_online_budget_fold.py']
    with (a.output/'tests.txt').open('w') as f:
        subprocess.run([worker,'-m','pytest','-q',*tests],stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([worker,'-c','import torch; assert torch.cuda.is_available()'],check=True)
    fold=a.output/'online-budget-protocol'
    # Use the numerical environment, which has the protocol builder dependencies.
    subprocess.run([worker,'-c',
        'from benchmarks.three_domain.online_validation import build_fold; import sys; build_fold(sys.argv[1],sys.argv[2])',
        str(source/'protocol.npz'),str(fold)],check=True)
    def run(output,stage,capacity,budget,updates,protocol,calibration_only=False):
        output.mkdir(parents=True,exist_ok=True)
        spec=dict(entity='harrisonzhu',project='KronHiPPO-STGP',campaign='fair-three-domain-wandb-20260929',
            dataset='covid',method='ohsvgp',split_seed=a.seed,training_seed=a.seed,stage=stage,
            source_commit=a.release,worker_python=worker,
            input_files=[str(protocol/'protocol.npz'),str(protocol/'protocol.json')],
            inducing_size=capacity,max_iterations=budget,online_update_steps=updates,
            main_table_admitted=False)
        if stage=='final':
            spec.update(qualification_record=str(a.output/'qualification.json'),expected_steps=143,
                expected_sites=10,hidden_delay_steps=1,initial_observed_sites=52,predictive_family='gaussian')
        (output/'spec.json').write_text(json.dumps(spec,indent=2))
        command=[worker,'scripts/run_covid_ohsvgp_own_theta.py','--protocol-npz',str(protocol/'protocol.npz'),
            '--protocol-json',str(protocol/'protocol.json'),'--output-dir',str(output),'--seed',str(a.seed),
            '--kernel','rbf','--inducing-size',str(capacity),'--rff-sample-size','256',
            '--basis-grid-size','1024','--calibration-iterations',str(budget),
            '--task1-min-steps',str(budget),'--task1-check-interval','5',
            '--calibration-batch-size','256','--update-steps',str(updates),'--delayed-observations','--device','cuda']
        if calibration_only:command.append('--calibration-only')
        subprocess.run([sys.executable,'scripts/run_tracked_experiment.py','--spec',str(output/'spec.json'),
            '--output',str(output),'--',*command],check=True)
        return json.loads((output/('calibration.json' if calibration_only else 'result.json')).read_text())
    candidates=[]
    for capacity in [32,64]:
        budget=500
        result=run(a.output/f'calibration-m{capacity}-b{budget}','validation',capacity,budget,1,source,True)
        if result['best_validation_iteration']>=budget:
            budget=1000
            result=run(a.output/f'calibration-m{capacity}-b{budget}','validation',capacity,budget,1,source,True)
        if result['best_validation_iteration']>=budget:
            raise RuntimeError('Initial validation best checkpoint remains at budget boundary')
        candidates.append(dict(capacity=capacity,budget=budget,result=result))
    selected=min(candidates,key=lambda c:c['result']['best_validation_nll'])
    capacity,budget=selected['capacity'],selected['budget']
    online=[]
    for updates in [1,5,20]:
        result=run(a.output/f'online-budget-u{updates}','validation',capacity,budget,updates,fold)
        online.append(dict(updates=updates,nlpd=result['overall_current_block']['nll'],
            update_seconds=result['timing']['mean_block_update_seconds']))
    best=min(online,key=lambda c:c['nlpd'])
    if best['updates']==20:
        result=run(a.output/'online-budget-u80','validation',capacity,budget,80,fold)
        online.append(dict(updates=80,nlpd=result['overall_current_block']['nll'],
            update_seconds=result['timing']['mean_block_update_seconds']))
        # A materially improving endpoint requires a larger validated search,
        # rather than silently declaring the largest attempted budget adequate.
        if online[-1]['nlpd'] < best['nlpd']-0.01:
            (a.output/'online-budget-incomplete.json').write_text(json.dumps(online,indent=2))
            raise RuntimeError('Online budget still improves at maximum; extend qualification')
        best=min(online,key=lambda c:c['nlpd'])
    selection=dict(initial_candidates=candidates,selected_initial=selected,
        online_candidates=online,selected_online=best,
        selection_scope='Formal Task-1 only: spatial capacity and prefix 40/12 chronological update budget')
    (a.output/'selection.json').write_text(json.dumps(selection,indent=2))
    qualification=dict(status='passed',method='ohsvgp',dataset='covid',source_commit=a.release,
        tests=tests,selection=selection,main_table_admitted=False)
    (a.output/'qualification.json').write_text(json.dumps(qualification,indent=2))
    run(a.output,'final',capacity,budget,best['updates'],source)


if __name__=='__main__':main()
