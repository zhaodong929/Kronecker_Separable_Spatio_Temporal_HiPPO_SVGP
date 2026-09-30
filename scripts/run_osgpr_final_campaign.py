#!/usr/bin/env python3
"""Official Bui OSGPR with own initial and chronological update-budget tuning."""
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
    os.environ['HIPPO_TEST_DEVICE']='cuda'
    worker=str(a.compute_root/'env-osgpr/bin/python')
    cuda_root=a.compute_root/'env-osgpr/lib/python3.11/site-packages/nvidia/cuda_nvcc'
    if not (cuda_root/'nvvm/libdevice/libdevice.10.bc').is_file():
        raise FileNotFoundError('TensorFlow GPU runtime requires the installed CUDA libdevice')
    os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir='+str(cuda_root)
    source=a.compute_root/f"protocol/{'covid-v2' if a.dataset=='covid' else a.dataset}/seed{a.seed}"
    expected_steps,expected_sites,initial_sites={'covid':(143,10,52),'pems':(50100,65,260),'era5':(1674,200,800)}[a.dataset]
    fold_function={'covid':'build_fold','pems':'build_pems_fold','era5':'build_era5_fold'}[a.dataset]
    tests=['tests/test_osgpr_release_boundary.py','tests/test_osgpr_graph_optimizer.py']
    if a.dataset=='era5':tests += ['tests/test_era5_information_boundary.py','tests/test_era5_adapter_boundary.py::test_actual_era5_no_release_adapter[osgpr]']
    with (a.output/'tests.txt').open('w') as f:
        subprocess.run([worker,'-m','pytest','-q',*tests],stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([worker,'-c',"import tensorflow as tf; assert tf.config.list_physical_devices('GPU')"],check=True)
    fold=a.output/'online-budget-protocol'
    subprocess.run([str(a.compute_root/'env-routeb/bin/python'),'-c',
        f'from benchmarks.three_domain.online_validation import {fold_function}; import sys; {fold_function}(sys.argv[1],sys.argv[2])',
        str(source/'protocol.npz'),str(fold)],check=True)
    def run(output,stage,mt,ms,budget,updates,protocol,calibration_only=False):
        output.mkdir(parents=True,exist_ok=True)
        spec=dict(entity='harrisonzhu',project='KronHiPPO-STGP',campaign='fair-three-domain-wandb-20260929',
            dataset=a.dataset,method='osgpr',split_seed=a.seed,training_seed=a.seed,stage=stage,
            source_commit=a.release,worker_python=worker,
            input_files=[str(protocol/'protocol.npz'),str(protocol/'protocol.json')],
            temporal_inducing=mt,spatial_inducing=ms,calibration_steps_per_block=budget,
            online_steps_per_update=updates,initial_optimizer_execution='graph',initial_block_times=10 if a.dataset=='covid' else 256,
            online_optimizer_execution='graph',main_table_admitted=False)
        if stage=='final':
            spec.update(qualification_record=str(a.output/'qualification.json'),expected_steps=expected_steps,
                expected_sites=expected_sites,hidden_delay_steps=None if a.dataset=='era5' else 1,initial_observed_sites=initial_sites,predictive_family='gaussian')
        (output/'spec.json').write_text(json.dumps(spec,indent=2))
        command=[worker,'scripts/run_official_bui_osgpr_era5.py','--protocol-npz',str(protocol/'protocol.npz'),
            '--protocol-json',str(protocol/'protocol.json'),'--output',str(output/'result.json'),
            '--blockwise-output',str(output/'blocks.csv'),'--predictions-output',str(output/'predictions.npz'),
            '--seed',str(a.seed),'--mt',str(mt),'--ms',str(ms),'--adaptive',
            '--adaptive-calibration-steps',str(budget),'--adaptive-online-steps',str(updates),
            '--device','cuda','--calibration-block-size',
            '10' if a.dataset=='covid' else '256','--initial-optimizer-execution',
            'graph','--online-optimizer-execution','graph']
        if a.dataset!='era5':command.append('--delayed-observations')
        if calibration_only:command.append('--task1-validation-only')
        subprocess.run([sys.executable,'scripts/run_tracked_experiment.py','--spec',str(output/'spec.json'),
            '--output',str(output),'--',*command],check=True)
        return json.loads((output/'result.json').read_text())
    candidates=[]
    for mt,ms in [(2,16),(2,32),(4,16),(4,32)]:
        scores=[]
        for budget in [100,400]:
            result=run(a.output/f'calibration-mt{mt}-ms{ms}-b{budget}','validation',mt,ms,budget,5,source,True)
            scores.append(dict(mt=mt,ms=ms,budget=budget,nlpd=result['metrics']['gaussian_nlpd'],result=result))
        if scores[-1]['nlpd'] < scores[0]['nlpd']-0.01:
            result=run(a.output/f'calibration-mt{mt}-ms{ms}-b800','validation',mt,ms,800,5,source,True)
            scores.append(dict(mt=mt,ms=ms,budget=800,nlpd=result['metrics']['gaussian_nlpd'],result=result))
            if scores[-1]['nlpd'] < scores[-2]['nlpd']-0.01:
                scores[-1]['requires_larger_budget']=True
        candidates.extend(scores)
    selected=min(candidates,key=lambda c:c['nlpd'])
    # Extend only a currently selected endpoint. A larger budget that worsens
    # validation resolves the earlier endpoint; it does not force its selection.
    while selected.get('requires_larger_budget'):
        extended_budget=2*selected['budget']
        if extended_budget>6400:
            (a.output/'initial-budget-incomplete.json').write_text(json.dumps(candidates,indent=2))
            raise RuntimeError('Selected initial configuration improves at 6400; extend qualification')
        result=run(a.output/f"calibration-mt{selected['mt']}-ms{selected['ms']}-b{extended_budget}",
            'validation',selected['mt'],selected['ms'],extended_budget,5,source,True)
        candidate=dict(mt=selected['mt'],ms=selected['ms'],budget=extended_budget,
            nlpd=result['metrics']['gaussian_nlpd'],result=result,
            requires_larger_budget=result['metrics']['gaussian_nlpd']<selected['nlpd']-0.01)
        selected['requires_larger_budget']=False
        selected['extended_budget_checked']=extended_budget
        candidates.append(candidate)
        selected=min(candidates,key=lambda c:c['nlpd'])
    mt,ms,budget=selected['mt'],selected['ms'],selected['budget']
    online=[]
    for updates in [5,20,80,320,1280]:
        if updates>80:
            best_so_far=min(online,key=lambda c:c['nlpd'])
            if best_so_far is not online[-1] or online[-1]['nlpd']>=min(c['nlpd'] for c in online[:-1])-0.01:
                break
        result=run(a.output/f'online-budget-u{updates}','validation',mt,ms,budget,updates,fold)
        online.append(dict(updates=updates,nlpd=result['final']['nll'],
            update_seconds=result['timing']['mean_block_update_seconds']))
    best=min(online,key=lambda c:c['nlpd'])
    if best['updates']==1280 and best['nlpd'] < min(c['nlpd'] for c in online[:-1])-0.01:
        (a.output/'online-budget-incomplete.json').write_text(json.dumps(online,indent=2))
        raise RuntimeError('Online validation materially improves at maximum budget; extend qualification')
    selection=dict(initial_candidates=candidates,selected_initial=selected,
        online_candidates=online,selected_online=best,
        selection_scope='Formal Task-1 only: spatial capacity plus chronological online-budget fold; '+a.dataset)
    selection['projected_stream_update_seconds']=best['update_seconds']*expected_steps
    (a.output/'selection.json').write_text(json.dumps(selection,indent=2))
    qualification=dict(status='passed',method='osgpr',dataset=a.dataset,source_commit=a.release,
        tests=tests,qualification_test_device='cuda',selection=selection,main_table_admitted=False)
    (a.output/'qualification.json').write_text(json.dumps(qualification,indent=2))
    run(a.output,'final',mt,ms,budget,best['updates'],source)


if __name__=='__main__':main()
