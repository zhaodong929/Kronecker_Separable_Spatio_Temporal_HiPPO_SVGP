#!/usr/bin/env python3
"""Measure exact Gaussian continuation against each completed reference run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compute-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-job',type=int,default=294321)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    worker=str(a.compute_root/'env-jax-gpu/bin/python')
    subprocess.run([worker,'-c',"import jax; assert jax.default_backend()=='gpu'"],check=True)
    rows=[]
    for seed in range(5,10):
        reference=a.compute_root/f'results/fair-three-domain-wandb-20260929/covid/st_svgp/seed{seed}/job-{a.reference_job}'
        exit_record=json.loads((reference/'exit.json').read_text())
        if exit_record['exit_code']!=0:raise ValueError('A successful reference is required')
        frozen=reference/'task1-frozen.npz'
        with np.load(frozen,allow_pickle=False) as state:capacity=int(state['spatial_inducing'])
        protocol=a.compute_root/f'protocol/covid-v2/seed{seed}'
        output=a.output/f'seed{seed}'
        command=[worker,'baselines/covid_long_setting_b/adapters/run_st_svgp.py',
            '--protocol-npz',str(protocol/'protocol.npz'),'--protocol-json',str(protocol/'protocol.json'),
            '--seed',str(seed),'--output-dir',str(output),'--spatial-inducing',str(capacity),
            '--frozen-task1-state',str(frozen),'--online-inference-steps','1','--online-backend','stateful']
        with (a.output/f'seed{seed}-worker.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        errors={}
        with np.load(reference/'predictions.npz') as expected,np.load(output/'predictions.npz') as actual,np.load(protocol/'protocol.npz') as source:
            for key in ['pred_mean','pred_var']:
                np.testing.assert_allclose(actual[key],expected[key],rtol=1e-5,atol=1e-6)
                errors[key]=float(np.max(np.abs(actual[key]-expected[key])))
            for key,value in [('y_true',source['stream_y'][:,source['test_indices']]),('times',source['stream_times']),('test_indices',source['test_indices'])]:
                np.testing.assert_array_equal(actual[key],value)
        result=json.loads((output/'result.json').read_text())
        parent=json.loads((reference/'result.json').read_text())
        rows.append(dict(seed=seed,reference=str(reference),frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
            max_absolute_error=errors,steps=143,task1_fit_seconds_from_reference=parent['task1_seconds'],
            initial_filter_seconds=result['initial_filter_and_prefix_seconds'],
            online_seconds=result['online_seconds_total'],mean_online_seconds=result['online_seconds_per_week'],
            full_continuation_seconds=result['continuation_total_seconds'],
            reference_replay_online_seconds=parent['online_seconds_total'],main_table_admitted=False))
        (a.output/'measurements.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(rows[-1]),flush=True)
    (a.output/'result.json').write_text(json.dumps(dict(status='five_full_reference_checks_passed',
        measurements=rows,main_table_admitted=False,scope='Same posterior and frozen fit; exclusive A30 inference timing'),indent=2))


if __name__=='__main__':main()
