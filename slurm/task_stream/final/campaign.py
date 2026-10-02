#!/usr/bin/env python3
"""Final orchestration outside the immutable numerical-source snapshot."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def completed(attempt, proof):
    try:
        if json.loads((attempt/'proof.json').read_text()) != proof: return False
        r = json.loads((attempt/'run/result.json').read_text())
        p = r['provenance']
        if r['status'] != 'completed' or p['stage'] != 'final': return False
        for k in ('source_commit', 'source_sha256', 'configuration_sha256'):
            if p[k] != proof[k]: return False
        if p['refit_budget'] != proof['refit_budget']: return False
        if any(p['input_files'].get(k) != v for k,v in proof['input_files'].items()): return False
        import math
        if not all(math.isfinite(r['metrics'][k]) for k in ('nlpd','rmse','crps')): return False
        artifacts = json.loads((attempt/'run/artifacts.json').read_text())
        for name in ('predictions.npz','task-metrics.json','configuration.json','fit-budget.json'):
            if digest(attempt/'run'/name) != artifacts[name]['sha256']: return False
        return True
    except (OSError, ValueError, KeyError, TypeError): return False


def qualify(c, destination, method, dataset, helpers):
    results=c/'results'
    if method in ('osgpr','st_svgp'):
        parity=results/'task-stream-memory-parity-20260930-44d78a8'/method/'job-294587/result.json'
        shape=results/f'task-stream-memory-shapes-20260930-44d78a8-{dataset}'/method/'job-294588/result.json'
    else:
        parity=results/('task-stream-compact-20260930-eedc07f' if method=='mgpvae' else 'task-stream-tiny-20260930-fcff705')/method/('job-294570/result.json' if method=='mgpvae' else 'job-294564/result.json')
        shape=results/f'task-stream-shapes-20260930-eedc07f-{dataset}'/method/'job-294573/result.json'
    evidence=[]
    for kind,path in [('device_parity',parity),('full_size_shape',shape)]:
        record=json.loads(path.read_text())
        assert record['status']=='completed' and record['method']==method
        assert record['qualification'] in ('cpu_gpu_parity_passed','independentfits_completed_and_fixed_state_prediction_parity','shape_only_passed')
        if kind=='device_parity': assert record['qualification']!='shape_only_passed'
        evidence.append(dict(kind=kind,path=str(path),sha256=digest(path),record=record))
    path=destination/'qualifications'/f'{dataset}-{method}.json'
    helpers.atomic_json(path,dict(status='passed',method=method,dataset=dataset,evidence=evidence,
        scope='device numerics and full initial-data shape qualification; subsequent selection complete',
        convergence_qualified=False,main_table_admitted=False))
    return path


def build(c, source, destination, campaign, helpers):
    from benchmarks.task_stream.data import PreparedData
    from benchmarks.task_stream.provenance import feature_identity, source_identity
    from benchmarks.task_stream.refit_budget import validate_refit_plan
    study=c/'studies/common-time-budget-20260930'
    summary=json.loads((study/'state/selection/study-summary.json').read_text())
    assert summary['complete'] and len(summary['groups']) == 65
    revision=(source/'SOURCE_COMMIT').read_text().strip()
    identity=source_identity(source)
    tasks=[]; proofs=helpers.InputProofs(); geometries={}
    expected={(d,s,m) for d,seeds in [('era5',range(5)),('covid',range(5,10)),('pems',range(1,4))]
              for s in seeds for m in helpers.ENVIRONMENTS}
    actual=set()
    for g in summary['groups']:
        key=(g['dataset'],g['split_seed'],g['method'])
        if key in actual: raise ValueError('Duplicate final group')
        actual.add(key)
        selection=Path(g['selection']); selected=json.loads(selection.read_text())
        assert selected['status']=='selected' and selected['all_declared_candidates_completed']
        assert len(selected['candidates'])==3
        winner=Path(selected['winner']['path'])
        assert digest(winner/'result.json') == selected['winner']['result_sha256']
        config,config_hash=helpers.canonical_configuration(json.loads((winner.parent/'configuration.json').read_text()))
        assert selected['configuration_sha256']==config_hash
        assert selected['source_commit']==revision and selected['source_sha256']==identity
        artifacts=json.loads((winner/'artifacts.json').read_text())
        assert digest(winner/'fit-budget.json')==artifacts['fit-budget.json']['sha256']
        prepared=c/'protocol/task-stream-20260930'/g['dataset']/f"seed{g['split_seed']}"
        if str(prepared) not in geometries:
            data=PreparedData.load(prepared)
            geometries[str(prepared)]=(data.selection_stream.identity(),
                feature_identity(data.selection_stream.times,data.selection_features),
                data.selection_stream.initial().values.size,data.stream.initial().values.size)
            del data
        protocol,features,selected_rows,final_rows=geometries[str(prepared)]
        assert selected['selection_protocol_sha256']==protocol
        assert selected['selection_features_sha256']==features
        plan=validate_refit_plan(selected['refit_budget'], config,
            json.loads((winner/'fit-budget.json').read_text()),selected_rows,final_rows)
        inputs=proofs.get(prepared)
        winner_result=json.loads((winner/'result.json').read_text())
        assert all(winner_result['provenance']['input_files'].get(p)==h for p,h in inputs.items())
        qualification=qualify(c,destination,g['method'],g['dataset'],helpers)
        name=f"{g['dataset']}-seed{g['split_seed']}-{g['method']}"
        tasks.append(dict(id=name,dataset=g['dataset'],split_seed=g['split_seed'],configuration=config,
            prepared=str(prepared),selection=str(selection),qualification_record=str(qualification),output=str(destination/'runs'/name),
            proof=dict(source_commit=revision,source_sha256=identity,configuration_sha256=config_hash,
                input_files=inputs,selection_sha256=digest(selection),qualification_sha256=digest(qualification),refit_budget=plan)))
    assert actual==expected
    # Interleave domains; method order distributes each framework across workers.
    tasks.sort(key=lambda t:(t['split_seed'],t['configuration']['method']))
    helpers.atomic_json(destination/'manifest.json',tasks)
    helpers.atomic_json(destination/'preflight.json',dict(status='passed',groups=len(tasks),
        source_commit=revision,source_sha256=identity,manifest_sha256=digest(destination/'manifest.json'),
        main_table_admitted=False,convergence_qualified=False,campaign=campaign))
    print(json.dumps(dict(status='preflight_passed',groups=len(tasks))))


def run(tasks, c, source, destination, campaign, index, helpers):
    state=destination/'state';state.mkdir(parents=True,exist_ok=True)
    with (state/f'worker-{index}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assigned=tasks[index::3]; started=time.monotonic(); current=None; stop=False
        status=dict(status='running',worker=index,assigned=len(assigned),completed=[],failed=[],active=None,
                    job_id=os.environ.get('SLURM_JOB_ID'))
        def persist():
            status['elapsed_seconds']=time.monotonic()-started
            helpers.atomic_json(state/f'worker-{index}.json',status)
        def signal_handler(signum,frame):
            nonlocal stop
            stop=True
            if current is not None and current.poll() is None: current.send_signal(signum)
        signal.signal(signal.SIGTERM,signal_handler);signal.signal(signal.SIGINT,signal_handler)
        cache=helpers.InputProofs()
        persist()
        for task in assigned:
            if stop: break
            proof=task['proof'];logical=Path(task['output'])
            helpers.verify_source(source,proof['source_commit'],proof['source_sha256'])
            assert cache.get(task['prepared'])==proof['input_files']
            assert digest(task['selection'])==proof['selection_sha256']
            assert digest(task['qualification_record'])==proof['qualification_sha256']
            reusable=[p for p in (logical/'attempts').glob('*') if completed(p,proof)]
            if reusable:
                status['completed'].append(dict(id=task['id'],attempt=str(sorted(reusable)[-1]),reused=True));persist();continue
            attempt=logical/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
            helpers.atomic_json(attempt/'configuration.json',task['configuration'])
            helpers.atomic_json(attempt/'proof.json',proof)
            method=task['configuration']['method'];worker=c/helpers.ENVIRONMENTS[method]/'bin/python'
            spec=dict(entity='harrisonzhu',project='KronHiPPO-STGP',campaign=campaign,dataset=task['dataset'],
                method=method,split_seed=task['split_seed'],training_seed=task['configuration']['seed'],stage='final',
                source_commit=proof['source_commit'],source_sha256=proof['source_sha256'],worker_python=str(worker),
                main_table_admitted=False,convergence_qualified=False,refit_budget=proof['refit_budget'],
                qualification_record=task['qualification_record'],
                input_files=[str(attempt/'configuration.json'),task['selection'],task['qualification_record'],*proof['input_files']],
                timing_scope='one dedicated A30 GPU; fixed selected refit work and complete final task stream')
            helpers.atomic_json(attempt/'spec.json',spec)
            command=[str(c/'env-tracking/bin/python'),str(source/'scripts/run_tracked_experiment.py'),
                '--spec',str(attempt/'spec.json'),'--output',str(attempt/'run'),'--',str(worker),
                str(source/'scripts/run_task_stream.py'),'--prepared',task['prepared'],
                '--configuration',str(attempt/'configuration.json'),'--selection',task['selection'],
                '--output',str(attempt/'run'),'--stage','final']
            status['active']=dict(id=task['id'],attempt=str(attempt));persist()
            with (attempt/'supervisor.log').open('w') as log:
                current=subprocess.Popen(command,cwd=source,stdout=log,stderr=subprocess.STDOUT)
                code=current.wait();current=None
            valid=code==0 and completed(attempt,proof)
            terminal=dict(status='completed' if valid else 'failed',id=task['id'],attempt=str(attempt),exit_code=code)
            helpers.atomic_json(attempt/'terminal.json',terminal);helpers.atomic_json(logical/'latest.json',terminal)
            status['completed' if valid else 'failed'].append(terminal);status['active']=None;persist()
        status['status']='interrupted' if stop else 'failed' if status['failed'] else 'completed';persist()
        return 1 if stop or status['failed'] else 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['build','run'])
    parser.add_argument('--compute-root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--worker',type=int,choices=range(3))
    a=parser.parse_args();sys.path.insert(0,str(a.source))
    from scripts import run_task_budget_campaign as helpers
    if a.mode=='build': build(a.compute_root,a.source,a.destination,a.campaign,helpers);return 0
    preflight=json.loads((a.destination/'preflight.json').read_text())
    assert digest(a.destination/'manifest.json')==preflight['manifest_sha256']
    assert a.worker is not None
    return run(json.loads((a.destination/'manifest.json').read_text()),a.compute_root,a.source,
               a.destination,a.campaign,a.worker,helpers)

if __name__=='__main__':sys.exit(main())
