import importlib.util
import json
from pathlib import Path
import types
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('final_campaign',ROOT/'slurm/task_stream/final/campaign.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture_attempt(tmp_path):
    a=tmp_path/'attempt';r=a/'run';r.mkdir(parents=True)
    proof=dict(source_commit='abc',source_sha256='def',configuration_sha256='ghi',
               refit_budget={'effective_iterations':7},input_files={'data':'hash'},selection_sha256='sel')
    (a/'proof.json').write_text(json.dumps(proof))
    p={k:proof[k] for k in ('source_commit','source_sha256','configuration_sha256','refit_budget','input_files')}
    result=dict(status='completed',provenance=dict(p,stage='final'),metrics=dict(nlpd=1.,rmse=1.,crps=1.))
    (r/'result.json').write_text(json.dumps(result))
    artifacts={}
    for name in ('predictions.npz','task-metrics.json','configuration.json','fit-budget.json'):
        (r/name).write_text('fixture');artifacts[name]={'sha256':m.digest(r/name)}
    (r/'artifacts.json').write_text(json.dumps(artifacts))
    return a,proof


def test_completed_requires_final_and_intact_artifacts(tmp_path):
    a,p=fixture_attempt(tmp_path)
    assert m.completed(a,p)
    f=a/'run/result.json';d=json.loads(f.read_text());d['provenance']['stage']='validation';f.write_text(json.dumps(d))
    assert not m.completed(a,p)
    d['provenance']['stage']='final';f.write_text(json.dumps(d))
    (a/'run/predictions.npz').write_text('changed')
    assert not m.completed(a,p)


def test_reuse_binds_selection_and_fixed_refit(tmp_path):
    a,p=fixture_attempt(tmp_path)
    changed=dict(p,selection_sha256='other')
    assert not m.completed(a,changed)
    f=a/'run/result.json';d=json.loads(f.read_text());d['provenance']['refit_budget']['effective_iterations']=8
    f.write_text(json.dumps(d));assert not m.completed(a,p)


def test_nonfinite_metrics_rejected(tmp_path):
    a,p=fixture_attempt(tmp_path)
    f=a/'run/result.json';d=json.loads(f.read_text());d['metrics']['nlpd']=float('nan');f.write_text(json.dumps(d))
    assert not m.completed(a,p)


def test_failed_run_does_not_stop_other_assigned_tasks(tmp_path,monkeypatch):
    def atomic(p,d):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
    selection=tmp_path/'selection.json';selection.write_text('{}')
    proof=dict(source_commit='a',source_sha256='b',selection_sha256=m.digest(selection),
               refit_budget={},input_files={})
    task=dict(id='x',configuration=dict(method='kronhippo_svgp',seed=0),proof=proof,
              output=str(tmp_path/'out'),prepared=str(tmp_path/'prepared'),selection=str(selection),
              dataset='covid',split_seed=5)
    tasks=[dict(task,id=str(i),output=str(tmp_path/f'out{i}')) for i in range(4)]
    helpers=types.SimpleNamespace(atomic_json=atomic,verify_source=lambda *a:None,
        InputProofs=lambda:types.SimpleNamespace(get=lambda *a:{}),ENVIRONMENTS={'kronhippo_svgp':'env-routeb'})
    commands=[]
    class Process:
        def __init__(self,command,**kw):commands.append(command)
        def wait(self):return 1
    monkeypatch.setattr(m.subprocess,'Popen',Process)
    assert m.run(tasks,tmp_path,tmp_path,tmp_path/'campaign','test',0,helpers)==1
    assert len(commands)==2
    assert all(c[c.index('--stage')+1]=='final' and '--selection' in c and '--max-tasks' not in c for c in commands)
    d=json.loads((tmp_path/'campaign/state/worker-0.json').read_text())
    assert len(d['failed'])==2 and d['status']=='failed'
