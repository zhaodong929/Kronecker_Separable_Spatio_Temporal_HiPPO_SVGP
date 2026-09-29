#!/usr/bin/env python3
"""Move the live record to the requested team without restarting computation.

Uses the same attempt ID in the destination project, links the source run,
retains every event in the final artifact, and does not report this host's
system telemetry as if it were the allocated GPU node.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote-output', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--entity', required=True)
    p.add_argument('--project', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    import wandb
    from benchmarks.three_domain.tracking import numeric_metrics
    run, offset = None, 0
    for poll in range(720):
        script = '''import json,pathlib
p=pathlib.Path(OUTPUT)
folders=sorted((p/'tracking').glob('*'))
if len(folders)!=1:raise RuntimeError('Expected one source attempt')
d=folders[0]
result={'attempt':d.name,'provenance':json.loads((d/'provenance.json').read_text()),'source':json.loads((d/'wandb.json').read_text())}
events=[];offset=OFFSET
with (d/'events.jsonl').open('rb') as f:
 f.seek(offset)
 while True:
  line=f.readline()
  if not line or not line.endswith(b'\\n'):break
  events.append(json.loads(line));offset=f.tell()
result.update(events=events,offset=offset)
if (d/'terminal.json').exists():result['terminal']=json.loads((d/'terminal.json').read_text())
if (p/'exit.json').exists():result['exit']=json.loads((p/'exit.json').read_text())
print(json.dumps(result))
'''.replace('OUTPUT',repr(a.remote_output)).replace('OFFSET',str(offset))
        try:
            result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                'corgi','/usr/bin/python3 -c '+shlex.quote(script)], capture_output=True, text=True, timeout=90, check=True)
            data = json.loads(result.stdout)
            if run is None:
                provenance = data['provenance']
                spec = provenance['spec']
                run = wandb.init(entity=a.entity,project=a.project,id=data['attempt'],resume='allow',
                    name=provenance['logical_id']+'/'+data['attempt'],group=spec['campaign'],
                    job_type=spec['stage'],tags=[spec['dataset'],spec['method'],spec['stage']],
                    config={**provenance,'source_run_url':data['source']['url'],
                        'record_role':'Team record of the SAME computation; not an independent replicate',
                        'destination_entity':a.entity,'destination_project':a.project},
                    dir=str(a.output),save_code=False,settings=wandb.Settings(x_disable_stats=True,init_timeout=30))
                for phase in ('train','validation','online','refit','system'):
                    run.define_metric(phase+'/step')
                    run.define_metric(phase+'/*',step_metric=phase+'/step')
                (a.output/'wandb.json').write_text(json.dumps(dict(url=run.url,id=run.id),indent=2))
                run.summary['main_table_admitted'] = False
            # Thirty-second bins reproduce a legible history from the beginning,
            # even when this bridge attaches after the computation has started.
            bins = {}
            for event in data['events']:
                bins[(event['phase'],int(event['time_unix']//30))] = event
            for event in sorted(bins.values(),key=lambda e:e['time_unix']):
                phase = event['phase']
                run.log({phase+'/step':event['step'],**numeric_metrics(event['metrics'],phase)})
            offset = data['offset']
            (a.output/'progress.json').write_text(json.dumps(dict(offset=offset,poll=poll,
                source_output=a.remote_output,last_time=time.time(),url=run.url),indent=2))
            if 'exit' in data:
                if 'terminal' not in data:
                    raise RuntimeError('Source exited without supervisor terminal record')
                local = a.output/'source-output'
                local.mkdir(exist_ok=True)
                subprocess.run(['rsync','-a','--exclude=wandb/','--exclude=wandb-staging/',
                    'corgi:'+a.remote_output+'/',str(local)+'/'],check=True,timeout=600)
                artifact = wandb.Artifact('run-'+data['attempt'],type='experiment-record',
                    metadata={'source_run_url':data['source']['url'],'same_computation':True})
                artifact.add_dir(str(local))
                run.log_artifact(artifact)
                run.summary.update(data['terminal'])
                run.summary['source_run_url'] = data['source']['url']
                code = data['exit']['exit_code']
                run.finish(exit_code=code)
                (a.output/'terminal.json').write_text(json.dumps(dict(status='synchronized',
                    computation_exit_code=code,source_terminal=data['terminal']),indent=2))
                return code
        except Exception as error:
            with (a.output/'errors.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.time(),error=type(error).__name__+': '+str(error)))+'\n')
        time.sleep(60)
    raise RuntimeError('Team synchronization timed out; source results are intact')


if __name__ == '__main__':
    raise SystemExit(main())
