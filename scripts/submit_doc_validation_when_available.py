#!/usr/bin/env python3
"""Retry a quota-limited, uniquely named submission every ten minutes."""
import argparse
import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


def ssh(command):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        'gpucluster2', shlex.join(command)], capture_output=True, text=True, timeout=90)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--vault-note', type=Path, required=True)
    p.add_argument('--followup-json', type=Path, help='Explicit final-array follow-up plan; omitted means baseline validation')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(a.followup_json.read_text()) if a.followup_json else {}
    name = plan.get('job_name','hippo-val-'+a.release[:8])
    c = '/vol/bitbucket/nk523/hipposvgp-fair-20260929'
    job = None
    def record_failure(record):
        (a.output/'terminal.json').write_text(json.dumps(record, indent=2))
        with a.vault_note.open('a') as note:
            note.write(f'\n## 自動投入の未完了 {record["time"]}\n\n{record["status"]}。記録: `{a.output}/terminal.json`。投入完了とは扱わない。\n')
    for _ in range(144):
        record = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            # Reconcile first: an SSH timeout can follow a successful sbatch.
            q = ssh(['squeue', '-h', '-u', 'nk523', '-n', name, '-o', '%A'])
            if q.returncode:
                raise RuntimeError(q.stderr[-1000:])
            jobs = set(re.findall(r'^\d+$', q.stdout, flags=re.M))
            if len(jobs) > 1:
                raise RuntimeError('Duplicate matching submissions; manual reconciliation required')
            if jobs:
                job = int(jobs.pop())
            else:
                submission = ['sbatch', '--parsable', '--job-name='+name, '--chdir='+c,
                    '--output='+c+'/logs/'+name+'-%A_%a.log']
                if plan.get('array'): submission.append('--array='+plan['array'])
                submission += [c+'/releases/'+a.release+'/source/'+plan.get('sbatch',
                    'slurm/fair_three_domain/baseline_validation.sbatch'),a.release]
                receipt = c+'/submissions/'+name+'.jobid'
                script = ('if test -s '+shlex.quote(receipt)+'; then cat '+shlex.quote(receipt)+
                    '; else '+shlex.join(submission)+' > '+shlex.quote(receipt+'.tmp')+
                    ' && mv '+shlex.quote(receipt+'.tmp')+' '+shlex.quote(receipt)+
                    ' && cat '+shlex.quote(receipt)+'; fi')
                # A remote, locked receipt survives loss of the SSH reply,
                # including a job that finishes before the next queue poll.
                r = ssh(['bash','-c','mkdir -p '+shlex.quote(c+'/submissions')+' && '+
                    shlex.join(['flock','-x','-w','10',receipt+'.lock','bash','-c',script])])
                ids = re.findall(r'^(\d+)(?:;[^\n]+)?$', r.stdout, flags=re.M)
                if r.returncode == 0 and len(ids) == 1:
                    job = int(ids[0])
                elif 'QOSMaxSubmitJobPerUserLimit' in r.stderr:
                    record['status'] = 'waiting_for_submission_quota'
                else:
                    record['status'] = 'submission_failed_requires_review'
                    record['error'] = r.stderr[-2500:]
                    record_failure(record)
                    return 1
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
            record['error'] = str(error)
        if job is not None:
            record.update(status='submitted', job=job, source_commit=a.release, followup_plan=plan)
        (a.output/'latest.json').write_text(json.dumps(record, indent=2))
        with (a.output/'attempts.jsonl').open('a') as log:
            log.write(json.dumps(record)+'\n')
        if job is not None:
            (a.output/'submission.json').write_text(json.dumps(record, indent=2))
            with a.vault_note.open('a') as note:
                scope=plan.get('scope','OHSVGP/OSGPR/ST-SVGP/MGPVAEのTask-1 validation。最終比較65本には数えない')
                note.write(f'\n## 後続ジョブの自動投入 {record["time"]}\n\nDoC job **{job}**、release `{a.release}`。{scope}。\n')
            template=plan.get('result_template',c+'/results/fair-three-domain-wandb-20260929/baseline-validation/job-{job}')
            return subprocess.call([sys.executable, str(Path(__file__).with_name('watch_doc_tracked_job.py')),
                '--job', str(job), '--result-template', template.replace('{job}',str(job)),
                '--seeds', *map(str,plan.get('seeds',[0])), '--output', str(a.output/'monitor'), '--vault-note', str(a.vault_note),
                '--kind', plan.get('kind','baseline-validation'), '--max-polls', '145'])
        time.sleep(600)
    record['status'] = 'submission_timeout_unverified'
    record_failure(record)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
