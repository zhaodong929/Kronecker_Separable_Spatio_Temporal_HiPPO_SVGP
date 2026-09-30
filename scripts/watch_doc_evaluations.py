#!/usr/bin/env python3
"""Evaluate newly completed monitored runs, with ten-minute discovery intervals."""
import argparse
import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def candidates(root):
    found = {}
    for path in sorted(root.rglob('latest.json')):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for row in record.get('results', []):
            if row.get('exit', {}).get('exit_code') != 0:
                continue
            match = re.fullmatch(
                r'/vol/bitbucket/nk523/hipposvgp-fair-20260929/results/'
                r'fair-three-domain-wandb-20260929/(covid|pems|era5)/'
                r'(kronhippo_svgp|osgpr|ohsvgp|st_svgp|mgpvae)/seed(\d+)/job-(\d+)',
                row.get('path', ''))
            if not match:
                continue
            if not any(a.get('terminal.json', {}).get('status') == 'completed_and_verified'
                       for a in row.get('attempts', [])):
                continue
            found[row['path']] = match.groups()
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--monitors', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--release', required=True)
    p.add_argument('--vault-note', type=Path, required=True)
    p.add_argument('--max-polls', type=int, default=1008)
    a = p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', a.release):
        raise ValueError('Full immutable release SHA required')
    a.output.mkdir(parents=True, exist_ok=True)
    state_path = a.output/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    c = '/vol/bitbucket/nk523/hipposvgp-fair-20260929'
    source = c+'/releases/'+a.release+'/source'
    for _ in range(a.max_polls):
        for root, (dataset, method, seed, job) in candidates(a.monitors).items():
            if state.get(root, {}).get('status') == 'synchronized':
                continue
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            protocol = c+'/protocol/'+('covid-v2' if dataset == 'covid' else dataset)+'/seed'+seed+'/protocol.npz'
            # Credentials remain inside the remote process and are never printed.
            code = '\n'.join([
                'import json,os,pathlib,subprocess',
                f'root=pathlib.Path({root!r})',
                "marker=root/'evaluation-tracking.json'",
                "if marker.exists() and json.loads(marker.read_text()).get('status')=='synchronized': raise SystemExit(0)",
                f'os.chdir({source!r})',
                f'os.environ["WANDB_API_KEY"]=pathlib.Path({(c+"/credentials/wandb-api-key")!r}).read_text().strip()',
                'os.environ["WANDB_DISABLE_GIT"]="true"',
                'os.environ["OMP_NUM_THREADS"]="2"',
                'os.environ["OPENBLAS_NUM_THREADS"]="2"',
                f'subprocess.run({[c+"/env-tracking/bin/python", "scripts/evaluate_three_domain_run.py", "--protocol", protocol, "--run", root, "--method", method]!r},check=True)',
                f'subprocess.run({[c+"/env-tracking/bin/python", "scripts/publish_three_domain_evaluation.py", "--run", root, "--dataset", dataset]!r},check=True)',
                "assert json.loads(marker.read_text())['status']=='synchronized'",
            ])
            log = a.output/f'{dataset}-{method}-seed{seed}-job{job}.log'
            try:
                with log.open('a') as stream:
                    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                        'corgi','/usr/bin/python3 -c '+shlex.quote(code)],
                        stdout=stream, stderr=subprocess.STDOUT, timeout=900)
                status = 'synchronized' if result.returncode == 0 else 'evaluation_failed_requires_review'
            except (OSError, subprocess.TimeoutExpired) as error:
                status = 'evaluation_failed_requires_review'
                with log.open('a') as stream:
                    stream.write('\n'+str(error)+'\n')
            previous = state.get(root, {}).get('status')
            state[root] = dict(status=status, time=now, log=str(log), main_table_admitted=False)
            temporary = state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2)+'\n')
            temporary.replace(state_path)
            if previous != status:
                with a.vault_note.open('a') as note:
                    note.write(f'\n共通評価の自動確認 {now}: `{dataset}/{method}/seed{seed}/job{job}` '
                               f'**{status}**。元データ照合・元単位採点・W&B評価artifact同期。'
                               f'主表採用は別途審査。詳細: `{log}`。\n')
        (a.output/'heartbeat.json').write_text(json.dumps(dict(
            time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            synchronized=sum(x['status']=='synchronized' for x in state.values()),
            requires_review=sum(x['status']!='synchronized' for x in state.values()))))
        time.sleep(600)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
