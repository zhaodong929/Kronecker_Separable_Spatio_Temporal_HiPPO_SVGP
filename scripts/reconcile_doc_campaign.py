#!/usr/bin/env python3
"""Refresh submission ledger from per-run receipts and verified evaluation state."""
import argparse
import datetime
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MONITORS = ROOT/'outputs/2026-09-29-doc-campaign'
PATTERN = re.compile(r'.*/(covid|pems|era5)/(kronhippo_svgp|osgpr|ohsvgp|st_svgp|mgpvae)/seed(\d+)/job-(\d+)')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    ledger_path = ROOT/'benchmarks/three_domain/submissions.json'
    ledger = json.loads(ledger_path.read_text())
    lookup = {(r['dataset'], r['method'], r['split_seed']): r for r in ledger['rows']}
    state = json.loads((MONITORS/'common-evaluation-watch/state.json').read_text())

    def target(path):
        match = PATTERN.fullmatch(path)
        if not match:
            return None
        dataset, method, seed, job = match.groups()
        row = lookup[('pems_bay' if dataset == 'pems' else dataset, method, int(seed))]
        if (row.get('job_id') or 0) > int(job):
            return None  # Historical failures cannot override a later retry.
        if row.get('job_id') != int(job):
            row.update(job_id=int(job), run_url=None, submission_status='submitted_with_per_split_validation_gate')
        return row

    for receipt in sorted(MONITORS.rglob('submission.json')):
        submitted = json.loads(receipt.read_text())
        plan = submitted.get('followup_plan', {})
        if plan.get('kind') != 'final':
            continue
        for seed in plan['seeds']:
            row = target(plan['result_template'].format(seed=seed, job=submitted['job']))
            if row and row['submission_status'] != 'completed_and_verified':
                row['pending_requirement'] = 'GPU qualification, Task-1 selection, full final stream and independent evaluation pending'

    for monitor in sorted(MONITORS.rglob('latest.json')):
        for observed in json.loads(monitor.read_text()).get('results', []):
            path = observed.get('path', '')
            row = target(path)
            if row is None:
                continue
            attempts = [r for r in observed.get('attempts', [])
                        if r.get('terminal.json', {}).get('status') == 'completed_and_verified']
            if observed.get('exit', {}).get('exit_code') == 0 and attempts:
                run_id = attempts[-1]['wandb.json']['id']
                row['run_url'] = 'https://wandb.ai/harrisonzhu/KronHiPPO-STGP/runs/'+run_id
                if state.get(path, {}).get('status') == 'synchronized':
                    row['submission_status'] = 'completed_and_verified'
                    row['pending_requirement'] = 'Source-aligned common evaluation and W&B artifacts verified; manuscript admission remains pending'
            elif 'exit' in observed:
                row['submission_status'] = 'failed_requires_review'
                row['pending_requirement'] = 'Inspect terminal records and recover before counting this comparison complete'
    ledger['as_of'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    destination=args.output or ledger_path
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(ledger, indent=2)+'\n');temporary.replace(destination)
    summary = dict(submitted=sum(r.get('job_id') is not None for r in ledger['rows']),
                   verified=sum(r['submission_status']=='completed_and_verified' for r in ledger['rows']))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
