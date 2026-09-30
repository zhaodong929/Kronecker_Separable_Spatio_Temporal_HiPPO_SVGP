#!/usr/bin/env python3
"""Publish the complete target grid so unsubmitted comparisons stay visible."""
import argparse
import json
from pathlib import Path
import wandb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('benchmarks/three_domain/submissions.json'))
    a=p.parse_args();data=json.loads(a.manifest.read_text());rows=data['rows']
    keys=('dataset','method','split_seed','stage','submission_status','job_id','run_url','pending_requirement')
    assert len(rows)==65 and len({(r['dataset'],r['method'],r['split_seed']) for r in rows})==65
    with wandb.init(entity=data['entity'],project=data['project'],id='campaign-20260929',
        resume='allow',name='Comparison campaign — submission ledger',group=data['campaign'],
        job_type='campaign-index',tags=['campaign-index'],save_code=False,
        settings=wandb.Settings(x_disable_stats=True)) as run:
        run.log({'campaign/submission_ledger':wandb.Table(columns=list(keys),data=[[r[k] for k in keys] for r in rows])})
        run.summary.update(dict(planned_comparisons=65,
            submitted_comparisons=sum(r['job_id'] is not None for r in rows),
            unsubmitted_comparisons=sum(r['job_id'] is None for r in rows),
            verified_completed_comparisons=sum(r['submission_status']=='completed_and_verified' for r in rows),
            failed_requires_review_comparisons=sum(r['submission_status']=='failed_requires_review' for r in rows),
            submitted_not_yet_verified_comparisons=sum(r['job_id'] is not None and r['submission_status'] not in
                {'completed_and_verified','failed_requires_review'} for r in rows),
            as_of=data['as_of'],
            status_scope='Submission snapshot; use individual runs for live execution status',
            ablations=False,compute='DoC only; at most three allocated GPUs'))
        artifact=wandb.Artifact('comparison-submission-ledger',type='campaign-manifest')
        artifact.add_file(str(a.manifest));run.log_artifact(artifact)
        print(run.url)


if __name__=='__main__':main()
