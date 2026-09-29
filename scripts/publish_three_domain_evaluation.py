#!/usr/bin/env python3
"""Attach independently verified scores to the original team experiment."""
import argparse
import json
from pathlib import Path
import wandb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',choices=['covid','pems','era5'],required=True)
    a=p.parse_args()
    evaluation=json.loads((a.run/'common-evaluation.json').read_text())
    if evaluation.get('status')!='complete_source_verified':raise ValueError('Verified complete evaluation required')
    attempts=[]
    for path in (a.run/'tracking').glob('*/wandb.json'):
        terminal=path.with_name('terminal.json')
        if terminal.exists() and json.loads(terminal.read_text()).get('status')=='completed_and_verified':attempts.append(path)
    if not attempts:raise ValueError('No verified finished experiment tracking record')
    original=json.loads(max(attempts,key=lambda p:p.stat().st_mtime).read_text())
    entity,project='harrisonzhu','KronHiPPO-STGP'
    api=wandb.Api(timeout=30)
    parent=api.run(f"{entity}/{project}/{original['id']}")
    artifacts=list(parent.logged_artifacts())
    if parent.state!='finished' or not artifacts:raise ValueError('Original team run/artifact has not synchronized')
    evaluator_hash=evaluation['evaluator_source_sha256']['benchmarks/three_domain/evaluation.py'][:8]
    identity=f"eval-{original['id']}-{evaluator_hash}"
    logs=a.run/'evaluation-logs';logs.mkdir(exist_ok=True)
    with wandb.init(entity=entity,project=project,id=identity,resume='allow',dir=str(logs),
        name=f"Verified scores: {a.dataset}/{evaluation['method']}/seed{evaluation['split_seed']}",
        group='fair-three-domain-wandb-20260929',job_type='evaluation',save_code=False,
        config=dict(parent_run=parent.url,dataset=a.dataset,method=evaluation['method'],
            split_seed=evaluation['split_seed'],evaluator_hash=evaluator_hash,
            main_table_admitted=False),settings=wandb.Settings(x_disable_stats=True)) as run:
        for artifact in artifacts:run.use_artifact(artifact)
        scores=evaluation['original_scale_scores']
        run.summary.update({f'evaluation/{k}':v for k,v in scores.items()})
        run.summary.update(dict(status='complete_source_verified',metric_scale=evaluation['metric_scale'],
            main_table_admitted=False,original_run=parent.url))
        run.log({'evaluation/coverage':wandb.Table(columns=['nominal','observed'],
            data=list(zip(scores['levels'],scores['coverage'])))})
        artifact=wandb.Artifact(f"verified-scores-{original['id']}",type='evaluation')
        artifact.add_file(str(a.run/'common-evaluation.json'))
        for independent in sorted(a.run.glob('independent-*.json')):
            artifact.add_file(str(independent))
        run.log_artifact(artifact)
        link=run.url
    parent.summary.update({**{f'verified_original_scale/{k}':v for k,v in scores.items()},
        'independent_evaluation_url':link,'source_protocol_verified':True,
        'verified_metric_scale':evaluation['metric_scale'],'main_table_admitted':False})
    (a.run/'evaluation-tracking.json').write_text(json.dumps(dict(status='synchronized',
        id=identity,url=link,original_run=parent.url,main_table_admitted=False),indent=2))
    print(link)


if __name__=='__main__':main()
