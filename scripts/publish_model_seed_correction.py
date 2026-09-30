#!/usr/bin/env python3
"""Append a transparent seed-provenance correction; preserve original run configs."""
import json,hashlib
from pathlib import Path
import wandb
root=Path(__file__).resolve().parents[1]
ledger=json.loads((root/'benchmarks/three_domain/submissions.json').read_text())
rows=[r for r in ledger['rows'] if r['method']=='kronhippo_svgp' and r['submission_status']=='completed_and_verified']
assert len(rows)==8
record=dict(scope='Metadata correction only; no numerical results changed',actual_model_seed=0,reason='COVID/PEMS batch calibration uses default model_seed=0; strict online Torch and RFF builder use fixed seed=0. Original top-level training_seed mistakenly repeated the spatial split seed.',rows=[dict(dataset=r['dataset'],split_seed=r['split_seed'],original_top_level_training_seed=r['split_seed'],actual_model_seed=0,parent_run=r['run_url']) for r in rows],source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['scripts/run_iclr_era5_routeb_batch.py','scripts/run_iclr_era5_routeb_strict_online.py','slurm/fair_three_domain/covid_tracked_worker.sh','slurm/fair_three_domain/pems_tracked_worker.sh']})
out=root/'outputs/2026-09-29-doc-campaign/seed-provenance-correction';out.mkdir(parents=True,exist_ok=True);path=out/'correction.json';path.write_text(json.dumps(record,indent=2))
with wandb.init(entity='harrisonzhu',project='KronHiPPO-STGP',id='seed-provenance-20260930',resume='allow',job_type='provenance-correction',group=ledger['campaign'],config=dict(scope=record['scope']),save_code=False,settings=wandb.Settings(x_disable_stats=True)) as run:
    artifact=wandb.Artifact('proposal-model-seed-correction',type='provenance-correction');artifact.add_file(str(path))
    for name in record['source_sha256']:artifact.add_file(str(root/name),name=name)
    run.log_artifact(artifact);run.summary['affected_runs']=8;url=run.url
api=wandb.Api()
for row in rows:
    parent=api.run('harrisonzhu/KronHiPPO-STGP/'+row['run_url'].rsplit('/',1)[1])
    parent.summary['provenance/model_seed_actual']=0
    parent.summary['provenance/spatial_split_seed']=row['split_seed']
    parent.summary['provenance/model_seed_correction']=url
    parent.summary.update()
print(url)
