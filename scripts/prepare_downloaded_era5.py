#!/usr/bin/env python3
"""Wait for CDS, verify/rebuild all splits, stage atomically on DoC, release gate."""
import argparse,datetime,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--downloads',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ready',type=Path,required=True);p.add_argument('--release',required=True);p.add_argument('--vault-note',type=Path,required=True);a=p.parse_args()
status=a.ready.with_name('era5-preparation-status.json');a.ready.parent.mkdir(parents=True,exist_ok=True)
def record(state,**kwargs):
    status.write_text(json.dumps(dict(status=state,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs),indent=2))
try:
    for _ in range(1008):
        d=json.loads((a.downloads/'download-status.json').read_text())
        if d.get('status')=='all_downloaded':break
        if d.get('status')=='failed_requires_review':raise RuntimeError('CDS retrieval failed; inspect its status')
        record('waiting_for_cds');time.sleep(600)
    else:raise TimeoutError('CDS wait exceeded seven days')
    record('verifying_source_and_building_protocols')
    subprocess.run(['/tmp/hipposvgp-audit-env-20260929/bin/python','scripts/prepare_fair_era5.py','--downloads',str(a.downloads),'--output',str(a.output)],check=True)
    evidence=json.loads((a.output/'source-verification.json').read_text());assert evidence['status']=='all_sites_full_period_target_and_372_hour_covariates_verified'
    hashes={str(f.relative_to(a.output)):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in sorted(a.output.glob('seed*/protocol.*'))}
    assert len(hashes)==10
    (a.output/'protocol-sha256.json').write_text(json.dumps(hashes,indent=2))
    c='/vol/bitbucket/nk523/hipposvgp-fair-20260929';stage=c+'/protocol/era5-staging-'+a.release[:8];final=c+'/protocol/era5'
    record('copying_verified_protocols')
    subprocess.run(['ssh','-o','BatchMode=yes','corgi',f'mkdir -p {shlex.quote(stage)}'],check=True)
    subprocess.run(['rsync','-a','--partial',str(a.output)+'/',f'corgi:{stage}/'],check=True)
    code=f'''import hashlib,json,os,pathlib,sys
sys.path.insert(0,{(c+'/releases/'+a.release+'/source')!r})
from baselines.era5_protocol import ERA5Protocol
stage=pathlib.Path({stage!r});final=pathlib.Path({final!r})
for name,expected in json.loads((stage/'protocol-sha256.json').read_text()).items():
    assert hashlib.file_digest((stage/name).open('rb'),'sha256').hexdigest()==expected
for seed in range(5):
    p=ERA5Protocol(stage/f'seed{{seed}}/protocol.npz');assert p.task1().locations.size==800
    assert all(p.week(i).delayed_hidden is None for i in [0,500,1673])
assert not final.exists(),'Do not overwrite an existing protocol'
stage.rename(final)
print('Verified five ERA5 splits on DoC')
'''
    subprocess.run(['ssh','-o','BatchMode=yes','corgi','OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '+c+'/env-routeb/bin/python -c '+shlex.quote(code)],check=True)
    gate=dict(status='source_and_protocol_verified_on_doc',source_verification=evidence,source_commit=a.release,remote_protocol=final,protocol_sha256=hashes)
    tmp=a.ready.with_suffix('.tmp');tmp.write_text(json.dumps(gate,indent=2));tmp.replace(a.ready);record('ready_for_gpu_qualification')
    with a.vault_note.open('a') as f:f.write('\nERA5取得・全地点照合・5分割生成・DoC入力hash検証完了。投入gate: `'+str(a.ready)+'`。各方式のGPU検証と最終比較は別途必要。\n')
except Exception as e:
    record('failed_requires_review',error=str(e))
    with a.vault_note.open('a') as f:f.write('\nERA5データ準備に要確認エラー。`'+str(status)+'`。投入gateは作成しない。\n')
    raise
