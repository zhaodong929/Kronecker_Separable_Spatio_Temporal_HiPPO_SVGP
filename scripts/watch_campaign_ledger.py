#!/usr/bin/env python3
"""Refresh the live 65-run ledger; publish only material changes every ten minutes."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--vault-note',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
manifest=a.output/'submissions.json';marker=a.output/'published.json';last=json.loads(marker.read_text())['sha256'] if marker.exists() else None
for _ in range(2160):
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        subprocess.run(['/usr/bin/python3','scripts/reconcile_doc_campaign.py','--output',str(manifest)],check=True,stdout=subprocess.DEVNULL)
        rows=json.loads(manifest.read_text())['rows'];fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
        counts=dict(verified=sum(r['submission_status']=='completed_and_verified' for r in rows),ever_submitted=sum(r.get('job_id') is not None for r in rows),failed_requires_review=sum(r['submission_status']=='failed_requires_review' for r in rows))
        if fingerprint!=last:
            with (a.output/'publisher.log').open('a') as log:
                subprocess.run(['/tmp/hipposvgp-audit-env-20260929/bin/python','scripts/publish_campaign_index.py','--manifest',str(manifest)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180,env={**os.environ,'WANDB_DISABLE_GIT':'true'})
            last=fingerprint;marker.write_text(json.dumps(dict(time=now,sha256=last,**counts)))
            with a.vault_note.open('a') as f:f.write(f'\n全比較の状態更新 {now}: '+json.dumps(counts)+'。ever_submittedは失敗履歴も含み、完了数ではない。W&B campaign indexを同期。\n')
        status=dict(status='watching',time=now,**counts)
    except Exception as e:status=dict(status='ledger_update_failed_requires_review',time=now,error=str(e))
    (a.output/'heartbeat.json').write_text(json.dumps(status,indent=2));time.sleep(600)
