#!/usr/bin/env python3
"""Durable monthly ERA5-Land retrieval; submit three requests before waiting."""
import argparse,datetime,hashlib,json,os
from pathlib import Path
import cdsapi

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plan=json.loads(a.plan.read_text());status=a.output/'download-status.json'
state=json.loads(status.read_text()) if status.exists() else {'dataset':plan['dataset'],'requests':[]}
assert state['dataset']==plan['dataset']
def save():
    state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
client=cdsapi.Client(quiet=True,debug=False,retry_max=5,timeout=120)
try:
    # Record each returned request ID immediately. Resume existing remote jobs;
    # stopping the local downloader must not submit duplicate CDS computations.
    for request in plan['requests']:
        target=a.output/f"era5-land-{request['year']}-{request['month']}.nc"
        found=[r for r in state['requests'] if r['request']==request]
        assert len(found)<=1
        item=found[0] if found else {'request':request,'target':str(target),'status':'prepared'}
        if not found:state['requests'].append(item);save()
        if target.exists() or item.get('request_id'):continue
        remote=client.client.submit(plan['dataset'],request)
        item.update(request_id=remote.request_id,status='submitted');save()
    for item in state['requests']:
        target=Path(item['target'])
        if not target.exists():
            item['status']='waiting_or_downloading';save()
            partial=target.with_suffix('.partial');client.client.get_remote(item['request_id']).download(str(partial));partial.replace(target)
        item.update(status='downloaded',bytes=target.stat().st_size,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest());save()
    state['status']='all_downloaded';save()
except Exception as e:
    secret=Path(os.environ['CDSAPI_RC']).read_text().split('key:')[1].strip()
    state.update(status='failed_requires_review',message=str(e).replace(secret,'[REDACTED]'));save();raise SystemExit(1)
