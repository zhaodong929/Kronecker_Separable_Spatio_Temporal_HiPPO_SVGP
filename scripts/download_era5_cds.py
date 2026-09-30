#!/usr/bin/env python3
"""Retrieve the exact ERA5-Land source window; credentials stay outside artifacts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import cdsapi

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plan=json.loads(a.plan.read_text());state={'dataset':plan['dataset'],'requests':[]}
def save():
    state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp=a.output/'download-status.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(a.output/'download-status.json')
client=cdsapi.Client(quiet=True,debug=False,retry_max=5,timeout=120)
for request in plan['requests']:
    target=a.output/f"era5-land-{request['year']}-{request['month']}.nc"
    item={'request':request,'target':str(target),'status':'requesting'};state['requests'].append(item);save()
    try:
        if not target.exists():
            partial=target.with_suffix('.partial')
            client.retrieve(plan['dataset'],request,str(partial));partial.replace(target)
        item.update(status='downloaded',bytes=target.stat().st_size,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest());save()
    except Exception as e:
        secret=Path(os.environ['CDSAPI_RC']).read_text().split('key:')[1].strip()
        item.update(status='failed',message=str(e).replace(secret,'[REDACTED]'));save();raise SystemExit(1)
state['status']='all_downloaded';save()
