#!/usr/bin/env python3
"""Low-frequency, read-only Slurm monitor with no accounting-DB dependency."""
import argparse, datetime, json, pathlib, re, subprocess, time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--job',type=int,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--interval',type=int,default=600)
p.add_argument('--tasks',type=int,default=3)
a=p.parse_args()
if a.interval < 60: p.error('interval must be at least 60 seconds')
a.output.mkdir(parents=True,exist_ok=True)
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
last=None
for _ in range(73):
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpucluster2',
           f'scontrol -o show job {a.job}'],text=True,capture_output=True,timeout=90)
        rows=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in r.stdout.splitlines() if 'JobId=' in line]
        record={'time':now,'returncode':r.returncode,'jobs':rows,'stderr':r.stderr}
    except (subprocess.TimeoutExpired,OSError) as error:
        rows=[]; record={'time':now,'error':str(error)}
    with (a.output/'monitor.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    state=[(row.get('ArrayTaskId'),row.get('JobState'),row.get('ExitCode')) for row in rows]
    if state!=last: print(json.dumps({'time':now,'state':state}),flush=True); last=state
    if len(rows)==a.tasks and all(row.get('JobState') in terminal for row in rows):
        success=all(row.get('JobState')=='COMPLETED' and row.get('ExitCode')=='0:0' for row in rows)
        (a.output/'terminal.json').write_text(json.dumps({'scheduler_success':success,**record},indent=2)+'\n')
        raise SystemExit(0 if success else 1)
    time.sleep(a.interval)
raise SystemExit('Monitoring limit exceeded; completion is unverified')
