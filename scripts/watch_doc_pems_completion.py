#!/usr/bin/env python3
"""Durable ten-minute monitor; verify terminal artifacts and write a vault note."""
import argparse,datetime,hashlib,json,pathlib,re,shlex,subprocess,time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--job',type=int,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--vault-note',type=pathlib.Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
remote='''import json,pathlib,hashlib
root=pathlib.Path('/vol/bitbucket/nk523/hipposvgp-fair-20260929/results/pems')
rows=[]
for seed in (1,2,3):
 p=root/f'seed{seed}'/'online-JOBID'
 row={'seed':seed,'path':str(p)}
 for name in ('exit','completion'):
  if (p/(name+'.json')).exists():row[name]=json.loads((p/(name+'.json')).read_text())
 if 'completion' in row:
  d=json.loads((p/'result.json').read_text())
  row['valid_counts']=d['num_blocks']==50100 and d['delayed_observation_rows']==50099*65
  row['commit']=(p/'source_commit.txt').read_text().strip()
  row['hashes']={}
  for name in ('predictions.npz','result.json'):
   h=hashlib.sha256()
   with (p/name).open('rb') as f:
    for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
   row['hashes'][name]=h.hexdigest()
 rows.append(row)
print(json.dumps(rows))
'''.replace('JOBID',str(a.job))
def ssh(host,command):
 return subprocess.run(['/usr/bin/ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,command],capture_output=True,text=True,timeout=120)
def finish(status,record):
 (a.output/'terminal.json').write_text(json.dumps({'status':status,**record},indent=2)+'\n')
 with a.vault_note.open('a') as f:
  f.write(f"\n## 自動監視の終端確認 {record['time']}\n\nジョブ {a.job}: **{status}**。詳細・結果ハッシュ: `{a.output}/terminal.json`。主表採用の判定とは別。\n")
 print(status,flush=True)
for attempt in range(73):
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();record={'time':now,'job':a.job}
 try:
  r=ssh('gpucluster2',f'scontrol -o show job {a.job}')
  states=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in r.stdout.splitlines() if 'JobId=' in line]
  record.update(states=states,scheduler_returncode=r.returncode,scheduler_stderr=r.stderr)
  r=ssh('corgi','/usr/bin/python3 -c '+shlex.quote(remote))
  record.update(artifact_returncode=r.returncode,artifact_stderr=r.stderr)
  artifacts=json.loads(r.stdout) if r.returncode==0 else []
  record['artifacts']=artifacts
  success=len(artifacts)==3 and all(x.get('exit',{}).get('exit_code')==0 and x.get('completion',{}).get('status')=='complete' and x.get('valid_counts') for x in artifacts)
  ended=len(states)==3 and all(x.get('JobState') in terminal for x in states)
  exits=len(artifacts)==3 and all('exit' in x for x in artifacts)
  status='COMPLETED_AND_VERIFIED' if success else 'FAILED_OR_INCOMPLETE' if ended or exits else None
 except Exception as error:
  record['error']=str(error);status=None
 with (a.output/'monitor.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 (a.output/'latest.json').write_text(json.dumps(record,indent=2)+'\n')
 if status:
  finish(status,record);raise SystemExit(0 if success else 1)
 time.sleep(600)
finish('MONITOR_TIMEOUT_UNVERIFIED',record)
raise SystemExit(2)
