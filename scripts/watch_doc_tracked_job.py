#!/usr/bin/env python3
"""Ten-minute, process-independent terminal verification for tracked DoC jobs."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True, type=int)
    p.add_argument('--result-template', required=True, help='Remote path with {seed}')
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--vault-note', type=Path, required=True)
    p.add_argument('--max-polls', type=int, default=73)
    p.add_argument('--kind', choices=['final', 'baseline-validation'], default='final')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    script = '''import pathlib,json,hashlib
rows=[]
for name in PATHS:
 p=pathlib.Path(name); row={'path':name}
 if (p/'exit.json').exists():row['exit']=json.loads((p/'exit.json').read_text())
 attempts=[]
 for d in sorted((p/'tracking').glob('*')):
  item={'attempt':d.name}
  for f in ('terminal.json','wandb.json'):
   if (d/f).exists():item[f]=json.loads((d/f).read_text())
  attempts.append(item)
 row['attempts']=attempts
 for method in ('ohsvgp','osgpr','st_svgp'):
  marker=p/method/'completed.json'
  if marker.exists():row.setdefault('validation_completed',[]).append(method)
 if row.get('exit',{}).get('exit_code')==0:
  row['sha256']={}
  for f in ('result.json','predictions.npz'):
   if not (p/f).is_file():continue
   h=hashlib.sha256()
   with (p/f).open('rb') as stream:
    for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
   row['sha256'][f]=h.hexdigest()
 rows.append(row)
print(json.dumps(rows))
'''.replace('PATHS', repr([a.result_template.format(seed=s) for s in a.seeds]))
    status = 'monitor_timeout_unverified'
    for attempt in range(a.max_polls):
        record = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), job=a.job)
        try:
            r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                'corgi','/usr/bin/python3 -c '+shlex.quote(script)], capture_output=True, text=True, timeout=90)
            record['ssh_exit_code'] = r.returncode
            if r.returncode:
                record['error'] = r.stderr[-1000:]
            else:
                rows = json.loads(r.stdout)
                record['results'] = rows
                if all('exit' in row for row in rows):
                    if a.kind == 'baseline-validation':
                        success = all(row['exit']['exit_code'] == 0 and
                            set(row.get('validation_completed',[])) == {'ohsvgp','osgpr','st_svgp'} for row in rows)
                    else:
                        success = all(row['exit']['exit_code'] == 0 and row.get('sha256',{}).keys() >= {'result.json','predictions.npz'} and
                            any(x.get('terminal.json',{}).get('status') == 'completed_and_verified' for x in row['attempts']) for row in rows)
                    status = 'completed_and_verified' if success else 'failed_or_incomplete'
                    record['status'] = status
        except Exception as error:
            record['error'] = str(error)
        (a.output/'latest.json').write_text(json.dumps(record,indent=2))
        with (a.output/'monitor.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        if 'status' in record:
            break
        time.sleep(600)
    record['status'] = status
    (a.output/'terminal.json').write_text(json.dumps(record,indent=2))
    with a.vault_note.open('a') as f:
        f.write(f"\n## W&B再実行の終了確認 {record['time']}\n\nDoC {a.job}: **{status}**。記録: `{a.output}/terminal.json`。全比較・主表採用とは別の判定。\n")
    return 0 if status == 'completed_and_verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
