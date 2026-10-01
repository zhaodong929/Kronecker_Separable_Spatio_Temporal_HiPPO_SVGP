"""Sequential numerical qualification before any shared-GPU paper runs."""
import json
import os
from pathlib import Path
import subprocess
import sys

C = Path(sys.argv[1]).resolve()
code = C / 'source'
release = (code / 'SOURCE_COMMIT').read_text().strip()
campaign = 'gh200-port-qualification-20261001'
methods = {'kronhippo_svgp': 'env-routeb', 'ohsvgp': 'env-routeb',
           'st_svgp': 'env-jax-gpu', 'mgpvae': 'env-jax-gpu', 'osgpr': 'env-osgpr'}
results = []
for method, environment in methods.items():
    worker = C / environment / 'bin/python'
    if not worker.exists():
        results.append(dict(method=method, status='environment_missing'))
        continue
    output = C / 'results' / campaign / os.environ['SLURM_JOB_ID'] / method
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = output.parent / (method + '-spec.json')
    configuration = C / 'configurations' / (method + '.json')
    prepared = C / 'tiny'
    spec.write_text(json.dumps(dict(entity='harrisonzhu', project='KronHiPPO-STGP',
        campaign=campaign, dataset='synthetic-qualification', method=method,
        split_seed=0, training_seed=0, stage='qualification', source_commit=release,
        worker_python=str(worker), main_table_admitted=False,
        input_files=[str(configuration)] + [str(prepared/name) for name in
            ('manifest.json', 'stream.npz', 'selection-stream.npz', 'features.npz')]
            + [str(p) for p in sorted((C/'environment-records').glob('*.txt'))],
        timing_scope='GH200 ARM numerical diagnostic; not a paper speed measurement')))
    command = [str(C/'env-tracking/bin/python'), str(code/'scripts/run_tracked_experiment.py'),
        '--spec', str(spec), '--output', str(output), '--', str(worker),
        str(code/'scripts/qualify_task_stream_device.py'), '--prepared', str(prepared),
        '--configuration', str(configuration), '--output', str(output)]
    status = subprocess.run(command, cwd=code, check=False).returncode
    results.append(dict(method=method, exit_code=status, output=str(output),
                        status='passed' if status == 0 else 'failed'))
summary = C/'results'/campaign/os.environ['SLURM_JOB_ID']/'summary.json'
summary.parent.mkdir(parents=True, exist_ok=True)
summary.write_text(json.dumps(dict(results=results, main_table_admitted=False), indent=2))
sys.exit(0 if all(r['status'] == 'passed' for r in results) else 1)
