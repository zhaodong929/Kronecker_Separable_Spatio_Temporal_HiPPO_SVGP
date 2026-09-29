#!/usr/bin/env python3
"""Predeclared COVID Task-1 capacity/budget checks; no stream/test selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['ohsvgp', 'osgpr', 'st_svgp', 'mgpvae'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--release', required=True)
    p.add_argument('--compute-root', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    env, tests, capacities, budgets = {
        'ohsvgp': ('env-routeb', ['tests/test_ohsvgp_release_boundary.py',
            'tests/test_ohsvgp_lengthscale_learning.py'], [32, 64], [500, 1000]),
        'osgpr': ('env-osgpr', ['tests/test_osgpr_release_boundary.py'], [16, 32], [25, 100]),
        'st_svgp': ('env-jax-gpu', ['tests/test_st_svgp_dense.py',
            'tests/test_causal_mean.py'], [16, 32], [250, 500]),
        'mgpvae': ('env-jax-gpu', ['tests/test_mgpvae_spatial_moments.py',
            'tests/test_mgpvae_selected.py', 'tests/test_mgpvae_partial.py',
            'tests/test_mgpvae_official_filter.py', 'tests/test_mgpvae_mixture_metrics.py'], [2, 4], [250, 500]),
    }[a.method]
    worker = str(a.compute_root/env/'bin/python')
    protocol = a.compute_root/'protocol/covid-v2/seed5'
    inputs = [str(protocol/'protocol.npz'), str(protocol/'protocol.json')]
    a.output.mkdir(parents=True, exist_ok=True)
    os.environ['MGPVAE_SOURCE'] = str(a.compute_root/'official/MGPVAE')
    # These tests run on the allocated GPU node, before any tuning candidates.
    with (a.output/'tests.txt').open('w') as log:
        subprocess.run([worker, '-m', 'pytest', '-q', *tests], stdout=log,
            stderr=subprocess.STDOUT, cwd=root, check=True)
    if a.method == 'osgpr':
        probe = "import tensorflow as tf; assert tf.config.list_physical_devices('GPU'); print(tf.config.list_physical_devices('GPU'))"
    elif a.method in {'st_svgp', 'mgpvae'}:
        probe = "import jax; assert jax.default_backend() == 'gpu'; print(jax.devices())"
    else:
        probe = "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
    with (a.output/'gpu-probe.txt').open('w') as log:
        subprocess.run([worker, '-c', probe], stdout=log, stderr=subprocess.STDOUT, check=True)
    outcomes = []
    for capacity in capacities:
        for budget in budgets:
            output = a.output/f'capacity{capacity}-budget{budget}'
            output.mkdir()
            common = ['--protocol-npz', inputs[0], '--protocol-json', inputs[1], '--seed', '5']
            if a.method == 'ohsvgp':
                command = [worker, 'scripts/run_covid_ohsvgp_own_theta.py', *common,
                    '--output-dir', str(output), '--kernel', 'rbf', '--inducing-size', str(capacity),
                    '--rff-sample-size', '256', '--basis-grid-size', '1024',
                    '--calibration-iterations', str(budget), '--task1-check-interval', '25',
                    '--task1-min-steps', str(budget), '--calibration-batch-size', '256',
                    '--calibration-only', '--device', 'cuda']
                result = output/'calibration.json'
            elif a.method == 'osgpr':
                command = [worker, 'scripts/run_official_bui_osgpr_era5.py', *common,
                    '--output', str(output/'result.json'), '--blockwise-output', str(output/'blocks.csv'),
                    '--mt', '2', '--ms', str(capacity), '--adaptive',
                    '--adaptive-calibration-steps', str(budget), '--task1-validation-only',
                    '--device', 'cuda']
                result = output/'result.json'
            elif a.method == 'st_svgp':
                command = [worker, 'baselines/covid_long_setting_b/adapters/run_st_svgp.py', *common,
                    '--output-dir', str(output), '--spatial-inducing', str(capacity),
                    '--task1-iterations', str(budget), '--task1-check-interval', '25',
                    '--task1-min-steps', str(budget), '--task1-validation-only']
                result = output/'task1_validation.json'
            else:
                command = [worker, 'scripts/run_mgpvae_causal.py', *common,
                    '--official-source', os.environ['MGPVAE_SOURCE'], '--output-dir', str(output),
                    '--latent', str(capacity), '--iterations', str(budget), '--check-every', '25',
                    '--prediction-samples', '128', '--validation-only']
                result = output/'calibration.json'
            spec = dict(entity='harrisonzhu', project='KronHiPPO-STGP',
                campaign='fair-three-domain-wandb-20260929', dataset='covid', method=a.method,
                split_seed=5, training_seed=5, stage='validation', source_commit=a.release,
                worker_python=worker, input_files=inputs, capacity=capacity, max_budget=budget,
                qualification_tests=tests, test_data_used=False, main_table_admitted=False,
                purpose='Task-1-only budget and capacity selection; online-update budgets remain unqualified')
            (output/'spec.json').write_text(json.dumps(spec, indent=2))
            subprocess.run([sys.executable, 'scripts/run_tracked_experiment.py',
                '--spec', str(output/'spec.json'), '--output', str(output), '--', *command],
                cwd=root, check=True)
            payload = json.loads(result.read_text())
            outcomes.append(dict(capacity=capacity, budget=budget, result=payload))
            (a.output/'validation-results.json').write_text(json.dumps(outcomes, indent=2))
    (a.output/'completed.json').write_text(json.dumps(dict(status='validation_candidates_complete',
        formal_comparisons_submitted=False, main_table_admitted=False)))


if __name__ == '__main__':
    main()
