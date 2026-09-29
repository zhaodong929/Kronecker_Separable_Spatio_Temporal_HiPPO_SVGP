"""Actual official-adapter execution at the current-hidden release boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def test_ohsvgp_current_hidden_is_used_only_after_its_release(tmp_path):
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(42)
    arrays = dict(calibration_y=rng.normal(size=(52,52)),stream_y=rng.normal(size=(143,52)),
        calibration_phi=np.ones((52,52,1)),stream_phi=np.ones((143,52,1)),
        calibration_times=np.arange(52)/51.,stream_times=np.arange(52,195)/51.,
        coordinates=rng.normal(size=(52,2)),train_indices=np.arange(42),
        fit_indices=np.arange(38),validation_indices=np.arange(38,42),
        test_indices=np.arange(42,52),block_start=np.arange(143),block_stop=np.arange(1,144))
    meta = dict(split_seed=5,xlag=dict(delay_weeks=1),
                target_standardization=dict(fit_scope='Task-1 visible locations only'))
    def run(name, data):
        path = tmp_path/name;path.mkdir()
        np.savez(path/'protocol.npz',**data)
        (path/'protocol.json').write_text(json.dumps(meta))
        command = [sys.executable,str(root/'scripts/run_covid_ohsvgp_own_theta.py'),
            '--protocol-npz',str(path/'protocol.npz'),'--protocol-json',str(path/'protocol.json'),
            '--output-dir',str(path/'run'),'--kernel','rbf','--inducing-size','4',
            '--rff-sample-size','16','--basis-grid-size','32','--calibration-iterations','2',
            '--task1-check-interval','1','--task1-min-steps','2','--calibration-batch-size','128',
            '--max-blocks','3','--delayed-observations','--seed','5','--device','cpu']
        subprocess.run(command,cwd=root,check=True,capture_output=True,text=True,timeout=90,
                       env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        result = json.loads((path/'run/result.json').read_text())
        assert result['num_initial_observed_space'] == 52
        assert result['delayed_observation_rows'] == 20
        with np.load(path/'run/predictions.npz') as a:
            return a['pred_mean'].copy(),a['pred_var'].copy()
    base = run('base',arrays)
    changed = {**arrays,'stream_y':arrays['stream_y'].copy()}
    changed['stream_y'][0,42:] += 30
    predicted = run('current_hidden',changed)
    np.testing.assert_array_equal(base[0][0],predicted[0][0])
    np.testing.assert_array_equal(base[1][0],predicted[1][0])
    assert np.max(np.abs(base[0][1]-predicted[0][1])) > 1e-8
    changed = {**arrays,'stream_y':arrays['stream_y'].copy()}
    changed['stream_y'][2:] += 30
    predicted = run('future',changed)
    np.testing.assert_array_equal(base[0][:2],predicted[0][:2])
