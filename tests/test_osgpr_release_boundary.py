"""Exercise the pinned Bui model through its actual causal adapter."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def test_osgpr_hidden_observation_changes_only_predictions_after_release(tmp_path):
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(12)
    coords = rng.normal(size=(6, 2))
    arrays = dict(calibration_y=rng.normal(size=(4, 6)), stream_y=rng.normal(size=(3, 6)),
        task1_calibration_mean=np.zeros((4, 6)), task1_stream_mean=np.zeros((3, 6)),
        calibration_times=np.arange(4.), stream_times=np.arange(4., 7.), coordinates=coords,
        train_indices=np.arange(4), fit_indices=np.arange(3), validation_indices=np.array([3]),
        test_indices=np.array([4, 5]), block_start=np.arange(3), block_stop=np.arange(1, 4),
        inducing_coords_ms2=coords[:2])
    def run(name, values):
        output = tmp_path/name
        output.mkdir()
        np.savez(output/'protocol.npz', **values)
        (output/'protocol.json').write_text(json.dumps({'task1_observed_indices': list(range(6))}))
        command = [sys.executable, str(root/'scripts/run_official_bui_osgpr_era5.py'),
            '--protocol-npz', str(output/'protocol.npz'), '--output', str(output/'result.json'),
            '--blockwise-output', str(output/'blocks.csv'), '--predictions-output', str(output/'predictions.npz'),
            '--mt', '2', '--ms', '2', '--adaptive', '--adaptive-calibration-steps', '2',
            '--adaptive-online-steps', '2', '--delayed-observations', '--seed', '5', '--device', 'cpu']
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=180,
            env={**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                 'TF_NUM_INTEROP_THREADS': '2', 'TF_NUM_INTRAOP_THREADS': '2'})
        assert result.returncode == 0, result.stderr[-5000:]
        metadata = json.loads((output/'result.json').read_text())
        assert metadata['num_initial_observed_space'] == 6
        assert metadata['delayed_observation_rows'] == 4
        with np.load(output/'predictions.npz') as a:
            np.testing.assert_array_equal(a['times'], arrays['stream_times'])
            return a['pred_mean'].copy(), a['pred_var'].copy()
    base = run('base', arrays)
    changed = {**arrays, 'stream_y': arrays['stream_y'].copy()}
    changed['stream_y'][0, 4:] += 10
    released = run('released', changed)
    np.testing.assert_array_equal(base[0][0], released[0][0])
    np.testing.assert_array_equal(base[1][0], released[1][0])
    assert np.max(np.abs(base[0][1] - released[0][1])) > 1e-8
    changed = {**arrays, 'stream_y': arrays['stream_y'].copy()}
    changed['stream_y'][2] += 10
    future = run('future', changed)
    np.testing.assert_array_equal(base[0][:2], future[0][:2])
