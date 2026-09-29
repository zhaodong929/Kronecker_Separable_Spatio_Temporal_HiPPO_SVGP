"""Exercise the real runner, including delayed assimilation and saved predictions."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np


def test_actual_runner_never_uses_unreleased_targets(tmp_path):
    root=Path(__file__).resolve().parents[1]
    r=np.random.default_rng(14);coords=r.normal(size=(5,2))
    data=dict(calibration_times=np.arange(5.),calibration_y=r.normal(size=(5,5)),
      calibration_phi=np.ones((5,5,1)),stream_times=np.arange(5.,8.),
      stream_y=r.normal(size=(3,5)),stream_phi=np.ones((3,5,1)),coordinates=coords,
      train_indices=np.arange(3),test_indices=np.arange(3,5),inducing_coords_ms2=coords[:2],
      block_start=np.arange(3),block_stop=np.arange(1,4))
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'theta.json').write_text(json.dumps({'learned_theta':dict(ell_t=1.,ell_s=[1.,1.],kernel_variance=1.,noise_std=.3)}))
    def run(label,arrays):
        path=tmp_path/label;path.mkdir();np.savez(path/'protocol.npz',**arrays)
        command=[sys.executable,str(root/'scripts/run_iclr_era5_routeb_strict_online.py'),
          '--protocol-npz',str(path/'protocol.npz'),'--protocol-json',str(tmp_path/'protocol.json'),
          '--theta-json',str(tmp_path/'theta.json'),'--output',str(path/'result.json'),
          '--blockwise-output',str(path/'blocks.csv'),'--predictions-output',str(path/'predictions.npz'),
          '--representation','analytic_hippo_rff','--mt','3','--ms','2','--rff-sample-size','16',
          '--seed','1','--device','cpu','--solver-backend','torch','--dtype','float64',
          '--delayed-observations','--task1-posterior-init','--temporal-factor-device','solver']
        subprocess.run(command,cwd=root,check=True,capture_output=True,text=True,timeout=90,
                       env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        meta=json.loads((path/'result.json').read_text())
        assert meta['delayed_observation_rows']==4
        assert meta['task1_posterior_initialization_rows']==15
        with np.load(path/'predictions.npz') as p:return p['pred_mean'].copy(),p['pred_var'].copy()
    base=run('base',data)
    hidden={**data,'stream_y':data['stream_y'].copy()};hidden['stream_y'][0,3:]+=100
    changed=run('hidden',hidden)
    np.testing.assert_array_equal(base[0][0],changed[0][0])
    np.testing.assert_array_equal(base[1][0],changed[1][0])
    assert np.max(np.abs(base[0][1]-changed[0][1]))>1e-4
    future={**data,'stream_y':data['stream_y'].copy()};future['stream_y'][2]+=200
    changed=run('future',future)
    np.testing.assert_array_equal(base[0][:2],changed[0][:2])
    np.testing.assert_array_equal(base[1][:2],changed[1][:2])
