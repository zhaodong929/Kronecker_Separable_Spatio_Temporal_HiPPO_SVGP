"""Actual no-release adapters must ignore every hidden stream label."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.parametrize('method',['kronhippo_svgp','ohsvgp','osgpr'])
def test_actual_era5_no_release_adapter(tmp_path,method):
    if method=='osgpr':pytest.importorskip('gpflow')
    else:pytest.importorskip('torch')
    root=Path(__file__).resolve().parents[1];rng=np.random.default_rng(51)
    coords=rng.normal(size=(800,2));initial=rng.normal(size=(4,800));beta=initial[:,:720].sum()/(4*720+1e-3)
    values=dict(calibration_y=initial,stream_y=rng.normal(size=(3,800)),calibration_phi=np.ones((4,800,1)),stream_phi=np.ones((3,800,1)),task1_calibration_mean=np.full((4,800),beta),task1_stream_mean=np.full((3,800),beta),task1_ridge_beta=np.array([beta]),calibration_times=np.arange(4.),stream_times=np.arange(4.,7.),coordinates=coords,train_indices=np.arange(720),fit_indices=np.arange(720),validation_indices=np.arange(720,800),test_indices=np.arange(720,800),inducing_coords_ms2=coords[:2],block_start=np.arange(3),block_stop=np.arange(1,4))
    meta=dict(protocol_id='era5_land_hourly_causal',development_protocol=True,hidden_label_policy='never released',task1_observed_indices=list(range(720)),split_seed=0)
    def run(name,data):
        out=tmp_path/name;out.mkdir();np.savez(out/'protocol.npz',**data);(out/'protocol.json').write_text(json.dumps(meta))
        common=['--protocol-npz',str(out/'protocol.npz'),'--protocol-json',str(out/'protocol.json'),'--seed','0','--device',os.environ.get('HIPPO_TEST_DEVICE','cpu')]
        artifacts=['--output',str(out/'result.json'),'--blockwise-output',str(out/'blocks.csv'),'--predictions-output',str(out/'predictions.npz')]
        if method=='ohsvgp':
            command=['scripts/run_covid_ohsvgp_own_theta.py',*common,'--protocol-kind','era5','--output-dir',str(out),'--kernel','rbf','--inducing-size','4','--rff-sample-size','16','--basis-grid-size','32','--calibration-iterations','2','--task1-check-interval','1','--task1-min-steps','2','--calibration-batch-size','64','--update-steps','1']
        elif method=='osgpr':
            command=['scripts/run_official_bui_osgpr_era5.py',*common,*artifacts,'--mt','2','--ms','2','--adaptive','--adaptive-calibration-steps','2','--adaptive-online-steps','2','--initial-ell-t','1','--initial-ell-s','2','2']
        else:
            (out/'theta.json').write_text(json.dumps({'learned_theta':dict(ell_t=1.,ell_s=[1.,1.],kernel_variance=1.,noise_std=.3)}))
            command=['scripts/run_iclr_era5_routeb_strict_online.py',*common,*artifacts,'--theta-json',str(out/'theta.json'),'--representation','analytic_hippo_rff','--mt','3','--ms','2','--rff-sample-size','16','--solver-backend','torch','--dtype','float64','--task1-posterior-init','--temporal-factor-device','cpu','--temporal-bessel-backend','scipy']
        r=subprocess.run([sys.executable,*command],cwd=root,capture_output=True,text=True,errors='replace',timeout=180,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TF_NUM_INTEROP_THREADS':'2','TF_NUM_INTRAOP_THREADS':'2'})
        assert r.returncode==0,r.stderr[-5000:]
        assert json.loads((out/'result.json').read_text())['delayed_observation_rows']==0
        with np.load(out/'predictions.npz') as a:return a['pred_mean'].copy(),a['pred_var'].copy()
    before=run('original',values);changed={**values,'stream_y':values['stream_y'].copy()};changed['stream_y'][:,720:]+=1000
    after=run('hidden',changed)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    changed['stream_y'][2,:720]+=1000;future=run('future',changed)
    for a,b in zip(before,future):np.testing.assert_array_equal(a[:2],b[:2])
