#!/usr/bin/env python3
"""Recover ERA5-Land source, verify legacy alignment, build fit-only hourly protocols."""
import argparse,hashlib,json,re,sys,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baselines.covid_long_setting_b.development import _ridge
from benchmarks.three_domain.geometry import farthest_indices
from benchmarks.three_domain.era5_alignment import legacy_row_indices
VARIABLES={'d2m':0,'t2m':1,'skt':2,'u10':29,'v10':30,'sp':31,'tp':32}
TOLERANCE={'d2m':.005,'t2m':.005,'skt':.005,'u10':.005,'v10':.005,'sp':1.,'tp':1e-6}

from benchmarks.three_domain.era5_features import features

def main():
    p=argparse.ArgumentParser();p.add_argument('--downloads',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import xarray as xr
    preparation_names=['scripts/prepare_fair_era5.py','benchmarks/three_domain/era5_features.py','benchmarks/three_domain/era5_alignment.py','benchmarks/three_domain/geometry.py','baselines/covid_long_setting_b/development.py']
    preparation_code={name:(ROOT/name).read_bytes() for name in preparation_names}
    preparation_hashes={name:hashlib.sha256(value).hexdigest() for name,value in preparation_code.items()}
    preparation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    public=ROOT/'baselines/external/harrisonzhu508_HIPPOSVGP/data/era5/processed_timeseries_4'
    files=sorted(x for x in (public/'task_1/sequences').glob('*.npz') if not x.stem.endswith('_scaled'))
    coords=np.array([[float(x) for x in re.match(r'lat_([-\d.]+)_lon_([-\d.]+)',f.stem).groups()] for f in files]);assert coords.shape==(1000,2)
    norm=(coords-coords.mean(0))/coords.std(0)
    oldroot=ROOT/'autodl_results/rtx4090_gpu_only_published_20260820T145100Z/benchmark/protocol/task1_10'
    with np.load(oldroot/'seed0/protocol.npz') as old:
        order=np.argmin(((old['coordinates'][:,None]-norm[None])**2).sum(-1),axis=1)
        np.testing.assert_allclose(norm[order],old['coordinates'],rtol=0,atol=1e-10)
        assert len(np.unique(order))==1000
        oldy=np.concatenate([old['calibration_y'],old['stream_y']])
    coords=coords[order];files=[files[i] for i in order];norm=norm[order]
    raw=[];dates=[];source=[]
    for f in sorted(a.downloads.glob('era5-land-2020-*.nc')):
        with xr.open_dataset(f,engine='h5netcdf') as d:
            subset=d[list(VARIABLES)].sel(latitude=xr.DataArray(coords[:,0],dims='site'),longitude=xr.DataArray(coords[:,1],dims='site'),method='nearest',tolerance=1e-6)
            raw.append(np.stack([subset[v].transpose('valid_time','site').values for v in VARIABLES],axis=-1));dates.append(subset.valid_time.values)
        source.append(dict(file=f.name,sha256=hashlib.file_digest(f.open('rb'),'sha256').hexdigest()))
    raw=np.concatenate(raw);dates=np.concatenate(dates)
    expected=np.datetime64('2020-01-01T00','h')+np.arange(1860).astype('timedelta64[h]');fullraw=raw;raw=raw[:1860];dates=dates[:1860]
    np.testing.assert_array_equal(dates,expected);assert raw.shape==(1860,1000,7) and np.isfinite(raw).all()
    errors={v:0. for v in VARIABLES};oldrecovered=np.empty((1860,1000));legacy_gaps=[];fullerror=0.
    for j,f in enumerate(files):
        public_values=[]
        for task in ['task_1','task_2']:
            with np.load(public/task/'sequences'/f.name) as data:
                t=np.concatenate([data['time_'+s] for s in ['train','val','test']]);v=np.concatenate([data['data_'+s] for s in ['train','val','test']],axis=1)
                public_values.append(v[:,np.argsort(t)])
        v=np.concatenate(public_values,axis=1)
        scale,offset=np.linalg.lstsq(np.column_stack([oldy[:186,j],np.ones(186)]),v[0,:186],rcond=None)[0]
        oldrecovered[:,j]=oldy[:,j]*scale+offset
        index,gap=legacy_row_indices(oldrecovered[:,j],fullraw[:,j,0],TOLERANCE['d2m'])
        fullerror=max(fullerror,float(np.max(abs(fullraw[index,j,0]-oldrecovered[:,j]))))
        if gap is not None:legacy_gaps.append(dict(site=j,file=f.name,location=coords[j].tolist(),omitted_utc_hour_index=gap,omitted_utc=str(np.datetime64('2020-01-01T00','h')+np.timedelta64(gap,'h')),legacy_offset_after_gap=1))
        for k,(name,idx) in enumerate(VARIABLES.items()):errors[name]=max(errors[name],float(np.max(abs(fullraw[index[:372],j,k]-v[idx]))))
    for name,err in errors.items():
        if err>TOLERANCE[name]:raise ValueError(f'Source overlap mismatch: {name} error {err}, tolerance {TOLERANCE[name]}')
    uncorrected_error=float(np.max(abs(raw[:,:,0]-oldrecovered)))
    if fullerror>TOLERANCE['d2m']:raise ValueError(f'Long target alignment mismatch {fullerror}')
    a.output.mkdir(parents=True,exist_ok=True)
    evidence=dict(status='all_sites_full_period_target_and_372_hour_covariates_verified',source_files=source,source_dataset='reanalysis-era5-land',overlap_max_absolute_errors=errors,full_target_max_absolute_error_K=fullerror,uncorrected_legacy_target_max_absolute_error_K=uncorrected_error,legacy_omitted_hours=legacy_gaps,output_time_policy='Canonical CDS UTC grid; legacy per-site dropped hours restored, never shift source values to imitate the old misalignment',tolerances=TOLERANCE,not_bitwise_identical=True)
    evidence.update(preparation_source_sha256=preparation_hashes,preparation_git_commit=preparation_commit)
    for name,value in preparation_code.items():
        dst=a.output/'preparation-source'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(value)
    (a.output/'source-verification.json').write_text(json.dumps(evidence,indent=2))
    np.savez_compressed(a.output/'raw-recovered.npz',values=raw,coordinates=coords,times=dates)
    for seed in range(5):
        with np.load(oldroot/f'seed{seed}/protocol.npz') as old:
            arrays={k:old[k].copy() for k in ['train_indices','test_indices','fit_indices','validation_indices']}
            np.testing.assert_allclose(old['coordinates'],norm,rtol=0,atol=1e-10)
        fit=arrays['fit_indices'];center=float(raw[:186,fit,0].mean());scale=float(raw[:186,fit,0].std());y=(raw[:,:,0].astype(float)-center)/scale
        phi,featuremeta=features(raw[:,:,1:],norm,fit);beta=_ridge(phi[:186].astype(float),y[:186],fit)
        arrays.update(calibration_weather=raw[:186,:,1:],calibration_y=y[:186],stream_y=y[186:],calibration_phi=phi[:186],stream_phi=phi[186:],coordinates=norm,calibration_times=np.arange(186)/185.,stream_times=np.arange(186,1860)/185.,task1_ridge_beta=beta,task1_calibration_mean=np.einsum('tsp,p->ts',phi[:186],beta),task1_stream_mean=np.einsum('tsp,p->ts',phi[186:],beta),block_start=np.arange(1674),block_stop=np.arange(1,1675))
        for count in [16,30,32,64,128]:arrays[f'inducing_coords_ms{count}']=norm[fit][farthest_indices(norm[fit],count)]
        out=a.output/f'seed{seed}';out.mkdir(exist_ok=False);np.savez_compressed(out/'protocol.npz',**arrays)
        metadata=dict(schema_version=2,dataset='era5_land',protocol_id='era5_land_hourly_causal',split_seed=seed,num_calibration_times=186,num_stream_times=1674,num_locations=1000,task1_observed_indices=arrays['train_indices'].tolist(),hidden_label_policy='never released',delayed_target_steps=None,stream_batch_hours=1,legacy_stream_batch_hours=10,protocol_change='All methods predict each hour after observing only current/past visible sites; no future within a multi-hour batch',target_standardization=dict(mean=center,scale=scale,fit_scope='initial fitting sites only',metric_scale='kelvin'),weather_features=featuremeta,source=evidence)
        (out/'protocol.json').write_text(json.dumps(metadata,indent=2));print(f'Prepared ERA5 seed {seed}',flush=True)
if __name__=='__main__':main()
