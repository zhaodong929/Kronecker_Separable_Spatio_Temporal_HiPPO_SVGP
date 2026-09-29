#!/usr/bin/env python3
"""Create paired splits from sensor metadata, then export corrected protocols."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stvgp_kronecker.data.traffic import _read_traffic_hdf, _read_coordinate_table, _normalise_sensor_id, make_spatial_split


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    args=ap.parse_args()
    raw=args.data_root/'pems_bay'
    frame=_read_traffic_hdf(raw/'PEMS-BAY.h5')
    if frame.shape != (52116,325):raise ValueError(f'Unexpected canonical PEMS shape: {frame.shape}')
    ids=tuple(_normalise_sensor_id(x) for x in frame.columns)
    ci,cv=_read_coordinate_table(raw/'graph_sensor_locations_bay.csv')
    import numpy as np
    lookup=dict(zip(ci,cv));coords=np.array([lookup[i] for i in ids])
    # No label is used to construct the split.
    descriptor=SimpleNamespace(name='pems_bay',num_sensors=325,sensor_ids=ids,coordinates_lat_lon=coords)
    args.output.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        out=args.output/f'seed{seed}'
        if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
        out.mkdir()
        split=make_spatial_split(descriptor,seed=seed)
        manifest=out/'split.json'
        manifest.write_text(json.dumps(dict(schema_version=1, task1_steps=2016,
                                            split=split.as_dict()),indent=2)+'\n')
        subprocess.run([sys.executable,str(ROOT/'scripts/export_traffic_external_gp_protocol.py'),
            '--data-root',str(args.data_root),'--split-manifest',str(manifest),
            '--output-dir',str(out),'--road-distance-csv',str(raw/'distances_bay_2017.csv'),
            '--include-features'],check=True,cwd=ROOT)
        provenance=dict(dataset='DCRNN PEMS-BAY',hdf_source='https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view',
            metadata_source='https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph',
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir() if p.is_file()})
        (out/'raw_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
if __name__=='__main__':main()
