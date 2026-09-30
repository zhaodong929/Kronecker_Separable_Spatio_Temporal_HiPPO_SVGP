#!/usr/bin/env python3
"""ERA5 proposal: own initial validation followed by hourly causal final stream."""
import argparse,json,subprocess,sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--release',required=True);p.add_argument('--compute-root',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
worker=str(a.compute_root/'env-routeb/bin/python');protocol=a.compute_root/f'protocol/era5/seed{a.seed}'
inputs=[str(protocol/'protocol.npz'),str(protocol/'protocol.json')];common=['--protocol-npz',inputs[0],'--protocol-json',inputs[1]]
tests=['tests/test_era5_adapter_boundary.py::test_actual_era5_no_release_adapter[kronhippo_svgp]','tests/test_era5_information_boundary.py','tests/test_strict_online_release_boundary.py','tests/test_multi_geometry.py','tests/test_routeb_torch_backend.py','tests/test_temporal_analytic_gradients.py']
with (a.output/'tests.txt').open('w') as f:subprocess.run([worker,'-m','pytest','-q',*tests],stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([worker,'-c','import torch; assert torch.cuda.is_available()'],check=True)
def run(output,stage,mt,ms,budget,command):
    output.mkdir(parents=True,exist_ok=True)
    spec=dict(entity='harrisonzhu',project='KronHiPPO-STGP',campaign='fair-three-domain-wandb-20260929',dataset='era5',method='kronhippo_svgp',split_seed=a.seed,training_seed=0,stage=stage,source_commit=a.release,worker_python=worker,input_files=inputs,temporal_inducing=mt,spatial_inducing=ms,iterations=budget,hidden_delay_steps=None,initial_observed_sites=800,main_table_admitted=False)
    if stage=='final':spec.update(expected_steps=1674,expected_sites=200,qualification_record=str(a.output/'qualification.json'))
    (output/'spec.json').write_text(json.dumps(spec,indent=2));subprocess.run([sys.executable,'scripts/run_tracked_experiment.py','--spec',str(output/'spec.json'),'--output',str(output),'--',worker,*command],check=True)
candidates=[]
for mt,ms in [(32,32),(64,32),(32,64),(64,64)]:
    for budget in [250,500,1000]:
        output=a.output/f'calibration-mt{mt}-ms{ms}-b{budget}'
        run(output,'validation',mt,ms,budget,['scripts/run_iclr_era5_routeb_batch.py',*common,'--output-dir',str(output),'--data-part','calibration','--target-mode','joint_xlag','--representation','analytic_hippo_rff','--mt',str(mt),'--ms',str(ms),'--rff-sample-size','256','--training-objective','vfe','--iterations',str(budget),'--learning-rate','0.02','--validation-every','5','--early-stopping-patience-validations','8','--split-seed',str(a.seed),'--model-seed','0','--device','cuda','--dtype','float64','--evaluation-backend','torch','--objective-optimization-version','E3'])
        result=json.loads((output/'result.json').read_text())
        if result['best_iteration']<budget:break
    if result['best_iteration']>=budget:raise RuntimeError('Proposal initial validation still improves at budget boundary')
    candidates.append(dict(mt=mt,ms=ms,path=str(output/'result.json'),nlpd=result['best_validation_nll'],selected_iteration=result['best_iteration']))
best=min(candidates,key=lambda x:x['nlpd']);(a.output/'selection.json').write_text(json.dumps(dict(candidates=candidates,selected=best),indent=2))
(a.output/'qualification.json').write_text(json.dumps(dict(status='passed',method='kronhippo_svgp',dataset='era5',source_commit=a.release,tests=tests,main_table_admitted=False),indent=2))
run(a.output,'final',best['mt'],best['ms'],best['selected_iteration'],['scripts/run_iclr_era5_routeb_strict_online.py',*common,'--theta-json',best['path'],'--representation','analytic_hippo_rff','--mt',str(best['mt']),'--ms',str(best['ms']),'--rff-sample-size','256','--seed',str(a.seed),'--device','cuda','--solver-backend','torch','--dtype','float64','--task1-posterior-init','--temporal-factor-device','cpu','--temporal-bessel-backend','scipy','--checkpoint',str(a.output/'checkpoint.pt'),'--checkpoint-every','200','--output',str(a.output/'result.json'),'--blockwise-output',str(a.output/'blocks.csv'),'--predictions-output',str(a.output/'predictions.npz')])
