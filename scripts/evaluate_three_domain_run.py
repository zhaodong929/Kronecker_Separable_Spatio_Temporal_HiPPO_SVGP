#!/usr/bin/env python3
"""Verify one complete run against source protocol and score on original units."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.three_domain.evaluation import gaussian_scores,mixture_scores_from_records,restore_score_scale


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--method',choices=['kronhippo_svgp','osgpr','ohsvgp','st_svgp','mgpvae'],required=True)
    a=p.parse_args()
    metadata=json.loads(a.protocol.with_suffix('.json').read_text())
    if metadata.get('development_protocol'):raise ValueError('Formal evaluation cannot use a development fold')
    with np.load(a.protocol,allow_pickle=False) as source,np.load(a.run/'predictions.npz',allow_pickle=False) as result:
        sites=source['test_indices'];truth=source['stream_y'][:,sites]
        for key,value in [('y_true',truth),('test_indices',sites),('times',source['stream_times'])]:
            np.testing.assert_array_equal(result[key],value)
        mean,var=result['pred_mean'],result['pred_var']
        if mean.shape!=truth.shape or var.shape!=truth.shape or not np.isfinite(mean).all() or not np.isfinite(var).all() or np.any(var<=0):
            raise ValueError('Invalid complete observation-space predictions')
        if a.method=='mgpvae':
            records=json.loads((a.run/'online-metrics.json').read_text())
            scores=mixture_scores_from_records(truth,mean,records)
            family='finite Gaussian decoder mixture; supplied moment variance is not used as its density'
        else:
            scores=gaussian_scores(truth,mean,var);family='Gaussian observation predictive distribution'
    normalization=metadata.get('target_standardization',metadata.get('target_standardisation'))
    if not normalization:raise ValueError('Missing target normalization for common-scale scoring')
    restored=restore_score_scale(scores,normalization['scale'])
    saved_result=json.loads((a.run/'result.json').read_text())
    if metadata.get('xlag',{}).get('delay_weeks')==1 or metadata.get('delayed_target_steps')==1:
        if saved_result.get('delayed_observation_rows')!=(len(truth)-1)*len(sites):
            raise ValueError('Delayed-label count does not match full protocol')
    record=dict(status='complete_source_verified',method=a.method,split_seed=metadata['split_seed'],
        main_table_admitted=False,predictive_family=family,target_normalization=normalization,
        evaluator_source_sha256={name:sha256(Path(__file__).resolve().parents[1]/name) for name in
            ['scripts/evaluate_three_domain_run.py','benchmarks/three_domain/evaluation.py']},
        metric_scale='restored log1p admissions per 100k' if metadata.get('dataset','').startswith('cdc_') else 'original target units',
        standardized_scores=scores,original_scale_scores=restored,
        expected_shape=list(truth.shape),protocol_sha256=sha256(a.protocol),
        predictions_sha256=sha256(a.run/'predictions.npz'),
        audit=dict(exact_source_truth_sites_times=True,finite_positive_observation_variance=True,
            delayed_labels_verified=True,selection_and_implementation_admission='separate review required'))
    (a.run/'common-evaluation.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(method=a.method,seed=metadata['split_seed'],status=record['status'],scores=restored)))


if __name__=='__main__':main()
