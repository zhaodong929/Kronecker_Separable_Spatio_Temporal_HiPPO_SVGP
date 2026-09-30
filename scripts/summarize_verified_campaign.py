#!/usr/bin/env python3
"""Summarize source-verified matched splits, without manuscript auto-admission."""
import argparse,csv,json,math,statistics
from pathlib import Path

METHODS=('kronhippo_svgp','osgpr','ohsvgp','st_svgp','mgpvae')
SPLITS={'covid':range(5,10),'pems':range(1,4),'era5':range(5)}
METRICS=('rmse','nlpd','crps','ece','coverage90')


def summarize(records,ledger):
    expected={(d,m,s) for d,seeds in SPLITS.items() for m in METHODS for s in seeds}
    current={('pems' if r['dataset']=='pems_bay' else r['dataset'],r['method'],r['split_seed']):r for r in ledger['rows']}
    if set(current)!=expected:raise ValueError('Ledger does not match the declared 65 comparisons')
    rows={};protocols={}
    for record in records:
        path=Path(record['path']);dataset,method,seed_text,job_text=path.parts[-4:]
        key=(dataset,method,int(seed_text.removeprefix('seed')))
        if key not in expected:raise ValueError('Unexpected comparison')
        if (current[key]['submission_status']!='completed_and_verified'
                or current[key]['job_id']!=int(job_text.removeprefix('job-'))):continue
        if key in rows:raise ValueError('Duplicate current verified result')
        evaluation=record['common-evaluation.json']
        if evaluation['status']!='complete_source_verified':raise ValueError('Unverified evaluation')
        if evaluation['method']!=method or evaluation['split_seed']!=key[2]:raise ValueError('Evaluation identity mismatch')
        if not evaluation['audit']['exact_source_truth_sites_times']:raise ValueError('Source alignment failed')
        protocol_key=(dataset,key[2]);digest=evaluation['protocol_sha256']
        if protocol_key in protocols and protocols[protocol_key]!=digest:raise ValueError('Methods use different protocols')
        protocols[protocol_key]=digest
        scores={k:float(evaluation['original_scale_scores'][k]) for k in METRICS}
        if not all(math.isfinite(x) for x in scores.values()):raise ValueError('Nonfinite score')
        rows[key]=dict(dataset=dataset,method=method,split_seed=key[2],job_id=current[key]['job_id'],
            training_seed_actual=current[key].get('training_seed_actual'),
            **scores,protocol_sha256=digest,predictions_sha256=evaluation['predictions_sha256'],
            metric_scale=evaluation['metric_scale'],run_url=current[key].get('run_url'),
            source_path=str(path),main_table_admitted=False)
    complete_datasets=[d for d,seeds in SPLITS.items() if all((d,m,s) in rows for m in METHODS for s in seeds)]
    aggregate=[]
    for dataset in complete_datasets:
        for method in METHODS:
            group=[rows[(dataset,method,s)] for s in SPLITS[dataset]]
            item=dict(dataset=dataset,method=method,n_splits=len(group),main_table_admitted=False)
            for metric in METRICS:
                values=[r[metric] for r in group]
                item[metric+'_mean']=statistics.mean(values);item[metric+'_std']=statistics.stdev(values)
            aggregate.append(item)
    return dict(expected=65,verified=len(rows),all_comparisons_complete=len(rows)==65,
        complete_datasets=complete_datasets,missing=[dict(dataset=d,method=m,split_seed=s) for d,m,s in sorted(expected-rows.keys())],
        per_split=[rows[k] for k in sorted(rows)],aggregate=aggregate,main_table_admitted=False,
        notes=['Standard deviation is across paired spatial splits (sample standard deviation, ddof=1).',
               'Training seeds are recorded separately: split variation is not an independent training-seed replication study.',
               'COVID original target units are log1p admissions per 100000, not raw counts.',
               'Timing, implementation, selection and manuscript admission require separate review.',
               'A dataset is aggregated only after all five methods finish every declared split.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();report=summarize(json.loads(a.records.read_text()),json.loads(a.ledger.read_text()))
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for name in ['per_split','aggregate']:
        path=a.output/(name+'.csv')
        with path.open('w',newline='') as handle:
            values=report[name]
            if values:
                writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    print(json.dumps({k:report[k] for k in ['verified','expected','complete_datasets','all_comparisons_complete']}))
    if not report['all_comparisons_complete'] and not a.allow_partial:raise SystemExit(2)


if __name__=='__main__':main()
