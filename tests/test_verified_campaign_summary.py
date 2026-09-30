import copy
import pytest
from scripts.summarize_verified_campaign import summarize,METHODS,SPLITS,METRICS


def fixture_records():
    records=[];rows=[]
    for dataset,seeds in SPLITS.items():
        for method in METHODS:
            for seed in seeds:
                rows.append(dict(dataset='pems_bay' if dataset=='pems' else dataset,method=method,split_seed=seed,job_id=1,submission_status='completed_and_verified'))
                records.append({'path':f'/results/{dataset}/{method}/seed{seed}/job-1',
                    'common-evaluation.json':dict(status='complete_source_verified',method=method,split_seed=seed,
                        audit=dict(exact_source_truth_sites_times=True),protocol_sha256=f'{dataset}-{seed}',
                        predictions_sha256='prediction',metric_scale='original target units',
                        original_scale_scores={k:float(seed) for k in METRICS})})
    return records,dict(rows=rows)


def test_only_complete_matched_datasets_are_aggregated():
    records,ledger=fixture_records();report=summarize(records[:-1],ledger)
    assert report['verified']==64 and not report['all_comparisons_complete']
    assert report['complete_datasets']==['covid','pems'] and len(report['aggregate'])==10
    complete=summarize(records,ledger)
    assert complete['all_comparisons_complete'] and len(complete['aggregate'])==15
    assert complete['aggregate'][0]['rmse_mean']==7
    assert not complete['main_table_admitted']


def test_mismatched_protocol_and_duplicate_result_are_rejected():
    records,ledger=fixture_records();changed=copy.deepcopy(records)
    changed[5]['common-evaluation.json']['protocol_sha256']='wrong'
    with pytest.raises(ValueError,match='different protocols'):summarize(changed,ledger)
    with pytest.raises(ValueError,match='Duplicate'):summarize(records+[records[0]],ledger)


def test_superseded_completed_attempt_is_not_counted():
    records,ledger=fixture_records();ledger['rows'][0]['job_id']=2
    ledger['rows'][0]['submission_status']='failed_requires_review'
    report=summarize(records,ledger)
    assert report['verified']==64 and 'covid' not in report['complete_datasets']
