import copy
from dataclasses import asdict
import json
import pytest
from benchmarks.task_stream.factory import Configuration, METHODS
from scripts.task_budget_study import policy, digest, validate_manifest, collect, sha256


def tasks(tmp_path):
    rule = policy(); result = []
    for dataset, seeds in rule['split_seeds'].items():
        for seed in seeds:
            for method in METHODS:
                for seconds in rule['seconds']:
                    name = f'{dataset}-{seed}-{method}-{seconds}'
                    result.append(dict(id=name, dataset=dataset, split_seed=seed, stage='validation',
                        output=str(tmp_path/name), configuration=asdict(Configuration(method,
                            initial_iterations=1000000, initial_max_seconds=seconds,
                            learning_rate=.001, online_iterations=5, seed=seed, device='cuda')),
                        study=dict(policy=rule, policy_sha256=digest(rule),
                            selection_initial_rows=30, final_initial_rows=60)))
    return result, rule


def test_complete_study_has_equal_opportunities_and_no_missing_groups(tmp_path):
    manifest, rule = tasks(tmp_path)
    assert len(manifest) == 195 and len(validate_manifest(manifest, rule)) == 65
    with pytest.raises(ValueError, match='all three'):
        validate_manifest(manifest[:-1], rule)
    with pytest.raises(ValueError, match='group'):
        validate_manifest(manifest[:-3], rule)
    with pytest.raises(ValueError, match='all three'):
        validate_manifest(manifest+[manifest[0]], rule)


@pytest.mark.parametrize('field,value', [('learning_rate', .1), ('initial_max_seconds', 999.),
    ('initial_iterations', 500), ('online_iterations', 10), ('seed', 99), ('spatial_inducing', 99)])
def test_silent_method_specific_settings_are_rejected(tmp_path, field, value):
    manifest, rule = tasks(tmp_path)
    manifest[0]['configuration'][field] = value
    with pytest.raises(ValueError): validate_manifest(manifest, rule)


def test_partial_campaign_does_not_select_from_incomplete_candidates(tmp_path):
    manifest, rule = tasks(tmp_path)
    result = collect(manifest, rule, tmp_path/'summary')
    assert not result['complete'] and len(result['groups']) == 65
    assert not (tmp_path/'summary/selections').exists()
    assert all(r['status'] == 'pending' for r in result['groups'])


def test_false_policy_identity_cannot_bypass_common_rule(tmp_path):
    manifest, rule = tasks(tmp_path)
    manifest = copy.deepcopy(manifest)
    manifest[0]['study']['policy']['early_stopping'] = True
    with pytest.raises(ValueError, match='policy'): validate_manifest(manifest, rule)


def test_complete_group_freezes_winning_actual_work_but_missing_group_stays_pending(tmp_path):
    from pathlib import Path
    manifest, rule = tasks(tmp_path)
    for index, task in enumerate(manifest[:3]):
        attempt = Path(task['output'])/'attempts'/'a'
        run = attempt/'run'; run.mkdir(parents=True)
        c = task['configuration']; seconds = c['initial_max_seconds']
        proof = dict(source_commit='a'*40, source_sha256='source', configuration_sha256=digest(c), input_files={})
        provenance = dict(proof, stage='validation', selection_protocol_sha256='protocol',
                          selection_features_sha256='features')
        record = dict(policy='synchronized_wall_time_v1', scope=rule['scope'], max_seconds=seconds,
            elapsed_seconds=seconds+1, completed_steps=10+index, stop_reason='wall_time_budget_reached',
            overshoot_seconds=1, minimum_completed_steps=1, convergence_claimed=False)
        values = {'configuration.json':dict(configuration=c, provenance=provenance, protocol_sha256='protocol'),
            'fit-budget.json':record, 'task-metrics.json':[], 'predictions.npz':{},
            'result.json':dict(status='completed', ordinary_latency_valid=True, protocol_sha256='protocol',
                metrics=dict(nlpd=[2.,1.,1.1][index]), provenance=provenance)}
        for name, value in values.items(): (run/name).write_text(json.dumps(value))
        (run/'artifacts.json').write_text(json.dumps({name:dict(sha256=sha256(run/name)) for name in values}))
        (Path(task['output'])/'latest.json').write_text(json.dumps(dict(status='completed',attempt=str(attempt),proof=proof)))
    result = collect(manifest, rule, tmp_path/'summary')
    assert not result['complete']
    assert result['groups'][0]['status']=='selected_within_declared_budget'
    selected = json.loads(Path(result['groups'][0]['selection']).read_text())
    assert selected['all_declared_candidates_completed']
    assert selected['refit_budget']['selected_steps']==11
    assert selected['refit_budget']['effective_iterations']==11
    assert not selected['convergence_qualified']
    assert all(r['status']=='pending' for r in result['groups'][1:])
