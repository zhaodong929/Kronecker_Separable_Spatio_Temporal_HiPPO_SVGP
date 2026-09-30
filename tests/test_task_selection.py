import hashlib
import json
import pytest
from scripts.select_task_configuration import select


def candidate(root, name, *, nlpd=1., stage='validation', status='completed', protocol='validation-split',
              result_protocol=None, method='kronhippo_svgp', source='source-a', iterations=3,
              source_hash='source-content-a', feature_hash='selection-features-a'):
    path = root/name; path.mkdir()
    configuration = dict(method=method, initial_iterations=iterations)
    configuration_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    provenance = dict(source_commit=source, stage=stage, selection_protocol_sha256=protocol,
        configuration_sha256=configuration_hash, source_sha256=source_hash,
        selection_features_sha256=feature_hash)
    (path/'configuration.json').write_text(json.dumps(dict(configuration=configuration, provenance=provenance, protocol_sha256=protocol)))
    (path/'result.json').write_text(json.dumps(dict(status=status, protocol_sha256=result_protocol or protocol,
        metrics=dict(nlpd=nlpd, rmse=99.), provenance=provenance)))
    return path


def test_selection_uses_finite_validation_nlpd_with_stable_tie_break(tmp_path):
    a = candidate(tmp_path, 'a', nlpd=.4, iterations=3)
    b = candidate(tmp_path, 'b', nlpd=.2, iterations=4)
    c = candidate(tmp_path, 'c', nlpd=.2, iterations=5)
    result = select([a, b, c])
    reversed_result = select([c, b, a])
    assert result['winner'] == reversed_result['winner']
    assert result['winner']['nlpd'] == .2
    assert result['criterion'] == 'query_weighted_observation_nlpd'
    assert not result['main_table_admitted'] and not result['convergence_qualified']


@pytest.mark.parametrize('kwargs', [dict(stage='final'), dict(stage='integration'), dict(stage='ablation'),
    dict(status='failed'), dict(status='running'), dict(result_protocol='final-evaluation'),
    dict(nlpd=float('nan')), dict(nlpd=float('inf'))])
def test_selection_never_uses_final_incomplete_or_invalid_score(tmp_path, kwargs):
    invalid = candidate(tmp_path, 'invalid', nlpd=-999., **kwargs) if 'nlpd' not in kwargs else candidate(tmp_path, 'invalid', **kwargs)
    with pytest.raises(ValueError): select([invalid])


@pytest.mark.parametrize('kwargs', [dict(method='mgpvae'), dict(source='different-source'), dict(protocol='different-split'),
    dict(source_hash='same-commit-changed-source'), dict(feature_hash='changed-features')])
def test_selection_rejects_incomparable_candidates(tmp_path, kwargs):
    a = candidate(tmp_path, 'a')
    b = candidate(tmp_path, 'b', **kwargs)
    with pytest.raises(ValueError, match='differ'): select([a, b])


def test_selection_rejects_mutated_configuration_and_empty_list(tmp_path):
    path = candidate(tmp_path, 'candidate')
    config = json.loads((path/'configuration.json').read_text())
    config['configuration']['initial_iterations'] = 123
    (path/'configuration.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='hash'): select([path])
    with pytest.raises(ValueError, match='At least one'): select([])


@pytest.mark.parametrize('field', ['stage', 'source_commit', 'source_sha256', 'selection_features_sha256',
                                  'selection_protocol_sha256', 'configuration_sha256'])
def test_selection_rejects_result_from_different_run(tmp_path, field):
    path = candidate(tmp_path, 'candidate')
    result = json.loads((path/'result.json').read_text())
    result['provenance'][field] = 'different-value'
    (path/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='[Pp]rovenance'):
        select([path])


def test_selection_rejects_configuration_artifact_with_different_protocol(tmp_path):
    path = candidate(tmp_path, 'candidate')
    record = json.loads((path/'configuration.json').read_text())
    record['protocol_sha256'] = 'different-protocol'
    (path/'configuration.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='provenance|protocol'):
        select([path])
