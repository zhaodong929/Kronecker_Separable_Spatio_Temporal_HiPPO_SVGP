"""Stage-specific data exposure with a single immutable selection configuration."""
from dataclasses import asdict
import hashlib
import json
import numpy as np
import pytest
from benchmarks.task_stream.factory import Configuration


def fingerprint(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def test_oh_validation_and_refit_resolve_one_immutable_hashed_policy():
    config = Configuration('ohsvgp', 500, .001, batch_rows=1024, initial_expected_passes=2.)
    before = asdict(config); identity = fingerprint(config)
    validation = config.resolve_initial_budget(2016*208)
    refit = config.resolve_initial_budget(4032*260)
    assert validation['effective_iterations'] == 819
    assert refit['effective_iterations'] == 2048
    assert validation['observation_rows'] == 419328
    assert refit['sampled_rows'] == 2048*1024
    assert 2 <= refit['expected_row_exposures'] < 2.001
    assert validation['policy'] == refit['policy'] == 'expected_row_exposures_v1'
    assert asdict(config) == before and fingerprint(config) == identity
    changed = Configuration('ohsvgp', 500, .001, batch_rows=1024, initial_expected_passes=6.)
    assert fingerprint(changed) != identity


@pytest.mark.parametrize('method', ['ohsvgp', 'kronhippo_svgp', 'osgpr', 'st_svgp', 'mgpvae'])
def test_fixed_budget_is_unchanged_without_exposure_policy(method):
    config = Configuration(method, 17, .001)
    for rows in (3, 1048320):
        resolved = config.resolve_initial_budget(rows)
        assert resolved['effective_iterations'] == 17
        assert resolved['policy'] == 'fixed_steps_v1'
        if method != 'ohsvgp':
            assert resolved['rows_per_step'] == rows
            assert resolved['expected_row_exposures'] == 17


def test_oh_floor_and_short_minibatch_and_fractional_rounding():
    config = Configuration('ohsvgp', 500, .001, initial_expected_passes=2.)
    assert config.resolve_initial_budget(1680)['effective_iterations'] == 500
    config = Configuration('ohsvgp', 1, .001, initial_expected_passes=2.1)
    assert config.resolve_initial_budget(3)['effective_iterations'] == 3
    assert config.resolve_initial_budget(3)['rows_per_step'] == 3


@pytest.mark.parametrize('value', [True, False, np.bool_(True), 0, -1., float('nan'), float('inf'), '2'])
def test_oh_policy_rejects_invalid_exposure(value):
    with pytest.raises(ValueError, match='finite and positive'):
        Configuration('ohsvgp', 2, .001, initial_expected_passes=value)


@pytest.mark.parametrize('method', ['kronhippo_svgp', 'osgpr', 'st_svgp', 'mgpvae'])
def test_non_oh_rejects_exposure_policy(method):
    with pytest.raises(ValueError, match='only to OHSVGP'):
        Configuration(method, 2, .001, initial_expected_passes=2.)


@pytest.mark.parametrize('rows', [True, 0, -1, 2.5])
def test_budget_requires_actual_integer_observation_count(rows):
    with pytest.raises(ValueError, match='positive integer'):
        Configuration('ohsvgp', 2, .001).resolve_initial_budget(rows)
