import copy
import pytest
from benchmarks.task_stream.measurement import Measurements, profile_record
from benchmarks.task_stream.reporting import table_rows, export_tables, GATES


def test_profile_is_separate_with_explicit_partial_coverage():
    kwargs = dict(provenance='counter export hash', profiled_task_ids=[1], expected_task_ids=[0, 1, 2],
        replay_policy='no_replay', profiler_seconds=100, timing_run_id='timing', profile_run_id='profile')
    record = profile_record([dict(id='launch1', dadd=2, dmul=4, dfma=5)], **kwargs)
    assert record['fp64_add_mul_fma'] == 16 and record['task_coverage'] == 1/3
    assert record['extrapolated_to_total'] is False
    with pytest.raises(ValueError, match='separately'):
        profile_record([], **dict(kwargs, profile_run_id='timing'))
    assert profile_record([], **kwargs)['fp64_add_mul_fma'] is None


def records():
    return [dict(run_id=m, dataset='test', method=m, seed=0, hardware_id='gpu-hash',
        result=dict(status='completed', protocol_sha256='protocol-hash', metrics={'rmse': 1.}, timing={}),
        qualification=dict(paired_run_ids=['b' if m == 'a' else 'a'],
            gates={g: {'passed': True, 'evidence_sha256': 'test-hash'} for g in GATES})) for m in ['a', 'b']]


def test_tables_require_gate_evidence_and_actual_paired_protocol_hardware(tmp_path):
    r = records()
    assert all(row['main_table_admitted'] for row in table_rows(r))
    for key in ['hardware_id', 'seed']:
        changed = copy.deepcopy(r); changed[1][key] = 'different'
        assert not any(row['main_table_admitted'] for row in table_rows(changed))
    changed = copy.deepcopy(r); changed[1]['result']['protocol_sha256'] = 'different'
    assert not any(row['main_table_admitted'] for row in table_rows(changed))
    r[0]['qualification']['gates']['implementation_parity'] = {'passed': True}
    assert not table_rows(r)[0]['main_table_admitted']
    export_tables(r, tmp_path)
    assert 'NA' in (tmp_path / 'comparison.csv').read_text()
    assert 'FP64 subset' in (tmp_path / 'comparison.tex').read_text()


def test_sync_failure_does_not_poison_measurement_lock():
    def fail():
        raise RuntimeError('sync failure')
    measurement = Measurements(fail)
    with pytest.raises(RuntimeError):
        with measurement.phase('online'):
            pass
    assert not measurement.active


def test_reporting_selects_native_metrics_and_explicit_units(tmp_path):
    r = records()
    for record in r:
        record['result'].update(metrics_native={'rmse': 7., 'nlpd': 2., 'crps': 3., 'ece': .1},
                                metric_scale='miles per hour')
    rows = export_tables(r, tmp_path)
    assert rows[0]['rmse'] == 7.
    assert rows[0]['metric_scale'] == 'miles per hour'
    assert 'metric_scale' in (tmp_path/'comparison.csv').read_text().splitlines()[0]
    assert 'miles per hour' in (tmp_path/'comparison.tex').read_text()
