import json
import pytest
from benchmarks.three_domain.reuse_validation import digest,original_validation_directory


def copy_record(destination,source):
    destination.mkdir()
    (destination/'result.json').write_bytes((source/'result.json').read_bytes())
    (destination/'reused-validation.json').write_text(json.dumps(dict(
        status='verified_validation_reused',source_directory=str(source),
        result_sha256=digest(source/'result.json'))))


def test_multi_stage_reuse_resolves_original_attempt(tmp_path):
    original=tmp_path/'original';original.mkdir();(original/'result.json').write_text('{"score":1}')
    first=tmp_path/'first';second=tmp_path/'second'
    copy_record(first,original);copy_record(second,first)
    assert original_validation_directory(second,'result.json')==original
    (first/'result.json').write_text('{"score":2}')
    with pytest.raises(ValueError,match='source changed'):
        original_validation_directory(second,'result.json')


def test_changed_copy_and_cycle_are_rejected(tmp_path):
    original=tmp_path/'original';original.mkdir();(original/'result.json').write_text('{}')
    copied=tmp_path/'copied';copy_record(copied,original)
    (copied/'result.json').write_text('[]')
    with pytest.raises(ValueError,match='copy changed'):
        original_validation_directory(copied,'result.json')
    (copied/'result.json').write_text('{}')
    record=json.loads((copied/'reused-validation.json').read_text());record['source_directory']=str(copied)
    (copied/'reused-validation.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='cycle'):
        original_validation_directory(copied,'result.json')
