import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.three_domain.tracking import emit
from scripts.run_tracked_experiment import journal_rows, validate_outputs


def test_journal_preserves_every_event_and_waits_for_complete_line(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("HIPPO_EVENT_PATH", str(path))
    before = np.random.get_state()
    for step in range(17):
        emit("online", step, {"loss": np.float64(step), "nan": float("nan")})
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    with path.open("ab") as f:
        f.write(b'{"phase":"online"')
    rows, offset = journal_rows(path, 0)
    assert [r["step"] for r in rows] == list(range(17))
    assert all(r["metrics"]["nan"] is None for r in rows)
    assert journal_rows(path, offset) == ([], offset)
    with path.open("ab") as f:
        f.write(b',"step":17,"metrics":{}}\n')
    assert journal_rows(path, offset)[0][0]["step"] == 17


def test_successful_process_is_not_sufficient_for_valid_predictions(tmp_path):
    (tmp_path / "result.json").write_text('{"status":"complete"}')
    np.savez(tmp_path / "predictions.npz", y_true=np.ones((2, 3)),
             pred_mean=np.ones((2, 3)), pred_var=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="nonpositive"):
        validate_outputs(tmp_path, 2, 3)
    np.savez(tmp_path / "predictions.npz", y_true=np.ones((1, 3)),
             pred_mean=np.ones((1, 3)), pred_var=np.ones((1, 3)))
    with pytest.raises(ValueError, match="shape"):
        validate_outputs(tmp_path, 2, 3)


@pytest.mark.parametrize("fail", [False, True])
def test_real_offline_supervisor_records_success_and_failure(tmp_path, fail):
    pytest.importorskip("wandb")
    root = Path(__file__).resolve().parents[1]
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(dict(entity="local-test", project="hippo-test",
        campaign="test", dataset="pems_bay", method="kronhippo_svgp", split_seed=1,
        training_seed=1, stage="integration")))
    command = "from benchmarks.three_domain.tracking import emit; emit('train',1,{'loss':2.0}); raise SystemExit(%d)" % (7 if fail else 0)
    result = subprocess.run([sys.executable, str(root / "scripts/run_tracked_experiment.py"),
        "--spec", str(spec), "--output", str(tmp_path / "output"), "--mode", "offline",
        "--poll-seconds", "0.05", "--", sys.executable, "-c", command],
        cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == (7 if fail else 0), result.stdout + result.stderr
    records = list((tmp_path / "output/tracking").iterdir())
    assert len(records) == 1
    terminal = json.loads((records[0] / "terminal.json").read_text())
    assert terminal["status"] == ("failed" if fail else "process_complete")
    assert terminal["main_table_admitted"] is False
    events = [json.loads(line) for line in (records[0] / "events.jsonl").read_text().splitlines()]
    assert next(e for e in events if e["phase"] == "train")["metrics"]["loss"] == 2.
    manifest = json.loads((records[0] / "artifact_manifest.json").read_text())
    assert any(k.endswith("events.jsonl") for k in manifest)
