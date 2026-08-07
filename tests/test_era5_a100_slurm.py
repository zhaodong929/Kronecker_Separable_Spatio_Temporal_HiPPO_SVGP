from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_era5_a100_manifest_job.py"
SUBMIT = ROOT / "slurm/era5_a100/submit_all.sh"
ARRAY = ROOT / "slurm/era5_a100/run_manifest_array.sbatch"
WORKER = ROOT / "slurm/era5_a100/run_manifest_worker.sbatch"
PREPARE = ROOT / "slurm/era5_a100/prepare_protocol.sbatch"
EFFICIENCY = ROOT / "slurm/era5_a100/run_efficiency.sbatch"
REPORT = ROOT / "slurm/era5_a100/generate_report.sbatch"
GPFLOW_SELECT = ROOT / "slurm/era5_a100/select_gpflow_tier.sbatch"
SPEC = importlib.util.spec_from_file_location("era5_a100_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_submit_contract_uses_generated_manifests_and_records_dependencies() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    prepare = PREPARE.read_text(encoding="utf-8")
    array = ARRAY.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")
    efficiency = EFFICIENCY.read_text(encoding="utf-8")
    report = REPORT.read_text(encoding="utf-8")
    gpflow_select = GPFLOW_SELECT.read_text(encoding="utf-8")

    assert "from scripts.build_era5_a100_manifests import build_manifests" in submit
    assert "outputs = build_manifests(" in submit
    assert "stage2_jobs" not in submit
    assert "stage3_jobs" not in submit
    assert 'MANIFEST_DIR="${BENCHMARK_ROOT}/manifests"' in submit
    assert 'JOB_IDS_JSON="${BENCHMARK_ROOT}/slurm_job_ids.json"' in submit
    assert '"dependency": dependency or None' in submit
    assert '"submitted_at": submitted_at' in submit
    assert '"created_at": datetime.now(timezone.utc).isoformat()' in submit
    assert 'JOB_SUBMITTED_AT["${name}"]=' in submit
    for name in (
        "shared_batch_short",
        "official_long_preflight",
        "official_long_full",
        "online_short",
        "online_long",
        "efficiency",
    ):
        assert f'expected = output_path / f"{{name}}.jsonl"' in submit
        assert f'"{name}"' in submit

    assert 'submit_single prepare "afterok:${JOB_IDS[validation]}"' in submit
    assert 'submit_workers shared_batch_gpflow_preflight "afterok:${JOB_IDS[prepare]}"' in submit
    assert 'submit_single gpflow_tier_selection' in submit
    assert 'afterany:${JOB_IDS[shared_batch_gpflow_preflight]}' in submit
    assert 'submit_workers shared_batch_short "afterok:${JOB_IDS[gpflow_tier_selection]}"' in submit
    assert 'submit_workers official_long_preflight "afterany:${JOB_IDS[shared_batch_short]}"' in submit
    assert 'submit_workers official_long_full "afterany:${JOB_IDS[official_long_preflight]}"' in submit
    assert 'submit_workers online_short "afterany:${JOB_IDS[official_long_full]}"' in submit
    assert 'submit_workers online_short_postprocess "afterany:${JOB_IDS[online_short]}"' in submit
    assert 'submit_workers online_long "afterany:${JOB_IDS[online_short_postprocess]}"' in submit
    assert 'submit_single efficiency "afterany:${JOB_IDS[online_long]}"' in submit
    assert 'if [[ -z "${PARTITION}" ]]' in submit
    assert "cancel_submitted_jobs" in submit
    assert 'WORKER_SCRIPT="${SCRIPT_DIR}/run_manifest_worker.sbatch"' in submit
    assert "avoids submitting one Slurm array element per record" in submit
    assert 'emit_indices("shared_preflight", "shared_batch_short", "gpflow_feasibility_preflight")' in submit
    assert 'emit_indices("online_short_models", "online_short", "online")' in submit
    assert 'emit_indices("online_short_postprocess", "online_short", "postprocess")' in submit
    assert (
        'afterany:${JOB_IDS[shared_batch_short]}:${JOB_IDS[official_long_full]}:'
        '${JOB_IDS[online_short_postprocess]}:${JOB_IDS[online_long]}:${JOB_IDS[efficiency]}'
    ) in submit

    assert "slurm/manifests" not in prepare
    assert "slurm/manifests" not in array
    assert "indices[worker::workers]" in worker
    assert 'failures=$((failures + 1))' in worker
    assert 'if [[ "${failures}" -ne 0 ]]; then exit 1; fi' in worker
    assert "slurm/manifests" not in efficiency
    assert 'MANIFEST_DIR="${BENCHMARK_ROOT}/manifests"' in prepare
    assert '--stage prepare --include-legacy' in prepare
    assert "${BENCHMARK_ROOT}/manifests/${MANIFEST_NAME}.jsonl" in array
    assert "MANIFEST_COUNT=" in efficiency
    assert "summarize_era5_a100_efficiency.py" in efficiency
    assert "not_instrumented" in efficiency
    assert '"--metrics"' in efficiency
    assert "audit_era5_a100_shared_online.py" in report
    assert "generate_era5_a100_shared_online_report.py" in report
    assert "select_era5_gpflow_tier.py" in gpflow_select


def test_submit_dry_run_requires_partition_and_uses_bounded_workers(tmp_path: Path) -> None:
    common = [
        "bash",
        str(SUBMIT),
        "--repo",
        str(ROOT),
        "--env",
        str(tmp_path / "envs"),
        "--benchmark",
        str(tmp_path / "benchmark"),
        "--max-gpus",
        "3",
        "--dry-run",
    ]
    missing = subprocess.run(common, check=False, capture_output=True, text=True)
    assert missing.returncode == 2
    assert "--partition is required" in missing.stderr

    bounded = subprocess.run(
        [*common, "--partition", "a100"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert bounded.returncode == 0, bounded.stderr
    assert "shared_batch_short" in bounded.stdout
    assert "workers=3 records=141" in bounded.stdout
    assert "array=0,1,2,3" not in bounded.stdout


def test_qos_worker_executes_disjoint_manifest_strides(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    manifest_dir = benchmark / "manifests"
    manifest_dir.mkdir(parents=True)
    rows = []
    for index in range(6):
        output_dir = tmp_path / f"record-{index}"
        artifact = output_dir / "artifact.txt"
        rows.append(
            {
                "id": f"record-{index}",
                "argv": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])",
                    str(artifact),
                    str(index),
                ],
                "cwd": str(tmp_path),
                "output_dir": str(output_dir),
                "complete": [str(artifact)],
                "timeout_seconds": 10,
            }
        )
    (manifest_dir / "shared_batch_short.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    for worker in range(2):
        env = os.environ.copy()
        env["SLURM_ARRAY_TASK_ID"] = str(worker)
        env["ROUTEB_PY"] = sys.executable
        result = subprocess.run(
            [
                "bash",
                str(WORKER),
                "--repo",
                str(ROOT),
                "--benchmark",
                str(benchmark),
                "--manifest-name",
                "shared_batch_short",
                "--indices",
                "0-5",
                "--worker-count",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr

    for index in range(6):
        artifact = tmp_path / f"record-{index}" / "artifact.txt"
        assert artifact.read_text(encoding="utf-8") == str(index)


def test_partial_submission_failure_rolls_back_created_jobs(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "sbatch-count"
    cancelled = tmp_path / "cancelled.txt"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        f"count=$(cat '{counter}' 2>/dev/null || echo 0)\n"
        "count=$((count + 1))\n"
        f"echo \"$count\" >'{counter}'\n"
        "if [[ $count -eq 5 ]]; then exit 1; fi\n"
        "echo $((70000 + count))\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    scancel = fake_bin / "scancel"
    scancel.write_text(
        "#!/usr/bin/env bash\n" f"printf '%s\\n' \"$*\" >'{cancelled}'\n",
        encoding="utf-8",
    )
    scancel.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["PYTHON"] = sys.executable
    result = subprocess.run(
        [
            "bash",
            str(SUBMIT),
            "--repo",
            str(ROOT),
            "--env",
            str(tmp_path / "envs"),
            "--benchmark",
            str(tmp_path / "benchmark"),
            "--partition",
            "a100",
            "--max-gpus",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "cancelling this pipeline's partial jobs" in result.stderr
    assert cancelled.read_text(encoding="utf-8").strip() == "70001 70002 70003 70004"


def write_manifest(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def run_manifest(manifest: Path, repo: Path, *, index: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--index",
            str(index),
            "--repo",
            str(repo),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def test_manifest_executes_argv_without_shell_interpolation(tmp_path: Path) -> None:
    output = tmp_path / "result.txt"
    shell_marker = tmp_path / "shell_marker"
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        {
            "id": "literal-argv",
            "argv": [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])",
                str(output),
                f"$(touch {shell_marker})",
            ],
            "cwd": str(tmp_path),
            "complete": [str(output)],
            "timeout_seconds": 10,
        },
    )

    result = run_manifest(manifest, tmp_path)

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == f"$(touch {shell_marker})"
    assert not shell_marker.exists()
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_complete_record_is_skipped(tmp_path: Path) -> None:
    output = tmp_path / "result.txt"
    marker = tmp_path / "ran.txt"
    output.write_text("already complete", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        {
            "id": "skip-me",
            "argv": [
                sys.executable,
                "-c",
                "from pathlib import Path; Path(r'%s').write_text('ran')" % marker,
            ],
            "cwd": str(tmp_path),
            "complete": [str(output)],
        },
    )

    result = run_manifest(manifest, tmp_path)

    assert result.returncode == 0
    assert "SKIP complete" in result.stdout
    assert not marker.exists()


@pytest.mark.parametrize(
    ("name", "code", "expected"),
    [
        ("signal", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)", "signal"),
        ("oom", "raise SystemExit('CUDA out of memory')", "oom"),
        ("nonfinite", "import json; json.dump({'rmse': float('nan')}, open('result.json', 'w'))", "nonfinite"),
        ("error", "raise SystemExit('ordinary failure')", "error"),
    ],
)
def test_failure_classification(tmp_path: Path, name: str, code: str, expected: str) -> None:
    manifest = tmp_path / f"{name}.jsonl"
    record = {
        "id": name,
        "argv": [sys.executable, "-c", code],
        "cwd": str(tmp_path),
        "complete": [str(tmp_path / "result.json")],
        "timeout_seconds": 10,
    }
    write_manifest(manifest, record)

    result = run_manifest(manifest, tmp_path)

    assert result.returncode != 0
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == expected
    assert status["failure_class"] == expected


def test_timeout_is_classified(tmp_path: Path) -> None:
    manifest = tmp_path / "timeout.jsonl"
    write_manifest(
        manifest,
        {
            "id": "timeout",
            "argv": [sys.executable, "-c", "import time; time.sleep(10)"],
            "cwd": str(tmp_path),
            "complete": [str(tmp_path / "result.json")],
            "timeout_seconds": 0.1,
        },
    )

    result = run_manifest(manifest, tmp_path)

    assert result.returncode != 0
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "timeout"


def test_manifest_count_and_array_index_ignore_blank_lines(tmp_path: Path) -> None:
    manifest = tmp_path / "records.jsonl"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    manifest.write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", f"open(r'{first}', 'w').write('1')"],
                "cwd": str(tmp_path),
                "complete": [str(first)],
            }
        )
        + "\n\n"
        + json.dumps(
            {
                "argv": [sys.executable, "-c", f"open(r'{second}', 'w').write('2')"],
                "cwd": str(tmp_path),
                "complete": [str(second)],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert MODULE.manifest_count(manifest) == 2
    result = run_manifest(manifest, tmp_path, index=1)
    assert result.returncode == 0, result.stderr
    assert second.read_text(encoding="utf-8") == "2"
    assert not first.exists()
