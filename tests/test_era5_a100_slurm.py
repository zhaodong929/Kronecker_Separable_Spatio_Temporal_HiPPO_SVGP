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
PERSISTENT = ROOT / "slurm/era5_a100/run_persistent_pipeline_worker.sbatch"
PREPARE = ROOT / "slurm/era5_a100/prepare_protocol.sbatch"
EFFICIENCY = ROOT / "slurm/era5_a100/run_efficiency.sbatch"
REPORT = ROOT / "slurm/era5_a100/generate_report.sbatch"
GPFLOW_SELECT = ROOT / "slurm/era5_a100/select_gpflow_tier.sbatch"
SPEC = importlib.util.spec_from_file_location("era5_a100_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_submit_contract_uses_exactly_three_persistent_workers() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    persistent = PERSISTENT.read_text(encoding="utf-8")
    prepare = PREPARE.read_text(encoding="utf-8")
    array = ARRAY.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")
    efficiency = EFFICIENCY.read_text(encoding="utf-8")
    report = REPORT.read_text(encoding="utf-8")
    gpflow_select = GPFLOW_SELECT.read_text(encoding="utf-8")

    assert 'SBATCH_ARGS=(' in submit
    assert '--array="0-2%3"' in submit
    assert '--time="72:00:00"' in submit
    assert submit.count('raw="$(sbatch ') == 1
    assert "submit_single" not in submit
    assert "submit_workers" not in submit
    assert "--dependency" not in submit
    assert 'JOB_IDS_JSON="${BENCHMARK_ROOT}/slurm_job_ids.json"' in submit
    assert '"created_at": datetime.now(timezone.utc).isoformat()' in submit
    assert 'if [[ -z "${PARTITION}" ]]' in submit
    assert '"submission_mode": "three_persistent_a100_workers"' in submit
    assert '"submitted_job_count": 3' in submit

    assert "#SBATCH --gpus=1" in persistent
    assert "#SBATCH --time=72:00:00" in persistent
    assert 'SCRIPT_DIR="${REPO_ROOT}/slurm/era5_a100"' in persistent
    assert 'if [[ "${WORKER_COUNT}" != "3" ]]' in persistent
    assert "run_leader_stage validation" in persistent
    assert "run_leader_stage prepare" in persistent
    assert "run_parallel_stage gpflow_preflight" in persistent
    assert "run_leader_stage gpflow_tier" in persistent
    assert "run_parallel_stage shared_batch" in persistent
    assert "run_parallel_stage official_long_full" in persistent
    assert "run_parallel_stage online_short " in persistent
    assert "run_parallel_stage online_short_postprocess" in persistent
    assert "run_parallel_stage online_long" in persistent
    assert 'if [[ "${WORKER_ID}" -ne 0 ]]' in persistent
    assert "run_efficiency.sbatch" in persistent
    assert "generate_report.sbatch" in persistent

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
    assert '"--nvtx"' in efficiency
    assert '"--nvtx-include"' in efficiency
    assert "parse_era5_a100_ncu_flops.py" in efficiency
    assert "ncu_flops.json" in efficiency
    assert "audit_era5_a100_shared_online.py" in report
    assert "generate_era5_a100_shared_online_report.py" in report
    assert "select_era5_gpflow_tier.py" in gpflow_select


def test_submit_dry_run_requires_partition_and_submits_one_three_worker_array(tmp_path: Path) -> None:
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
    assert "single submission" in bounded.stdout
    assert "array=0-2%3" in bounded.stdout
    assert "gpus=3" in bounded.stdout
    assert "time=72:00:00" in bounded.stdout
    assert "shared_batch_short" not in bounded.stdout


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


def test_real_submit_invokes_sbatch_once(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "sbatch-count"
    arguments = tmp_path / "sbatch-arguments.txt"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        f"count=$(cat '{counter}' 2>/dev/null || echo 0)\n"
        "count=$((count + 1))\n"
        f"echo \"$count\" >'{counter}'\n"
        f"printf '%s\\n' \"$*\" >'{arguments}'\n"
        "echo 70001\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)

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
    assert result.returncode == 0, result.stderr
    assert counter.read_text(encoding="utf-8").strip() == "1"
    submitted = arguments.read_text(encoding="utf-8")
    assert "--partition=a100" in submitted
    assert "--array=0-2%3" in submitted
    assert "--time=72:00:00" in submitted
    assert str(PERSISTENT) in submitted
    metadata = json.loads(
        (tmp_path / "benchmark" / "slurm_job_ids.json").read_text(encoding="utf-8")
    )
    assert metadata["submitted_job_count"] == 3
    assert metadata["jobs"]["persistent_pipeline"]["job_id"] == "70001"


def test_efficiency_wrapper_parses_common_ncu_flop_artifact(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    manifest_dir = benchmark / "manifests"
    manifest_dir.mkdir(parents=True)
    output = benchmark / "efficiency" / "profiles" / "batch" / "routeb" / "seed0"
    result = output / "result.json"
    command = (
        "from pathlib import Path; import json; "
        "p=Path(r'" + str(result) + "'); p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_text(json.dumps({'scope':'task1_2','branch':'batch','method':'routeb','seed':0}))"
    )
    record = {
        "job_id": "profile-routeb",
        "argv": [sys.executable, "-c", command],
        "scope": "task1_2",
        "branch": "batch",
        "method": "routeb",
        "seed": 0,
        "output_dir": str(output),
        "expected": [str(result)],
        "timeout_seconds": 30,
        "precision": "float64",
        "hardware_class": "NVIDIA A100",
        "ncu": {
            "enabled": True,
            "range": "era5_batch_update",
            "target": "last",
            "work_unit": "one_full_fit_optimization_update",
        },
        "compute_contract": {
            "schema_version": 2,
            "baseline_family": "routeb",
            "data_access_unit": "full_fit_dataset",
            "measurement_scope": "optimization_update",
            "work_unit": "one_full_fit_optimization_update",
            "native_work_unit": "one_full_fit_optimization_update",
            "comparison_group": "batch_full_fit_update",
            "required_measurement_backend": "nsight_compute_executed_gpu_flops",
            "comparison_status": "pending_common_hardware_counter",
        },
    }
    (manifest_dir / "efficiency.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/usr/bin/env bash\necho 'NVIDIA A100-SXM4-80GB'\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)
    ncu = fake_bin / "ncu"
    ncu.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == \"--import\" ]]; then\n"
        "  printf '\"Metric Name\",\"Metric Value\"\\n'\n"
        "  printf '\"smsp__sass_thread_inst_executed_op_dadd_pred_on.sum\",\"10\"\\n'\n"
        "  printf '\"smsp__sass_thread_inst_executed_op_dmul_pred_on.sum\",\"20\"\\n'\n"
        "  printf '\"smsp__sass_thread_inst_executed_op_dfma_pred_on.sum\",\"30\"\\n'\n"
        "  exit 0\n"
        "fi\n"
        "report=''\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  case \"$1\" in\n"
        "    --export) report=\"$2\"; shift 2 ;;\n"
        "    --) shift; \"$@\"; rc=$?; : > \"$report\"; exit $rc ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        "exit 2\n",
        encoding="utf-8",
    )
    ncu.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ROUTEB_PY": sys.executable,
            "NCU_BIN": str(ncu),
        }
    )
    completed = subprocess.run(
        ["bash", str(EFFICIENCY), "--repo", str(ROOT), "--benchmark", str(benchmark)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    ncu_flops = json.loads((output / "ncu_flops.json").read_text(encoding="utf-8"))
    assert ncu_flops["measurement_backend"] == "nsight_compute"
    assert ncu_flops["nsight_executed_gpu_flops"] == 90.0
    assert ncu_flops["nsight_flops_per_unit"] == 90.0
    assert (benchmark / "efficiency" / "era5_a100_flop_ratios.csv").is_file()


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
