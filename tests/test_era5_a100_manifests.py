from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from cloud.autodl_era5.run_benchmark import load_manifest_jobs, stage3_jobs
from scripts.build_era5_a100_manifests import (
    build_manifests,
    load_base_config,
    load_spec,
    python_paths,
)
from scripts.select_era5_gpflow_tier import evaluate_preflight


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "cloud/autodl_era5/benchmark_a100_shared_online_v2.json"
MANIFEST_NAMES = {
    "shared_batch_short.jsonl",
    "official_long_preflight.jsonl",
    "official_long_full.jsonl",
    "online_short.jsonl",
    "online_long.jsonl",
    "efficiency.jsonl",
}
ROUTE_B_BATCH = "scripts/run_iclr_era5_routeb_batch.py"
ROUTE_B_ONLINE = "scripts/run_iclr_era5_routeb_strict_online.py"
POSTPROCESS = "scripts/summarize_task2_online_segments.py"
CAPACITIES = {(128, 128), (128, 64), (64, 64)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def argument(row: dict, flag: str) -> str:
    argv = row["argv"]
    return argv[argv.index(flag) + 1]


def test_gpflow_preflight_selection_uses_only_runnability(tmp_path: Path) -> None:
    manifest = tmp_path / "shared_batch_short.jsonl"
    rows = []
    for index, peak in enumerate((4096.0, 6144.0)):
        output = tmp_path / f"preflight{index}"
        output.mkdir()
        (output / "status.json").write_text(
            json.dumps({"status": "complete", "wall_seconds": 10.0}) + "\n",
            encoding="utf-8",
        )
        (output / "result.json").write_text(
            json.dumps({"resources": {"peak_cuda_allocated_mib": peak}}) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "kind": "gpflow_feasibility_preflight",
                "output_dir": str(output),
                "expected": [str(output / "result.json")],
            }
        )
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    selected, decisions = evaluate_preflight(
        manifest,
        full_iterations=100,
        preflight_iterations=2,
        max_peak_mib=72 * 1024,
        max_estimated_seconds=6 * 3600,
    )
    assert selected is True
    assert all(row["accepted_8192"] for row in decisions)

    rows[1]["output_dir"] = str(tmp_path / "preflight1")
    (tmp_path / "preflight1" / "status.json").write_text(
        json.dumps({"status": "runtime_error", "wall_seconds": 10.0}) + "\n",
        encoding="utf-8",
    )
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    selected, decisions = evaluate_preflight(
        manifest,
        full_iterations=100,
        preflight_iterations=2,
        max_peak_mib=72 * 1024,
        max_estimated_seconds=6 * 3600,
    )
    assert selected is False
    assert decisions[1]["accepted_8192"] is False


def build(tmp_path: Path, *, gpflow_tier: str | None = None) -> tuple[dict[str, Path], Path]:
    benchmark = tmp_path / "benchmark"
    outputs = build_manifests(
        spec_path=SPEC,
        benchmark_root=benchmark,
        gpflow_tier=gpflow_tier,
    )
    return outputs, benchmark


def test_exact_manifests_use_benchmark_manifests_root(tmp_path: Path) -> None:
    outputs, benchmark = build(tmp_path)
    assert MANIFEST_NAMES.issubset({path.name for path in outputs.values()})
    assert all(path.parent == benchmark / "manifests" for path in outputs.values())
    assert MANIFEST_NAMES.issubset({path.name for path in (benchmark / "manifests").glob("*.jsonl")})


def test_shared_short_matrix_has_producers_and_required_families(tmp_path: Path) -> None:
    outputs, _ = build(tmp_path)
    rows = read_jsonl(outputs["shared_batch_short"])
    assert len(rows) == 143
    assert Counter(row["kind"] for row in rows) == Counter(
        {
            "protocol": 11,
            "calibration": 10,
            "batch": 120,
            "gpflow_feasibility_preflight": 2,
        }
    )
    assert all(row["schema_version"] == 1 for row in rows)
    assert all(isinstance(row["argv"], list) for row in rows)
    assert all(
        argument(row, "--target-mode") != "direct"
        for row in rows
        if "--target-mode" in row["argv"]
    )

    protocol = [row for row in rows if row["kind"] == "protocol"]
    shared_protocol = [row for row in protocol if row["method"] == "export_shared_protocol"]
    official_protocol = [
        row for row in protocol if row["method"] == "export_official_stvgp_protocol"
    ]
    assert len(shared_protocol) == 1
    assert len(official_protocol) == 10
    assert all(
        "/protocol/task1_2/seed" in artifact
        or "/protocol/task1_10/seed" in artifact
        or artifact.endswith("manifest.json")
        for artifact in shared_protocol[0]["expected"]
    )
    assert all(
        artifact.endswith("/data.npz")
        for row in official_protocol
        for artifact in row["expected"]
    )
    available = set(shared_protocol[0]["expected"])
    for row in rows:
        if row["kind"] == "protocol":
            available.update(row["expected"])
            continue
        missing = [
            dependency
            for dependency in row["dependencies"]
            if dependency not in available and not Path(dependency).exists()
        ]
        assert not missing, f"{row['job_id']} has no preceding producer for {missing}"
        available.update(row["expected"])

    calibration = [row for row in rows if row["kind"] == "calibration"]
    assert len(calibration) == 10
    assert {row["method"] for row in calibration} == {
        "routeb_joint_analytic_hippo_rff",
        "routeb_joint_inducing_points",
    }
    assert {row["seed"] for row in calibration} == {0, 1, 2, 3, 4}
    assert all(argument(row, "--data-part") == "calibration" for row in calibration)
    assert all("--include-conditional-residual-variance" in row["argv"] for row in calibration)

    assert sum(row["method"] == "xlag_mean_only" for row in rows) == 5
    assert sum(row["method"] == "official_st_vgp_full" for row in rows) == 5
    assert sum(row["method"].startswith("official_st_svgp_ms") for row in rows) == 10
    assert sum(row["method"].startswith("official_mf_st_svgp_ms") for row in rows) == 10
    assert not any(
        row["method"] == "official_st_vgp_full" and "--num-spatial-inducing" in row["argv"]
        for row in rows
    )
    assert all(
        "--fixed-spatial-inducing" in row["argv"]
        for row in rows
        if row["method"].startswith(("official_st_svgp_", "official_mf_st_svgp_"))
    )

    routeb = [
        row
        for row in rows
        if row["scope"] == "task1_2" and len(row["argv"]) > 1 and row["argv"][1] == ROUTE_B_BATCH
    ]
    assert len(routeb) == 60
    assert {
        (int(argument(row, "--mt")), int(argument(row, "--ms"))) for row in routeb
    } == CAPACITIES
    assert {
        argument(row, "--target-mode") for row in routeb
    } == {"shared_xlag_residual", "joint_xlag"}
    assert {
        argument(row, "--representation") for row in routeb
    } == {"analytic_hippo_rff", "inducing_points"}
    assert all("--include-conditional-residual-variance" in row["argv"] for row in routeb)
    assert all(row["variance_protocol"] == "full_joint_conditional" for row in routeb)

    gpflow_preflight = [row for row in rows if row["kind"] == "gpflow_feasibility_preflight"]
    gpflow_selected = [row for row in rows if row["method"].startswith("gpflow_feasibility_")]
    assert len(gpflow_preflight) == 2
    assert {row["seed"] for row in gpflow_preflight} == {0}
    assert {
        (int(argument(row, "--mt")), int(argument(row, "--ms"))) for row in gpflow_preflight
    } == {(128, 64), (64, 128)}
    assert all(row["selection"] == {"family": "gpflow_feasibility", "role": "preflight", "tier": "8192", "decision_seed": 0} for row in gpflow_preflight)
    assert len(gpflow_selected) == 10
    assert {row["selection"]["tier"] for row in gpflow_selected} == {"8192"}
    assert all(
        int(argument(row, "--mt")) * int(argument(row, "--ms")) == 8192
        for row in gpflow_selected
    )


def test_gpflow_fallback_keeps_preflight_out_of_main_results(tmp_path: Path) -> None:
    outputs, _ = build(tmp_path, gpflow_tier="4096")
    rows = read_jsonl(outputs["shared_batch_short"])
    preflight = [row for row in rows if row["kind"] == "gpflow_feasibility_preflight"]
    selected = [row for row in rows if row["method"].startswith("gpflow_feasibility_")]
    assert {
        (int(argument(row, "--mt")), int(argument(row, "--ms"))) for row in preflight
    } == {(128, 64), (64, 128)}
    assert {row["selection"]["tier"] for row in selected} == {"4096"}
    assert {
        (int(argument(row, "--mt")), int(argument(row, "--ms"))) for row in selected
    } == {(64, 64), (32, 128)}
    assert all(row["seed"] in {0, 1, 2, 3, 4} for row in selected)


def test_long_preflight_and_full_are_real_and_disjoint(tmp_path: Path) -> None:
    outputs, _ = build(tmp_path)
    preflight = read_jsonl(outputs["official_long_preflight"])
    full = read_jsonl(outputs["official_long_full"])
    assert len(preflight) == 3
    assert len(full) == 15
    assert {row["kind"] for row in preflight} == {"official_preflight"}
    assert {row["kind"] for row in full} == {"official_full"}
    assert {row["seed"] for row in preflight} == {0}
    assert {row["method"] for row in preflight} == {
        "official_preflight_st_vgp_full",
        "official_preflight_st_svgp_ms64",
        "official_preflight_st_svgp_ms128",
    }
    assert {row["method"] for row in full} == {
        "official_st_vgp_full",
        "official_st_svgp_ms64",
        "official_st_svgp_ms128",
    }
    assert all(row["argv"][1] == "scripts/run_official_stvgp_legacy.py" for row in preflight + full)
    assert all(argument(row, "--iterations") == "2" for row in preflight)
    assert all("official_protocol/task1_10" in argument(row, "--data-npz") for row in preflight)
    assert all("--use-xlag-mean" in row["argv"] for row in preflight + full)
    assert not any(
        row["method"].endswith("st_vgp_full") and "--num-spatial-inducing" in row["argv"]
        for row in preflight + full
    )
    assert all(
        "--fixed-spatial-inducing" in row["argv"]
        for row in preflight + full
        if row["method"].startswith(("official_preflight_st_svgp_", "official_st_svgp_"))
    )
    long_text = " ".join(" ".join(row["argv"]) for row in preflight + full)
    assert ROUTE_B_BATCH not in long_text
    assert "scripts/run_official_gpflow_svgp_era5.py" not in long_text
    assert "scripts/run_official_markovflow_stsvgp_era5.py" not in long_text


def _online_rows(outputs: dict[str, Path], name: str) -> tuple[list[dict], list[dict]]:
    rows = read_jsonl(outputs[name])
    return [row for row in rows if row["kind"] == "online"], [
        row for row in rows if row["kind"] == "postprocess"
    ]


def test_online_scopes_have_independent_routeb_matrices_and_task2_jobs(tmp_path: Path) -> None:
    outputs, benchmark = build(tmp_path)
    base_config = load_base_config(SPEC, load_spec(SPEC))
    expected_stage3 = stage3_jobs(base_config, benchmark, python_paths(base_config))
    for name, scope in (("online_short", "task1_2"), ("online_long", "task1_10")):
        models, postprocess = _online_rows(outputs, name)
        assert {row["scope"] for row in models} == {scope}
        assert len(models) == 55
        expected_baselines = {
            job.identifier
            for job in expected_stage3
            if job.scope == scope and ROUTE_B_ONLINE not in job.command
        }
        assert expected_baselines.issubset({row["job_id"] for row in models})
        routeb = [row for row in models if row["argv"][1] == ROUTE_B_ONLINE]
        assert len(routeb) == 30
        assert {
            (int(argument(row, "--mt")), int(argument(row, "--ms"))) for row in routeb
        } == CAPACITIES
        assert Counter(
            (argument(row, "--representation"), argument(row, "--mt"), argument(row, "--ms"))
            for row in routeb
        ) == Counter(
            {
                (representation, str(mt), str(ms)): 5
                for representation in ("analytic_hippo_rff", "inducing_points")
                for mt, ms in CAPACITIES
            }
        )
        assert all("--include-conditional-residual-variance" in row["argv"] for row in routeb)
        assert all(row["variance_protocol"] == "full_joint_conditional" for row in routeb)
        assert {row["seed"] for row in routeb} == {0, 1, 2, 3, 4}
        if scope == "task1_2":
            assert len(postprocess) == len(models) == 55
            assert all(row["argv"][1] == POSTPROCESS for row in postprocess)
            assert all(Path(row["expected"][0]).name == "task2_online_segments.csv" for row in postprocess)
            assert all(Path(row["expected"][1]).name == "task2_online_segments.json" for row in postprocess)
            assert {
                row["method"].removeprefix("postprocess_task2_online_segments_")
                for row in postprocess
                if row["method"].startswith("postprocess_task2_online_segments_routeb_")
            } == {row["method"] for row in routeb if row["seed"] in {0, 1, 2, 3, 4}}
        else:
            assert not postprocess


def test_online_dependencies_are_produced_by_prior_manifests(tmp_path: Path) -> None:
    outputs, _ = build(tmp_path)
    shared = read_jsonl(outputs["shared_batch_short"])
    producers = {artifact for row in shared for artifact in row["expected"]}
    calibration_theta = {
        artifact
        for row in shared
        if row["kind"] == "calibration"
        for artifact in row["expected"]
        if artifact.endswith("result.json")
    }
    assert len(calibration_theta) == 10
    for name in ("online_short", "online_long"):
        models, postprocess = _online_rows(outputs, name)
        for row in models:
            assert set(row["dependencies"]).issubset(producers)
        assert all(
            set(row["dependencies"]).issubset(
                {artifact for model in models for artifact in model["expected"]}
            )
            for row in postprocess
        )
    assert any(
        "calibration/routeb_joint_analytic_hippo_rff/seed0/result.json" in dependency
        for row in read_jsonl(outputs["online_short"])
        if row["kind"] == "online"
        for dependency in row["dependencies"]
    )


def test_efficiency_contains_common_counter_profiles_and_cpu_references(tmp_path: Path) -> None:
    outputs, _ = build(tmp_path)
    rows = read_jsonl(outputs["efficiency"])
    probes = [row for row in rows if row["kind"] == "efficiency_probe"]
    profiles = [row for row in rows if row["kind"] == "efficiency_profile"]
    records = [row for row in rows if row["kind"] == "efficiency_record"]
    summary = [row for row in rows if row["kind"] == "efficiency_summary"]
    assert len(probes) == 6
    assert len(profiles) == 40
    assert len(records) == 9
    assert len(summary) == 1
    assert {row["table"] for row in probes} == {"Table2A", "Table3A", "Table3B"}
    assert {row["seed"] for row in probes} == {0}
    assert all(row["warmup"] == 10 and row["repeats"] >= 30 for row in probes)
    assert all(
        row["objective"] and row["unit"] and row["execution"] == {"gpu_count": 1, "serial": True}
        for row in probes
    )
    assert {row["measurement_status"] for row in records} == {"cpu_not_applicable"}
    assert all(row["compute_contract"]["comparison_status"] == "not_applicable" for row in records)
    assert {row["branch"] for row in profiles} == {"batch", "online"}
    assert all(row["ncu"] == {"enabled": True, "range": row["ncu"]["range"], "target": "last", "work_unit": row["ncu"]["work_unit"]} for row in profiles)
    assert all(row["measurement_status"] == "scheduled_common_hardware_counter" for row in profiles)
    assert all(row["precision"] == "float64" and row["hardware_class"] == "NVIDIA A100" for row in profiles)
    assert any(row["method"] == "official_st_vgp_full" for row in profiles)
    assert any(row["method"].startswith("gpflow_feasibility_") for row in profiles)
    assert any(row["configuration"].get("method", "").startswith("markovflow_") for row in records)
    stochastic = [row for row in profiles if row["method"].startswith("gpflow_")]
    assert stochastic
    assert all(row["compute_contract"]["data_access_unit"] == "stochastic_minibatch" for row in stochastic)
    assert all(row["compute_contract"]["work_unit"] == "one_full_data_pass" for row in stochastic)
    assert all(row["compute_contract"]["native_work_unit"] == "one_minibatch_optimization_update" for row in stochastic)
    shared_rows = read_jsonl(outputs["shared_batch_short"])
    assert all(row["kind"] != "batch" for row in shared_rows if row["scope"] == "calibration")
    assert summary[0]["execution"] == {"gpu_count": 1, "serial": True}
    assert summary[0]["argv"][1] == "scripts/summarize_era5_a100_efficiency.py"
    assert "--output" in summary[0]["argv"]
    assert "--ratio-output" in summary[0]["argv"]
    assert "--markdown-output" in summary[0]["argv"]


def test_efficiency_profiles_preserve_the_selected_hardware_label(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    outputs = build_manifests(
        spec_path=SPEC,
        benchmark_root=benchmark,
        hardware_class="NVIDIA RTX 6000 Ada",
    )
    rows = read_jsonl(outputs["efficiency"])
    profiles = [row for row in rows if row["kind"] == "efficiency_profile"]
    assert profiles
    assert {row["hardware_class"] for row in profiles} == {"NVIDIA RTX 6000 Ada"}


def test_manifests_are_deterministic_and_runner_accepts_argv_jsonl(tmp_path: Path) -> None:
    outputs, benchmark = build(tmp_path)
    before = {name: path.read_text(encoding="utf-8") for name, path in outputs.items()}
    outputs_again = build_manifests(spec_path=SPEC, benchmark_root=benchmark)
    assert {name: path.read_text(encoding="utf-8") for name, path in outputs_again.items()} == before
    for path in outputs.values():
        jobs = load_manifest_jobs(path)
        assert jobs
        assert all(job.command == list(job.command) for job in jobs)
    combined_jobs = load_manifest_jobs(outputs["combined"])
    assert len(combined_jobs) == sum(len(load_manifest_jobs(path)) for name, path in outputs.items() if name != "combined")
