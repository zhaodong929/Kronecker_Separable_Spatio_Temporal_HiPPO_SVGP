#!/usr/bin/env bash
# Run every efficiency manifest record serially in this exclusive allocation.
# Nsight is only claimed when an ncu wrapper actually ran.
#SBATCH --job-name=era5-a100-efficiency
#SBATCH --exclusive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=era5_a100_efficiency_%j.out
#SBATCH --error=era5_a100_efficiency_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO}}"
ENV_ROOT="${ERA5_ENV_ROOT:-${ENV_ROOT:-${HOME}/stvgp_envs}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/results/experiments_era5_ohsvgp_heldout_fullspace/paper_ready/ICLR Formal experiment/iclr_era5_full_benchmark}"
MANIFEST=""
MANIFEST_NAME=""
GPU_NAME_REGEX="${ERA5_GPU_NAME_REGEX:-a100}"
MIN_GPU_MEMORY_GIB="${ERA5_MIN_GPU_MEMORY_GIB:-75}"
REQUIRE_NCU=0

usage() {
  cat >&2 <<'EOF'
Usage: run_efficiency.sbatch [--manifest PATH | --manifest-name efficiency]
                            [--repo PATH] [--env PATH] [--benchmark PATH]
                            [--gpu-name-regex REGEX] [--min-gpu-memory-gib GIB]
                            [--require-ncu]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --manifest-name) MANIFEST_NAME="$2"; shift 2 ;;
    --repo) REPO_ROOT="$2"; shift 2 ;;
    --env) ENV_ROOT="$2"; shift 2 ;;
    --benchmark) BENCHMARK_ROOT="$2"; shift 2 ;;
    --gpu-name-regex) GPU_NAME_REGEX="$2"; shift 2 ;;
    --min-gpu-memory-gib) MIN_GPU_MEMORY_GIB="$2"; shift 2 ;;
    --require-ncu) REQUIRE_NCU=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! "${MIN_GPU_MEMORY_GIB}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "--min-gpu-memory-gib must be a positive number" >&2
  exit 2
fi

if [[ -n "${MANIFEST_NAME}" ]]; then
  if [[ "${MANIFEST_NAME}" != "efficiency" ]]; then
    echo "Unknown ERA5 efficiency manifest name: ${MANIFEST_NAME}" >&2
    exit 2
  fi
  if [[ -n "${MANIFEST}" ]]; then
    echo "Use only one of --manifest and --manifest-name" >&2
    exit 2
  fi
  MANIFEST="${BENCHMARK_ROOT}/manifests/${MANIFEST_NAME}.jsonl"
fi

REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
BENCHMARK_ROOT="$(cd "${BENCHMARK_ROOT}" && pwd)"
MANIFEST="${MANIFEST:-${BENCHMARK_ROOT}/manifests/efficiency.jsonl}"
ROUTEB_PY="${ROUTEB_PY:-${ENV_ROOT}/routeb/bin/python}"
RUNNER="${REPO_ROOT}/scripts/run_era5_a100_manifest_job.py"
SUMMARY="${REPO_ROOT}/scripts/summarize_era5_a100_efficiency.py"
PARSER="${REPO_ROOT}/scripts/parse_era5_a100_ncu_flops.py"

if [[ ! -x "${ROUTEB_PY}" ]]; then
  echo "Route B Micromamba Python is missing: ${ROUTEB_PY}" >&2
  exit 1
fi
if [[ ! -f "${RUNNER}" || ! -f "${SUMMARY}" || ! -f "${PARSER}" ]]; then
  echo "Efficiency runner, parser, or summarizer is missing under ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Efficiency manifest is missing: ${MANIFEST}" >&2
  exit 1
fi

GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
GPU_NAME="${GPU_INFO%%,*}"
GPU_MEMORY_MIB="${GPU_INFO#*,}"
GPU_MEMORY_MIB="${GPU_MEMORY_MIB//[[:space:]]/}"
if [[ -z "${GPU_NAME}" || ! "${GPU_MEMORY_MIB}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || ! [[ "${GPU_NAME,,}" =~ ${GPU_NAME_REGEX,,} ]] \
  || ! awk -v value="${GPU_MEMORY_MIB}" -v minimum="${MIN_GPU_MEMORY_GIB}" 'BEGIN { exit !(value >= minimum * 1024) }'; then
  echo "Common-counter profiling requires GPU regex ${GPU_NAME_REGEX} with at least ${MIN_GPU_MEMORY_GIB} GiB; found: ${GPU_INFO:-unavailable}" >&2
  exit 1
fi

MANIFEST_COUNT="$("${ROUTEB_PY}" - "${MANIFEST}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
print(sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()))
PY
)"
if [[ ! "${MANIFEST_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Efficiency manifest has no non-empty records: ${MANIFEST}" >&2
  exit 1
fi

TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/era5-a100-efficiency.XXXXXX")"
trap 'rm -rf -- "${TEMP_ROOT}"' EXIT

NCU_BIN="${NCU_BIN:-}"
if [[ -n "${NCU_BIN}" ]]; then
  NCU_BIN="$(command -v "${NCU_BIN}" || true)"
else
  NCU_BIN="$(command -v ncu || true)"
fi
if [[ "${REQUIRE_NCU}" -eq 1 && ( -z "${NCU_BIN}" || ! -x "${NCU_BIN}" ) ]]; then
  echo "--require-ncu was requested but Nsight Compute (ncu) is unavailable" >&2
  exit 1
fi
# Count executed arithmetic instructions for the float64 A100 path.  These
# are instruction counts, not a peak-throughput percentage.  The report keeps
# them separate from analytical and framework-profiler estimates.
NCU_METRICS="${NCU_METRICS:-smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum}"

failures=0
not_instrumented=0
ncu_parse_failures=0
processed=0
index=0
while IFS= read -r line || [[ -n "${line}" ]]; do
  if [[ -z "${line//[[:space:]]/}" ]]; then
    continue
  fi
  ncu_requested="$("${ROUTEB_PY}" - "${MANIFEST}" "${index}" <<'PY'
import json
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
wanted = int(sys.argv[2])
record_index = 0
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    if record_index == wanted:
        record = json.loads(line)
        value = record.get("ncu", False)
        if isinstance(value, str):
            value = value.strip().lower() not in {"", "0", "false", "no", "off"}
        elif isinstance(value, dict):
            value = value.get("enabled", True)
        print("1" if bool(value) else "0")
        break
    record_index += 1
else:
    raise SystemExit(f"manifest index not found: {wanted}")
PY
)"

  run_manifest="${MANIFEST}"
  run_index="${index}"
  if [[ "${ncu_requested}" == "1" ]]; then
    if [[ -n "${NCU_BIN}" && -x "${NCU_BIN}" ]]; then
      run_manifest="${TEMP_ROOT}/ncu_${index}.jsonl"
      "${ROUTEB_PY}" - "${MANIFEST}" "${index}" "${run_manifest}" \
        "${NCU_BIN}" "${NCU_METRICS}" "${REPO_ROOT}" <<'PY'
import json
import os
from pathlib import Path
import sys

source, wanted, target, ncu, metrics, repo = sys.argv[1:]
wanted = int(wanted)
record_index = 0
for line in Path(source).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    if record_index == wanted:
        record = json.loads(line)
        argv = record.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            raise SystemExit("ncu=true requires an argv string list")
        profile = record.get("ncu")
        if not isinstance(profile, dict):
            raise SystemExit("ncu=true requires an ncu object with range and work_unit")
        range_name = str(profile.get("range", "")).strip()
        work_unit = str(profile.get("work_unit", "")).strip()
        if not range_name or not work_unit:
            raise SystemExit("ncu profile requires non-empty range and work_unit")
        output_value = record.get("output_dir")
        if output_value is None:
            raise SystemExit("ncu profile requires output_dir")
        output = Path(os.path.expandvars(os.path.expanduser(str(output_value))))
        if not output.is_absolute():
            output = Path(repo) / output
        ncu_profile = {
            "range": range_name,
            "work_unit": work_unit,
            "report": str(output / "ncu_profile.ncu-rep"),
            "csv": str(output / "ncu_raw.csv"),
            "result": str(output / "result.json"),
            "flops": str(output / "ncu_flops.json"),
        }
        record["argv"] = [
            ncu,
            "--target-processes", "all",
            "--nvtx",
            "--nvtx-include", range_name + "/",
            "--metrics", metrics,
            "--export", ncu_profile["report"],
            "--force-overwrite",
            "--",
            *argv,
        ]
        env = record.get("env", {})
        if not isinstance(env, dict):
            raise SystemExit("ncu profile env must be an object")
        env.update({
            "ERA5_NCU_PROFILE_RANGE": range_name,
            "ERA5_NCU_PROFILE_TARGET": str(profile.get("target", "last")),
        })
        record["env"] = env
        record["ncu_requested"] = True
        record["ncu_executed"] = True
        record["ncu_metrics"] = metrics
        record["ncu_profile"] = ncu_profile
        Path(target).write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        break
    record_index += 1
else:
    raise SystemExit(f"manifest index not found: {wanted}")
PY
      run_index=0
    else
      "${ROUTEB_PY}" - "${MANIFEST}" "${index}" "${REPO_ROOT}" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

manifest, wanted, repo = sys.argv[1:]
wanted = int(wanted)
record_index = 0
for line in Path(manifest).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    if record_index == wanted:
        record = json.loads(line)
        output_value = record.get("output_dir")
        if output_value is None:
            artifacts = record.get("complete", record.get("expected", []))
            output = Path(artifacts[0]).parent if artifacts else Path(manifest).parent / "jobs" / f"{wanted:06d}"
        else:
            output = Path(os.path.expandvars(os.path.expanduser(str(output_value))))
        if not output.is_absolute():
            output = Path(repo) / output
        status_value = record.get("status_path") or (output / "status.json")
        status = Path(os.path.expandvars(os.path.expanduser(str(status_value))))
        if not status.is_absolute():
            status = Path(repo) / status
        status.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "job": str(record.get("id", f"array-{wanted}")),
            "manifest": str(Path(manifest).resolve()),
            "array_index": wanted,
            "status": "not_instrumented",
            "classification": "not_instrumented",
            "failure_class": "not_instrumented",
            "ncu_requested": True,
            "ncu_executed": False,
            "reason": "ncu=true but Nsight Compute (ncu) is unavailable; no Nsight claim was made",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = status.with_name(f".{status.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(status)
        break
    record_index += 1
else:
    raise SystemExit(f"manifest index not found: {wanted}")
PY
      echo "NOT_INSTRUMENTED efficiency index=${index}: ncu is unavailable" >&2
      not_instrumented=$((not_instrumented + 1))
      index=$((index + 1))
      processed=$((processed + 1))
      continue
    fi
  fi

  runner_succeeded=0
  if "${ROUTEB_PY}" "${RUNNER}" --manifest "${run_manifest}" --index "${run_index}" --repo "${REPO_ROOT}"; then
    runner_succeeded=1
  else
    rc=$?
    echo "Efficiency record ${index} failed with status ${rc}; continuing" >&2
    failures=$((failures + 1))
  fi
  if [[ "${ncu_requested}" == "1" && "${runner_succeeded}" == "1" ]]; then
    ncu_paths="$("${ROUTEB_PY}" - "${run_manifest}" "${run_index}" <<'PY'
import json
from pathlib import Path
import sys

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()[int(sys.argv[2])])
profile = record.get("ncu_profile", {})
for key in ("report", "csv", "result", "work_unit", "flops"):
    value = profile.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit("ncu profile metadata is incomplete")
print("\t".join(profile[key] for key in ("report", "csv", "result", "work_unit", "flops")))
PY
)"
    IFS=$'\t' read -r ncu_report ncu_csv ncu_result ncu_work_unit ncu_flops <<< "${ncu_paths}"
    if [[ -f "${ncu_report}" ]] \
      && "${NCU_BIN}" --import "${ncu_report}" --csv --page raw > "${ncu_csv}" \
      && "${ROUTEB_PY}" "${PARSER}" \
        --csv "${ncu_csv}" \
        --manifest-record "${run_manifest}" \
        --result "${ncu_result}" \
        --work-unit "${ncu_work_unit}" \
        --output "${ncu_flops}"; then
      echo "NCU_COMPLETE efficiency index=${index}: ${ncu_flops}" >&2
    else
      echo "NCU_PARSE_FAILED efficiency index=${index}: no completed common-counter artifact" >&2
      ncu_parse_failures=$((ncu_parse_failures + 1))
    fi
  fi
  index=$((index + 1))
  processed=$((processed + 1))
done < "${MANIFEST}"

if [[ "${processed}" -ne "${MANIFEST_COUNT}" ]]; then
  echo "Efficiency manifest changed while reading: expected ${MANIFEST_COUNT}, processed ${processed}" >&2
  exit 1
fi

SUMMARY_DIR="${BENCHMARK_ROOT}/efficiency"
mkdir -p "${SUMMARY_DIR}"
SUMMARY_RC=0
if "${ROUTEB_PY}" "${SUMMARY}" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --output "${SUMMARY_DIR}/era5_a100_efficiency.csv" \
  --ratio-output "${SUMMARY_DIR}/era5_a100_flop_ratios.csv" \
  --markdown-output "${SUMMARY_DIR}/era5_a100_efficiency.md"; then
  :
else
  SUMMARY_RC=$?
  echo "ERA5 A100 efficiency summarizer failed with status ${SUMMARY_RC}" >&2
fi

echo "Efficiency records processed=${processed} failures=${failures} not_instrumented=${not_instrumented} ncu_parse_failures=${ncu_parse_failures}"
if [[ "${failures}" -ne 0 || "${SUMMARY_RC}" -ne 0 \
  || ( "${REQUIRE_NCU}" -eq 1 && ( "${not_instrumented}" -ne 0 || "${ncu_parse_failures}" -ne 0 ) ) ]]; then
  exit 1
fi
