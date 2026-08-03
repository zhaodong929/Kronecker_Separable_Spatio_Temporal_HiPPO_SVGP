# AutoDL RTX 4090 ERA5 Stage 2+ Benchmark

This package reruns the ERA5 experiments from Stage 2 onward under one shared
protocol and records the artifacts needed for an ICLR-style comparison.

## What is included

- Controlled batch/full-history experiments with a shared X-lag mean.
- Strict online streaming experiments with Task-1-only calibration and no replay.
- Five paired spatial split seeds (`0,1,2,3,4`).
- Short Task-2 and long Task-2--10 streams.
- Runtime, first/steady iteration, time-to-best, prediction time, RSS, CUDA
  memory, persistent state, replay-buffer size, and FLOP metadata.
- Reproducibility audit and report/figure generation.

The default modern-GPU matrix contains Route B HiPPO, Route B ordinary temporal
inducing points, GPflow SVGP, Bui OSGPR, Maddox StreamingSGPR, official OHSVGP,
and X-lag-only controls. Bayes-Newton ST-SVGP/MF-ST-SVGP and Markovflow v0.0.13
are optional legacy compatibility runs.

## Interpretation boundary

The official Bayes-Newton and Markovflow paper-era environments cannot execute
natively on an RTX 4090 without changing their dependency stack and therefore
their implementation. They are retained as pinned CPU compatibility paths.
Their timings are reported separately and must not be used in a same-GPU
speedup claim. A failed legacy run is classified as OOM, timeout, dependency
error, or GPU incompatibility; it is never converted into a predictive score.

Route B now uses PyTorch on the selected GPU for both batch hyperparameter
learning and the structured Schur/Sylvester posterior and prediction path.
Strict-online Route B keeps its sufficient-statistic state on the GPU and only
returns predictions and metrics to the host. This is the same finite-DTC,
Kronecker/Sylvester algorithm as the NumPy/SciPy reference; no dense
``(M_s M_t) x (M_s M_t)`` precision is introduced.

The analytic HiPPO implementation also has a pure-PyTorch CUDA path for the
RFF features, spherical-Bessel/Miller recurrence, temporal covariances, and
their gradients. The default strict-online configuration is deliberately
hybrid: it builds these small sequential temporal factors on CPU and transfers
them to the CUDA-resident posterior. Use ``--temporal-factor-device solver``
for the full-CUDA ablation. Ordinary temporal inducing-point factors default to
CUDA because their dense kernel construction benefits from the GPU.

## 1. Prepare data locally

Run this in the project checkout that already contains both processed datasets:

```bash
bash cloud/autodl_era5/pack_processed_data.sh \
  /mnt/d/era5_task1_10_processed.tar.zst
```

Upload both the archive and its `.sha256` file to the AutoDL data disk. The
archive contains only processed Task 1--2 and Task 1--10 data; it does not
contain experiment results or Python environments.

## 2. Create the AutoDL instance

Recommended instance:

- RTX 4090 24 GiB.
- At least 16 CPU cores and 64 GiB host RAM for the modern matrix.
- Ubuntu 22.04 or 24.04 image with an NVIDIA driver compatible with CUDA 12.1.
- At least 120 GiB free data-disk space for environments, logs, checkpoints,
  predictions, and figures.

The publication protocol defaults to `float64`. This favors numerical parity,
but consumer Ada GPUs have weak FP64 throughput. A separate `float32` run may be
used as a speed/precision ablation; do not silently mix it into the primary table.

## 3. Clone and install

```bash
cd /root/autodl-tmp
git clone --branch codex/autodl-era5-benchmark \
  git@github.com:zhaodong929/Kronecker_Separable_Spatio_Temporal_HiPPO_SVGP.git
cd Kronecker_Separable_Spatio_Temporal_HiPPO_SVGP

bash cloud/autodl_era5/unpack_processed_data.sh \
  /root/autodl-tmp/era5_task1_10_processed.tar.zst

bash cloud/autodl_era5/setup_autodl.sh
bash cloud/autodl_era5/validate_environments.sh

# Runs NumPy-vs-Torch CPU checks and an additional CPU-vs-CUDA check on AutoDL.
${AUTODL_ENV_ROOT:-/root/autodl-tmp/stvgp_envs}/routeb/bin/python \
  -m pytest -q tests/test_routeb_torch_backend.py \
  tests/test_temporal_analytic_gradients.py
```

The setup script clones every official baseline at an exact commit and verifies
the checkout. It creates independent Route B, GPflow, and Maddox environments
under `/root/autodl-tmp/stvgp_envs`.

To install the optional legacy CPU environments:

```bash
INCLUDE_LEGACY=1 bash cloud/autodl_era5/setup_autodl.sh
```

Use a high-memory CPU instance for those runs. A 24 GiB RTX 4090 does not solve
the legacy host-memory requirements.

## 4. Run a one-seed pilot first

Use `tmux` or another persistent shell. All jobs are resumable and write an
atomic `status.json`.

```bash
export AUTODL_ENV_ROOT=/root/autodl-tmp/stvgp_envs
export BENCHMARK_ROOT=/root/autodl-tmp/iclr_era5_stage2plus

${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py \
  --stage prepare

${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py \
  --stage stage2 --scope calibration --seed 0

${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py \
  --stage stage2 --scope task1_2 --seed 0

${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py \
  --stage stage3 --scope task1_2 --seed 0
```

Inspect failures and GPU use before starting the long matrix:

```bash
find "${BENCHMARK_ROOT}" -name status.json -print0 \
  | xargs -0 grep -H '"status"'
nvidia-smi
```

## 5. Run the publication matrix

```bash
bash cloud/autodl_era5/run_all.sh
```

The default matrix consists of 144 jobs:

- 1 shared protocol export.
- 70 Stage-2 jobs, including 10 Task-1 Route-B calibration jobs.
- 70 strict-online jobs.
- 1 efficiency aggregation job.
- 2 audit/report jobs.

Jobs run sequentially to keep GPU-memory accounting interpretable. Completed
artifacts are skipped. Use `--force` only when intentionally replacing a run.

Useful filters:

```bash
# List commands without executing them.
${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py --list

# Dry-run the long stream only.
${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py \
  --stage stage3 --scope task1_10 --dry-run

# Run one method family.
${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py \
  --stage stage3 --method-pattern 'routeb|ohsvgp'

# Retry only a known failed command after diagnosis.
${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py \
  --stage stage3 --scope task1_10 --seed 2 \
  --method-pattern 'official_ohsvgp' --force
```

Optional legacy attempts are enabled explicitly:

```bash
${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py \
  --stage prepare --include-legacy
${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py \
  --stage stage2 --include-legacy --method-pattern 'official|markovflow'
```

To isolate the Route B CPU/GPU bottlenecks, run the matched backend profiler
on one exported protocol. For HiPPO it records NumPy CPU, Torch CPU, full CUDA,
and CPU-Bessel/CUDA-posterior hybrid results separately:

```bash
${AUTODL_ENV_ROOT}/routeb/bin/python \
  scripts/profile_routeb_backend_bottlenecks.py \
  --protocol-npz "${BENCHMARK_ROOT}/protocol/task1_2/seed0/protocol.npz" \
  --protocol-json "${BENCHMARK_ROOT}/protocol/task1_2/seed0/protocol.json" \
  --theta-json "${BENCHMARK_ROOT}/calibration/routeb_joint_analytic_hippo_rff/seed0/result.json" \
  --output-dir "${BENCHMARK_ROOT}/efficiency/routeb_hippo_backend_profile" \
  --representation analytic_hippo_rff --mt 128 --ms 128 --repeats 5
```

## 6. Audit and report

The normal `run_all.sh` path performs these steps. They can be regenerated
without rerunning models:

```bash
${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py \
  --stage stage4 --force
${AUTODL_ENV_ROOT}/routeb/bin/python cloud/autodl_era5/run_benchmark.py \
  --stage stage5 --force
```

Principal outputs:

```text
${BENCHMARK_ROOT}/efficiency/efficiency_per_run.csv
${BENCHMARK_ROOT}/efficiency/efficiency_summary.csv
${BENCHMARK_ROOT}/audit.json
${BENCHMARK_ROOT}/report/report.md
${BENCHMARK_ROOT}/report/report.tex
${BENCHMARK_ROOT}/report/report.pdf        # when latexmk is installed
${BENCHMARK_ROOT}/report/artifact_manifest.json
```

Install TeX during setup with `INSTALL_TEX=1` if a PDF is required on the
instance. Markdown, CSV, JSON, PNG, and vector-PDF figures do not require TeX.

## 7. Download results

```bash
bash cloud/autodl_era5/collect_results.sh \
  "${BENCHMARK_ROOT}" \
  /root/autodl-tmp/iclr_era5_stage2plus_results.tar.zst
```

Download the archive and checksum before releasing the instance. Do not commit
the result archive, processed ERA5 data, environments, or generated PDFs to Git.

## Publication checklist

- Use the same completed seed set in every paired claim.
- Report mean and standard deviation over five spatial splits.
- Keep batch/full-history and strict-online tables separate.
- Charge Task-1 calibration to online end-to-end time.
- Separate first-iteration/JIT cost from steady-state time.
- Verify and report the Route B NumPy/Torch/CUDA parity test before timing.
- Include host-to-device block-factor transfer in strict-online update time.
- Compare FLOPs only when `flops_scope` is compatible.
- Report OOM and timeout as execution outcomes.
- Preserve `command.txt`, `environment.json`, `status.json`, `run.log`,
  `resource_usage.txt`, and `nvidia_smi.csv` for every run.
