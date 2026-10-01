# GH200 migration preparation, 2026-10-01

The user requested GH200 shared-GPU execution after the earlier DoC-only plan.
This directory prepares qualification; it does **not** admit or launch the 65
final scientific runs. Keep the running DoC selection campaign intact.

## Actual state

- SSH alias `u6tj.aip2.isambard`, account `brics.u6tj`, architecture aarch64.
- Clifton authentication refreshed successfully using the existing local login
  automation. No credentials are committed or included in experiment artifacts.
- Deployment root: `/scratch/u6tj/nk523.u6tj/hipposvgp-gh200-20261001`.
- Numerical source is the unchanged `ea54966e3790bdf33114605048065ac0e98f68c8`
  snapshot, with pinned official source. These orchestration files are separate.
- Isolated Torch, tracking and JAX environments were installed. Torch uses a
  read-only inherited distribution; both local and inherited inventories are
  saved. JAX 0.4.35/Objax 1.8/BayesNewton 1.1 is an ARM port candidate, **not**
  qualified equivalence to the DoC JAX 0.4.23 environment.
- Tiny fixture and representative ERA5 seed 0, COVID seed 5, PEMS seed 1 data
  transferred. These are preparation copies, not the complete final campaign.
- TensorFlow container preparation submission was **rejected**, without a job
  ID: `AssocGrpCPUMinutesLimit`. A 1-GPU/5-minute `sbatch --test-only` was also
  rejected with the same accounting error. The exact parent allocation balance
  was not exposed by the user association query.
- **No GH200 GPU qualification or final experiment has started.**

## Gates after allocation becomes available

1. Complete TensorFlow container installation, using the pinned NVIDIA
   `24.02-tf2-py3` ARM image and GPflow 2.9.0. This installation script is
   syntax-checked only. Validate imports, CUDA availability and dependency
   compatibility before claiming the environment works.
2. Run `qualify.sbatch DEPLOYMENT_ROOT`: all five methods must pass the existing
   CPU/GPU diagnostic, with failures retained in W&B and a local summary.
3. Run full-size shape probes and compare outputs with the DoC references.
   CPU/GPU agreement within the ARM environment alone is not cross-platform
   equivalence. Do not relax numerical thresholds to force a pass.
4. Measure matched workloads sequentially and at concurrency 2 and 4 on one
   allocated GPU. Increase only when measured completed work per allocated GPU
   hour improves, memory has headroom and outputs remain equivalent. Limit CPU
   threads and disable JAX/TF bulk preallocation. Use process co-location/MPS
   only within the owned allocation; do not change shared-node MIG settings.
5. Transfer all prepared splits and selected validation proofs. Preserve the
   numerical source identity and configuration hashes. Bind the original DoC
   artifact paths inside the container rather than rewriting configurations or
   stripping the final-run provenance guard. Verify file hashes after transfer.
6. Freeze the selected refit iteration counts. Do not apply wall-clock training
   caps again under GPU contention. Start the complete 65-run manifest only
   after all required groups and port checks pass, retaining independent W&B
   runs and failure records. Final execution orchestration remains to implement.

Shared-GPU runs can supply accuracy results after qualification. Their timing
is operational throughput, not the paper's dedicated-device speed comparison.
Dedicated speed measurements require identical hardware/allocation conditions
for all methods. Environment-port development, selection, ablations and dedicated
timing measurements were excluded from the rough GH200 runtime scenario.

NVIDIA TensorFlow release reference:
https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-24-09.html
(historical release table includes 24.02/TF 2.15.0).
Isambard allocation reference:
https://docs.isambard.ac.uk/user-documentation/guides/slurm-advanced/
