# Task-stream device qualification on DoC

This is an integration gate, not a paper experiment. Nothing in this directory automatically submits a job. The immutable source snapshot, prepared data and all five configurations must be reviewed and deployed first.

`qualify.sbatch` allocates one A30 GPU per method and an array concurrency cap of three. The five workers use the existing isolated Torch, GPflow and JAX environments. CPU reference and GPU runs execute sequentially within their allocation; they never split a GPU. Submit only one qualification array at a time, or explicitly account for other running arrays: `%3` limits this array, not the whole account. Account limits still apply.

Every worker is wrapped by `scripts/run_tracked_experiment.py`, writing to `harrisonzhu/KronHiPPO-STGP`. Its journal and recursive artifacts retain both CPU/GPU runs, checkpoints, task distributions, worker logs, runtime probes and parity errors. W&B unavailability follows the existing supervisor's offline fallback; artifact upload is not proof of numerical qualification.

## Inputs

- Release: exact Git commit with `source.sha256` and accessible pinned official sources.
- Prepared directory: `stream.npz`, `selection-stream.npz`, `features.npz`, `manifest.json`, using `PreparedData` format. Provide a small real protocol for parity first, with at least two evaluation tasks.
- Configuration directory: `kronhippo_svgp.json`, `osgpr.json`, `ohsvgp.json`, `st_svgp.json`, `mgpvae.json`. Use actual valid `Configuration` entries, including a readable pinned MGPVAE `official_source` path. Seeds must be deterministic; this runner preserves the configured seeds and model sizes.
- Existing credential file and environment paths follow the earlier DoC campaign. Credentials are read only and never written to artifacts.

The runner caps initial and online optimization iterations at two and restricts execution to two tasks. It invokes the actual `run_task_stream.py --stage integration --max-tasks 2` in separate CPU and GPU subprocesses. All five methods must expose a real GPU in their own framework; absence is a failure. Mean/variance parity defaults to `rtol=1e-4, atol=1e-6`, with no automatic tolerance relaxation. Initial refitting is included, so a failure can originate in training as well as filtering; inspect trained checkpoints before attributing it.

## Reviewable submission template

Run on the DoC login node after deployment, replacing all placeholders:

```bash
sbatch slurm/task_stream/qualify.sbatch COMMIT PREPARED_TINY CONFIGURATIONS CAMPAIGN-parity parity
```

After reviewing all five results, submit one domain at a time with its full spatial size, full initial period and first two actual tasks:

```bash
sbatch slurm/task_stream/qualify.sbatch COMMIT PREPARED_FULL_DOMAIN CONFIGURATIONS CAMPAIGN-domain-shapes shapes
```

`shapes` skips the CPU reference and records `shape_only_passed`; it must not be cited as CPU/GPU parity or convergence. It still caps fitting at two steps. Large MGPVAE initial compilation or dense baseline geometry may fail here: that is precisely a qualification result, not permission to change the problem silently.

The initial parity probe checks numerical device availability and `hardware.txt` records GPU identity/driver. Compare timings only after a separate controlled benchmark on the same GPU model and allocation policy; these diagnostic job durations and heavily capped fits are not paper runtime estimates. Full-run selection, convergence, paired-mask validation and held-out evaluation remain separate gates. Every output sets `main_table_admitted=false`.

Check job state infrequently (for example every ten minutes). The account queue was empty when checked during preparation on 2026-09-30; recheck before submission. This wrapper does not cancel or modify unrelated jobs.
