# Reproducible comparison tracking

Project: https://wandb.ai/harrisonzhu/KronHiPPO-STGP

`run_tracked_experiment.py` is a framework-independent supervisor. A separate
DoC `env-tracking` pins wandb 0.25.1, leaving official model environments intact.
The solver writes every supplied event to an append-only JSONL file with no
network calls. W&B receives the most recent point per phase every 30 seconds;
full events, CSV traces and predictions are retained as artifacts, not reduced
to the dashboard sampling. CPU RSS is sampled for the worker process tree.
W&B also records its system metrics; do not confuse supervisor RSS with worker RSS.

Identity: campaign group, dataset, method, spatial split, training seed, stage,
logical_id and distinct attempt ID. `validation`, `qualification`, `final`, and
`integration` are separate job types. An attempt never overwrites an earlier
run. Checkpoint continuation must explicitly link its parent attempt in the spec.
A final run needs a passed method/dataset qualification record. Numerical run
completion does not mean manuscript admission.

Artifacts include exact command, pinned source identity, input SHA256 hashes,
protocol and split JSON, source snapshot, local diff, worker package versions,
Slurm allocation, GPU model/driver, stdout/stderr, all event and CSV records,
selected and last training states, final checkpoint, predictions, and result
validation. Credentials and the whole process environment are never captured.
Raw dataset arrays remain on DoC, identified by checksums and provenance.
Artifacts are uploaded after the child exits, so changing checkpoints are not
uploaded halfway through a write. All training checkpoint files are retained.

Network failure during initialization falls back to offline W&B. Streaming
telemetry failure cannot terminate the child; the complete local journal stays
available. Finalization errors create a sync-required record. Offline W&B runs
can be uploaded with `wandb sync PATH`; a telemetry replay may be needed if the
SDK itself failed during streaming. Do not declare synchronization successful
until the server lists the run and artifacts.

PEMS rerun: `pems_tracked.sbatch RELEASE CAMPAIGN ENTITY` starts independent
Task-1 calibration and then all 50,100 online steps, one exclusive A30 per split,
three array tasks maximum. The common numerical tests run on each node before
training. Artifacts preserve selected and last optimizer states separately.
Online checkpoints are written every 500 steps. Aggregate shape, finite positive
variance and exact delayed observation count are checked before success.

The target remains all 65 comparisons from `plan.json`. This infrastructure and
the PEMS launcher do not imply the remaining adapters/data are qualified. ERA5
long target/split arrays exist in the archived Task 1–10 NPZs; original
covariate feature reconstruction remains unresolved. COVID long source and the
remaining baseline qualifications also remain prerequisites.

Official W&B reference: https://docs.wandb.ai/ref/python/experiments/run/
Offline sync: https://docs.wandb.ai/ja/support/models/articles/can-i-run-wandb-offline
