# Task-end comparison pipeline

This supersedes observation-by-observation *evaluation* in the old three-domain campaign. The scientific task is reconstruction of hidden locations **after the current task's visible observations arrive**. Observation-wise official Markov updates remain observation-wise. No current hidden target reaches an adapter or a label-derived feature. Historical feature rows are frozen at their original release.

| Dataset | Initial observations | Subsequent task | Evaluation |
|---|---|---|---|
| ERA5-Land | 336 hourly times, 800/1000 sites | 24 hours, final 12 hours | 64 tasks; 200 sites never revealed |
| COVID | 52 weekly times, all 52 sites | 1 week | 143 tasks; 10 held sites released next task |
| PEMS-BAY | 4032 five-minute times, 260/325 sites | 12 samples / 1 hour | 672 tasks; 65 held sites released next task |

Five paired ERA5 masks, five COVID masks and three PEMS masks give **65 main runs** across KronHiPPO-SVGP, Bui OSGPR, spatial OHSVGP, Aalto ST-SVGP and causal MGPVAE. No extra baseline is silently substituted. Ablation code was subsequently explicitly requested by the user and is separate from these 65 runs.

## Data and selection

`prepare_task_stream.py` writes the final and internal-selection `TaskStream` files plus feature tables and manifests. All target/feature scaling uses only the inner fitting time/site subset. Internal validation occupies the second initial week for ERA5/PEMS, or weeks 41–52 for COVID, and uses the final task/release convention. The final held sites do not occur in the internal stream. The final initial posterior uses all legally observed initial data.

ERA5 uses the verified CDS raw reconstruction, not legacy per-site row positions. 1860 UTC hours run from 2020-01-01 00:00 through 2020-03-18 **11:00 inclusive**. 12:00 is the exclusive end. The supplied historical description's 171 ten-hour blocks cannot follow an initial 186 hours within 1860 total hours: the remainder is 1674 hours. Its Task-1 scaler and old block count are not inherited.

The ERA5 weather L10, COVID released L4 and PEMS Road-context L10 families are retained. Full source, exact feature transforms, release rules, dates, paired masks and hashes are in each prepared manifest. ERA5 is retrospective interpolation using contemporaneous reanalysis covariates, not weather forecasting. PEMS's simulated one-hour release delay is a deliberate protocol change from the previous five-minute delay.

`run_task_stream.py` runs a fully specified candidate. Each method learns its own initial parameters using its own objective; GP baselines use a fixed initial ridge mean while the proposed method learns a joint trend/residual posterior. `select_task_configuration.py` chooses query-weighted validation observation NLPD, refuses final/integration scores, and binds the selected configuration to source contents, protocol, features and result provenance. Initial optimizer budget/capacity candidates still require resource and convergence qualification; two-step integration runs are not selected scientific models.

## Inference and corrections

- Kron: fixed geometry uses Schur–Sylvester; differing initial/released geometries use a sum of Kronecker terms. Each old geometry's sufficient statistics are retained with its own spatial factor. Matérn spectral draws now exclusively use the model generator; the former chi-square global RNG dependency broke paired reproducibility.
- OSGPR: official collapsed VFE and adaptive updates. Product-kernel spatial amplitudes are fixed at one, avoiding unidentifiable scales lost during theta serialization. Round-tripping legacy nonunit spatial amplitudes retains the product amplitude.
- OH: pinned multidimensional row-stream HiPPO, explicit initialization grid and microbatches, official stochastic ELBO with correct minibatch scaling. Initial kernel fitting is followed by frozen-kernel updates; the already documented frozen log-amplitude correction remains external to upstream source. MC draws are isolated per run/step. CPU and GPU stochastic optimizers are not required to follow identical random trajectories.
- ST/MGP: task-end official smoothing/temporal conditionals; filtered endpoint continues. For next-task delayed labels, restore the state before the previous task and recompute that bounded interval with visible and newly released labels exactly once. No full-history replay or posterior reset is introduced.
- MGP: explicit existing spatial-covariance pushforward correction, compact diagonal projection and selected-site inference; nontrivial location reordering is tested. Full-size GPU training/prediction remains a separate numerical gate.

## Logging, timing and ablations

The W&B supervisor owns network access and targets `harrisonzhu/KronHiPPO-STGP`. The pipeline writes a durable local journal, exact per-task predictions (including every MGP Gaussian-mixture component), targets/query coordinates, metrics, config/source/data identities, timing, resources, latent marginals and explicit numeric checkpoints. Artifacts are atomically written before a completion event. A failed run leaves its completed tasks and failure record. Checkpoints do **not** yet have an independently qualified automatic restoration path; mid-fit optimizer recovery is unsupported and explicitly recorded.

Online timing covers released-label construction, delayed assimilation, prepared-feature lookup and mean evaluation, update/transfer, task smoothing and observation-distribution preparation with device synchronization. Initial fitting, scoring and serialization are separate. Shared raw-data/feature preparation is recorded separately and is not part of this per-method online timer. Whole-process RSS is distinguished from GPU memory. Actual publication timings require the same device and exclusive execution; qualification jobs are diagnostics.

FP64 add/multiply/FMA counters are an explicitly incomplete arithmetic subset, not total model FLOPs. Missing counts stay `NA`. Profile execution, kernel replay, task coverage and omitted operations are recorded; partial task counts are never multiplied into a claimed whole-run total. Actual profiler acquisition is not yet qualified by these unit tests.

Matched predictive arms are joint transfer, zero trend/residual cross-information (current **and** accumulated historical coupling), and identity transfer at the same changing basis. All reuse the same selected configuration and deterministic fitted parameters/random features; a fit fingerprint verifies the actual common fit. The zero-cross arm does not isolate only historical revision. Dense-versus-structured solver comparison operates on the identical bounded SPD system, reporting residuals; it is a diagnostic solve-only comparison, not automatically a paper-ready throughput claim.

`reporting.py` refuses main-table admission without explicit evidence and matching protocol/seed/hardware pairing. Passing small correctness tests does not imply convergence, GPU scale qualification, or completed main experiments.
