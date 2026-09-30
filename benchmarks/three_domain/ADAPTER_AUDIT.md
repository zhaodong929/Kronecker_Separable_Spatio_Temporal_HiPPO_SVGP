# Official sources and explicit adaptations (2026-09-29)

The target methods are exactly the five in `plan.json`. Qualification and
validation runs are not final comparisons. The submission ledger deliberately
retains the unsubmitted rows.

## Source pins

| Baseline | Official source | Commit/version |
|---|---|---|
| OSGPR | thangbui/streaming_sparse_gp | d95081b8a67316981088172124bc44e3c2235e49 |
| OHSVGP | harrisonzhu508/HIPPOSVGP | a1bff1bc81629316af92c97bde53d2792f4d8025 |
| ST-SVGP | AaltoML/spatio-temporal-GPs | c5b929e1fc07b14ff9671dd1d66b3b8041e2a2ce; BayesNewton 1.1 |
| MGPVAE | harrisonzhu508/MGPVAE | c9a80a05fca66b2911c8e7cb0deccb39d261a5b8 |

External model source checkouts are unchanged. The following are local adapter
changes; they must be disclosed in the experiment description.

## OSGPR

Use official SGPR initialization and OSGPR_VFE posterior transfer. Kernel,
noise and inducing coordinates are optimized using released data only. Honor
the manifest's initial observation set (all 52 COVID sites). Time units are
fixed by the initial time span; predictions retain original timestamps.
Task-1 fitting sites alone train the shared residual mean during validation.
Persist posterior, inducing coordinates, kernel parameters and block traces.

An actual adapter test perturbs hidden labels and future observations: current
predictions are invariant, predictions after release change, and each delayed
observation is counted once. Initial/online iteration budgets still need
validation; the historical 25/5 setting is not declared converged.

## OHSVGP + spatial kernel

The official multivariate RFF construction receives time and spatial inputs.
An adapter subclass corrects the frozen old-covariance amplitude factor:
divide by `exp(log_sf)`, not `log_sf`. This prevents division by zero at unit
amplitude and preserves the old covariance after the upstream multiplication.
Initial minibatches use the unbiased full-data VE-minus-KL scaling. Fixed base
RFF draws retain differentiable lengthscale dependence. Prediction is chunked
to avoid an upstream dense query-query allocation. Lazy Legendre transitions
match the pinned implementation. Dense numerical gradients and actual
current/next-step release-boundary tests pass.

## ST-SVGP

BayesNewton's MarkovVariationalGP is unchanged. Removed JAX APIs are mapped to
their current equivalents in `baselines/bayesnewton_compat.py`. Initial targets
are selected by public site ID, and the shared covariate mean is subtracted
before inference and restored after prediction. Select checkpoints using
Task-1 validation sites. With Gaussian likelihood and all spatial inducing
points, an independent dense GP gives matching means and observation variances
(approximately 1e-8 maximum difference in the qualification fixture).

Online inference rebuilds a posterior from legally arrived history. Timing
includes history construction, inference, prediction and cache release. A
history window is an explicit approximation and must be validated; do not call
this implementation replay-free. Long-horizon PEMS cost remains unqualified.

## MGPVAE

Preserve the official kernel, encoder/decoder and ELBO terms. An explicit
adapter corrects the spatial covariance pushforward in `energy`: for `f=L u`,
use `L Cov(u) L.T`, whereas the pinned source left-multiplies by `L` only.
Independent dense-moment tests verify the correction; the KL term is unchanged.
This is a disclosed correctness fix, not an unmodified-official result.
Also, the pinned kernel's internal transformed temporal parameter names are
interchanged in its accessors. We leave that parameterization unchanged and
log effective `lengthscale_time` and `variance_time` values.

Selected-row filtering never sends unreleased targets to the encoder. The
official identity measurement matrix makes a scalar update per site exactly
equivalent to the dense selected-row reference; tests include delayed releases.
Keep the state before the previous step and recompute that step jointly when
its hidden labels arrive, then process the current visible labels.

Initially unobserved sites are appended after observed sites in the Cholesky
ordering. Their prior extension agrees with the original model's spatial
conditional before release, verified numerically. The parameters are frozen
online. A CPU COVID integration completed initial training, refitting on all
52 sites, and three online steps with the expected 20 delayed labels. This
does not establish convergence or final benchmark readiness.

Observation-space scores use Gaussian decoder mixtures, not a Gaussian
moment approximation. Store latent moments, decoder checkpoint, sample count
and per-step random seed to reconstruct the mixture. Validation records
128/256/512-sample sensitivity before selecting the final sampling budget.

## Remaining admission gates

- Method/dataset-specific capacity, training/online budgets and convergence.
- Real A30 timing/memory, including middle/end history for ST-SVGP.
- MGPVAE Monte Carlo stability and full-run resource measurements.
- ERA5's long labels/splits exist; the long multivariate covariate source is
  missing. Archived means trained on validation sites cannot replace it in a
  clean comparison. Do not drop covariates silently.
- Final cross-method metrics, independent time/site/scale checks, and revised
  manuscript wording for the proposal's multi-geometry solver adaptation.

## Gaussian ST-SVGP continuation qualification

The optional `--online-backend stateful` adapter retains the official kernel,
likelihood, inducing locations and frozen parameters. Its Gaussian update is
qualified against unit-natural-gradient BayesNewton prefix replay, including
one-step delayed labels. Spatial residual covariance enters prediction, not
site precision. The replay backend remains the default and remains in the
already submitted COVID release.

A real COVID seed5 three-step integration, using the same trained checkpoint,
agreed with replay to maximum absolute mean error 1.30e-7 and variance error
3.37e-9. Small synthetic sparse comparisons also pass. These are correctness
checks, not final benchmark measurements. Stateful timing separately records
initial filtering/prefix cost and subsequent update/prediction cost. PEMS GPU
qualification and long-run numerical stability remain pending.

## Interpretation of the mean function

The proposal jointly updates its Bayesian linear coefficients and GP state
online. Current external adapters use an initial-data ridge mean and freeze
that mean online. They share the covariates and legal observation boundaries,
but their mean-learning dynamics are not identical. Accordingly this is an
end-to-end comparison of the stated models/adaptations; accuracy differences
cannot be attributed solely to the GP inference algorithm or solver. This
limitation must appear in the manuscript. No additional ablation is scheduled
and no running configuration is silently changed to remove the distinction.


## OHSVGP final qualification pipeline

The external adapters retain the same ridge mean fitted on Task-1 fit sites;
OHSVGP no longer refits that mean on all initial sites while the other external
methods freeze it. All legally initial sites still enter its GP residual refit.
The first optimization iteration is included in validation checkpoints.

Each COVID split selects M=32/64 using its own spatial validation, checking
500 iterations and extending to 1000 when the best checkpoint is at the endpoint.
A separate chronological 40/12 split of the formal initial 52 weeks selects
online steps from 1/5/20, extending to 80 if 20 is best. Material improvement
at that final endpoint raises a qualification failure requiring review.
This internal fold rebuilds normalization and lag features from its training
prefix and never reads the formal 143-week stream. Tests independently perturb
the complete formal stream and future internal labels to check this boundary.
The formal test remains separate; none of these gates establish main-table
admission. GPU tests run again before candidate training and final execution.

OSGPR final qualification searches Cartesian initial grids (Mt, Ms) in
{2,4} x {16,32}, with official adaptive pseudo-input optimization. It compares
100/400 initial steps per block, extends to 800 if NLPD improves by over 0.01,
and requests review if the selected configuration still materially improves
at that endpoint. Online budgets 5/20/80 use the same internal 40/12 temporal
fold; an improving endpoint also blocks final evaluation. These thresholds
are declared before viewing the final stream, and are validation-budget gates,
not a proof of global optimization convergence. Previous 25/5 settings are
not automatically treated as sufficient.

For long MGPVAE streams, cache the official stationary spatial projection for
one query grid while parameters remain frozen. This avoids repeated inducing
covariance factorizations without caching the evolving posterior prediction.
Tests compare the cached prediction after a later state update to a fresh
official spatial conditional, in addition to the official-prefix checks.
Already submitted COVID jobs retain their immutable earlier release.

The PEMS ST-SVGP launcher first runs GPU correctness checks, then fits one
initial iteration on the real full 2016-step initial protocol and compares
three online steps of official replay with continuation using the same frozen
checkpoint. A failed equivalence or resource check prevents final evaluation.
Only after that check does it select 16/32 spatial inducing points using
Task-1 validation (500 iterations, 1000 when endpoint-best), refit and execute
all 50,100 steps. This is a queued qualification-and-final pipeline, not a
claim that PEMS GPU qualification or convergence has already succeeded.
Continuation writes atomic state/prediction checkpoints every 500 steps,
including the state before the preceding update needed for delayed labels.
Automatic resume from those checkpoints is not yet implemented.

MGPVAE's repeated checkpoint selection now computes only the RMSE/NLPD used
for model choice and Monte Carlo sensitivity. It avoids the quadratic-in-sample
CRPS calculation at every validation checkpoint; the selection NLPD is exactly
unchanged (tested). Final evaluation still computes full mixture CRPS and CDF
coverage. The PEMS pipeline first exercises the entire 2016-step initial window,
refit, and three legal online updates on its allocated GPU before candidate
training. GPU memory/runtime feasibility is an empirical gate, not assumed.

The stored PEMS protocol contains only a 32-point inducing grid. For a missing
candidate size, the traffic adapter derives the same deterministic farthest
point design used by the exporter from visible-site coordinates only; no
labels enter grid construction. Existing stored grids remain unchanged.

PEMS OSGPR/OHSVGP online-budget selection uses a separate day-1–6/day-7 fold
within Task 1: all 260 originally available sensors initialize the model,
234 are visible during the internal stream and the original 26 validation
sensors become delayed query sites. Original 65 final held-out targets never
enter this fold. Target and dynamic-feature normalization are refitted on the
shorter prefix and its 234 fitting sensors. Stored affine graph-context features
are renormalized without changing their original legal context sensor set.
Perturbation tests cover original held-out labels, formal stream, and future
internal labels/features. A real PEMS OHSVGP three-step CPU integration passed
with all 260 initial sites and exactly 52 delayed rows; it is not a convergence
or timing measurement.

For PEMS OSGPR only, initial batches contain 256 time slices instead of 10,
reducing repeated optimizer/model construction while keeping the official loss
and all initial observations. This initial batching is recorded and used in
its own validation. Online blocks remain one real time step. PEMS jobs request
24h (the A30 partition allows 72h) and lower scheduling priority. Dispatch waits
until both shorter COVID baseline arrays have actual submission receipts so
long PEMS jobs cannot fill the submission quota ahead of them.

## Common post-run evaluation

`evaluate_three_domain_run.py` independently matches truth, site order and
original timestamps to the complete source protocol before scoring. Gaussian
CRPS and central coverage are analytic; all methods use ten coverage levels
0.05, 0.15, ..., 0.95. For MGPVAE it pools the saved actual decoder-mixture
scores and coverage, never computes a Gaussian density from moment variances.
Scores are restored from per-split standardized units: RMSE/CRPS multiply by
target scale, NLPD adds log(scale), coverage is invariant. This prevents split
normalization differences from distorting cross-split summaries. Independent
Gaussian/one-component-mixture and affine-restoration tests pass. Source and
prediction hashes and evaluator hashes are saved; successful scoring does not
approve the implementation or manuscript claim.

A completed COVID seed5 reference was subsequently checked over all 143 online
weeks against continuation using the same trained state: maximum absolute mean
error 1.822e-7 and variance error 1.554e-9. This CPU correctness comparison is
not a GPU speed claim. A separate exclusive-A30 measurement job repeats this
full comparison for all five completed reference splits. It records parent
initial-fit cost, new initial-filter cost and online continuation time separately.
It is the same replicate, not five additional independent experiments. Use
these qualified continuation timings for a consistent implementation comparison
with PEMS; retain original replay timings as reference costs.

## Qualification device clarification

Earlier OHSVGP/OSGPR release-boundary tests explicitly requested CPU even when
launched on an allocated GPU node. They verified causality with the actual
models, and the candidate training used the GPU, but those boundary tests were
not themselves GPU executions. Before any OHSVGP/OSGPR final runs were submitted,
the tests were made device-selectable and their Slurm launchers now explicitly
set `HIPPO_TEST_DEVICE=cuda`. Local CPU tests retain the CPU default. The
qualification record states the actual requested test device. ST-SVGP/MGPVAE
JAX tests already selected the allocated GPU backend.

PEMS MGPVAE now uses an optional JAX implementation of the same exact
finite-mixture CRPS/NLPD/PIT formulas, keeping all 512 decoder samples. CPU
SciPy remains the independent reference. Tests with 1/17/64 components pass;
a real-sized 512-component/65-query check also agrees to 1e-10 tolerance.
On the local two-thread CPU qualification, SciPy scored that batch in 5.107s
and warmed JAX in 0.0775s; this is not an A30 benchmark. GPU checks repeat before
PEMS execution. Coverage means explicitly reduce in float64 (JAX's boolean
mean otherwise defaults to float32). Scoring time is logged separately from
model update/prediction, and the score call synchronizes before the next step.

PEMS Bui OSGPR initial optimization optionally wraps the unchanged official loss
and Adam steps in a TensorFlow graph (no XLA). Two CPU tests compare SGPR and
official OSGPR eager/graph parameters and predictions to 1e-9 after eight
steps; both passed in 36.85s on corgi. The PEMS launch repeats these on CUDA
before training. Online optimization remains eager. No GPU speedup is claimed
before measurement. Official source files remain unchanged.

OSGPR COVID job 294345 failed before calibration: actual CUDA boundary tests
exposed TensorFlow libdevice lookup failure (a non-UTF8 diagnostic initially
masked it). The remaining array was cancelled; no final results admitted.
The campaign now explicitly points XLA to the installed NVIDIA cuda_nvcc
libdevice directory and decodes subprocess error logs with replacement so
future diagnostics remain readable. Model/objective/data are unchanged.
GPU boundary and graph/eager parity gates must pass on the replacement job.

## OHSVGP validation-budget extension

COVID job 294351 seed5/6 correctly stopped before final evaluation because
Task-1 chronological validation NLPD continued to improve at 80 updates:
seed5 20→80: 0.86112→0.75418; seed6: 0.76570→0.62477. Extend the same optimizer
search to 320, then 1280 only while the endpoint materially improves (>0.01).
A still-improving 1280 endpoint remains a failed qualification, not admission.
Selection never accesses the formal stream.

Successful earlier validation trials can be reused only with identical complete
configuration, input hashes, numerical dependency source hashes, an unchanged
result artifact hash, and a finished team W&B run with an artifact. Reuse records
retain the original source and run URL. No final run is reused by this mechanism.
This path was checked against the actual seed5/u80 artifact; a changed update
budget was correctly rejected. Numerical model/official source stays unchanged.

## ERA5 source recovery and no-release protocol (2026-09-30)

CDS authentication and a two-hour seven-variable NetCDF retrieval succeeded.
Full-window retrieval and all-site source alignment are required before submission.
The recovered run retains the 1,000-site spatial splits and 186/1,674-hour window.
All methods use hourly current/past visible-site conditioning; the 200 hidden
sites are never released. This explicitly replaces historical 10-hour batch
conditioning, and is a revised causal comparison, not an exact reproduction of
those historical predictions. The manuscript protocol and timing must reflect it.

The common 133-column feature family comprises seven coordinate/time columns and
six weather variables with current, ten lags and ten differences. Target and
weather normalization use initial fitting sites only; time and lag continuity
are preserved across historical task boundaries. All external methods use the
same initial-fit ridge mean. The proposal updates its Bayesian mean online, so
this remains an end-to-end model comparison rather than an isolated solver claim.

`ERA5Protocol` never exposes hidden stream labels. Actual proposal and OHSVGP
no-release adapter tests pass on CPU after changing all hidden stream labels and
future visible labels. The Bui counterpart is included and must pass in its
isolated TensorFlow environment; ST continuation must match the official prefix
reference without delayed labels. Each method repeats appropriate qualification
on its allocated GPU. MGPVAE and ST also run full-initial real-data resource/parity
gates. No ERA5 comparison is admitted by merely passing a protocol unit test.

Existing COVID/PEMS releases remain immutable. A stale traffic test fixture used
`_metadata` instead of the production `metadata` attribute; the fixture is corrected.

### Long-stream optimizer qualification

COVID OHSVGP Task-1 validation selected 1,280 Adam steps at lr=0.001 for seeds
5/6 (about 19.5 seconds per delayed-plus-visible update on A30). Blindly applying
this optimizer setting to 50,100 PEMS steps would take over eleven days. PEMS and
ERA5 therefore select the online learning rate from 0.001/0.01/0.05 jointly with
update count using only the initial chronological fold. Initial calibration
learning rate is unchanged. The official Adam objective and variational parameters
are retained. Non-COVID updates use an online batch limit of 1,024, keeping each
current site's set together (260 or 800 sites) instead of splitting it at the
unrelated initial-calibration minibatch size 256. This batch choice is used in
both validation and final evaluation, and recorded in the spec. Actual no-release
OHSVGP mutation tests also pass with lr=0.05 and batch limit 1,024.

PEMS/ERA5 Bui online updates can use the already qualified TensorFlow graph
optimizer. This is the same official loss and Adam sequence as eager execution;
eager/graph SGPR and OSGPR parameter/prediction parity passed on CPU and CUDA.
PEMS OSGPR/OHSVGP allocation time is increased to the partition's 72-hour maximum;
this is an upper allocation bound, not an assertion that every selected run fits.
Existing submitted COVID releases and recorded timing remain unchanged.

### Separate spatial split and training seeds

The proposal's completed COVID/PEMS calibration uses the batch runner's explicit
default `model_seed=0`, and the online temporal builder also fixes its RFF seed
to 0. Those numerical runs agree in both phases. Their original top-level W&B
spec called the spatial split seed `training_seed`; the lower-level saved args
and source retain the actual value. Treat this as a provenance-field correction,
not a numerical rerun. ERA5 proposal calibration now explicitly passes
`--model-seed 0` and declares `training_seed=0`, matching its online builder.
All ledger rows record `training_seed_actual` separately from `split_seed`.
External methods use their explicit `--seed`, equal to the recorded split seed.

### ERA5 legacy time-index defect found during full-site source checks

The January CDS source contains all 744 physical hours and the seven requested
variables. The public 372-row files for two sites omit one physical hour:
(50.3,1.8) drops 2020-01-13 05:00 UTC (index 293); (52.8,1.0) drops 03:00 UTC
(index 291). Their subsequent synthetic time indices are one hour behind their
actual values. With that single-row omission accounted for, **all seven variables
match bit-for-bit** at these sites over the public prefix. The official processing
notebook shows per-location `dropna()` followed by synthetic sequence indices,
consistent with this observed defect; do not assume equal row numbers imply equal
physical timestamps.

The rerun uses the canonical CDS UTC grid, restoring those hours rather than
shifting future weather/targets into the present. The reference-alignment gate
may diagnose a single omitted source hour, but must still meet the original
strict tolerances and reject unexplained differences. The mapping is used only
for legacy provenance checks, never to alter the new target/covariate time grid.
All-site full-period alignment remains pending February/March retrieval. The
preparation code and hashes are snapshotted alongside the data-verification report.

### 2026-09-30: OSGPR initial-budget recovery
COVID seed5 job294350 failed the qualification gate before any final stream because the selected initial configuration still improved at 800 updates. Array294359 (seeds6–9) was cancelled to replace slow eager execution consistently. Official SGPR/OSGPR losses, parameters and Adam remain unchanged; the already CPU/CUDA parity-qualified graph mode is now used for both initial and online optimization on all domains. Candidate endpoints that remain globally selected and improve materially are extended 800→1600→3200→6400. Worsening extension resolves the earlier endpoint without forcing a worse setting; an improving selected 6400 endpoint still blocks final evaluation. Chronological initial-fold online budgets similarly extend 80→320→1280 only when the selected endpoint improves materially. Selection uses only Task-1 validation, never final labels. All previous validation W&B artifacts remain preserved; new attempts repeat selection with fresh provenance.

### 2026-09-30: full-PEMS memory qualification failures
MGPVAE job294374 failed before final evaluation: the official differentiable scan retained 36.65 GiB of buffers for initial fitting (2016 times, 234 fitting sites, 2 latents), exceeding the A30. Pending array294383 was cancelled. An explicit `--rematerialize-scans` adapter wraps the unmodified official scan bodies in JAX checkpointing: identical float64 objective, data, stochastic draws and gradients; backward computation recomputes temporary buffers. CPU and per-job GPU tests compare actual corrected official energy and every parameter gradient. Both latent capacities 2 and 4 must pass full-initial fit/refit/three-step resource pilots before capacity selection.
ST-SVGP job294379 passed initial fit but ran out of memory during full-prefix replay with the fitting model still live. Release the unused fitting model and compilation caches after copying frozen parameters to host, before online inference. No model, objective, labels or precision changes. Pending PEMS tail294387 and ERA5 head294386 were cancelled for this fix; remaining unsubmitted ST/MGP dispatchers are held until replacement releases are verified.

### 2026-09-30: OSGPR graph reuse for long streams
The first PEMS chronological validation measured approximately 7.8–7.9 s per time step at only five Adam updates: two newly traced graphs per time (delayed hidden, current visible). Extrapolation to 50,100 steps exceeds 108 hours and the 72-hour allocation. Jobs294381/294382 were cancelled before final evaluation. Reuse the pinned official OSGPR model graph by observation shape, assign legal X/Y and old posterior variables, copy current trainable parameters, and reset every Adam slot and iteration counter to its fresh initial value before each update. Copy optimized parameters back to the current official model; losses, gradients, assimilation order and posterior calculations remain unchanged. Cache is local to one runner process. CPU comparisons across alternating observation counts and changing prior states match all parameters and predictions at rtol1e-9/atol1e-10; graph tracing stays bounded. GPU versions of those checks and the actual causal hidden-label release test are mandatory in replacement jobs.

### 2026-09-30: ERA5 proposal initial-budget recovery
Job294385 passed GPU tests and completed all twelve initial validation trials, but capacity(64,64) remained at its best endpoint at 1000 steps. The same qualification standard as baselines blocks final evaluation. Extend capacity fitting through 2000/4000/8000 only when its best iteration remains at the budget boundary; eight-validation-check patience is unchanged. Seed0 earlier validation results are reusable only with identical configuration, protocol hashes, transitive numerical source hashes, result artifact hashes and a finished W&B run with artifact. Other splits fit their own validation. No final result reuse. Array294391 was cancelled before repeating the known budget limitation; all proposal ERA5 split dispatchers are replaced.

### 2026-09-30: MGPVAE initial-time validation marginals
ERA5 job294389 passed rematerialized initial training but official `predict(t,t,...)` then materialized a 5.75 GiB full-state smoother tensor twice, exhausting the A30. For validation exactly at initial observation times, use the same official `update_posterior` function and spatial conditional B,C; under the pinned identity measurement, the spatial latent covariance is diagonal. Contract B against its mean and B-squared against its variance, adding the diagonal of C. This removes full-state temporal interpolation and preserves the intended initial-time predictive moments. The adapter fails closed if the pinned identity measurement changes. Tests compare all initial means/variances against untouched `model.predict` before and after parameter changes. Real full-spatial-size, three-time-prefix reference/direct parity is an additional GPU resource-pilot gate for both latent capacities. Completed COVID uses its original reference validation path.


### 2026-09-30 ST-SVGP endpoint qualification recovery
Official BayesNewton temporal_conditional applies 1e-8 bridge jitter even at an observed endpoint. Reproduce this with the last two filtered states and official RTS/interpolation, preserving filtered states for future updates. Use official collapsed pseudo-site precision with its 1e-12 regularizer. Four CPU parity tests (with/without delayed releases and collapsed precision) passed. Real ERA5 seed0 three-step predictions against untouched official GPU replay (294395) match mean max 4.44e-15 and variance max 4.97e-15; original rtol1e-5/atol1e-6 unchanged. Evidence: DoC qualification-work/st-svgp-endpoint-era5-check/reference-parity.json. PEMS full-history reference alone runs on allocated CPU to avoid its GPU spatial-conditional memory peak; fitting, stateful qualification and final remain GPU. Initial validation extends through 8000 only if the best checkpoint remains at the boundary; admission gate retained. Earlier GPU replay OOM/parity failures are preserved.


### 2026-09-30 compact official MGPVAE interpolation / OHSVGP initial-budget recovery
The first direct MGPVAE initial-marginal candidate was held before final execution: official predict applies a 1e-8 temporal bridge regularizer even at observed times. The adapter now invokes pinned official RTS and temporal_conditional on each independent 2x2 spatial block, then the same spatial conditional. This preserves the bridge without allocating T x latent x (2*space)^2 matrices. Two CPU tests, including dt reduced 200-fold to expose endpoint effects and parameter changes, pass against complete official prediction (rtol1e-7,atol1e-8). Full-geometry GPU prefix parity and both latent-capacity resource pilots remain mandatory. Old pending 294405 cancelled; no result from that held candidate is admitted.
OHSVGP ERA5 seed0 (294392) reached its 1000-step initial-validation boundary. Extend budgets 500,1000,2000,4000,8000 only while validation selects the endpoint; never waive convergence. Prior successful calibration may be reused only after matching actual transitive numerical files, configuration, input hashes, result hash, terminal state and finished W&B artifact. The dependency list now names protocol/archive/ERA5 and imported official-helper scripts explicitly rather than unrelated ST/MGP adapters. Original active OHSVGP candidates continue to a terminal state; replacements wait for their monitor terminal and reuse verified completed work. Unstarted old ERA5 elements 294400_2 / 294403_4 cancelled, unsubmitted old PEMS tail stopped.

Validation follow-up (05:05 UTC): the compact MGPVAE initial bridge also passes the real ERA5 seed0 geometry on DoC CPU (720 fit sites, 80 query sites, three initial times): mean max difference 1.0095e-10, variance 1.0557e-9, unchanged rtol1e-6/atol1e-7. Evidence: qualification-work/mgpvae-initial-bridge-era5-cpu.json. Five combined compact-marginal, rematerialized-gradient and spatial-covariance tests passed in 34.28 seconds. GPU gates remain mandatory. Verified OHSVGP reuse of actual ERA5 seed0 b1000 succeeds; deliberately changed budget is rejected. Execution release ebab2b1a298e6502041b9f3f9b8606e2b931a146 is pushed; all replacement dispatch units registered, OHSVGP active-seed replacements wait for old terminal records.


### 2026-09-30 OHSVGP prediction basis reuse (05:16 UTC)
Long-domain validation repeatedly invokes official pred_f in bounded 512-query chunks. Each chunk recomputed the same deterministic HiPPO get_Z twice. A separate execution wrapper now memoizes only get_Z within one prediction call, retains the untouched official pred_f/matrix operations, and restores the method in finally. No cache survives a call, parameter change, or exception; fitting, gradients, RNG draws and all reference-worker source files are unchanged. This also makes previous immutable calibration reuse valid after the existing config/input/source/artifact checks; its historical runtime remains historical, not attributed to the faster wrapper.
Three CPU tests passed (22.46s): bitwise chunked prediction parity across lengthscale changes, cache invalidation/exception restoration, and the actual current-hidden/future-label information-boundary tests for both execution entrypoints. Real trained PEMS seed1 M32/RFF256/grid1024, 4096 validation queries on DoC CPU two threads: reference 5.413s, cached 0.366s (~14.8x), outputs bitwise identical. This is a prediction-only CPU measurement, not an end-to-end GPU speedup claim. Evidence: qualification-work/oh-prediction-cache-real.json. Each full GPU job must pass the same cache parity test before validation/final. Old budget recovery units were stopped before any submission (local and remote receipts absent), replaced with unique basis recovery names.


### 2026-09-30 MGPVAE compact training recovery (05:54 UTC)
PEMS head294414 passed all18 GPU tests in57.37s, real-prefix initial prediction parity (mean6.94e-17,variance2.50e-16), and latent2 full-initial fit/refit + three online steps (121.53s supervisor). Latent4 failed its first training step: XLA temporary21.02GiB, including multiple [2016,4,234,234] covariance-gradient buffers. It did not enter main evaluation. ERA5 head294415 was cancelled to avoid repeating the same dense-gradient peak; tails remained gated/unsubmitted.
New long-domain option uses the exact identity diag(L diag(v) L.T)=(L**2)v for the already-corrected spatial covariance pushforward. Pinned measurement identity is checked at model construction; official filter/smoother, pseudo likelihood, KL, draws and precision remain unchanged. Keep dense corrected_energy as reference. CPU tests for latent2 and4 at PEMS-sized time intervals, before/after parameter changes, verify posterior cross-site covariance is exactly diagonal, loss1e-10 and every parameter gradient1e-9/1e-10 against the dense correction (2 passed,71.40s). Initial fitting/validation executables are also released before all-site refitting. Full GPU parity and both capacity resource pilots remain required. Completed COVID uses its unchanged dense objective.
ST-SVGP PEMS294406 has now passed full real-initial CPU reference vs GPU continuation: mean max3.43e-9, variance4.16e-10 at unchanged tolerances, after9 GPU tests (157.83s), and entered initial validation. OHSVGP cached-prediction ERA5 seed2 job294418 passed15 GPU tests (49.74s); first500-step calibration completed in130.19s. Its reference worker/objective remains unchanged.


### 2026-09-30 predeclared convergence gate consistency
ST/OHSVGP workers already implement a predeclared objective/ELBO plateau gate: relative change below0.001 between two ten-check moving-median windows, after the configured minimum iterations. Their campaign controller previously ignored a positive plateau status if the validation argmin happened to be the last checkpoint. Honor the existing convergence status at that boundary; exhausting the budget without a positive plateau still requires extension. No threshold, objective, validation labels or worker numerical code changed. Four decision tests cover positive plateau at the endpoint, unresolved endpoints, interior validation selection and invalid checkpoint indices. PEMS ST seed1 b500 actually remains unconverged (relative change0.001854), so its currently running b1000 extension is appropriate and continues. ERA5 OH seed2 b4000 also remains unconverged and continues to b8000.
Add strictly verified ST calibration reuse to preserve expensive completed work if a controller recovery is needed. Actual PEMS seed1 b500 (294406) passed numerical-source/config/input/result-hash and finished-W&B-artifact checks; its unconverged status is preserved. Evidence: qualification-work/st-svgp-validation-reuse-checked. Initial reference-prediction/GPU tests still run on any replacement. Unsubmitted ST/OH dispatchers are moved to the corrected controller; healthy in-flight calibration continues.

### OHSVGP initial-budget extension after the full 8000-step check

ERA5 seed2 job294418 completed all M32 budgets through8000, but the validation optimum remained at8000 (NLPD0.18078537995132593) and the existing ELBO plateau statistic was0.04439746486227815, above0.001. It correctly failed the gate; these are not final results. Extend the same conditional doubling schedule to16000/32000 without changing loss, optimizer, precision, stopping statistic, data split or capacity. Completed calibrations remain immutable and are eligible for strict source/input/config/artifact-verified reuse. New budgets rerun training from the same seed; no unsupported partial optimizer/RNG resume. A failure at the new ceiling still prevents final submission. Earlier resolved candidates need not extend.

### MGPVAE independent-site official training filter

Pinned H is exactly I_sites ⊗ [1,0], stationary/transition matrices are already site blocks, and pseudo-observation noise is diagonal. Apply the unmodified official temporal `kalman_filter` independently per site, then sum log likelihoods; keep official process noise, Cholesky solves, RTS, objective, float64, random decoder draws and spatial Lss mixing. The factory checks identity H and the adapter rejects masks/parallel mode outside this qualification. Official checkout remains untouched; completed COVID defaults unchanged.

Full posterior, objective and every gradient match the dense reference in 3 CPU tests (65.77 seconds), latent2/4, two parameter states, repeated/tiny/irregular time steps. Full-spatial real PEMS Task-1 prefix: 234 sites ×64 times, loss max error3.64e-12 and gradient max error2.71e-11, unchanged tolerances1e-10 for loss and rtol1e-9/atol1e-10 for gradients. Warm CPU loss+gradient2 latent:7.652→0.658 seconds;4 latent:15.790→1.232 seconds. These are CPU prefix timings, not GPU full-run speed claims. Evidence `outputs/mgpvae-sitewise-full-spatial-cpu-qualification.json`. New long-domain campaigns require full initial-period GPU loss/all-gradient parity for latent2/4, plus existing tests and resource/causality pilots, before calibration or final evaluation. Healthy old compact PEMS partial calibration is preserved as superseded work; no reuse of incomplete optimizer state.

### CPU execution investigated, not substituted into the GPU campaign

DoC corgi, two CPU threads, unchanged6a952dc OSGPR worker: full PEMS seed1 initial period then first20 stream steps at selected mt2/ms16,100 initial steps and20 online steps. RMSE/NLPD/coverage/std match the ongoing GPU294411 prefix with largest metric discrepancy2.54e-10. Median update/prediction CPU0.5006/0.0784 seconds versus GPU1.3889/0.1087. However learned hyperparameters exceed the strict rtol1e-9/atol1e-10 comparison (largest violating absolute difference1.50e-9). Do not describe the complete device-parity gate as passed or loosen it post hoc. Keep this a diagnostic and continue the formal GPU jobs. Also Slurm --test-only --gpus=0 on a30 explicitly assigns1GPU; it does not establish an additional CPU-only concurrency lane. No production hardware change is made.

### Full geometry GPU qualification and latest ST timing confirmed

MGPVAE release6135b49: PEMS234×2016 and ERA5 720×186 initial rectangles both pass dense-vs-sitewise loss and all-gradient checks, latent2/4, at unchanged tolerances. PEMS GPU warm loss+gradient12.5268→3.21975 seconds and15.2971→3.97371 seconds; ERA5 5.49144→0.832288 and8.53337→1.43704 seconds. Largest full geometry loss discrepancy1.05e-9, gradient discrepancy2.50e-9 (relative-plus-absolute checks pass all elements). These are training-step timings, not final-stream timing. Jobs294435 and294459 retain W&B full-initial-sitewise-parity artifacts.

Latest endpoint-aware ST continuation release807cdd1, job294528, compares all143 COVID steps for all5 frozen official fits at the original rtol1e-5/atol1e-6. Largest mean difference1.49e-7, variance7.59e-9. Exclusive A30 online times2.049–2.409 seconds per full split, initial filtering5.779–6.218 seconds, plus original recorded Task-1 fit time. Use these latest endpoint-aware measurements rather than the historical approximately1.24-second direct-endpoint measurement. Evidence `outputs/st-svgp-endpoint-continuation-20260930/measurements.json`; qualification only, not an extra replicate.
