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
