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
