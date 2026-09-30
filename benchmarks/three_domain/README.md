# Three-domain comparison rerun

Scope: the ERA5, COVID and PEMS main comparisons only. The 65 candidate final
runs in `plan.json` compare KronHiPPO-SVGP, OSGPR, OHSVGP with a spatial
kernel, ST-SVGP and MGPVAE on all three domains (5/5/3 paired splits). Ablations
and mechanism experiments are excluded. A candidate is not a completed or
publication-admitted result.

Use the official repositories at fixed commits; put information-order and data
translation in adapters. Keep historical branches and archived exploratory
results intact. New predictions must go to a new campaign directory.

Implemented fixes: PEMS scaling requires explicit available Task-1 sensor
indices (calibration subset during selection); the protocol exporter records
them. The COVID Gaussian evaluator uses exact quantiles, without Monte Carlo
ranking noise. Existing historical reports are not silently regenerated.

MGPVAE: `baselines/mgpvae` preserves the pinned upstream model and objective,
with explicit spatial-covariance correction and observed-site conditioning.
Delayed hidden observations are assimilated once by restoring the preceding
state and replaying the jointly available observations. Qualification compares
these states with the official legal-prefix reference. Decoder mixture NLPD,
CRPS and coverage use actual samples, with CPU/GPU scoring parity checks.
Five COVID splits have completed, passed independent source-aligned scoring,
and synchronized their W&B artifacts. PEMS/ERA5 full-initial resource gates and
final execution remain pending. Main-table admission is a separate review.

Data readiness: canonical DCRNN PEMS-BAY HDF and geometry are recovered. COVID
uses the recovered CDC admissions series with initial all-site observation and
one-week-delayed hidden labels, explicitly documented against the old snapshot.
ERA5-Land seven-variable January–March 2020 retrieval, all-site full-period
alignment, and five protocol splits verified on DoC on September 30. Six legacy
site series each omitted an hour; the rebuilt inputs restore the common UTC grid.
Do not substitute the public 372-hour series for the required 1,860 hours.
The revised ERA5 comparison uses hourly causal interpolation without hidden-label
release; historical ten-hour-batch results are not directly interchangeable.
Target and weather normalization use only initial fitting sites.

Branch map: the previous `codex/pems-a100-existing-gp` head is the integration
base; its ancestry includes the COVID exploratory archives and ERA5 archived
results. Those are audit evidence, not new results. `codex/fair-three-domain-comparison`
is the active correction/campaign branch. No remote branches are deleted or
force-pushed.

The 2026-09-29 baseline-selection policy is authoritative. No StreamingSGPR,
Kron-STGP, Persistence, IGNNK, LMC/ICM/FSDE or HiPPO-SVGPVAE runs belong to
this campaign. Legacy scripts retain historical identifiers; display names use
KronHiPPO-SVGP. Each external method needs independent legal validation.
Execution uses DoC; Vast.ai is disabled by user instruction. Previous 80/93
run estimates are obsolete. Runtime needs new per-method pilots, especially
ST-SVGP replay and MGPVAE adaptation; count scaling is not an ETA.

`PartialPrefixFilter` adds a reference for observed-site row selection and
historical delayed-label insertion, with full causal-prefix replay. It preserves
upstream mean-field covariance projection. Five focused checks cover official
full-observation parity, dense one-site conditioning, release boundaries, and
mixture metrics. Observed-site training and the common covariate mean are implemented;
per-dataset resource qualification and final admission remain separate gates.

Delayed visible/hidden updates now retain a sum of spatial Gram Kronecker
terms in `multi_geometry.py`. Sylvester-preconditioned CG solves this precision
with explicit residual checks (target 1e-9, accepted true relative residual at
most 1e-8). Unconverged solves fail. This is a multiple-geometry adaptation,
not the paper's single-Sylvester complexity guarantee. Dense joint-Gaussian
parity covers mean, variance, noncommuting geometries and rectangular temporal
transfers; a real-runner perturbation test checks release boundaries. GPU and
full-scale numerical qualification remain necessary before final admission.

The frozen online temporal builder optionally uses SciPy `spherical_jn` with
negative-argument parity, preserving the analytic HiPPO features. Training
retains the differentiable Torch implementation. The original Miller start
order grew with frequency times horizon and caused substantial inference cost.
Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_jn.html

Online checkpoints atomically retain the completed prefix and the pending
one-step-delayed observations. Resume verifies input, configuration and source
fingerprints. Tests require identical uninterrupted/resumed predictions and
reject a changed-input checkpoint. The full PEMS launcher validates a 100-step
prefix first, then resumes the same state through all 50,100 steps, saving every
500. Completion requires finite full-shaped outputs and exactly 50,099 * 65
delayed observations. Completion alone is not final table admission.
