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

MGPVAE: `baselines/mgpvae` uses the pinned upstream model, objective and filter
step. Its stateful visible-site filter is checked against upstream filtering
on every prefix. It avoids whole-prefix replay and future smoothing. Delayed
held-out observation updates are not implemented yet; this candidate is NOT
admitted as a full-information COVID/PEMS main-table comparison. Decoder
components are retained for mixture scoring rather than reporting an ELBO as
predictive NLPD. No claim of GPU parity or convergence follows from CPU tests.

Data readiness: canonical DCRNN PEMS-BAY HDF and coordinate/road files recovered.
The exact long ERA5 and COVID protocols referenced by the paper are missing.
Do not substitute ERA5 Task 1–2 for Task 1–10 or older COVID cases for admissions.

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
mixture metrics. Masked training, covariates, and final admission remain open.
