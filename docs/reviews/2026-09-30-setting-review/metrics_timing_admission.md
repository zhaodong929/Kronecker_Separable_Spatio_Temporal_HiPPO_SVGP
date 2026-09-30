# Evaluation, timing, statistical interpretation and admission review

Reviewer: root, independent code inspection and bounded tests, 2026-09-30. Frozen science code4f881c7; no source changes during this review. This report separates demonstrated errors from measurement limitations and hypothetical hardening.

## What the existing evaluation establishes

Gaussian RMSE, observation-NLPD, analytic CRPS and central-interval coverage are implemented consistently in `/data/nk523/projects/hipposvgp-fair-comparison/benchmarks/three_domain/evaluation.py:8`. `restore_score_scale` correctly multiplies RMSE/CRPS by target scale and adds log(scale) to NLPD (`:21`). It leaves coverage/ECE invariant. For COVID, “original scale” is log1p admissions per100000, not raw admission counts (`scripts/evaluate_three_domain_run.py:54`). Metrics in different target units must not be averaged across datasets.

The mixture evaluator uses actual Gaussian decoder mixtures rather than substituting the supplied moment variance as a Gaussian density. The finite-mixture NLPD uses logsumexp and CRPS uses component/pair terms (`benchmarks/three_domain/metrics.py:6`); central calibration uses the mixture CDF/PIT (`:33`). Pooled ECE is computed after pooling coverage, not by averaging per-time absolute errors (`evaluation.py:40`). Equal per-time averaging is appropriate here because the number of queried sites is constant within a run. Changing missingness/query counts would require weighting by actual query count.

Bounded tests under OMP/BLAS/MKL2, CUDA disabled: `test_common_evaluation.py`, `test_mgpvae_mixture_metrics.py`, `test_experiment_tracking.py`, `test_verified_campaign_summary.py`: **12 passed in9.05s**. These validate specified arithmetic/contracts, not end-to-end causal correctness of every model.

## Important limit: “source verified” is not independent validation of every inference computation

`/data/nk523/projects/hipposvgp-fair-comparison/scripts/evaluate_three_domain_run.py:28` compares archived truth, site IDs and timestamps exactly to the common archive and checks complete finite positive observation variances. It also checks delayed-label counts (`:45`). These are useful necessary conditions. They cannot detect actual model-time disagreement if every adapter writes the same archive timestamp array, as independently demonstrated for PEMS.

For MGPVAE, common evaluation reads the worker's `online-metrics.json` (`:35`), rechecks ordering/finite scores and recomputes RMSE from the saved predictive mean. It does **not** independently recompute all mixture NLPD/CRPS/coverage values from the model. Saved latent means/variances, decoder seeds, noise and refit model make that replay possible (`scripts/run_mgpvae_causal.py:253`–261). A selected COVID seed5 independent decoder replay exists from earlier work, but the common status is not proof of an all-run density replay. Report this distinction; require targeted independent density checks and corrected-model qualification, rather than pretending an identical scoring file is independent evidence.

The supervisor's `completed_and_verified` is set from process exit, output shapes and declared counts (`scripts/run_tracked_experiment.py:198`–214). It intentionally retains `main_table_admitted=false`. Qualification binding in that supervisor checks method/dataset/status (`:111`–116), not every source/input/split identity itself. Controllers, provenance manifests and immutable release checks supply additional safeguards. No actual wrong qualification binding was demonstrated here; strengthening binding would be hardening, not grounds to call all outputs invalid.

**Conclusion:** the34 completed/source-aligned/W&B-synchronized records are not34 scientifically admitted results. In particular, the independent OSGPR review found a real parameter-handoff defect despite these checks. Keep completion, protocol agreement, implementation validity, selection adequacy and manuscript admission as separate states.

## Timing: define the same decision-to-prediction boundary

Compare the wall time from receipt of newly available observations to a ready observation-space predictive distribution for that event. Include previous delayed-label assimilation, input feature preparation necessary for that event, representation transport, parameter/posterior update, and prediction; synchronize GPU work at the boundary. Separately report initial model selection across all tried configurations, selected refit/initial posterior cost, compilation, evaluation scoring, logging/checkpointing and recovery/qualification overhead. Scientific validation overhead is not method inference cost, but is part of actual campaign expenditure.

| Path | Current measured scope | Consequence |
|---|---|---|
| Proposal | Synchronized `factor_preparation`, `update`, `prediction` recorded separately; delayed assimilation is inside factor preparation | `update+prediction` alone omits real method work |
| ST-SVGP | Event timer includes protocol access, delayed update, current update and prediction; continuation residual also contains logging/checkpoint overhead | Current243s PEMS figure is online only, not candidate exploration or initialization |
| OSGPR | Event loop times delayed/current assimilation and prediction separately | Includes repeated optimizer work; compare as the chosen adaptive policy, not equal algebraic workload |
| OHSVGP | Event update includes sequential minibatch variational optimization and exporting state; prediction copies results to CPU | Kernel/mean policy and number of optimized steps need explicit disclosure |
| MGPVAE | Event timer includes filter and decoder component generation; metric computation separately timed | Dense mixture scoring is evaluation cost, not filtering cost; current code computes latent predictions once for archival checks and again inside decoder-components helper |

Evidence: proposal `/data/nk523/projects/hipposvgp-fair-comparison/scripts/run_iclr_era5_routeb_strict_online.py:757`–807, `:867`–904, `:1054`–1062; ST `/data/nk523/projects/hipposvgp-fair-comparison/baselines/covid_long_setting_b/adapters/run_st_svgp.py:432`–449; OSGPR `scripts/run_official_bui_osgpr_era5.py:461`–469; OH `scripts/run_covid_ohsvgp_own_theta.py:515`–561; MGP `scripts/run_mgpvae_causal.py:217`–250 and `baselines/mgpvae/filtering.py:66`.

Concrete archived PEMS proposal seed1 example, from `outputs/2026-09-29-doc-campaign/completed-review-inputs.json`: factor preparation2668.005s + update1050.885s + prediction632.158s = **4351.048s** factor/update/prediction subtotal before separately recorded feature work and other overhead, versus1683.043s if only update+prediction is reported. Process total4760.835s includes checkpoint320.779s. This is a reporting-boundary issue; the omitted components are already stored, so corrected totals need not trigger retraining.

ST PEMS seed1 stores online242.565s and continuation_total352.385s. The misleadingly named initial-filter/prefix109.820s is the difference between these timers: it includes initial filtering **and** excluded emit/checkpoint/stacking overhead (`run_st_svgp.py:449`–463, `:696`), not a pure filter measurement. `task1_seconds=1326.922` is the **final selected-configuration refit** on all legal initial sites (`:543`; `scripts/run_st_svgp_pems_campaign.py:98`), not candidate exploration. The earlier conversational description of these22minutes as candidate fitting was incorrect. The complete set of calibration artifacts is required for total tuning cost. Original COVID full-prefix replay timings and newer endpoint-continuation timing294528 represent different implementations; do not mix them in one unlabelled timing column or count the latter as new accuracy replications.

Different parameter policies are not automatically unfair: frozen-parameter Kalman updates and per-event optimizer steps are legitimate method choices if independently selected and disclosed. Their runtime ratio is an end-to-end implementation/policy comparison, not the isolated gain of analytic HiPPO or the Schur–Sylvester solver. OSGPR's old timings additionally come from a defective parameter handoff and cannot serve as final correct-baseline timing evidence.

## Memory: bounded inference state versus actual process memory

The theorem can bound retained sufficient-state size for fixed dimensions and fixed finite geometry count. Dataset arrays, temporal feature caches, all predictions, metric rows and audit archives can still grow with stream length. `scripts/run_mgpvae_causal.py:253` retains all latent/predictive arrays; ST's runner also retains prediction/information rows (`run_st_svgp.py:433`, `:449`). Proposal allocates full prediction grids and maintains checkpoint output arrays. Those are not proofs that the mathematical posterior state grows, but they prevent equating actual runner RSS with bounded inference-state memory.

The supervisor samples worker-tree RSS every30s (`scripts/run_tracked_experiment.py:95`, `:195`), which is not an exact memory peak and includes loader/archive state. W&B/system GPU traces and framework allocated/reserved values also differ: hardware capture via nvidia-smi lists devices (`:131`); device UUID attribution and phase resets are needed. A CUDA allocator reservation, actual tensor allocation, retained mathematical state and whole-process RSS are different quantities. No uniform cross-framework memory comparison has been admitted by this review.

Minimal valid memory reporting: explicit retained-state dimensions/bytes, peak allocated device memory for a declared phase and implementation, and host/process measurements labelled separately. Avoid asserting observed constant process memory solely from bounded-state equations, or claiming hardware-neutral memory ratios from different frameworks' allocator statistics.

## Statistical and selection interpretation

All prediction rows within a field are temporally/spatially dependent. Five ERA/COVID masks and three PEMS masks use the same physical period and overlapping locations. Across-split SD is descriptive split/optimization sensitivity, not an IID sampling SE over independent datasets. Training seed0 for proposal and split-linked training seeds for several baselines should be reported, not silently described as matched optimization-seed replications. Numerical equality of RNG seeds across different model families is not required for fair information access.

`scripts/summarize_verified_campaign.py:38` only aggregates a dataset after all five methods finish every declared split, reports sample SD, and keeps admissionfalse. That is appropriate bookkeeping. Any inferential claim should use paired split differences and account for temporal dependence; the millions of site-time rows are not millions of independent replicates. A daily/weekly paired error summary can be derived from existing predictions without training new models.

The existing run outcomes have already been inspected. A redesigned task must be justified by the paper's intended information/geometry claim, not by which period, reveal rule or feature set makes the proposal win. Freeze and record the revised contract before its comparative results; describe prior runs as development evidence. If a truly untouched confirmation partition is available it is preferable, but lack of one is a stated limitation rather than grounds to invent an unseen-test claim. No new ablation suite or baseline is needed merely to report these limits honestly.

## Admission checklist, not blanket rejection

- Arithmetic/source-alignment tests and exact protocol hashes remain reusable evidence.
- Metric/timing aggregation fixes alone generally require rescore/reaggregation, not retraining.
- A kernel-state handoff change, time-coordinate change used by a model, label-release change, or task information-set change requires affected own-selection/inference reruns unless exact equivalence is independently demonstrated.
- Main-table rows need a common task contract, correct method implementation, adequate predeclared tuning, complete common-scale scores and properly scoped claims. None follows merely from a successful W&B upload.

## Independent measurement follow-up

The independent `measurement_crosscheck.md` identifies additional nonidentical timer boundaries: proposal `feature_seconds` must be added to the factor/update/prediction subtotal; OSGPR/OHSVGP omit query-input and offset preparation between timers. Thus those subtotals are not yet fully comparable end-to-end latency. OHSVGP repeatedly materializes full-stream residuals inside an update, which adds avoidable horizon-dependent preparation without implying future-label leakage. All reported predictive scores concern scalar marginal distributions, not joint field uncertainty. ECE is pooled central-coverage discrepancy, not a proper score. These are measurement/claim qualifications, not additional demonstrated predictive defects.
