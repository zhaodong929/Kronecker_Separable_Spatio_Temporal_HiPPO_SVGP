# Independent OSGPR/OHSVGP review across ERA5, COVID and PEMS — 2026-09-30

This report audits the two external streaming baselines as methods, across all three settings. Code is the fair-comparison worktree `/data/nk523/projects/hipposvgp-fair-comparison`; absolute source paths below refer to that worktree unless explicitly remote. Original official repositories are locally clean at Bui `d95081b8a67316981088172124bc44e3c2235e49` and HIPPOSVGP `a1bff1bc81629316af92c97bde53d2792f4d8025` (git HEAD/status inspected). Changes are in adapters, not those official working trees. The stale general manifest `baselines/external/manifest.json` says several integrations are pending; it is not a final campaign specification.

Read-only source/metadata audit, plus a tiny CPU-only OSGPR reproduction on corgi using the exact active release `6a952dc2a1012592c0b74d1f9b01d9a1c8c898dc`; CUDA disabled, Python bytecode disabled, BLAS/OpenMP/TensorFlow threads capped at2. No jobs or source changed. Local OHSVGP tests were attempted but system Python lacks pytest; test implementations were inspected, not reported as newly passed.

## Main assessment

Both adapters use recognizable pinned official model cores and legal delayed-observation ordering. OSGPR performs causal kernel/noise/inducing optimization; OHSVGP learns its own initial kernel/noise, then freezes them and optimizes its variational state online. These different update policies are not intrinsically unfair, and equating their Adam iteration counts would not make them fair. Their common information boundaries, complete optimization costs, validated choices and faithful continuation matter.

A newly reproduced **major OSGPR continuation defect affects adaptive runs in all three datasets**: learned spatial kernel amplitudes are discarded at every model reconstruction. This requires repairing and rerunning affected OSGPR validation/finals before scientific admission. It does not establish that the official Bui equations are wrong or that all existing numerical predictions are undefined. OHSVGP has no equivalent demonstrated amplitude-loss defect; its explicit upstream amplitude correction is mathematically justified by the source and targeted tests. Significant open questions remain about OHSVGP initialization budget, integration-grid approximation, capacity adequacy, source-complete reuse and timing comparability.

## Method and domain matrix

| Item | OSGPR | OHSVGP + spatial kernel |
|---|---|---|
| Official core | Bui GPflow `OSGPR_VFE`; first prefix block uses GPflow SGPR | HIPPOSVGP `hipposvgp.multidim.HIPPOOSVGP`, explicit subclass correction |
| Inputs | scalar time + standardized latitude/longitude | same three coordinates, sorted time/latitude/longitude |
| Covariance family | Product of three 1D Matérn3/2 kernels | Three-dimensional ARD squared exponential; factorizes into time and spatial RBF factors |
| Inducing meaning | Cartesian product only at initialization, then all joint pseudo-input coordinates learned; M=mt×ms | One M-dimensional HiPPO interdomain state, no separate spatial-inducing Ms |
| Candidate sizes | mt∈{2,4}, ms∈{16,32}, total M∈{32,64,128} | M∈{32,64}, 256 sampled frequencies (sin/cos representation), initial basis grid capped1024 rows |
| Initial fitting | All observations of each initial block; full-batch Adam; posterior carried into next initial block | Random batches of256 observations; joint kernel/noise/variational calibration; select held-out-site checkpoint |
| Initial final posterior | Adaptive mode automatically carries prefix posterior, using all legal initial sites | New variational model refitted on all legal initial-site population with selected kernel/noise frozen |
| Online updates | Kernel, likelihood and pseudo-inputs via Adam; analytically collapsed variational posterior | Frozen kernel/noise/RFF; optimize whitened mean/covariance via Adam, transport official old posterior correction |
| Mean | Shared initial fit-site ridge offset frozen online | Same fit-site ridge offset frozen online |
| Causal history | Old inducing posterior/prior and current batch; no raw-data replay in numerical update | Old HiPPO/RFF state and variational posterior, current batch; no raw-data replay in numerical update |

Evidence: `/data/nk523/projects/hipposvgp-fair-comparison/scripts/run_official_bui_osgpr_era5.py:56`–82,140–147,346–407,465–508; `/data/nk523/projects/hipposvgp-fair-comparison/scripts/run_covid_ohsvgp_own_theta.py:33`,132–194,249–253,353–503,513–553; campaign candidate lists at `/data/nk523/projects/hipposvgp-fair-comparison/scripts/run_osgpr_final_campaign.py:65` and `/data/nk523/projects/hipposvgp-fair-comparison/scripts/run_ohsvgp_final_campaign.py:73`.

| Domain | Prefix and fit/validation | Final initial legal sites | Stream information | OSGPR initial block | OHSVGP online microbatch/rate search | Chronological tuning |
|---|---|---|---|---|---|---|
| ERA5 |186 hours;720fit/80validation within800 |800;200 hidden never observed |current800; no delayed hidden |256 timestamps, hence one full prefix block |1024 rows; rates .001/.01/.05 |130-hour prefix→56-hour tail,720visible/80hidden, hidden never released |
| COVID |52 weeks;38fit/4validation within42visible |all52, including later10hidden |current42 + previous10hidden |10 timestamps, six blocks |default256 rows; .001 |40-week prefix→12-week tail; original52-site information split |
| PEMS |2016 five-minute rows;234fit/26validation within260 |260;65hidden absent from prefix |current260 + previous65hidden |256 timestamps, eight blocks |1024 rows; rates .001/.01/.05 |1728 rows→288 tail; only original260 sites,234visible/26hidden |

Evidence: campaign domain flags `/data/nk523/projects/hipposvgp-fair-comparison/scripts/run_osgpr_final_campaign.py:25`–27,44–59 and `scripts/run_ohsvgp_final_campaign.py:24`–26,47,61–67,84; folds `/data/nk523/projects/hipposvgp-fair-comparison/benchmarks/three_domain/online_validation.py:8`,51,106. The PEMS fold exposes all260 sites initially, so its26 hidden validation sensors have history that the formal65 initially lack. COVID deliberately exposes all52 initially; this is not an unseen-location test. ERA5 remains permanent spatial holdout. These differences must remain visible in the experiment description.

## Major, reproduced: OSGPR discards learned spatial amplitudes

`/data/nk523/projects/hipposvgp-fair-comparison/scripts/run_official_bui_osgpr_era5.py:64` creates temporal variance theta.kernel_variance and spatial variances1,1. Line81 makes the entire product kernel trainable in adaptive mode. `theta_from_model` at line140 stores all three lengthscales but only the **temporal** amplitude at line145. Rebuilding at lines370 and471 therefore resets both learned spatial amplitudes to1. The effective product amplitude is not absorbed into the serialized temporal parameter. `old_kernel_covariance` is preserved at lines405/507, so this is an unintended hyperparameter restart of the new model, not simply harmless renaming of equivalent amplitudes.

Reproduction on the active release:

```
amplitude_trainable = [True, True, True]
variance_before = [0.8, 2.0, 3.0]
serialized kernel_variance = 0.8
variance_after = [0.8, 1.0, 1.0]
K_before = [[4.8, 3.2525184981], [3.2525184981, 4.8]]
K_after  = [[0.8, 0.5420864164], [0.5420864164, 0.8]]
```

A real one-step Adam fit on four synthetic inputs confirms this is reachable under training, not only manually constructed:

```
after_one_Adam_amplitudes = [0.7945057044, 0.9936904769, 0.9936904769]
effective_variance = 0.7845114296
serialized_variance = 0.7945057044
```

Graph reuse does not repair it. `/data/nk523/projects/hipposvgp-fair-comparison/baselines/osgpr_graph.py:35`–41 copies every new model trainable into the cached model. New model spatial variances are already reset. The existing eager/graph parity tests (`tests/test_osgpr_graph_optimizer.py:34`) compare execution engines given the same incoming model; they do not test kernel preservation across `theta_from_model→make_kernel`.

Scope: all `--adaptive` datasets; prefix transitions in COVID/PEMS and all online transitions in all domains. Historical frozen variance1-controlled runs are not affected by this specific defect. It also means logged `kernel_variance` and posterior-checkpoint theta do not fully describe the actual learned kernel. The official OSGPR objective can legitimately accept a changed new kernel, but this undocumented forced change biases the limited optimizer budget and invalidates claiming normal continuation of its learned hyperparameters.

Remedy: simplest stable parameterization is one trainable product amplitude and fixed unit spatial amplitudes; alternatively serialize and restore all amplitudes exactly, accepting redundant parameterization. Add a tiny roundtrip kernel test and multi-update parity test. Rerun OSGPR own validation and finals affected by changed optimization; old validation rankings cannot be certified reusable. Preserve old artifacts as diagnostics. Reusing unchanged raw data/splits/common ridge offsets is fine. If preserving old checkpoints for debugging, label their missing spatial amplitudes; theta alone cannot reconstruct their predictive model.

## OHSVGP fidelity, initialization and meaning of spatial adaptation

The “+spatial kernel” implementation is the official multidimensional input construction: 3D RBF frequencies feed one HiPPO state. It is **not** an OHSVGP temporal state tensored with a separate spatial inducing system. Sensor rows count as discrete HiPPO integration steps. Sorting uses `np.lexsort` over time and coordinates (`run_covid_ohsvgp_own_theta.py:132`), while `LazyHiPPOLegS` uses discrete indices `previous_steps+1…+batch_length` (`/data/nk523/projects/hipposvgp-fair-comparison/scripts/run_traffic_ohsvgp.py:70`). Thus a physical timestamp with800 observed sensors advances800 integration steps; one with42 advances42. Delayed observations are appended in arrival order even though their physical time is earlier than the current query. Time remains a coordinate in the kernel, but HiPPO's discrete measure is the row/arrival stream, unlike the proposed temporal-axis representation. This is a substantive baseline definition, not a numerical integration error by itself.

Initial integration uses1024 deterministic evenly spaced rows of sorted fitting design (`basis_grid`, line249), despite the likelihood population being much larger. `previous_steps` after initialization is the number of grid rows (line504), then grows with every actual online sensor observation (line553). Initial compression therefore changes the relative integration weight of initialization and stream. This is an adapter approximation beyond merely adding latitude/longitude to inputs. It deserves explicit description and a correctness/representation rationale; exact prediction-basis caching tests do not validate this approximation. Neither matching proposal's temporal order nor matching raw M values equates representational capacity.

Explicit corrections are soundly motivated:

- Upstream frozen-branch old covariance divides by `log_sf` (`/data/nk523/projects/hipposvgp-fair-comparison/baselines/external/harrisonzhu508_HIPPOSVGP/hipposvgp/multidim.py:171`) and later multiplies by exp(log_sf) at line274. Adapter subclass divides by exp(log_sf) at `run_covid_ohsvgp_own_theta.py:43`–54, preserving old covariance.
- Fixed base random draws are divided by current trainable lengthscales inside calibration (`run_covid_ohsvgp_own_theta.py:371`–394), so lengthscale gradients are not detached.
- Calibration uses beta=batch_size/population and multiplies ELBO by population/batch_size, giving an unbiased likelihood/KL-scaled objective. Upstream likelihood is Monte Carlo sampled even for Gaussian likelihood (`multidim.py:351`–366), so this is stochastic objective optimization, not a deterministic conjugate update.
- Predictor adds likelihood variance exactly once. Upstream `pred_f` has a misleading `flag_noise` docstring but its body returns latent variance without adding observation noise (`multidim.py:375`–427); adapter adds noise at `run_covid_ohsvgp_own_theta.py:219`.

OHSVGP final initialization is not just the38/720/234 fit-site state reused. It fixes the selected initial kernel/noise (`run_covid_ohsvgp_own_theta.py:472`–475), creates a **fresh variational posterior**, and refits using `protocol.task1().locations` (line457), i.e. all52/800/260 legal sites. It runs only `best_validation_iteration` minibatches (line490). This preserves the intended available-site boundary, but does not ensure every initial label is sampled or that the new all-site posterior converged. Concrete observed COVIDseed5 selection is best iteration5: five batches of256 can touch at most1280 of2704 legal initial observations. This is legal stochastic training, not leakage, but “all initial labels assimilated exactly” would be false, and an early hyperparameter-validation optimum is not a convergence certificate for a fresh all-site variational refit. Qualify the final refit separately using only legal initialization data; retain method-specific stochastic training rather than imposing equal iterations across methods.

## Own validation, capacity/budget gates and observed selections

Both methods now use their own validation; they are not merely borrowing the proposal's theta. Both use formal initialization only for model/budget selection and reserve the full later stream for final evaluation. Frozen fit-site means are shared with ST-SVGP/MGPVAE. This is a fair information contract but does not isolate solver quality because the proposal jointly updates its mean and GP whereas these baselines keep mean fixed.

OSGPR: full-batch Adam lr.01, initial per-block budgets100/400, extend800 and then doubling selected improving boundary to6400; candidates2×16,2×32,4×16,4×32. Online steps5/20/80, conditionally320/1280, improvement threshold.01 NLPD. See `/data/nk523/projects/hipposvgp-fair-comparison/scripts/run_osgpr_final_campaign.py:65`–106. Budget measures per-assimilation operations, so delayed streams usually perform twice the selected count per scored timestamp. Adam resets each assimilation, including graph-cache route (`baselines/osgpr_graph.py:42`). Iteration budget sufficiency is judged by validation budget comparisons, not a gradient/objective convergence trace. Spatial capacity is selected on the posterior after sequential prefix compression; this is a legitimate streaming-specific choice, distinct from full batch calibration.

OHSVGP: lr.001 initial, batch256, gradient norm clip20 and explicit parameter bounds (`run_covid_ohsvgp_own_theta.py:233`–238,398). Capacity32/64. Initial budgets500→1000→2000→4000→8000→16000→32000 stop when the chosen validation checkpoint is interior or objective plateau criterion holds (`/data/nk523/projects/hipposvgp-fair-comparison/benchmarks/three_domain/convergence.py:4`). An interior optimum resolves **selection**, not necessarily global convergence. Online updates1/5/20 plus80 for ERA5/PEMS, then320/1280 only for currently best learning rate if endpoint improves. COVID only lr.001, others .001/.01/.05 (`run_ohsvgp_final_campaign.py:73`–103). Selection plots should show stochastic variability and bounds; a hard maximum or very early optimum should not be described generically as “converged.”

Observed read-only remote latest selection records (not full admission certificates):

| Domain/method | Seeds inspected | Selected size/budget | Online updates |
|---|---|---|---|
| ERA5 OSGPR |0,3,4 |4×32; initial1600,1600,3200 per block |5 each |
| COVID OSGPR |5,8,9 |2×32/800;4×16/100;2×32/400 |5 each |
| PEMS OSGPR |1, job294411 |2×16/100 |20 |
| COVID OHSVGP |5–9 |M32,64,32,32,64; budget500; best iterations5,425,420,45,105 |1280,1280,320,1280,1280 |

These data came from `/vol/bitbucket/nk523/hipposvgp-fair-20260929/results/fair-three-domain-wandb-20260929/{dataset}/{method}/seed*/job-*/selection.json`, filtered to most recent existing selection per seed. OSGPR PEMS294411spec pins release6a952dc2. Local submission records pin OHSVGP ERA5job294499 and PEMSjob294532 to f74dd637. No completed OHSVGP ERA5/PEMSselection.json was found in that query, so do not assume a completed choice from active job names.

Potential major limitation: some OSGPR ERA5 selections are at the largest tested capacity128, and OHSVGP COVID selects64 on two splits; campaigns extend iterations but **not** capacity. This does not prove larger capacity would help, but prevents a strong claim that baselines have been capacity-qualified against proposal states of thousands of joint variables. Resolve using a predeclared adequate capacity/memory budget or validation boundary checks within the existing five methods, not extra baselines or mechanism ablations. Equal M is inappropriate because OHSVGP M and proposal Mt×Ms describe different representations. Equal information with honest accuracy/time/memory tradeoffs is defensible.

## Delay order, double counting, chronology and shared information

OSGPR sequence is delayed prior hidden block then current visible block (`run_official_bui_osgpr_era5.py:465`–508); old posterior updates after each, and each new label appears in one new likelihood batch. Repeating optimizer iterations on that fixed batch does not duplicate its likelihood. OHSVGP does the same at `run_covid_ohsvgp_own_theta.py:516`–553; old posterior changes only after a microbatch finishes all optimizer steps. No extra same-label re-assimilation was found in these loops. ERA5 flags exclude delayed hidden labels; PEMS flags require them. Boundary tests mutate current hidden/future labels and assert prior predictions invariant (`tests/test_osgpr_release_boundary.py:42`, `tests/test_ohsvgp_release_boundary.py:42`); these are meaningful tests, unlike self-reported counters alone. Their small synthetic executions do not certify every feature builder or every released artifact.

PEMS time discrepancy from the setting audit remains horizontal: OSGPR and OHSVGP consume naive archived timestamps (then affine rescale); proposed warm-start path and ST/MGP via protocol.week use uniform steps. The HDF DST jump is consistent with continuous physical five-minute sampling under America/Los_Angeles. Adopt one explicit elapsed-time convention and retain separate local calendar features. Initial PEMS calibration and its day7 chronological fold precede DST; a purely post-prefix time-convention repair can preserve those validation inputs/rankings **if hashes/dependencies are requalified**, while final OSGPR/OHSVGP streams crossing the jump need reevaluation. The amplitude fix independently invalidates OSGPR validation reuse.

## Timing, memory and immutable reuse

OSGPR update timer covers model construction, optimization and posterior extraction, including delayed assimilation; numpy extractions synchronize TensorFlow. Per-iteration tracking `emit` is within optimization (`run_official_bui_osgpr_era5.py:95`–97), so logging overhead contributes. Prediction timer excludes later score/archive/checkpoint writes; process total is a different scope. Initial calibrated-state extraction time is counted at lines369–407.

OHSVGP update timer at `run_covid_ohsvgp_own_theta.py:515` ends at555, after GPU state-export operations but without an explicit final CUDA synchronization. Prediction converts tensors to CPU at line220, which flushes queued work; some final update work can therefore be charged to prediction. Several earlier scalar checks synchronize within updates, so this is a boundary ambiguity, not proof of a large underestimate. Sum update+prediction is safer than interpreting each separately until standardized synchronized timers are used. It reports combined process time including calibration/refit, but no clean separate refit cost in final timing fields (line619). Proposal uses synchronized timer scopes; cross-method speed claims need aligned components and isolated hardware measurements.

Both runners keep entire protocol arrays and prediction archives in host memory for execution/evaluation. Algorithmic bounded persistent state and process peak RSS are different quantities. OSGPR maintains denseM×M posterior; OHSVGP denseM×M variational state plus HiPPO/RFF state. Report these alongside measured total memory; do not compare proposal's theoretical state bytes to another method's whole-process RSS.

OHSVGP validation reuse checks matching spec (except source/input paths), numerical dependency hashes, input hashes, successful tracking and artifact hashes; follows chains to original record (`/data/nk523/projects/hipposvgp-fair-comparison/benchmarks/three_domain/reuse_validation.py:11`–93). This is substantial provenance control. However dependencies at lines32–39 omit the actual cached wrapper `scripts/run_ohsvgp_cached_prediction.py` and cache implementation `baselines/ohsvgp_prediction.py`, and do not pin the executable environment's installed package hashes. Current cache is scoped deterministic prediction-basis memoization with a useful exact-parity/lifetime test (`tests/test_ohsvgp_prediction_cache.py:8`); mathematical reuse across that optimization is plausible and supported by test design, but “all numerical dependencies identical” overstates the hash gate. Include these files/environment in qualification. Changes to mean, kernel, likelihood, refit policy, integration grid, observations or tuning input generally require new validation. Pure logging/cache changes can preserve predictions after explicit parity checks, but old timing is not new timing.

OSGPR graph reuse is execution reuse, not saved scientific validation reuse: cached graph receives new batch/old-posterior tensors and resets Adam. Tests verify eager/graph equality on small models. It preserves both good and bad adapter behavior, including the amplitude-loss defect above. The current campaign does not offer an OSGPR saved-validation reuse-root; previously completed results must not be silently carried through a source fix merely because graph parity passed.

## Admission implications

1. Repair and requalify adaptive OSGPR amplitude continuation across all three datasets, then rerun its own validation/finals. No extra method needed.
2. Specify OHSVGP's actual multidimensional row-stream HiPPO definition,1024-row initialization grid, stochastic initial refit and frozen kernel; qualify final refit separately and justify capacity limits using validation, not final accuracy.
3. Keep unequal optimizer policies where they belong to the method, but record complete update costs and separate selection resolution from optimizer convergence. Compare common mean/input availability and independent tuning.
4. Resolve one cross-method time contract, explicit reveal semantics and actual block cadence. Update/refit parameters need not be identical for fairness.
5. Reuse clean data/protocol/unchanged validated components conservatively with hashes and numerical equivalence. Preserve pending/active artifacts; this review itself authorizes no termination or restart.
