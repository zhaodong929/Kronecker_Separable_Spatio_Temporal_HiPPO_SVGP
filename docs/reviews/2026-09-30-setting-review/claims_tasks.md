# Independent manuscript/experimental-task review — 2026-09-30

Scope: read-only audit of actual Overleaf manuscript and current comparison implementation, with historical source inspection. No source edits, jobs, commits, or external messages. Only this report is written. This review does not certify every method's numerical implementation or every remote result; those are separate team audits.

Policy read: /data/nk523/ObsidianVault/03-projects/hipposvgp/2026-09-29-baseline-selection-policy.md. Lines 7–9 restrict the new comparison to three datasets, accuracy/time/memory, with no new ablations; lines 15–32 prescribe KronHiPPO-SVGP + Bui OSGPR + spatial OHSVGP + Aalto ST-SVGP + causally adapted MGPVAE. The manuscript still uses KronHiPPO-STGP; name must be harmonized. Excluded methods and archived ablations are discussed below only to identify stale manuscript claims, not proposed for rerunning.

Snapshots inspected: fair-comparison HEAD 4f881c7; Overleaf HEAD f0e7e46; original workspace HEAD 0398b28. The original workspace is an earlier ERA5-oriented checkout without PEMS paths in its tracked HEAD. The comparison repository preserves later PEMS introduction commit 70f09c5.

## Principal findings

1. **High: paper and current experiment represent different ERA5 tasks.** Manuscript ERA5 uses 171 multi-time blocks across Tasks 2–10, whereas the current protocol scores 1,674 single-hour updates. A ten-time block that assimilates all visible targets before predicting all hidden targets permits later-within-block visible observations when predicting earlier-within-block times. Hourly online prediction does not. These are different information boundaries, not merely different speed measurements.
2. **High: PEMS is explicitly one-time online updating, not multi-time task batches.** Initialization is 2,016 timestamps; the stream has 50,100 single-time queries, each at 65 hidden sensors after current 260-visible assimilation and previous-step hidden-label assimilation. Multi-time initial optimization batches and prediction microbatches are not online tasks. This was already true in earliest located PEMS baseline commit 70f09c5; it is not a newly introduced 2026-09-30 interpretation.
3. **High: actual PEMS temporal inputs differ across methods.** Stored timestamps contain one 65-minute increment. Bui/OHSVGP use that stored coordinate; proposal/ST/MGPVAE reconstruct a regular five-minute coordinate. After the increment the latter differ by one hour. Whether the source records naive local time with a DST jump or actual missing elapsed observations remains unresolved, but different method inputs are established independently of that interpretation.
4. **High: delayed COVID/PEMS experiments do not instantiate the paper's single-Kronecker solver theorem.** The theorem requires unchanged spatial observation functionals. Current delayed updates alternate visible and hidden geometries and use a sum-Kronecker preconditioned CG extension. Its bounded number of geometry factors can preserve bounded inference state, but the exact one-Sylvester O(M³) guarantee is not the guarantee of these executions.
5. **High: current main table and winner claims are stale relative to authorized scope and revised protocols.** Its method sets differ by dataset; ST-SVGP is absent on ERA5/PEMS and MGPVAE absent everywhere. Old point rankings cannot establish superiority over the new five-method design.
6. **Medium/high: mean-model differences confound mechanism attribution.** Proposal continues joint beta/GP updates. Four baselines receive a Task-1-fitted frozen mean. This is a legitimate system comparison if disclosed and independently tuned; it cannot by itself show that analytical HiPPO or the solver caused an accuracy advantage.
7. **Medium: manuscript task descriptions omit several operational definitions that materially change their scientific interpretation.** COVID initial visibility, release latency, PEMS stream extent, temporal batching, timestamp convention, initial posterior initialization, and method-specific replay/update rules need explicit statements.

## Evidence register

All paths below are absolute. Line numbers refer to the inspected working copies.

- E1: /data/nk523/projects/hipposvgp-overleaf-6aba8fcfb715ddc40d02c0d2/main.tex:69 — abstract claims joint trend/residual, analytical HiPPO, O(M6) to O(M3), three-domain predictive evidence and stream-length-independent memory.
- E2: /data/nk523/projects/hipposvgp-overleaf-6aba8fcfb715ddc40d02c0d2/main.tex:354 — fixed spatial geometry premise; :360 single-Kronecker proposition explicitly assumes unchanged spatial inducing and observation functionals, Cartesian blocks and homoscedastic noise; :381 bounded state based on single spatial Gram factor.
- E3: /data/nk523/projects/hipposvgp-overleaf-6aba8fcfb715ddc40d02c0d2/main.tex:446 — all-online same-information/no-replay wording; :453–497 old heterogeneous baseline table; :524 aggregate lowest-RMSE/CRPS/NLPD claim.
- E4: /data/nk523/projects/hipposvgp-overleaf-6aba8fcfb715ddc40d02c0d2/appendix.tex:297 — ERA5 186 initialization times, 171 blocks, 800/200 sites, 133 mean features, archived frozen-offset versus joint-mean distinction; :299 block assimilation before held-out prediction.
- E5: /data/nk523/projects/hipposvgp-overleaf-6aba8fcfb715ddc40d02c0d2/appendix.tex:315 — COVID 52+143 weeks, 42/10 split, but initial 52-site visibility and one-week delayed release not specified.
- E6: /data/nk523/projects/hipposvgp-overleaf-6aba8fcfb715ddc40d02c0d2/appendix.tex:342 — PEMS nowcasting, 260/65 sites, labels after prediction, archived Ms32/Mt128/RFF512/road context; no 2016/50100 horizon or one-step latency.
- E7: /data/nk523/projects/hipposvgp-fair-comparison/baselines/traffic_protocol_n.py:21 — Task 1 exposes only visible sensors; :75–99 formal 325 sites/2016 initial/50100 stream/260 visible/65 hidden/234 fit/26 validation, one-step delay and seeds1–3.
- E8: /data/nk523/projects/hipposvgp-fair-comparison/baselines/covid_long_setting_b/protocol.py:168 — all Task-1 locations exposed; :180–212 previous hidden then current visible then current hidden query; :233–240 52+143,42/10,38/4.
- E9: /data/nk523/projects/hipposvgp-fair-comparison/baselines/era5_protocol.py:10 — one-time query, no hidden release; :24 186 initial/1674 online/800 visible/200 hidden/720 fit/80 validation; :29 initial hidden excluded.
- E10: /data/nk523/projects/hipposvgp-fair-comparison/scripts/export_traffic_external_gp_protocol.py:63 — scaler fits only visible-calibration indices; :73–85 road-context features and fixed initial ridge offset; :94–102 source times and singleton block starts/stops; :133–148 one-step release order.
- E11: /data/nk523/projects/hipposvgp-fair-comparison/scripts/iclr_era5_full_benchmark_protocol.py:111 — archived task-aware block extraction; :132–134 fallback ten-time blocks; :208 default block size10.
- E12: /data/nk523/projects/hipposvgp-fair-comparison/scripts/run_iclr_era5_routeb_strict_online.py:468 — uniform time reconstruction under Task-1 initialization; :505 multi-geometry selection when delayed release or changed initial geometry; :684 initial joint posterior update; :798 hidden-geometry assimilation; :868 current visible joint update; :993–994 reports sum-Kronecker CG and false fixed-geometry complexity applicability under delay.
- E13: /data/nk523/projects/hipposvgp-fair-comparison/stvgp_kronecker/joint_ssgp_kron/multi_geometry.py:1 — explicitly disclaims single-Sylvester complexity; :18 CG tolerance1e-9/max512; :57–87 residual checks/nonconvergence error; :147–158 transports and accumulates distinct Gram factors; :163 joint beta posterior; :196 predictive mean includes beta.
- E14: /data/nk523/projects/hipposvgp-fair-comparison/baselines/causal_mean.py:8 — fixed stored mean; :10 indexes it on reconstructed timeline; :18 rejects off-grid query.
- E15: /data/nk523/projects/hipposvgp-fair-comparison/scripts/run_covid_ohsvgp_own_theta.py:325 — actual stored stream times, initial-only affine normalization :326–332; :338–340 requires singleton blocks; :458–461 frozen fit-site mean; :517–522 previous hidden then current visible.
- E16: /data/nk523/projects/hipposvgp-fair-comparison/scripts/run_official_bui_osgpr_era5.py:282 — fixed initial offsets; :287–297 uses source stream times with initial-only scale.
- E17: /data/nk523/projects/hipposvgp-fair-comparison/scripts/run_mgpvae_causal.py:219 — obtains protocol.week times; :223–228 delayed/current updates; :234–236 observation-space Gaussian components with frozen mean.
- E18: /data/nk523/projects/hipposvgp-fair-comparison/baselines/covid_long_setting_b/adapters/run_st_svgp.py:428–447 — initial state then protocol.week-driven causal updates and fixed mean.
- E19: /data/nk523/projects/hipposvgp-fair-comparison/baselines/covid_long_setting_b/protocol.py:96 — recreates uniform chronological_stream_times from median initial cadence, ignoring actual online timestamps; :194/:202/:209 uses these for released/current/query times.
- E20: /data/nk523/projects/hipposvgp-fair-comparison/scripts/run_iclr_era5_routeb_strict_online.py:626–631 — result lists and complete prediction arrays grow with stream; :960–968 checkpoints scored truth and predictions; :990 persistent_state_bytes reports only inference state.
- E21: /data/nk523/projects/hipposvgp-overleaf-6aba8fcfb715ddc40d02c0d2/appendix.tex:157–173 — detailed conditional core-solve/state costs explicitly exclude data loading, statistics, Task1 calibration and all-point predictions.
- E22: /data/nk523/projects/hipposvgp-fair-comparison/scripts/run_era5_kronhippo_campaign.py:24–34 — current candidate capacities32/64, own validation, final1674-hour run; differs from archived Ms=Mt128 operating point.
- E23: /data/nk523/projects/hipposvgp-fair-comparison/benchmarks/three_domain/evaluation.py:8–17 — Gaussian metrics; :30–45 mixture aggregation/ECE. Manuscript Gaussian formula is /data/nk523/projects/hipposvgp-overleaf-6aba8fcfb715ddc40d02c0d2/appendix.tex:179–216.
- E24: /data/nk523/projects/hipposvgp-fair-comparison/stvgp_kronecker/data/traffic.py:265–272 — road features are current visible target-derived context, not exogenous channels; :314–326 current and past graph context; :410–413 legal initial scaling.
- E25: /data/nk523/projects/hipposvgp-fair-comparison/data/fair-three-domain-20260929/pems-v2/seed1/protocol.json:3–18 — delay1,50100 online,seed1, hash and scale declaration. Binary paired NPZ inspected directly.

## Horizontal comparison of task semantics

| Dimension | ERA5 current | COVID current | PEMS current |
|---|---|---|---|
| Initial labels |186 times ×800 sites|52 weeks ×all52 sites|2016 times ×260 sites|
| Scored query event |One hour ×200 hidden sites|One week ×10 hidden sites|One five-minute sample ×65 hidden sites|
| Number of events |1674|143|50100|
| Current observed sites |800|42|260|
| Past held-out labels |Never revealed|Previous stream week revealed once|Previous stream sample revealed once|
| Site generalization meaning |Sites absent from all target training|Sites already seen in initial year, and all but current stream label available later|Sites absent initially, then revealed with one-step delay|
| Initial fit/validation sites |720/80|38/4 drawn from current42-visible set|234/26 drawn from260-visible set|
| Typical dynamic mean information |Current/past meteorological channels|Protocol-provided mean features; future-hidden initial labels are legal under Task1 contract|Road-weighted current and lagged visible speeds; not independent covariates|
| Manuscript batching |171 multi-time blocks; differs|Weekly; compatible but release details omitted|No operational block extent stated|
| Fixed-geometry proposal theorem |Applicable if same800-site geometry initial/online and other assumptions hold|Not applicable to all52 initial then42/10 delayed geometries|Not applicable to260/65 delayed geometries|

Sources E4–E11/E24. This matrix establishes different task types, not one universal held-out-site forecasting task. Delayed release is lawful by declaration but means COVID/PEMS evaluate delayed sensor reporting/interpolation rather than permanently unobserved spatial locations. At stream step0 PEMS has no previous stream hidden label; subsequent steps do. All three are contemporaneous spatial prediction, not whole-field future forecasting.

There is no mathematical requirement that an online Cartesian block contain multiple timestamps. A singleton time crossed with many sites is a valid Cartesian block. Consequently PEMS singleton updates do not invalidate analytical HiPPO in themselves. Their implications are computational workload, release timing, and the alternating spatial geometry. The report does not recommend batching merely to obtain better or faster results.

## Horizontal five-method/claim matrix

All five methods must receive the same dataset-specific information boundary above. 'Same features' is not 'same mean inference.'

| Method | ERA5 | COVID | PEMS | Legitimate comparison / limitation |
|---|---|---|---|---|
|KronHiPPO-SVGP|Joint beta+GP; hourly visible-only; single-geometry solver|Joint beta+GP; all-site initial; delayed42/10; multi-geometry CG|Joint beta+GP; delayed260/65; multi-geometry CG; uniform online clock|Tests full proposed system; single-Sylvester guarantee restricted to fixed geometry|
|BuiOSGPR|Task1 fixed mean + adaptive sparse residual GP; own tuning|Same boundary with initial52 and delayed labels|Fixed mean, singleton stream, uses source timestamp jump|External sequential sparse-GP comparison; not identical trend update or kernel capacity|
|OHSVGP spatial|Task1 fixed mean + own calibrated HiPPO residual update|Initial52 and delayed labels|Fixed mean, singleton stream, uses source timestamp jump|Closest external HiPPO comparison; algorithmic differences include mean and representation/capacity|
|AaltoST-SVGP|Fixed mean + causal state continuation after frozen fit|All52 initial then delayed correction|Fixed mean + delayed correction; uniform online clock|Compare actual causal adapter, distinguish prior replay version and continuation version|
|causalMGPVAE|Fixed mean + learned nonlinear residual observation model with causal latent state|All52 initial and delayed correction|Uniform online clock; same legal current features through shared mean|Mixture predictive distribution requires observation-space proper scores, not Gaussian plug-in or ELBO|

Source code evidence E12–E18. This is a semantics matrix, not certification that all15 method×dataset cells are complete or qualified. The separate execution/provenance audit determines which cells have admissible results. The current MGPVAE residual adapter addresses a useful competing observation-model family, but does not test whether a standalone GPVAE without the shared covariate mean would be better. No additional model or ablation is requested here.

## PEMS task history and clock verification

Read-only historical command `git show 70f09c5:baselines/traffic_protocol_n.py` establishes2016/50100 at historical lines44–45 and delay1 at65–66. That commit is titled 'Add A100 PEMS external GP baseline suite'. Current exporter E10 explicitly assigns block_start[t]=t, block_stop[t]=t+1. Current OHSVGP raises if blocks are not singleton (E15). Initial OSGPR chunks of256 times mentioned in ADAPTER_AUDIT are optimizer batches only; they do not alter scored online events. The original workspace at0398b28 predates these tracked PEMS files, so it cannot provide affirmative evidence for a different PEMS multi-time design.

Direct NPZ inspection with NumPy, no file mutation:
- calibration_y shape=(2016,325), stream_y=(50100,325);
- all50100 block widths exactly1;
- stream_times increments:50098 increments of0.08333333h and one increment of1.08333333h;
- jump occurs between zero-based stream rows18167 and18168;
- max source-time versus reconstructed-time difference=1.0000000000040927h;
- final source time4343.916666666667h; reconstructed4342.916666666663h.

This is **fact** about the stored archive. Different actual model-time paths E12/E15/E16/E17/E18/E19 establish a **fact** of horizontal inconsistency. It is **unknown** whether source-time jump is civil-time/DST or missing elapsed observations. Therefore do not claim that one path has definitively dropped a physical missing hour, and do not correct all paths blindly to raw naive timestamps. Declare one canonical physical/model coordinate and a separate calendar-feature coordinate, document timezone/DST handling, then apply identically to all methods and archives. A timestamp audit must inspect actual model inputs; common output timestamps alone would not catch this.

## Claims: supported, limited, unsupported

**Supported conditional mathematical claim:** The manuscript theorem and costs concern fixed spatial geometry with a single Kronecker likelihood factor (E2/E21). Current ERA5 can instantiate those conditions. Existing matched-solver checks can support solver algebra independently, but no new mechanism experiments are proposed here.

**Supported structural inference, requiring measurement:** For COVID/PEMS a fixed finite set of recurring spatial Gram factors can have stream-length-independent retained inference-state dimensions. E13 merges only equal factors and stores their temporal factors. This does not prove constant process RSS/GPU memory, constant end-to-end latency, or O(M³) runtime independent of CG iterations. The exact finite Gaussian target is solved numerically to a tolerance; report residuals/iteration costs rather than calling it the same exact single-Sylvester operation.

**Unsupported unqualified application:** 'The proposed method reduces posterior computation from O(M6) to O(M3)' in the abstract invites readers to apply that cost to all three reported domains. Delayed-geometry runs need their actual solver qualification and scaling statement, or the abstract must explicitly scope the guarantee. This is a theory/application mismatch, not a refutation of the conditional theorem.

**Supported system-comparison scope:** A completed, matched, qualified five-method table can compare predictive quality under declared nowcasting boundaries and chosen validation-selected capacities. It can compare end-to-end update and prediction cost at those operating points.

**Unsupported mechanism attribution from that table alone:** Joint trend update, RFF/basis capacities, hyperparameter adaptation policies and observation models differ. Improved RMSE does not isolate analytical temporal evaluation, transport, or Schur solving. The current no-ablation policy means those causal component claims must remain theoretical or rely on explicitly separate already-existing evidence, not be inferred from new basic comparisons.

**Unsupported blanket no-replay wording:** E3 says all online experiments never replay prior blocks, but archived appendix baseline provenance includes causal ST-SVGP refits and old adapters have replay paths. The current state-continuation adapter may avoid full replay, but must be described by pinned release and the actual delayed correction policy. A replay-free proposed method does not imply every baseline uses no replay.

**Unsupported unqualified memory claim for executable:** E20 accumulates complete results/truth/prediction grids and checkpoint prefixes. These are not retained GP sufficient statistics, yet they count toward actual process memory. State boundedness and measured peak process/device memory must be reported separately; archive storage can grow linearly without refuting bounded inference state. The abstract's memory wording should say 'retained inference state' if that is the proven object.

**Unsupported preservation of archived winner claims:** E3 table uses excluded controls/baselines, missing required methods, old capacities and old ERA5 batching. Those results may remain historical results with their exact provenance; they cannot be numerically relabeled as the new benchmark. Changing normalization also changes numerical RMSE/NLPD scale, so old and new scalar scores should not be mixed.

**Limited qualitative evidence:** Manuscript main.tex:520 and appendix.tex:349–355 explicitly select sensors by full-stream proposal RMSE quantiles and the plotting interval by observed-speed range. A predeclared selection rule can be acceptable illustration, but selection is outcome-derived and cannot independently validate superiority or justify configurations. COVID trajectory selection also uses observed trajectories (appendix:319–321). Keep these separate from validation and aggregate inference.

**Metrics:** The Gaussian formula is proper for Gaussian predictive distributions; causalMGPVAE needs its actual observation-space mixture scores. E23 includes such machinery, but the revised paper must describe it and its fixed Monte Carlo budget. Score uncertainty is not quantified merely by spatial-split SD; split pairing, serial dependence, and training randomness differ. No assertion of statistical significance follows from a smaller mean alone.

## Minimum defensible redesign, without outcome-driven selection

1. Keep precisely the five authorized methods and three datasets. Preserve archived results and identify them as superseded protocols where necessary. No new ablations or excluded methods.
2. Freeze a single formal event contract before further final admission: initialization labels; calibration/validation partitions; number of query events; event size; release delay in samples and physical time; current covariate availability; time convention; normalization; initial posterior initialization; final queried locations.
3. Resolve ERA5 task identity explicitly. The minimal choice consistent with the current causal rerun is1674 hourly events, with manuscript rewritten to that setting. Restoring171 multi-time blocks is an alternative scientific task, not an implementation optimization; choose based on intended application, not which produces the favorable outcome. Do not pool either version.
4. Preserve PEMS2016+50100 single-time protocol unless a genuine application requirement calls for batched reporting. If batching is chosen, specify whether a block's later visible labels may help earlier hidden predictions, when hidden labels are released, and recompute all five methods on the same boundary. A pure compute microbatch must preserve per-time predictions and cannot assimilate later observations early.
5. Canonicalize PEMS time using source timezone evidence; ensure all five methods actually consume identical coordinate values and that model times are saved. Add a targeted clock-consistency admission check; source gap interpretation must be documented first. This is a correctness fix, not selecting task variants on held-out performance.
6. State that proposed delayed-geometry runs use a finite-sum extension and residual-checked CG. Retain the single-Sylvester theorem with its original assumptions; use measured time/memory for delayed tasks. Do not silently claim fixed-geometry theory applies or remove lawful delayed labels solely to satisfy theory.
7. Declare proposal joint adaptive beta versus common frozen initial mean for baselines. Restrict new empirical claims to full-system comparisons. Independent validation of each method and matched feature availability remain essential; numerical equality of parameter counts is not a universal fairness condition.
8. Report complete paired splits (ERA5/COVID5, PEMS3), statuses and exclusions by predetermined numerical/availability criteria. Select capacities/budgets on legal initial validation only. Current test outcomes must not decide protocol, clock, candidates, result inclusion, or whether to drop MGPVAE.
9. Measure initial fitting and online update/prediction separately, including delayed corrections, temporal factor formation, replay where used, and input-feature cost. Separate scoring/archive/checkpoint overhead, persistent GP state, and process/device peak memory. Use controlled hardware contention; concurrency runs do not establish intrinsic speed rankings.
10. Rewrite manuscript baseline list, dataset protocols, Gaussian/mixture metric definitions, solver applicability, task-specific initial visibility, and conclusions to match the admitted records. Until then, neither archived table rankings nor incomplete new results support the finalized five-method paper comparison.

## Outstanding unknowns

- Which completed remote runs use each pinned protocol/source revision, and which already crossed the PEMS clock jump? Local source plus a seed1 NPZ establishes the issue; affected result inventory requires the execution audit.
- Source timestamp timezone/DST semantics; no external data documentation was consulted in this sub-audit.
- Full numerical equivalence of each adapted baseline to its claimed official objective/causal filter; handled by dedicated method audits.
- Whether archived mechanism and baseline outcomes were produced with later corrected scaling, posterior initialization and conditional residual variance; no provenance assumption is made from manuscript numbers.
- Ultimate task intent if user means multi-time PEMS arrival batches: neither current code nor inspected PEMS introduction history implements that intent. It requires an explicit scientifically defined change, not reinterpretation of existing50100 rows.
