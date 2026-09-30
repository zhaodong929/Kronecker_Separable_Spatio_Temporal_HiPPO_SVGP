# Proposed initial-period selection and optimization budget

**Status: SUPERSEDED as an adoption proposal by the user’s common-fairness-criteria correction. No listed method-specific step ladder has been adopted or executed.**

The method-specific ladders below were planning scenarios, not a fair-comparison rule. Before launching candidate selection, fix common validation scoring, evaluation opportunities, stopping criteria, and computational/search limits. Method-specific iteration counts may follow those rules; they must not be assigned ad hoc. Matched iteration counts, data exposures, and elapsed computation are distinct quantities. Any finite-budget comparison must be labelled as such, with capped methods not called converged. Generic runner and exposure-accounting utilities do not authorize or define a scientific budget policy.

**Historical proposal follows.** The two-step CPU/GPU and full-shape runs are implementation/resource diagnostics. Their scores must not select scientific configurations. No proposed step count below proves convergence, and no completion-time promise follows from it. This proposal does not authorize 65 final runs before the required validation and resource qualification.

## Data and selection boundary

Use each spatial split's existing initial-period selection stream independently:

| Dataset | Internal initial fit | Subsequent validation | Final initial refit |
|---|---:|---:|---:|
| ERA5 | 168 hours, internal fit sites | 168 hours, 7 daily tasks | 336 hours, 800 visible sites |
| COVID | 40 weeks, all 42 outer-visible sites | 12 weekly tasks with internal query sites | 52 weeks, all 52 sites |
| PEMS | 2016 five-minute samples, internal fit sites | 2016 samples, 168 hourly tasks | 4032 samples, 260 visible sites |

Preserve the implemented task-end predictions and delayed-release rules inside validation. Select on query-weighted observation NLPD only; retain RMSE, CRPS, coverage and task-level errors as diagnostics. Never choose candidates, periods or stopping decisions using the final stream. A score-selected winner from one spatial split must not simply be transferred to another split: the former's observations can include the latter's permanently held-out sites. Runtime-only profiling of one split/domain is acceptable; all 13 splits still need their own valid selection record.

## Small candidate set

Use two capacity candidates per method and dataset, not a Cartesian sweep over every parameter. Proposed starting points:

| Method | Capacity candidates | Fixed learning rate for this small search | Initial optimization checkpoints |
|---|---|---:|---|
| KronHiPPO-SVGP | ERA5/PEMS: spatial 32 or 64, temporal 32; COVID: spatial 16 or 32, temporal 32 | 0.02, matching existing empirical-Bayes runner default | 50, 150, 450 full-data steps |
| Bui OSGPR | spatial 32 × temporal 4 or 8, giving 128 or 256 inducing points | 0.01, existing official-adapter default | 100, 300, 900 full-data steps |
| OHSVGP+spatial | inducing functionals 32 or 64; RFF 256 and initialization grid 1024 fixed | 0.001, existing adapter default | s0, 3s0, 9s0 as defined below |
| ST-SVGP | ERA5/PEMS: spatial 32 or 64; COVID: spatial 16 or 32 | 0.01, existing official-adapter default | 50, 150, 450 full-data steps |
| causal MGPVAE | latent dimension 2 or 4; encoder width 16; training MC 4, prediction MC 512 fixed | 0.001, existing adapter default | 100, 300, 900 full-data stochastic steps |

Counts are method-specific: OSGPR's Cartesian inducing count is the product, OHSVGP uses row-stream inducing functionals, and MGPVAE retains its own spatial state. They are not equal-capacity models because a parameter happens to have the same numeric value. Reject candidates exceeding available initial sites. Larger candidates must pass their own memory checks before admission; do not silently substitute a smaller capacity after OOM. Keep the current official model families and qualified numerical adapters unchanged.

This is a deliberately limited search, to be disclosed as such. If the fixed learning rate fails numerically or all checkpoints remain clearly improving, a predeclared fallback is one rerun at one third of that rate and one additional 3× budget rung. Record that extension; do not improvise repeated searches until the proposed method wins. Resource limits may leave a candidate unresolved rather than establish convergence.

## OHSVGP budget must count data exposure

Let N be the number of legal initial observation rows and b=min(1024,N). Define

    s0 = max(500, ceil(2*N/b))

and evaluate multipliers 1, 3 and 9. These correspond to at least 2, 6 and 18 *expected row exposures*, plus a minimum number of optimization steps. The current sampler draws independently without replacement within each minibatch, but does not traverse every observation exactly once per epoch; do not call these complete epochs.

For the final PEMS initial fit, N=4032×260=1,048,320. Two optimizer updates with b=1024 expose only about 0.002 expected passes. The proposed final s0 is 2048 updates, followed by 6144 and 18432. This is why applying the same raw step count as a full-data optimizer is not a defensible effort comparison. For ERA5's final fit, s0=525. For COVID the minimum 500 dominates. These are proposed finite budgets, not evidence that either method has converged.

**Current implementation limitation:** `run_task_stream.py` requires the final configuration hash to equal the selected configuration hash exactly. Changing `initial_iterations` between selection and refit currently invalidates that binding. Before adopting exposure-scaled OH refits, add an explicit hashed budget policy (minimum steps, expected-exposure multiplier, batch size, rounding rule), resolve its effective step count from the legal stage-specific initial rows, and record the resolved count and data identity. Do not silently override the selected integer. Until that policy exists, either retain the identical selected integer with its differing exposure disclosed, or leave this proposal unimplemented; the latter is preferable to falsely claiming exposure-matched refitting.

## Online optimization and validation cost

KronHiPPO, ST-SVGP and MGPVAE use their current algebraic/filter updates, with no artificial common optimizer count. For the two adaptive baselines only, propose online steps **1, 5 and 10**. OH applies this per declared row microbatch; OSGPR applies it per newly released batch. Record both effective updates and assimilated observations per task. Select on the complete initial-period validation stream with the same release boundary as the final experiment.

To keep the search small, first compare the two capacities and initial-fit rungs at online_steps=5. Then compare online_steps=1 and 10 from the selected initial fitted state. Restore that identical fitted state before every validation replay. This staged search is a restricted search, not the optimum of a joint capacity/budget grid. If continuation/checkpoint restoration is not implemented and qualified, rerun deterministically and include the duplicated fitting cost; do not claim checkpoint reuse. Final fit budgets and online settings must be fixed before final labels are scored.

## Optimization evidence and language

Log per-observation objective, all objective components, finite-gradient checks, parameter/clamp diagnostics, effective rows processed, runtime and memory. Full-data GP traces can use successive 20-step windows; OH and MGP traces need window averages/medians because their objectives include MC noise (and OH also samples rows). A tiny single-step loss change is not a convergence test. Diagnostic re-evaluation of stochastic objectives should use the same declared evaluation RNG keys and MC budget across checkpoints; it must not silently increase the training estimator's budget.

At each budget rung, store the entire validation score and paired per-task differences from the previous rung. A possible predeclared *budget-stability diagnostic* is less than 0.01 nats/query absolute NLPD change between the last two rungs, alongside less than 1e-3 relative deterministic-objective window change where meaningful. These are operational thresholds, not significance tests or a proof of an optimum. Stochastic losses need a noise-aware assessment; no fixed relative-loss threshold alone establishes their convergence. If the last rung still materially improves validation or the objective, apply the single declared extension or report optimization as unresolved. Do not automatically admit underqualified candidates to an accuracy-superiority table. A finite-compute comparison is possible only if framed and reported explicitly as such.

Use wording such as “selected within the predeclared validation budget” or “budget-stability check passed,” not “converged” merely because the loop terminated. Preserve all failures and budget-limited runs in W&B/Obsidian.

## Three-A30 execution plan

First measure compilation separately and steady-state step/task costs for the base capacities using initial data only. Estimate the cumulative ladder, both capacities, validation replays and final refits from those measurements before launching the campaign. The already completed two-step pilots are useful for resource failures but often too short to estimate steady-state training throughput.

Run at most one training process per allocated GPU and at most three simultaneous jobs initially. Schedule independent candidates/splits through those slots; avoid concurrent GPU sharing for paper timing until interference has been measured and isolated. Validate only at declared checkpoints rather than at every step, retain local durable journals, and poll jobs infrequently. Do not infer the total duration from the number of 65 final results: selection, failed qualification, refits and ablations are additional work.
