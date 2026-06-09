# Route B Structured Joint Posterior Report

## Summary

The current Route B implementation uses `structured_joint_ssgp_transfer` as the
main non-mean-field continual method. It keeps the joint likelihood natural
precision blocks

- `R_beta_beta`
- `R_beta_u`
- `R_uu = B_temporal kron G`
- `h_beta`
- `h_u = vec_F(H_info)`

and recovers the posterior through the Schur complement plus Sylvester solves.
The core Route B formulas were not changed in the continual-learning update.

The former sparse residual variance issue is resolved: predictive variance now
uses explicit `prior_point_variance`, and synthetic data carries
`gp_prior_variance`. Unit and non-unit kernel amplitude checks pass for
amplitudes `2.0` and `0.5`.

## Evaluation Modes

The experiment runner supports both

```bash
--eval-mode current
--eval-mode seen_history
--eval-mode future
--eval-mode all
```

and the backward-compatible form

```bash
--eval-modes current seen_history future
```

Internal names are `current`, `seen_history`, and `future`.

- `current`: after training block `n`, evaluate on block `n`.
- `seen_history`: after training block `n`, evaluate on all seen blocks
  `1,...,n`.
- `future`: after training block `n`, evaluate on block `n+1` when available.

For continual-learning claims, `seen_history` is the primary evaluation mode.
`current` is a within-block fit diagnostic, and `future` is exploratory.

All CSV/JSON rows include `regime_name` and `eval_mode`, plus predictive
variance decomposition columns:

- `avg_predictive_variance`
- `avg_interval_width90`
- `avg_sigma2`
- `avg_nu_star`
- `avg_u_posterior_term`
- `avg_beta_schur_term`

## Forgetting Score

For a metric `M`, the seen-history forgetting score at online block `n` is

```text
F_n(M) = average over j < n of [M_after_training_n_on_block_j - M_after_training_j_on_block_j]
```

Implemented metrics:

- `rmse_forgetting`
- `nll_forgetting`

Lower is better because RMSE and NLL are minimized. Missing per-block baselines
are skipped.

## Synthetic Regimes

The runner now supports

```bash
--synthetic-regime standard
--synthetic-regime long_memory
--synthetic-regime sparse_current
--synthetic-regime old_region_retention
--synthetic-regime all
```

Definitions:

- `standard`: existing synthetic generator.
- `long_memory`: longer temporal residual correlation, currently `ell_t=0.8`
  with the same spatial lengthscale family.
- `sparse_current`: trains on a fixed random subset of spatial locations per
  seed, controlled by `--missing-rate`; evaluation remains on the full spatial
  grid. The fixed subset preserves a single spatial Gram factor across the run.
- `old_region_retention`: diagnostic left-old-region evaluation. To avoid
  violating the structured Kronecker assumption that `G=C^T C` is fixed within
  a run, training uses the fixed full spatial factor and `seen_history`
  evaluation is restricted to the left spatial region. A true left-to-right
  changing observation-mask schedule is still a TODO because it needs
  block-specific spatial factors or region-specific states.

## Main Continual Results

Commands produced outputs under:

- `results/experiments_routeB_standard_confirmatory_all/`
- `results/experiments_routeB_long_memory_ablation/`
- `results/experiments_routeB_continual_standard/`
- `results/experiments_routeB_continual_long_memory/`
- `results/experiments_routeB_continual_sparse_current/`
- `results/experiments_routeB_continual_old_region/`
- `results/experiments_routeB_continual_sweep/`
- `results/experiments_routeB_calibration_sweep_rerun_current_routeB/`
- `results/routeB_experiment_report/`

### Calibration Diagnostics Rerun

This rerun uses the current Route B implementation with the same diagnostic
setup as the calibration/noise sweep: synthetic standard data, `num_time=20`,
`num_space=6`, `block_size=5`, `M_t=5`, `M_s=4`, two seeds,
`linear_dim=2`, noise in `{0.03, 0.05, 0.08, 0.10}`, beta-u coupling in
`{weak, medium, strong}`, and `eval_mode=all`. The reported slice below is
the strong beta-u coupling case. `ell_t` fitting is disabled, so this remains
a fixed `model_ell_t=0.25` diagnostic rather than the later initial-task
full-GP MLL protocol.

Strong coupling, current block:

| Noise | Method | RMSE | NLL | Cov90 | Avg var | Width90 | beta/Schur |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.03 | mean-field | 0.0563 | -1.2373 | 0.8375 | 0.0063 | 0.2179 | 0.000031 |
| 0.03 | Route B | 0.0496 | -1.3472 | 0.8833 | 0.0063 | 0.2174 | 0.000020 |
| 0.05 | mean-field | 0.0631 | -1.2752 | 0.9250 | 0.0084 | 0.2765 | 0.000086 |
| 0.05 | Route B | 0.0588 | -1.3135 | 0.9250 | 0.0084 | 0.2754 | 0.000040 |
| 0.08 | mean-field | 0.0788 | -1.0363 | 0.9667 | 0.0135 | 0.3696 | 0.000219 |
| 0.08 | Route B | 0.0764 | -1.0513 | 0.9708 | 0.0133 | 0.3673 | 0.000069 |
| 0.10 | mean-field | 0.0917 | -0.8751 | 0.9708 | 0.0181 | 0.4343 | 0.000343 |
| 0.10 | Route B | 0.0901 | -0.8856 | 0.9708 | 0.0178 | 0.4310 | 0.000088 |

Strong coupling, seen history:

| Noise | Method | RMSE | NLL | Cov90 | Avg var | Width90 | beta/Schur |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.03 | mean-field | 0.0889 | -0.9555 | 0.9042 | 0.1100 | 0.6526 | 0.000032 |
| 0.03 | Route B | 0.0551 | -1.1237 | 0.9566 | 0.1100 | 0.6521 | 0.000032 |
| 0.05 | mean-field | 0.0943 | -0.9005 | 0.9632 | 0.1121 | 0.6964 | 0.000087 |
| 0.05 | Route B | 0.0649 | -0.9755 | 0.9760 | 0.1121 | 0.6953 | 0.000070 |
| 0.08 | mean-field | 0.1077 | -0.6988 | 0.9847 | 0.1170 | 0.7686 | 0.000223 |
| 0.08 | Route B | 0.0831 | -0.7396 | 0.9899 | 0.1169 | 0.7665 | 0.000133 |
| 0.10 | mean-field | 0.1188 | -0.5711 | 0.9844 | 0.1215 | 0.8201 | 0.000348 |
| 0.10 | Route B | 0.0968 | -0.6036 | 0.9899 | 0.1213 | 0.8172 | 0.000176 |

In this current rerun, Route B improves RMSE and NLL over mean-field throughout
the strong-coupling current and seen-history slices. Coverage is equal or
higher for Route B, and the beta/Schur term remains smaller in current mode,
which is consistent with the sharper structured posterior from preserving
beta-u covariance.

### Standard Confirmatory Experiment

The standard confirmatory section is now reported as a mechanism diagnostic
rather than a single fixed setting. The experiment uses the standard synthetic
regime with `num_time=100`, `num_space=6`, `M_s=4`, observation noise `0.08`,
and `eval_modes=current seen_history`. The primary continual-learning mode is
`seen_history`.

Diagnostic grid:

- `M_t in {5, 8, 12, 16}`
- `block_size in {5, 10}`
- `gp_signal_strength in {0.5, 1.0, 1.5}`
- `beta_u_correlation_design in {weak, medium, strong}`
- `hyperfit_mode in {ell_only, noise_kernel}`

The `noise_kernel` protocol uses initial-task full-GP MLL to fit `ell_t`,
observation noise, and kernel variance, then freezes those values for all
later online tasks and all methods. This gives 144 configurations. The table
reports mean plus standard error over configurations from the one-seed
diagnostic scan, so it is a mechanism map rather than a final significance
test.

| Method | RMSE | NLL | Coverage90 | RMSE forgetting | NLL forgetting |
|---|---:|---:|---:|---:|---:|
| `mean_field_ssgp_transfer` | 0.1454 +/- 0.0039 | -0.4273 +/- 0.0238 | 0.9261 +/- 0.0018 | 0.0548 +/- 0.0032 | 0.5897 +/- 0.0164 |
| `structured_joint_ssgp_transfer` | 0.1788 +/- 0.0083 | -0.5129 +/- 0.0221 | 0.9486 +/- 0.0019 | 0.0794 +/- 0.0060 | 0.5445 +/- 0.0161 |

After fixing the NaN forgetting aggregation, `44 / 144` standard diagnostic
configurations satisfy all five Route B advantages over mean-field:

- lower RMSE;
- lower NLL;
- higher 90% coverage;
- lower RMSE forgetting;
- lower NLL forgetting.

All-metric-win distribution:

| Factor | Count |
|---|---:|
| `block_size=10` | 30 / 44 |
| `block_size=5` | 14 / 44 |
| `coupling=medium` | 26 / 44 |
| `coupling=strong` | 14 / 44 |
| `coupling=weak` | 4 / 44 |
| `hyperfit=noise_kernel` | 27 / 44 |
| `hyperfit=ell_only` | 17 / 44 |

The most stable mechanism conditions are:

| M_t | block_size | gp_signal | coupling | hyperfit |
|---|---:|---:|---|---|
| 5/8/12/16 | 10 | 1.0 | medium | ell_only or noise_kernel |
| 5/8/12/16 | 10 | 1.0 | strong | noise_kernel |
| 5/8/12/16 | 10 | 0.5 | medium | ell_only or noise_kernel |
| 5/8/12/16 | 10 | 0.5 | strong | noise_kernel, mainly M_t >= 12 |
| 5/8 | 10 | 1.5 | medium | ell_only, partly noise_kernel |
| 5 | 10 | 1.5 | weak/strong | noise_kernel |

Representative strong Route B wins:

| M_t | block | gp | coupling | fit | Route B RMSE | MF RMSE | Route B NLL | MF NLL |
|---:|---:|---:|---|---|---:|---:|---:|---:|
| 5 | 10 | 1.0 | medium | ell_only | 0.1078 | 0.1207 | -0.6819 | -0.5315 |
| 8 | 10 | 1.0 | medium | noise_kernel | 0.1068 | 0.1267 | -0.5624 | -0.4712 |
| 12 | 10 | 1.0 | strong | noise_kernel | 0.1121 | 0.1246 | -0.5504 | -0.4819 |
| 5 | 10 | 1.5 | medium | noise_kernel | 0.1321 | 0.1662 | -0.3790 | -0.1946 |
| 16 | 10 | 0.5 | medium | noise_kernel | 0.0851 | 0.0911 | -0.8403 | -0.7961 |

Interpretation: Route B is most consistently beneficial when beta-u coupling is
non-negligible and the online block is less myopic, especially with
`block_size=10` and medium/strong coupling. Full-GP MLL fitting of noise and
kernel variance further increases the frequency of all-metric wins. GP residual
signal strength does not need to be maximal: `gp_signal=1.0` and `0.5` are more
stable than `1.5` in this scan. Weak-coupling standard settings remain
mean-field-favorable or mixed. Simply increasing `M_t` is not sufficient; the
all-metric win count is largest at `M_t=5` in this one-seed diagnostic, so the
standard failure mode is not only temporal basis capacity.

### Earlier Short Continual Runs

Seen-history summary, averaged over seeds:

| Regime | Method | RMSE | NLL | Coverage90 | RMSE forgetting | NLL forgetting | Avg pred var | Avg width90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| standard | `mean_field_ssgp_transfer` | 0.162407 | -0.267239 | 0.903795 | 0.065811 | 0.728039 | 0.169530 | 0.953485 |
| standard | `no_transfer` | 0.325277 | 0.064274 | 0.942753 | 0.271781 | 1.298954 | 0.300255 | 1.402521 |
| standard | `structured_joint_ssgp_transfer` | 0.137435 | -0.503752 | 0.973957 | 0.056213 | 0.687057 | 0.169631 | 0.951632 |
| long_memory | `mean_field_ssgp_transfer` | 0.147727 | -0.276239 | 0.967452 | 0.055681 | 0.726788 | 0.237310 | 1.184841 |
| long_memory | `no_transfer` | 0.204724 | 0.017785 | 0.981798 | 0.129143 | 1.201334 | 0.343247 | 1.554945 |
| long_memory | `structured_joint_ssgp_transfer` | 0.298554 | -0.250203 | 0.963593 | 0.181582 | 0.860054 | 0.238830 | 1.187555 |
| sparse_current | `mean_field_ssgp_transfer` | 0.319807 | 0.029926 | 0.893175 | 0.190986 | 0.741732 | 0.235152 | 1.157587 |
| sparse_current | `no_transfer` | 0.313582 | 0.120486 | 0.952471 | 0.220952 | 1.292535 | 0.344544 | 1.548920 |
| sparse_current | `structured_joint_ssgp_transfer` | 0.333795 | -0.112374 | 0.910926 | 0.204364 | 0.853223 | 0.242662 | 1.171995 |
| old_region_retention | `mean_field_ssgp_transfer` | 0.506067 | 0.358199 | 0.814874 | 0.368173 | 1.318156 | 0.233423 | 1.141379 |
| old_region_retention | `no_transfer` | 0.253473 | -0.006102 | 0.964623 | 0.181954 | 1.347053 | 0.332204 | 1.491683 |
| old_region_retention | `structured_joint_ssgp_transfer` | 0.365615 | -0.112421 | 0.902763 | 0.248967 | 0.987656 | 0.239641 | 1.152521 |

Interpretation of the earlier short runs:

- On the shorter `standard` run, Route B is strongest on seen-history RMSE, NLL,
  and both forgetting scores. This did not fully survive the larger confirmatory
  run above.
- On `long_memory`, Route B improves NLL over `no_transfer`, but RMSE is worse
  and mean-field has the best RMSE/NLL. This is a negative result.
- On `sparse_current`, Route B has the best seen-history NLL, but not RMSE or
  coverage. This supports retained density information, not a blanket win.
- On `old_region_retention`, Route B has better NLL forgetting than
  `no_transfer`, but worse RMSE forgetting and coverage. Because this regime is
  currently a fixed-`G` left-region diagnostic, it should not be overclaimed as
  a full left-to-right masked-training benchmark.

### Long-Memory Ablation

The long-memory ablation grid uses `M_t in {5,8,12,16}`, `block_size in {5,10}`,
`ell_t in {0.5,0.8}`, `noise in {0.08,0.10}`, three seeds, and seen-history
evaluation with current-mode baselines for forgetting. Full numeric results are
saved in:

- `results/routeB_experiment_report/tables/long_memory_ablation_all_seen_history.csv`
- `results/routeB_experiment_report/plots/long_memory_ablation_rmse.png`
- `results/routeB_experiment_report/plots/long_memory_ablation_nll.png`
- `results/routeB_experiment_report/plots/long_memory_ablation_avg_predictive_variance.png`

This ablation is diagnostic rather than a positive claim. It tests whether
increasing temporal basis capacity improves Route B. Where larger `M_t`
improves Route B, basis capacity is implicated; where it does not, calibration
or transfer mismatch is more likely.

The synthetic linear regression basis used in these experiments is:

```text
Phi(t,s) = [1, t_scaled, s_centered, sin(2*pi*t_scaled)]
beta_true = [0.4, -0.7, 0.25, 0.15]
```

This can explain a global intercept, a linear time trend, a spatial offset, and
one smooth seasonal component. It cannot by itself represent arbitrary
long-memory residual trajectories, so the residual GP basis and temporal
projection are important. Time-dependence visualizations for one observed
location are saved at:

- `results/routeB_experiment_report/long_memory_time_dependence/long_memory_location_time_dependence.png`
- `results/routeB_experiment_report/long_memory_time_dependence_matched_ell/long_memory_location_time_dependence.png`

The first plot uses the current long-memory experiment convention:
data `ell_t=0.8`, model block-factor lengthscale `0.25`. The second is a
diagnostic with matched model lengthscale `0.8`. This makes explicit that the
long-memory behavior depends on both the linear basis and the temporal GP basis
used for residual transfer.

### Model Temporal Lengthscale Ablation

The follow-up quantitative ablation fixes `data ell_t=0.8` and sweeps:

```text
model ell_t in {0.25, 0.5, 0.8}
M_t in {5, 8, 12, 16}
block_size = 5
noise = 0.08
num_seeds = 3
eval_modes = current, seen_history
```

The previous long-memory experiments used the default model-side temporal
lengthscale `model ell_t=0.25`. The new results show that this was a major
setting mismatch.

Average over `M_t`:

| Method | model ell_t | RMSE | NLL | Coverage90 | RMSE forgetting | NLL forgetting |
|---|---:|---:|---:|---:|---:|---:|
| `no_transfer` | 0.25 | 0.2056 | 0.0196 | 0.9817 | 0.1299 | 1.2016 |
| `no_transfer` | 0.50 | 0.1861 | -0.2292 | 0.9708 | 0.1085 | 0.8987 |
| `no_transfer` | 0.80 | 0.1798 | -0.3570 | 0.9629 | 0.0986 | 0.7406 |
| `mean_field_ssgp_transfer` | 0.25 | 0.1833 | -0.2651 | 0.9657 | 0.0890 | 0.7596 |
| `mean_field_ssgp_transfer` | 0.50 | 0.0883 | -0.6948 | 0.9639 | 0.0035 | 0.3278 |
| `mean_field_ssgp_transfer` | 0.80 | 0.1779 | -0.5357 | 0.8991 | 0.0804 | 0.5021 |
| `structured_joint_ssgp_transfer` | 0.25 | 0.2505 | -0.2828 | 0.9694 | 0.1429 | 0.8179 |
| `structured_joint_ssgp_transfer` | 0.50 | 0.0893 | -0.7355 | 0.9732 | 0.0086 | 0.3421 |
| `structured_joint_ssgp_transfer` | 0.80 | 0.1395 | -0.7501 | 0.9463 | 0.0505 | 0.3230 |

Interpretation:

- Moving away from `model ell_t=0.25` dramatically improves Route B in
  long-memory.
- `model ell_t=0.8` gives the best Route B NLL and NLL forgetting, consistent
  with the true data-generating long-memory scale.
- `model ell_t=0.5` gives the best Route B RMSE and RMSE forgetting, suggesting
  that the best predictive mean calibration may prefer a slightly shorter
  inference lengthscale than the data-generating kernel under the current sparse
  basis and online transfer approximation.
- Therefore the previous long-memory failure was largely a temporal
  basis/kernel mismatch, but it is more precise to say the best model lengthscale
  must be tuned or fitted, not simply fixed to the true `data ell_t=0.8`.

First-batch fitted lengthscale was also tested with candidates `{0.25, 0.5,
0.8}`. This simple first-batch current-NLL rule selected `0.25` on the tested
seeds, so it did not recover the better long-history setting. A better
first-batch fitting criterion should use held-out points within the first batch
or a marginal-likelihood objective, rather than in-sample current NLL after
training.

Outputs:

- `results/routeB_experiment_report/tables/model_ell_ablation_seen_history.csv`
- `results/routeB_experiment_report/tables/model_ell_ablation_average_over_mt.csv`
- `results/routeB_experiment_report/tables/model_ell_first_batch_fit_seen_history.csv`
- `results/routeB_experiment_report/plots/model_ell_ablation_rmse.png`
- `results/routeB_experiment_report/plots/model_ell_ablation_nll.png`
- `results/routeB_experiment_report/plots/model_ell_ablation_coverage90.png`
- `results/routeB_experiment_report/plots/model_ell_ablation_rmse_forgetting.png`
- `results/routeB_experiment_report/plots/model_ell_ablation_nll_forgetting.png`

### Initial-Task Fitted Temporal Lengthscale

The previous short-`K` fitting experiment has been removed from the main
protocol. The experiment runner now follows the general online hyperparameter
setup: fit the temporal kernel lengthscale once on an initial calibration task,
then freeze it for all later online tasks.

```bash
--ell-t-fit-mode none|initial_task_fullgp
--initial-task-blocks INT
--initial-task-fraction FLOAT
--time-normalization none|expected_horizon|custom|initial_task
--time-scale FLOAT
--ell-t-grid-source manual|time_scale
--ell-t-grid-values ...
```

The implemented method is `initial_task_fullgp`. It evaluates an independent
batch/full-GP marginal likelihood on the initial task and selects the candidate
temporal lengthscale with the lowest initial-task marginal NLL.
The likelihood integrates out the linear coefficients under the same Gaussian
beta prior used by the online models:

```text
y_initial ~ N(0, K_t(ell_t) kron K_s + Phi S_beta Phi^T + sigma2 I).
```

No short-`K` fitting mode and no structured validation-NLL fitter is used as a
main protocol. No later online blocks or test labels are used by the selector.

After selection, `ell_t_star` is frozen and shared by:

- `no_transfer`;
- `mean_field_ssgp_transfer`;
- `structured_joint_ssgp_transfer`.

There is no per-method tuning and no online update to `ell_t`.

Time normalization is general rather than synthetic-specific:

```text
custom / expected_horizon: t_scaled = (t_raw - t0) / time_scale
initial_task:              t_scaled = (t_raw - t0) / initial_task_span
none:                      use raw time units
```

With `ell_t_grid_source=time_scale`, normalized model-time candidates are
`{0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6}`. In raw units they correspond
to those fractions of the selected time scale. Synthetic experiments may set
`time_scale=1.0` because their time axis is defined on that scale, but this is
not a synthetic-only rule. For real data, the scale should come from an
expected deployment horizon, known task duration, a user-provided unit, or a
initial-task full-GP fit. `time_normalization=initial_task` must be used
carefully because a short initial task can bias model selection toward short
memory.

New result columns are:

```text
data_ell_t, model_ell_t, selected_ell_t, fitted_ell_t, ell_t_fit_mode,
initial_task_blocks, initial_task_fraction, time_normalization, time_scale,
ell_t_grid_source, ell_t_grid, candidate_ell_t, candidate_score,
selected_candidate_score, initial_task_span, raw_time_span_if_available
```

The long-memory comparison uses exactly three settings:

| Setting | data ell_t | model ell_t protocol | Status |
|---|---:|---|---|
| mismatch | 0.8 | fixed 0.25 | retained negative diagnostic |
| oracle matched | 0.8 | fixed 0.8 | upper-bound diagnostic only |
| initial-task full-GP MLL | 0.8 | independent full-GP marginal NLL | primary non-oracle protocol |

New experiment outputs are saved under:

- `results/experiments_routeB_long_memory_initial_task_ellt_fit/`
- `results/experiments_routeB_standard_initial_task_ellt_fit/`
- `results/experiments_routeB_standard_initial_task_fullgp_ellt_fit/`

Actual fitted model lengthscales from the completed initial-task full-GP MLL
runs:

| Experiment | Seed | fitted model ell_t | selected initial-task score |
|---|---:|---:|---:|
| long_memory | 0 | 1.6 | -1.0444 |
| long_memory | 1 | 1.6 | -0.9480 |
| long_memory | 2 | 1.2 | -0.9823 |
| standard | 0 | 0.4 | -0.8339 |
| standard | 1 | 0.2 | -0.7907 |
| standard | 2 | 0.4 | -0.8755 |
| standard | 3 | 0.2 | -0.8379 |
| standard | 4 | 0.2 | -0.7415 |
| standard | 5 | 0.2 | -0.8813 |
| standard | 6 | 0.2 | -0.8204 |
| standard | 7 | 0.2 | -0.8873 |
| standard | 8 | 0.2 | -0.7895 |
| standard | 9 | 0.2 | -0.9067 |

Initial-task full-GP MLL seen-history summaries:

| Regime | Method | RMSE | NLL | Coverage90 | RMSE forgetting | NLL forgetting | fitted ell_t values |
|---|---|---:|---:|---:|---:|---:|---|
| long_memory | `mean_field_ssgp_transfer` | 0.0915 | -0.8923 | 0.9393 | 0.0049 | 0.0831 | 1.2, 1.6 |
| long_memory | `structured_joint_ssgp_transfer` | 0.0836 | -0.9626 | 0.9598 | 0.0035 | 0.0757 | 1.2, 1.6 |
| standard | `mean_field_ssgp_transfer` | 0.2569 | 0.0021 | 0.9120 | 0.1511 | 0.9032 | 0.2, 0.4 |
| standard | `structured_joint_ssgp_transfer` | 0.2946 | -0.0969 | 0.9336 | 0.1889 | 0.9415 | 0.2, 0.4 |

The full-GP MLL selector is retained as the main protocol because it is method
independent and is substantially stronger in the long-memory continual setting:
Route B has lower RMSE, lower NLL, and lower forgetting than mean-field there.
The standard confirmatory result is mixed: Route B has better NLL and coverage,
but mean-field still has lower RMSE and lower forgetting. This negative result
remains part of the report.

The existing model-lengthscale ablation table above remains unchanged; the
initial-task fitted results are reported as a new confirmatory section rather
than overwriting that diagnostic evidence.

## Calibration and Noise Sweep

The full sweep is saved in
`results/experiments_routeB_continual_sweep/`. It covers

```text
noise in {0.03, 0.05, 0.08, 0.10}
beta-u coupling in {weak, medium, strong}
eval modes = current, seen_history, future
methods = no_transfer, mean_field_ssgp_transfer, structured_joint_ssgp_transfer
```

Noise `0.08` is the recommended balanced synthetic setting for the current
experiments. It avoids the worst small-noise undercoverage while preserving a
nontrivial calibration stress.

The earlier densecheck conclusion still holds, but its interpretation must be
precise: Route B is sharper than mean-field because it preserves beta-u
covariance. This often improves RMSE/NLL but can reduce current-block coverage
at small noise. That behavior is not attributable to the old unit-amplitude bug.

## Validation Taxonomy

The validation evidence is separated into three categories.

1. **Sylvester solver numerical validation.** The reference is the dense
   numerical solve of the same linear system, `D_u z = q`. This validates the
   structured solver only; mean-field is not a solver baseline and should not be
   included in this table.
2. **Dense finite-dimensional posterior validation.** The reference is
   `Lambda_dense^{-1} h_dense`, the exact posterior of the same
   finite-dimensional Gaussian approximation. This is not the unknown
   data-generating truth.
3. **GP-generative synthetic prediction and calibration.** The synthetic
   generator samples `f ~ GP(0, k_t kron k_s)` on the full observed grid and
   then forms `y = Phi beta + f + epsilon`. Prediction claims should therefore
   use RMSE, NLL, coverage, ECE, and forgetting. `m_u` error is not a natural
   ground-truth metric in this setting because `u` is an inducing
   representation.

## Dense Finite-Dimensional Posterior Diagnostic

Dense small-problem diagnostic. The dense posterior below is the exact posterior
of the same fixed finite-dimensional Gaussian approximation, not a true GP
data-generating posterior:

| Quantity | Route B error | Mean-field error |
|---|---:|---:|
| m_beta error | 1.24321e-16 | 0.247243 |
| m_u error | 2.90855e-16 | 0.0992548 |
| S_beta_beta error | 2.41320e-16 | 0.0838744 |
| S_beta_u error | 1.34484e-16 | 0.291301 |
| predictive variance error | 0 | 0.282273 |
| cross covariance norm | 1.34484e-16 | 0.291301 |
| beta-u cross block norm | 0 | 1.59232 |

This verifies the algebraic contribution: Route B matches the dense
finite-dimensional joint posterior, including nonzero `S_beta_u`, while
mean-field sets that block to zero and differs when coupling is nonzero. It
should not be described as direct evidence that Route B is closer to the
unknown true GP posterior under arbitrary data generation.

## Plots

Generated plot directories:

- `results/experiments_routeB_continual_standard/plots/`
- `results/experiments_routeB_continual_long_memory/plots/`
- `results/experiments_routeB_continual_sparse_current/plots/`
- `results/experiments_routeB_continual_old_region/plots/`
- `results/experiments_routeB_continual_sweep/plots/`

Key plot files include:

- `seen_history_rmse_vs_noise.png`
- `seen_history_nll_vs_noise.png`
- `current_coverage_vs_noise.png`
- `seen_history_coverage_vs_noise.png`
- `current_beta_schur_vs_noise.png`
- `forgetting_rmse_over_blocks.png`
- `forgetting_nll_over_blocks.png`
- `rmse_over_blocks.png`
- `nll_over_blocks.png`

Coverage-vs-noise plots include a horizontal 0.9 line.

## Verification

Commands run:

```bash
uv run --no-sync pytest -q
uv run --no-sync python scripts/verify_joint_ssgp_kron_derivations.py
```

Results:

```text
34 passed
routeB_all_passed: true
```

Verification outputs:

- `results/verification/joint_ssgp_kron_verification.json`
- `results/verification/routeB_joint_ssgp_kron_verification.json`

## Baselines and ERA5

`projected_prior` is an old dense ablation. It does not preserve Route B
beta-u cross covariance and should not be presented as an equally principled
Route B variant. It is retained only as a diagnostic ablation.

ERA5 remains a lightweight probe. Do not claim Route B superiority on ERA5
unless the results show it. If ERA5 reporting is extended, include selected
`num_time`, `num_space`, `block_size`, `noise`, eval mode, standardization
status, and whether kernel variance is explicit.

## TODOs

- Implement a theoretically clean old-region benchmark with either
  block-specific spatial factors or separate region-specific states.
- Study calibration with learned noise or likelihood tempering instead of
  changing the Route B posterior formulas.
- Add a stable changing-basis dense oracle if future-mode claims are needed.
- Keep negative current/future and coverage results visible in paper-style
  reporting.
