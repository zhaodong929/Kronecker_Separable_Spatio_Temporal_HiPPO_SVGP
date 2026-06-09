# Joint SSGP Kronecker Test Report

## Test Commands

```bash
uv run --no-sync pytest -q
```

Latest result:

```text
32 passed in 4.26s
```

```bash
uv run --no-sync python scripts/verify_joint_ssgp_kron_derivations.py
```

Latest result:

```text
all_passed: true
routeB_all_passed: true
```

## Route B Tests Added

- Dense vs structured Gaussian block likelihood statistics.
- Dense vs structured joint old-likelihood transfer.
- Schur posterior recovery vs dense inverse.
- Structured predictive variance vs dense joint posterior.
- Predictive variance respects non-unit kernel amplitude.
- Mean-field variance differs under nonzero beta-u coupling.
- Fixed-basis streaming equals batch joint posterior.
- No-linear-mean GP-only reduction.
- Zero cross-feature sanity.

## Verification Outputs

- `results/verification/joint_ssgp_kron_verification.json`
- `results/verification/routeB_joint_ssgp_kron_verification.json`

The Route B JSON includes:

- `routeB_dense_vs_structured_likelihood`
- `routeB_joint_transfer_dense_vs_structured`
- `routeB_schur_mean_vs_dense`
- `routeB_schur_covariance_vs_dense`
- `routeB_predictive_variance_vs_dense`
- `routeB_predictive_variance_respects_kernel_amplitude`
- `routeB_fixed_basis_streaming_vs_batch`
- `routeB_gp_only_reduction`
- `routeB_cross_block_transfer`

## Experiment Outputs

Synthetic Route B:

- `results/experiments_routeB/joint_ssgp_kron_synthetic_report.json`
- `results/experiments_routeB/joint_ssgp_kron_synthetic_metrics.csv`
- `results/experiments_routeB/rmse_over_blocks.png`
- `results/experiments_routeB/nll_over_blocks.png`
- `results/experiments_routeB/coverage_plot.png`

Densecheck Route B:

- `results/experiments_routeB_densecheck/joint_ssgp_kron_synthetic_report.json`
- `results/experiments_routeB_densecheck/joint_ssgp_kron_synthetic_metrics.csv`

ERA5 Route B probe:

- `results/experiments_era5_routeB_probe/joint_ssgp_kron_era5_report.json`
- `results/experiments_era5_routeB_probe/joint_ssgp_kron_era5_metrics.csv`

Calibration diagnostics sweep:

- `results/experiments_routeB_calibration_sweep/joint_ssgp_kron_synthetic_report.json`
- `results/experiments_routeB_calibration_sweep/joint_ssgp_kron_synthetic_metrics.csv`

## Main Numerical Results

Synthetic weak-correlation regime:

| Method | RMSE | NLL | 90% coverage |
|---|---:|---:|---:|
| `no_transfer` | 0.082844 | -1.014475 | 0.858333 |
| `projected_prior` | 0.083927 | -0.961574 | 0.815278 |
| `ssgp_transfer` | 0.096103 | -0.707832 | 0.768056 |
| `structured_joint_ssgp_transfer` | 0.063279 | -1.123844 | 0.911111 |

Densecheck strong-correlation regime:

| Method | RMSE | NLL | 90% coverage |
|---|---:|---:|---:|
| `dense_reference_fixed_basis` | 0.173733 | 4.791053 | 0.443750 |
| `mean_field_ssgp_transfer` | 0.162765 | 1.508472 | 0.762500 |
| `structured_joint_ssgp_transfer` | 0.160084 | 1.497752 | 0.712500 |

ERA5 probe:

| Method | RMSE | NLL | 90% coverage |
|---|---:|---:|---:|
| `no_transfer` | 0.162079 | 3.521761 | 0.773333 |
| `projected_prior` | 0.377459 | 15.558729 | 0.546667 |
| `ssgp_transfer` | 0.247203 | 8.824390 | 0.626667 |
| `structured_joint_ssgp_transfer` | 0.245417 | 8.865501 | 0.593333 |

## Interpretation

Route B is now verified algebraically against dense references. On the main
synthetic run it improves RMSE, NLL, and coverage. On the densecheck stress run,
it improves RMSE/NLL slightly relative to mean-field but has lower 90% coverage.
On the lightweight ERA5 probe, `no_transfer` remains strongest; Route B should
not yet be claimed as an ERA5 improvement.

Calibration diagnostics show that increasing noise from `0.03` to `0.10`
raises coverage for both Route B and mean-field. In strong-coupling current-block
evaluation, Route B keeps slightly better RMSE/NLL than mean-field but its
beta/Schur variance term is smaller, confirming that retained beta-u covariance
sharpens uncertainty. Seen-history evaluation is better calibrated and Route B
is generally stronger there.
