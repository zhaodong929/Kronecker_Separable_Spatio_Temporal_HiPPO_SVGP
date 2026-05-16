# Joint SSGP Kronecker Implementation Report

This report summarizes the verification and experiment results for the new
implementation requested in `stvgp_kronecker/codex具体实现.txt`.

## Scope Implemented

The new implementation is isolated from the original baseline code. The baseline
training files were not rewritten:

- `stvgp_kronecker/st_model_batch.py`
- `stvgp_kronecker/st_model_online.py`
- `stvgp_kronecker/train_batch.py`
- `stvgp_kronecker/train_online.py`
- `stvgp_kronecker/train_online_joint.py`

New modules and scripts implement:

- joint online Gaussian process with linear mean model,
- SSGP-style old-likelihood-ratio transfer,
- Kronecker-preserving temporal HiPPO changing-basis update,
- Sylvester-compatible structured posterior solve,
- projected-prior ablation,
- no-transfer ablation,
- synthetic and ERA5 processed-data experiment runners.

## Files Added

- `stvgp_kronecker/joint_ssgp_kron/__init__.py`
- `stvgp_kronecker/joint_ssgp_kron/kron_utils.py`
- `stvgp_kronecker/joint_ssgp_kron/ssgp_transfer.py`
- `stvgp_kronecker/joint_ssgp_kron/structured_state.py`
- `stvgp_kronecker/joint_ssgp_kron/model.py`
- `stvgp_kronecker/joint_ssgp_kron/synthetic.py`
- `scripts/run_joint_ssgp_kron.py`
- `scripts/verify_joint_ssgp_kron_derivations.py`
- `scripts/run_joint_ssgp_kron_experiments.py`
- `tests/conftest.py`
- `tests/test_joint_ssgp_kron_derivations.py`
- `tests/test_joint_ssgp_kron_model.py`
- `docs/joint_ssgp_kron_readme.md`
- `docs/joint_ssgp_kron_test_report.md`

## Core Formulas Verified

The verification script checks the key formulas required by the implementation
instructions:

```text
L_on = K_on K_nn^{-1} = L_t kron I_s
R_o = B_o kron G
Lambda_old = (L_t^T B_o L_t) kron G
B_n = L_t^T B_o L_t + T_n^T T_n / sigma2
H_n = H_o L_t + C^T residual T_n / sigma2
Ks^{-1} M Kt^{-1} + G M B_n = H_n
```

## Verification Command

```bash
uv run --no-sync python scripts/verify_joint_ssgp_kron_derivations.py
```

Output file:

```text
results/verification/joint_ssgp_kron_verification.json
```

Result:

```text
all_passed: true
```

Detailed checks:

| Check | Result | Metric |
|---|---:|---:|
| Dense vs Kronecker transfer operator `L_on` | pass | 1.0436317585156084e-15 |
| Dense vs Kronecker old-likelihood transfer | pass | 7.617627280478494e-15 |
| Fixed-basis streaming mean vs batch mean | pass | 4.2096827333209885e-16 |
| Fixed-basis streaming precision vs batch precision | pass | 1.9445565319689927e-16 |
| No linear mean reduces to GP-only update | pass | exact tolerance check |
| No old data transfer vanishes | pass | `B_norm=85.68847908207124`, `H_norm=70.57513102414208` after new data |
| Projected-prior dense marginalization | pass | 0.0 |
| Structured old-likelihood information transfer | pass | 2.38273893514245e-15 |
| Synthetic feasibility check | pass | `RMSE=0.31233378498298625`, `NLL=6.014490610911911` |

## Unit Tests

Command:

```bash
uv run --no-sync pytest -q
```

Result:

```text
24 passed in 3.09s
```

The test suite includes:

- original baseline tests,
- derivation tests for the new Kronecker transfer identities,
- no-linear-mean and no-old-data checks,
- projected-prior dense marginalization check,
- one-block and multi-block model no-NaN tests,
- baseline import compatibility checks.

## Synthetic Experiment

Command:

```bash
uv run --no-sync python scripts/run_joint_ssgp_kron_experiments.py \
  --dataset synthetic \
  --num-seeds 3 \
  --num-time 40 \
  --num-space 6 \
  --block-size 5 \
  --mt 5 \
  --ms 4 \
  --noise 0.05 \
  --methods no_transfer projected_prior ssgp_transfer \
  --outdir results/experiments
```

Output files:

- `results/experiments/joint_ssgp_kron_synthetic_metrics.csv`
- `results/experiments/joint_ssgp_kron_synthetic_report.json`
- `results/experiments/coverage_plot.png`
- `results/experiments/rmse_over_blocks.png`
- `results/experiments/nll_over_blocks.png`

Mean metrics:

| Method | RMSE | MAE | NLL | 90% Coverage | Runtime/block sec |
|---|---:|---:|---:|---:|---:|
| `no_transfer` | 0.08363039123072417 | 0.07167108163841356 | -1.0123396624681873 | 0.8652777777777777 | 0.00035817687512462726 |
| `projected_prior` | 0.08059519187927497 | 0.06841531617252826 | -1.0181966814866525 | 0.85 | 0.0005020620416568514 |
| `ssgp_transfer` | 0.0963736536810708 | 0.08095406496202222 | -0.725893548939815 | 0.7833333333333333 | 0.0003295214165367118 |

Interpretation:

- All methods ran without NaN/Inf.
- All reported RMSE/NLL values are finite.
- 90% coverage is within the broad sanity range requested in the implementation
  instructions for all methods.
- On this small synthetic run, projected-prior had the best average RMSE/NLL,
  while SSGP transfer remained numerically stable and competitive enough for
  the requested feasibility check.

## ERA5 Processed-Data Probe

Command:

```bash
uv run --no-sync python scripts/run_joint_ssgp_kron_experiments.py \
  --dataset era5 \
  --num-seeds 1 \
  --num-time 40 \
  --num-space 6 \
  --block-size 5 \
  --mt 5 \
  --ms 4 \
  --noise 0.05 \
  --methods no_transfer projected_prior ssgp_transfer \
  --outdir results/experiments_era5_probe
```

Output files:

- `results/experiments_era5_probe/joint_ssgp_kron_era5_metrics.csv`
- `results/experiments_era5_probe/joint_ssgp_kron_era5_report.json`
- `results/experiments_era5_probe/coverage_plot.png`
- `results/experiments_era5_probe/rmse_over_blocks.png`
- `results/experiments_era5_probe/nll_over_blocks.png`

Mean metrics:

| Method | RMSE | MAE | NLL | 90% Coverage | Runtime/block sec |
|---|---:|---:|---:|---:|---:|
| `no_transfer` | 0.16207872827054987 | 0.1311438460984561 | 3.52176071427005 | 0.7733333333333333 | 0.0004925272005493753 |
| `projected_prior` | 0.37745938907777454 | 0.3126892096815112 | 15.55872927401004 | 0.5466666666666666 | 0.0003846848001558101 |
| `ssgp_transfer` | 0.24720343799670225 | 0.19960661973486887 | 8.8243904929984 | 0.6266666666666667 | 0.0003968732002249453 |

Interpretation:

- The new ERA5 processed-data path runs end to end on local `.npz` data.
- The loader aligns common timestamps across selected scaled ERA5 location files.
- This is a lightweight probe for the new method, not a replacement for the
  existing baseline ERA5 training scripts.
- On this small probe, `no_transfer` is strongest; `ssgp_transfer` is stable but
  not yet better than the simpler baseline. This suggests the next work should
  tune temporal basis construction, noise scale, and ERA5 feature design.

## Issues Found and Fixed During Implementation

1. Import path failure under `uv run --no-sync`:

```text
ModuleNotFoundError: No module named 'stvgp_kronecker'
ModuleNotFoundError: No module named 'scripts'
```

Fix:

- Added path bootstrap in scripts.
- Added `tests/conftest.py`.

2. Initial algebra checks failed at about `1e-5` because dense Kronecker inverse
and temporal-only solve used inconsistent jitter levels.

Fix:

- Algebra identity tests now use exact no-jitter Cholesky solves on already SPD
test matrices.

3. Initial ERA5 probe failed with:

```text
IndexError: index -1 is out of bounds for axis 0 with size 0
```

Cause:

- Some selected ERA5 files had fewer aligned common timestamps than the requested
`--num-time`.

Fix:

- The experiment runner now slices blocks using the actual loaded time length.

## Assumptions and Limitations

- Main scalable path does not compute `R_o = S_o^{-1} - K_oo^{-1}` from dense
  posterior covariance. It maintains `B_temporal` and `H_info` directly.
- Dense covariance materialization is only used in tests and projected-prior
  ablation.
- Synthetic experiments use RBF temporal inducing matrices rather than exact
  HiPPO-RFF integrals.
- ERA5 probe uses locally processed scaled `.npz` files and common timestamp
  alignment.
- If spatial observation pattern or spatial inducing locations change online,
  the single `B_o kron G` old-likelihood representation may no longer apply.
