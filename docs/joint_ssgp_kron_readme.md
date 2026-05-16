# Joint SSGP Kronecker HiPPO Implementation

This directory documents the new NumPy/SciPy CPU implementation added under
`stvgp_kronecker/joint_ssgp_kron/`. The baseline PyTorch training files are
left intact; the new code is isolated in new modules, scripts, and tests.

## Added Components

- `stvgp_kronecker/joint_ssgp_kron/kron_utils.py`: Kronecker products, dense
  test adapters, SPD solves, and the Sylvester-compatible precision solver.
- `stvgp_kronecker/joint_ssgp_kron/ssgp_transfer.py`: SSGP old-likelihood-ratio
  transfer and projected-prior ablation formulas.
- `stvgp_kronecker/joint_ssgp_kron/structured_state.py`: structured posterior
  state with `B_temporal` and `H_info`.
- `stvgp_kronecker/joint_ssgp_kron/model.py`: joint linear-mean model with
  SSGP transfer, no-transfer, projected-prior, prediction, and dense test hooks.
- `stvgp_kronecker/joint_ssgp_kron/synthetic.py`: synthetic separable GP data
  and consistent temporal/spatial projection factors.
- `scripts/verify_joint_ssgp_kron_derivations.py`: derivation checks and JSON
  report writer.
- `scripts/run_joint_ssgp_kron_experiments.py`: synthetic experiment runner.
- `tests/test_joint_ssgp_kron_*.py`: unit tests for formulas and model sanity.

## Main Formulas

With fixed spatial inducing locations and a temporal-only basis change,

```text
L_on = K_on K_nn^{-1} = L_t kron I_s
L_t  = K_on^(t) (K_nn^(t))^{-1}
```

The scalable implementation maintains old likelihood natural precision as

```text
R_o = B_o kron G
G   = C^T C
```

so the transferred old likelihood precision is

```text
Lambda_old = (L_t^T B_o L_t) kron G
```

For a new block with temporal projection `T_n`,

```text
B_n = L_t^T B_o L_t + T_n^T T_n / sigma2
H_n = H_o L_t + C^T residual T_n / sigma2
```

The posterior mean matrix solves

```text
Ks^{-1} M Kt^{-1} + G M B_n = H_n
```

without materializing the full Kronecker precision.

## Transfer Variants

- **SSGP-style old-likelihood-ratio transfer** is the default method. It carries
  `B_temporal` and `H_info` through the temporal changing basis.
- **Gaussian projected-prior transfer** is an ablation. It maps posterior moments
  by dense Gaussian marginalization and is intended only for small/medium tests.
- **No transfer** resets the GP old likelihood contribution between changing
  bases while still allowing the linear mean posterior to continue.

## Run Verification

```bash
uv run --no-sync python scripts/verify_joint_ssgp_kron_derivations.py
```

The script writes:

```text
results/verification/joint_ssgp_kron_verification.json
```

## Run Synthetic Experiments

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

Outputs:

```text
results/experiments/joint_ssgp_kron_synthetic_metrics.csv
results/experiments/joint_ssgp_kron_synthetic_report.json
results/experiments/coverage_plot.png
results/experiments/rmse_over_blocks.png
results/experiments/nll_over_blocks.png
```

## Run ERA5 Processed-Data Probe

The same script can stream the local processed ERA5 `.npz` files without using
the baseline training entry points:

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

This loader aligns common timestamps across selected scaled ERA5 location files.
It is a lightweight reproduction/probe for the new method, not a replacement for
the existing baseline ERA5 training scripts.

## Known Limitations

- If the spatial observation pattern changes per block, `G_j` changes and the
  old likelihood becomes a sum of Kronecker products, not one `B_o kron G`.
- If spatial inducing locations move online, `L_on` need not equal
  `L_t kron I_s`.
- If a dense unrestricted covariance is used and one computes
  `S_o^{-1} - K_oo^{-1}`, the result need not be a single Kronecker product.
- Non-Gaussian likelihoods may break the simple `A.T A / sigma2` likelihood
  precision structure.
- The synthetic experiments use consistent RBF temporal inducing matrices for
  verification. They do not require exact HiPPO-RFF integrals.
