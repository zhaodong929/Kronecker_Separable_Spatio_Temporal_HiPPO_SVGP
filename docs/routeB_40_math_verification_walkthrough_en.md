# Route B 40 Mathematical and Pipeline Verification Tests: Formula-Code-Logic Walkthrough

This document explains the current project's 40 verification tests in English. The first 34 tests validate the core Route B and Stage-1 mathematics and implementation. The last 6 tests validate the ERA5 loader and baseline pipeline. Each block states the formula or contract being checked, the relevant code snippet, and why the test supports the theory or experimental protocol.

Verification commands:

```bash
uv run --no-sync pytest -q
uv run --no-sync python scripts/verify_joint_ssgp_kron_derivations.py
```

Result: `40 passed, 1 warning`. The warning is a local CUDA driver message and does not affect CPU numerical verification.

## Python Syntax Quick Reference

The tests use a small set of common Python, NumPy, and PyTorch constructs:

- `=` assigns a computed value to a variable.
- `assert condition` is a test assertion; the test fails if the condition is false.
- `for ... in ...:` starts a loop; the indented block runs repeatedly.
- `if ...:` starts a conditional branch.
- `A @ B` means matrix multiplication.
- `A.T` means matrix transpose.
- `x[:3]` takes the first three entries; `x[3:5]` takes entries with indices 3 and 4.
- `dict['key']` retrieves a value from a dictionary.
- `.shape` reports array dimensions.
- `np.allclose(a, b)` and `torch.allclose(a, b)` check numerical equality up to tolerance.
- `np.isfinite(x)` checks that values are neither NaN nor infinity.
- `...` means that nonessential arguments are omitted in the document snippet.

## Summary Table

| # | Test | Group | Main verification target | Result |
|---:|---|---|---|---|
| 1 | `test_temporal_and_spatial_shape_consistency` | Stage-1 Kronecker STGP | Checks that the temporal and spatial kernel builders return matrices with the theoretical Kronecker STGP dimensions. | PASSED |
| 2 | `test_kronecker_projection_shapes` | Stage-1 Kronecker STGP | Checks that the Kronecker projection matrix has one row per spatio-temporal observation and one column per mixed inducing variable. | PASSED |
| 3 | `test_small_synthetic_batch_matches_dense_solution` | Stage-1 Kronecker STGP | Compares the model's batch posterior mean against the explicit dense Gaussian posterior formula. | PASSED |
| 4 | `test_small_synthetic_training_reduces_loss` | Stage-1 Kronecker STGP | Checks that the training objective decreases under optimization, ruling out sign errors, broken gradients, or missing parameter registration. | PASSED |
| 5 | `test_blockwise_forward_returns_consistent_shapes` | Stage-1 Kronecker STGP | Checks that blockwise forward prediction preserves shapes after splitting the time axis into online blocks. | PASSED |
| 6 | `test_online_recursion_matches_batch_solution` | Stage-1 Kronecker STGP | Checks that fixed-horizon online natural-parameter accumulation matches the one-shot batch posterior. | PASSED |
| 7 | `test_temporal_cross_covariance_is_consistent` | Stage-1 Kronecker STGP | Checks that temporal cross-covariance between two horizons satisfies the symmetry identity \(K_{ab}=K_{ba}^	op\). | PASSED |
| 8 | `test_online_local_horizon_transfer_updates_state` | Stage-1 Kronecker STGP | Checks that local-horizon online updates produce a finite transfer matrix and update the state across blocks. | PASSED |
| 9 | `test_online_predictive_variance_matches_dense_precision_solver` | Stage-1 Kronecker STGP | Checks that the structured precision/Sylvester predictive variance matches the dense precision correction. | PASSED |
| 10 | `test_load_processed_era5_task_aligns_locations` | ERA5 Data Contract | Checks that processed ERA5 per-location files are sorted and aligned onto a common time axis. | PASSED |
| 11 | `test_load_processed_era5_task_resplit_rebuilds_longer_validation` | ERA5 Data Contract | Checks that chronological re-splitting preserves time order and avoids leakage into validation or test windows. | PASSED |
| 12 | `test_load_processed_era5_tasks_concatenates_multiple_tasks` | ERA5 Data Contract | Checks that multiple ERA5 tasks can be concatenated over time while preserving shared spatial coordinates. | PASSED |
| 13 | `test_discover_and_count_processed_era5_tasks` | ERA5 Data Contract | Checks that ERA5 task discovery and location counting match the processed directory structure. | PASSED |
| 14 | `test_spatial_inducing_fps_spreads_across_domain` | Spatial Inducing Contract | Checks that farthest-point spatial inducing selection covers the spatial domain better than first-N selection. | PASSED |
| 15 | `test_Lon_kron_identity` | Kronecker Derivations | Checks that the dense old-to-new transfer operator reduces to a temporal transfer Kronecker an identity over fixed spatial inducing locations. | PASSED |
| 16 | `test_old_likelihood_transfer_kron_identity` | Kronecker Derivations | Checks that old likelihood precision remains Kronecker-structured after changing-basis transfer. | PASSED |
| 17 | `test_fixed_basis_streaming_equals_batch` | Kronecker Derivations | Checks that fixed-basis streaming natural-parameter accumulation equals the batch Gaussian posterior. | PASSED |
| 18 | `test_no_linear_mean_reduces_to_gp_only` | Kronecker Derivations | Checks that the model reduces to the GP-only SSGP update when the linear mean is removed. | PASSED |
| 19 | `test_no_old_data_transfer_zero` | Kronecker Derivations | Checks that zero old data produces zero transferred old-likelihood information. | PASSED |
| 20 | `test_projected_prior_dense_marginalization` | Kronecker Derivations | Checks projected-prior dense marginalization and structured transfer against dense reference formulas. | PASSED |
| 21 | `test_old_likelihood_dense_vs_structured_information_vector` | Kronecker Derivations | Checks that dense information-vector transfer matches the structured matrix form \(H_oL_t\). | PASSED |
| 22 | `test_model_one_block_no_nan` | Model Sanity | Checks that a one-block model update produces finite predictions and state values. | PASSED |
| 23 | `test_model_multi_block_no_nan` | Model Sanity | Checks that multi-block online transfer remains numerically finite. | PASSED |
| 24 | `test_baseline_imports_still_work` | Model Sanity | Checks that adding Route B did not break the public imports and legacy APIs. | PASSED |
| 25 | `test_routeB_dense_vs_structured_new_block_likelihood` | Route B Theory | Checks that the structured new-block joint likelihood statistics match the dense joint likelihood statistics. | PASSED |
| 26 | `test_routeB_dense_vs_structured_joint_old_likelihood_transfer` | Route B Theory | Checks that joint old-likelihood transfer with the beta-u cross block matches the dense coordinate transform. | PASSED |
| 27 | `test_routeB_schur_posterior_recovery_vs_dense_inverse` | Route B Theory | Checks that Schur-complement posterior recovery with structured solves matches a dense inverse. | PASSED |
| 28 | `test_routeB_cross_covariance_matches_dense_reference` | Route B Theory | Checks that Route B recovers the dense-reference beta-u posterior cross covariance. | PASSED |
| 29 | `test_mean_field_has_zero_cross_covariance_and_differs_when_coupling_nonzero` | Route B Theory | Checks that mean-field has zero beta-u cross covariance and differs from dense posterior when coupling is nonzero. | PASSED |
| 30 | `test_routeB_predictive_variance_matches_dense_joint_posterior_and_differs_from_mean_field` | Route B Theory | Checks that Route B predictive variance matches the dense joint posterior and differs from mean-field when cross covariance matters. | PASSED |
| 31 | `test_routeB_fixed_basis_streaming_equals_batch_joint_posterior` | Route B Theory | Checks that fixed-basis Route B streaming equals the batch joint posterior. | PASSED |
| 32 | `test_routeB_no_linear_mean_reduces_to_gp_only` | Route B Theory | Checks that Route B reduces to the GP-only SSGP update when there is no linear mean. | PASSED |
| 33 | `test_routeB_zero_cross_feature_sanity` | Route B Theory | Checks that when cross features are zero, the predictive variance decomposes into separate beta and GP terms. | PASSED |
| 34 | `test_predictive_variance_respects_kernel_amplitude` | Route B Theory | Checks that sparse conditional residual variance explicitly respects non-unit kernel amplitude. | PASSED |
| 35 | `test_loader_shapes_and_blocks` | ERA5 Baseline Pipeline | Checks that the ERA5 loader returns \(Y\), coordinates, features, and online blocks with the shapes expected by baselines and Route B. | PASSED |
| 36 | `test_loader_converts_to_routeb_factors` | ERA5 Baseline Pipeline | Checks that ERA5 loader outputs can be converted into Route B block factors without changing the structured joint model formulas. | PASSED |
| 37 | `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[PersistenceBaseline]` | ERA5 Baseline Pipeline | Checks that the persistence baseline predicts without reading future labels and returns finite positive variance. | PASSED |
| 38 | `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[ClimatologyBaseline]` | ERA5 Baseline Pipeline | Checks that the climatology baseline estimates mean and variance from seen history only, with no future-label leakage. | PASSED |
| 39 | `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[RidgeBaseline]` | ERA5 Baseline Pipeline | Checks that the ridge baseline has the expected closed-form fit, output shape, no-leakage future prediction, and residual variance. | PASSED |
| 40 | `test_gpytorch_baselines_smoke_if_available` | ERA5 Baseline Pipeline | Checks that the GPyTorch independent GP, SGPR, and SVGP baselines can train and return finite positive likelihood predictive variance. | PASSED |

## Test-by-Test Explanation

### 1. `test_temporal_and_spatial_shape_consistency`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: Stage-1 Kronecker STGP
- Result: `PASSED`

**Formula or Contract**

$K_{uu}^{(t)}\in\mathbb{R}^{M_t\times M_t},\ K_{fu}^{(t)}\in\mathbb{R}^{T\times M_t},\ K_{zz}^{(s)}\in\mathbb{R}^{M_s\times M_s},\ K_{xz}^{(s)}\in\mathbb{R}^{S\times M_s}$

**Code Snippet**

```python
temporal = model.build_temporal_covariances(times)
spatial_cov = model.build_spatial_covariances(spatial)
assert temporal.kuu_t.shape == (4, 4)
assert temporal.kfu_t.shape == (5, 4)
```

**Line-by-Line Implementation Logic**

- 1. `temporal = model.build_temporal_covariances(times)`: Builds temporal covariance objects from the input time coordinates.
- 2. `spatial_cov = model.build_spatial_covariances(spatial)`: Builds spatial covariance objects from the spatial coordinates.
- 3. `assert temporal.kuu_t.shape == (4, 4)`: Runs a test assertion; if this condition is false, the test fails.
- 4. `assert temporal.kfu_t.shape == (5, 4)`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `.shape` reports an array's dimensions, for example `(T, S)`.

**Verification Logic**

Checks that the temporal and spatial kernel builders return matrices with the theoretical Kronecker STGP dimensions.

### 2. `test_kronecker_projection_shapes`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: Stage-1 Kronecker STGP
- Result: `PASSED`

**Formula or Contract**

$A=A_t\otimes A_s,\quad A_t=K_{fu}^{(t)}K_{uu}^{(t)-1},\quad A_s=K_{xz}^{(s)}K_{zz}^{(s)-1}$

**Code Snippet**

```python
projection = model.build_projection(times, spatial)
assert projection.a_t.shape == (5, 4)
assert projection.a_s.shape == (4, 4)
assert torch.kron(projection.a_t, projection.a_s).shape == (20, 16)
```

**Line-by-Line Implementation Logic**

- 1. `projection = model.build_projection(times, spatial)`: Builds temporal and spatial projection factors for the sparse GP approximation.
- 2. `assert projection.a_t.shape == (5, 4)`: Runs a test assertion; if this condition is false, the test fails.
- 3. `assert projection.a_s.shape == (4, 4)`: Runs a test assertion; if this condition is false, the test fails.
- 4. `assert torch.kron(projection.a_t, projection.a_s).shape == (20, 16)`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `.shape` reports an array's dimensions, for example `(T, S)`.

**Verification Logic**

Checks that the Kronecker projection matrix has one row per spatio-temporal observation and one column per mixed inducing variable.

### 3. `test_small_synthetic_batch_matches_dense_solution`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: Stage-1 Kronecker STGP
- Result: `PASSED`

**Formula or Contract**

$\Lambda=K_{uu}^{-1}+\sigma^{-2}A^\top A,\quad m=\Lambda^{-1}\sigma^{-2}A^\top y$

**Code Snippet**

```python
precision = Kuu_inv + torch.reciprocal(sigma2) * (a_dense.T @ a_dense)
info = torch.reciprocal(sigma2) * (a_dense.T @ y.reshape(-1))
mean = torch.linalg.solve(precision, info)
assert torch.allclose(output['posterior_mean_u'], mean)
```

**Line-by-Line Implementation Logic**

- 1. `precision = Kuu_inv + torch.reciprocal(sigma2) * (a_dense.T @ a_dense)`: Uses a matrix transpose, usually to form a quadratic term or symmetry check.
- 2. `info = torch.reciprocal(sigma2) * (a_dense.T @ y.reshape(-1))`: Reshapes an array so it matches the vector or matrix convention used by the formula.
- 3. `mean = torch.linalg.solve(precision, info)`: Solves a linear system without explicitly forming a matrix inverse.
- 4. `assert torch.allclose(output['posterior_mean_u'], mean)`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `A @ B` means matrix multiplication.
- `.T` means matrix transpose.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `torch.linalg` and `np.linalg` provide linear-algebra routines such as solves and decompositions.

**Verification Logic**

Compares the model's batch posterior mean against the explicit dense Gaussian posterior formula.

### 4. `test_small_synthetic_training_reduces_loss`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: Stage-1 Kronecker STGP
- Result: `PASSED`

**Formula or Contract**

$\mathcal{L}_{20}<\mathcal{L}_{0}$

**Code Snippet**

```python
for _ in range(20):
    output = model(times, spatial, y, cache_posterior=False)
    output['loss'].backward()
    optimizer.step()
assert losses[-1] < losses[0]
```

**Line-by-Line Implementation Logic**

- 1. `for _ in range(20):`: Starts a loop; the indented lines below it run once per iteration.
- 2. `    output = model(times, spatial, y, cache_posterior=False)`: Computes `model(times, spatial, y, cache_posterior=False)` and stores the result in `output` for later checks.
- 3. `    output['loss'].backward()`: Runs backpropagation from the current loss to compute gradients.
- 4. `    optimizer.step()`: Updates trainable parameters using the gradients computed by backpropagation.
- 5. `assert losses[-1] < losses[0]`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `for ... in ...:` repeats the indented block.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.

**Verification Logic**

Checks that the training objective decreases under optimization, ruling out sign errors, broken gradients, or missing parameter registration.

### 5. `test_blockwise_forward_returns_consistent_shapes`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: Stage-1 Kronecker STGP
- Result: `PASSED`

**Formula or Contract**

$\{B_1,\ldots,B_N\},\quad \hat{Y}_{B_n}\in\mathbb{R}^{|B_n|\times S}$

**Code Snippet**

```python
blockwise = model.forward_blockwise(..., block_size=2)
assert len(blockwise.block_outputs) == 3
assert blockwise.block_outputs[-1]['train_mean'].shape[1] == spatial.shape[0]
```

**Line-by-Line Implementation Logic**

- 1. `blockwise = model.forward_blockwise(..., block_size=2)`: Computes `model.forward_blockwise(..., block_size=2)` and stores the result in `blockwise` for later checks.
- 2. `assert len(blockwise.block_outputs) == 3`: Runs a test assertion; if this condition is false, the test fails.
- 3. `assert blockwise.block_outputs[-1]['train_mean'].shape[1] == spatial.shape[0]`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `.shape` reports an array's dimensions, for example `(T, S)`.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that blockwise forward prediction preserves shapes after splitting the time axis into online blocks.

### 6. `test_online_recursion_matches_batch_solution`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: Stage-1 Kronecker STGP
- Result: `PASSED`

**Formula or Contract**

$\Lambda_N=\Lambda_0+\sum_n\sigma^{-2}A_n^\top A_n,\quad h_N=h_0+\sum_n\sigma^{-2}A_n^\top y_n$

**Code Snippet**

```python
batch_output = batch_model(..., materialize_posterior_cov=True)
for block in blocks:
    online_model.update_block(...)
assert torch.allclose(batch_output['posterior_mean_u'], online_model.state.m)
```

**Line-by-Line Implementation Logic**

- 1. `batch_output = batch_model(..., materialize_posterior_cov=True)`: Runs the batch model once to obtain the dense or one-shot reference result.
- 2. `for block in blocks:`: Starts a loop; the indented lines below it run once per iteration.
- 3. `    online_model.update_block(...)`: Processes one online block and updates the streaming state.
- 4. `assert torch.allclose(batch_output['posterior_mean_u'], online_model.state.m)`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `for ... in ...:` repeats the indented block.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that fixed-horizon online natural-parameter accumulation matches the one-shot batch posterior.

### 7. `test_temporal_cross_covariance_is_consistent`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: Stage-1 Kronecker STGP
- Result: `PASSED`

**Formula or Contract**

$K_{ab}=K_{ba}^{\top}$

**Code Snippet**

```python
cross_ab = builder.compute_kuu_t_cross(horizon_a, horizon_b)
cross_ba = builder.compute_kuu_t_cross(horizon_b, horizon_a)
assert torch.allclose(cross_ab, cross_ba.T)
```

**Line-by-Line Implementation Logic**

- 1. `cross_ab = builder.compute_kuu_t_cross(horizon_a, horizon_b)`: Computes a temporal cross-covariance matrix between two horizons.
- 2. `cross_ba = builder.compute_kuu_t_cross(horizon_b, horizon_a)`: Computes a temporal cross-covariance matrix between two horizons.
- 3. `assert torch.allclose(cross_ab, cross_ba.T)`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `.T` means matrix transpose.

**Verification Logic**

Checks that temporal cross-covariance between two horizons satisfies the symmetry identity \(K_{ab}=K_{ba}^	op\).

### 8. `test_online_local_horizon_transfer_updates_state`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: Stage-1 Kronecker STGP
- Result: `PASSED`

**Formula or Contract**

$u_o\mapsto u_n,\quad L_{on}=K_{on}K_{nn}^{-1}$

**Code Snippet**

```python
first = online_model.update_block(..., horizon=first_block_horizon)
second = online_model.update_block(..., horizon=second_block_horizon)
assert first['temporal_transfer'].shape == (4, 4)
```

**Line-by-Line Implementation Logic**

- 1. `first = online_model.update_block(..., horizon=first_block_horizon)`: Processes one online block and updates the streaming state.
- 2. `second = online_model.update_block(..., horizon=second_block_horizon)`: Processes one online block and updates the streaming state.
- 3. `assert first['temporal_transfer'].shape == (4, 4)`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `.shape` reports an array's dimensions, for example `(T, S)`.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that local-horizon online updates produce a finite transfer matrix and update the state across blocks.

### 9. `test_online_predictive_variance_matches_dense_precision_solver`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: Stage-1 Kronecker STGP
- Result: `PASSED`

**Formula or Contract**

$\operatorname{Var}(f_*)=k_{**}-a_*K_{uu}a_*^\top+a_*\Lambda^{-1}a_*^\top$

**Code Snippet**

```python
dense_latent_var = prior_diag - projected_prior_diag + dense_posterior_correction
assert torch.allclose(pred['latent_var'], dense_latent_var)
```

**Line-by-Line Implementation Logic**

- 1. `dense_latent_var = prior_diag - projected_prior_diag + dense_posterior_correction`: Computes the dense latent predictive variance reference.
- 2. `assert torch.allclose(pred['latent_var'], dense_latent_var)`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.

**Verification Logic**

Checks that the structured precision/Sylvester predictive variance matches the dense precision correction.

### 10. `test_load_processed_era5_task_aligns_locations`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: ERA5 Data Contract
- Result: `PASSED`

**Formula or Contract**

$Y\in\mathbb{R}^{T\times S}$ with shared sorted times over all selected locations

**Code Snippet**

```python
task = load_processed_era5_task(...)
assert task.train.times.tolist() == [0.0, 1.0, 2.0]
assert task.train.observations.shape == (3, 2)
```

**Line-by-Line Implementation Logic**

- 1. `task = load_processed_era5_task(...)`: Loads a processed ERA5 task and checks its ordering, alignment, or concatenation behavior.
- 2. `assert task.train.times.tolist() == [0.0, 1.0, 2.0]`: Runs a test assertion; if this condition is false, the test fails.
- 3. `assert task.train.observations.shape == (3, 2)`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `.shape` reports an array's dimensions, for example `(T, S)`.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that processed ERA5 per-location files are sorted and aligned onto a common time axis.

### 11. `test_load_processed_era5_task_resplit_rebuilds_longer_validation`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: ERA5 Data Contract
- Result: `PASSED`

**Formula or Contract**

$\{1,\ldots,T\}=\mathcal{T}_{train}\cup\mathcal{T}_{val}\cup\mathcal{T}_{test}$ chronologically

**Code Snippet**

```python
task = load_processed_era5_task(..., resplit=True)
assert task.train.times.tolist() == [0.0, 1.0, 2.0]
assert task.val.times.tolist() == [3.0, 4.0]
```

**Line-by-Line Implementation Logic**

- 1. `task = load_processed_era5_task(..., resplit=True)`: Loads a processed ERA5 task and checks its ordering, alignment, or concatenation behavior.
- 2. `assert task.train.times.tolist() == [0.0, 1.0, 2.0]`: Runs a test assertion; if this condition is false, the test fails.
- 3. `assert task.val.times.tolist() == [3.0, 4.0]`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that chronological re-splitting preserves time order and avoids leakage into validation or test windows.

### 12. `test_load_processed_era5_tasks_concatenates_multiple_tasks`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: ERA5 Data Contract
- Result: `PASSED`

**Formula or Contract**

$Y_{1:K}=[Y^{(1)};Y^{(2)};\ldots;Y^{(K)}]$ with shared spatial coordinates

**Code Snippet**

```python
task = load_processed_era5_tasks([task_1, task_2], resplit=True)
assert task.train.observations.shape == (6, 2)
assert task.test.times.tolist() == [9.0, 10.0, 11.0]
```

**Line-by-Line Implementation Logic**

- 1. `task = load_processed_era5_tasks([task_1, task_2], resplit=True)`: Loads a processed ERA5 task and checks its ordering, alignment, or concatenation behavior.
- 2. `assert task.train.observations.shape == (6, 2)`: Runs a test assertion; if this condition is false, the test fails.
- 3. `assert task.test.times.tolist() == [9.0, 10.0, 11.0]`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `.shape` reports an array's dimensions, for example `(T, S)`.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.

**Verification Logic**

Checks that multiple ERA5 tasks can be concatenated over time while preserving shared spatial coordinates.

### 13. `test_discover_and_count_processed_era5_tasks`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: ERA5 Data Contract
- Result: `PASSED`

**Formula or Contract**

$|\mathcal{S}_{task}|=\#\{\text{selected location files}\}$

**Code Snippet**

```python
task_dirs = discover_processed_era5_task_dirs(root, ['task_1', 'task_2'])
assert count_processed_era5_locations(task_dirs[0]) == 2
```

**Line-by-Line Implementation Logic**

- 1. `task_dirs = discover_processed_era5_task_dirs(root, ['task_1', 'task_2'])`: Discovers the processed ERA5 task directories requested by the loader.
- 2. `assert count_processed_era5_locations(task_dirs[0]) == 2`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.

**Verification Logic**

Checks that ERA5 task discovery and location counting match the processed directory structure.

### 14. `test_spatial_inducing_fps_spreads_across_domain`

- File: `stvgp_kronecker/tests/test_stage1.py`
- Group: Spatial Inducing Contract
- Result: `PASSED`

**Formula or Contract**

$Z_s=\operatorname{FPS}(X_s)$ should cover the spatial domain better than first-N selection

**Code Snippet**

```python
first = select_spatial_inducing_points(..., selection_method='first')
fps = select_spatial_inducing_points(..., selection_method='fps')
assert float(fps[:, 0].min()) == -10.0
assert float(fps[:, 0].max()) == 2.0
```

**Line-by-Line Implementation Logic**

- 1. `first = select_spatial_inducing_points(..., selection_method='first')`: Selects spatial inducing points using the simple first-N baseline.
- 2. `fps = select_spatial_inducing_points(..., selection_method='fps')`: Selects spatial inducing points using farthest-point sampling.
- 3. `assert float(fps[:, 0].min()) == -10.0`: Runs a test assertion; if this condition is false, the test fails.
- 4. `assert float(fps[:, 0].max()) == 2.0`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that farthest-point spatial inducing selection covers the spatial domain better than first-N selection.

### 15. `test_Lon_kron_identity`

- File: `tests/test_joint_ssgp_kron_derivations.py`
- Group: Kronecker Derivations
- Result: `PASSED`

**Formula or Contract**

$L_{on}=K_{on}K_{nn}^{-1}=(K_{on}^{t}K_{nn}^{t-1})\otimes I_s$

**Code Snippet**

```python
L_t = compute_Lt(K_on_t, K_nn_t)
L_dense = kron(K_on_t, Ks) @ inv(kron(K_nn_t, Ks))
L_kron = kron(L_t, I_s)
assert err < 1e-8
```

**Line-by-Line Implementation Logic**

- 1. `L_t = compute_Lt(K_on_t, K_nn_t)`: Computes `compute_Lt(K_on_t, K_nn_t)` and stores the result in `L_t` for later checks.
- 2. `L_dense = kron(K_on_t, Ks) @ inv(kron(K_nn_t, Ks))`: Forms a Kronecker product to build or compare the full spatio-temporal matrix.
- 3. `L_kron = kron(L_t, I_s)`: Forms a Kronecker product to build or compare the full spatio-temporal matrix.
- 4. `assert err < 1e-8`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `A @ B` means matrix multiplication.

**Verification Logic**

Checks that the dense old-to-new transfer operator reduces to a temporal transfer Kronecker an identity over fixed spatial inducing locations.

### 16. `test_old_likelihood_transfer_kron_identity`

- File: `tests/test_joint_ssgp_kron_derivations.py`
- Group: Kronecker Derivations
- Result: `PASSED`

**Formula or Contract**

$L_{on}^{\top}(B_o\otimes G)L_{on}=(L_t^{\top}B_oL_t)\otimes G$

**Code Snippet**

```python
Lambda_dense = L_dense.T @ kron(B_old, G) @ L_dense
Lambda_kron = kron(transfer_temporal_precision(B_old, L_t), G)
assert err < 1e-8
```

**Line-by-Line Implementation Logic**

- 1. `Lambda_dense = L_dense.T @ kron(B_old, G) @ L_dense`: Forms a Kronecker product to build or compare the full spatio-temporal matrix.
- 2. `Lambda_kron = kron(transfer_temporal_precision(B_old, L_t), G)`: Forms a Kronecker product to build or compare the full spatio-temporal matrix.
- 3. `assert err < 1e-8`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `A @ B` means matrix multiplication.
- `.T` means matrix transpose.

**Verification Logic**

Checks that old likelihood precision remains Kronecker-structured after changing-basis transfer.

### 17. `test_fixed_basis_streaming_equals_batch`

- File: `tests/test_joint_ssgp_kron_derivations.py`
- Group: Kronecker Derivations
- Result: `PASSED`

**Formula or Contract**

$\Lambda_{stream}=\Lambda_0+\sum_n H_n^\top H_n/\sigma^2=\Lambda_{batch}$

**Code Snippet**

```python
for H, y in blocks:
    Lambda_stream += H.T @ H / sigma2
    h_stream += H.T @ y / sigma2
assert mean_err < 1e-8 and prec_err < 1e-8
```

**Line-by-Line Implementation Logic**

- 1. `for H, y in blocks:`: Starts a loop; the indented lines below it run once per iteration.
- 2. `    Lambda_stream += H.T @ H / sigma2`: Uses a matrix transpose, usually to form a quadratic term or symmetry check.
- 3. `    h_stream += H.T @ y / sigma2`: Uses a matrix transpose, usually to form a quadratic term or symmetry check.
- 4. `assert mean_err < 1e-8 and prec_err < 1e-8`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `A @ B` means matrix multiplication.
- `.T` means matrix transpose.
- `for ... in ...:` repeats the indented block.

**Verification Logic**

Checks that fixed-basis streaming natural-parameter accumulation equals the batch Gaussian posterior.

### 18. `test_no_linear_mean_reduces_to_gp_only`

- File: `tests/test_joint_ssgp_kron_derivations.py`
- Group: Kronecker Derivations
- Result: `PASSED`

**Formula or Contract**

$\Phi=0\Rightarrow \beta=0,\quad H_u=C^\top YT/\sigma^2,\quad B=T^\top T/\sigma^2$

**Code Snippet**

```python
state = model.update_block_ssgp_transfer(y_vec=y, Phi=zeros, ...)
assert allclose(state.beta_mean, 0)
assert allclose(state.B_temporal, B_gp)
assert allclose(state.H_info, H_gp)
```

**Line-by-Line Implementation Logic**

- 1. `state = model.update_block_ssgp_transfer(y_vec=y, Phi=zeros, ...)`: Processes one online block and updates the streaming state.
- 2. `assert allclose(state.beta_mean, 0)`: Checks that two floating-point arrays are numerically close within tolerance.
- 3. `assert allclose(state.B_temporal, B_gp)`: Checks that two floating-point arrays are numerically close within tolerance.
- 4. `assert allclose(state.H_info, H_gp)`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that the model reduces to the GP-only SSGP update when the linear mean is removed.

### 19. `test_no_old_data_transfer_zero`

- File: `tests/test_joint_ssgp_kron_derivations.py`
- Group: Kronecker Derivations
- Result: `PASSED`

**Formula or Contract**

$B_o=0,\ H_o=0\Rightarrow B_{o\to n}=0,\ H_{o\to n}=0$

**Code Snippet**

```python
B_trans = transfer_temporal_precision(B_old_zero, L_t)
H_trans = transfer_information_matrix(H_old_zero, L_t)
assert allclose(B_trans, 0) and allclose(H_trans, 0)
```

**Line-by-Line Implementation Logic**

- 1. `B_trans = transfer_temporal_precision(B_old_zero, L_t)`: Computes `transfer_temporal_precision(B_old_zero, L_t)` and stores the result in `B_trans` for later checks.
- 2. `H_trans = transfer_information_matrix(H_old_zero, L_t)`: Computes `transfer_information_matrix(H_old_zero, L_t)` and stores the result in `H_trans` for later checks.
- 3. `assert allclose(B_trans, 0) and allclose(H_trans, 0)`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.

**Verification Logic**

Checks that zero old data produces zero transferred old-likelihood information.

### 20. `test_projected_prior_dense_marginalization`

- File: `tests/test_joint_ssgp_kron_derivations.py`
- Group: Kronecker Derivations
- Result: `PASSED`

**Formula or Contract**

$m_n=K_{no}K_{oo}^{-1}m_o,\quad S_n=K_{nn}+K_{no}K_{oo}^{-1}(S_o-K_{oo})K_{oo}^{-1}K_{on}$

**Code Snippet**

```python
m_proj, S_proj = projected_prior_transfer_dense(...)
assert projected_prior_error < 1e-8
assert structured_transfer_error < 1e-8
```

**Line-by-Line Implementation Logic**

- 1. `m_proj, S_proj = projected_prior_transfer_dense(...)`: Computes `projected_prior_transfer_dense(...)` and stores the result in `m_proj, S_proj` for later checks.
- 2. `assert projected_prior_error < 1e-8`: Runs a test assertion; if this condition is false, the test fails.
- 3. `assert structured_transfer_error < 1e-8`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks projected-prior dense marginalization and structured transfer against dense reference formulas.

### 21. `test_old_likelihood_dense_vs_structured_information_vector`

- File: `tests/test_joint_ssgp_kron_derivations.py`
- Group: Kronecker Derivations
- Result: `PASSED`

**Formula or Contract**

$h_{u,o\to n}=L_{on}^{\top}h_{u,o},\quad \operatorname{vec}(H_oL_t)=L_{on}^{\top}\operatorname{vec}(H_o)$

**Code Snippet**

```python
h_dense = L_dense.T @ vec_f(H_old)
h_kron = vec_f(transfer_information_matrix(H_old, L_t))
assert norm(h_dense - h_kron) < 1e-8
```

**Line-by-Line Implementation Logic**

- 1. `h_dense = L_dense.T @ vec_f(H_old)`: Uses a matrix transpose, usually to form a quadratic term or symmetry check.
- 2. `h_kron = vec_f(transfer_information_matrix(H_old, L_t))`: Computes `vec_f(transfer_information_matrix(H_old, L_t))` and stores the result in `h_kron` for later checks.
- 3. `assert norm(h_dense - h_kron) < 1e-8`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `A @ B` means matrix multiplication.
- `.T` means matrix transpose.

**Verification Logic**

Checks that dense information-vector transfer matches the structured matrix form \(H_oL_t\).

### 22. `test_model_one_block_no_nan`

- File: `tests/test_joint_ssgp_kron_model.py`
- Group: Model Sanity
- Result: `PASSED`

**Formula or Contract**

$\hat y=\Phi m_\beta + A\,\operatorname{vec}(M_u)$ finite

**Code Snippet**

```python
state = model.update_block_ssgp_transfer(...)
mean = Phi @ state.beta_mean + A @ vec_f(state.M_u)
assert np.all(np.isfinite(mean))
```

**Line-by-Line Implementation Logic**

- 1. `state = model.update_block_ssgp_transfer(...)`: Processes one online block and updates the streaming state.
- 2. `mean = Phi @ state.beta_mean + A @ vec_f(state.M_u)`: Uses matrix multiplication to implement the corresponding linear-algebra formula.
- 3. `assert np.all(np.isfinite(mean))`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `A @ B` means matrix multiplication.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that a one-block model update produces finite predictions and state values.

### 23. `test_model_multi_block_no_nan`

- File: `tests/test_joint_ssgp_kron_model.py`
- Group: Model Sanity
- Result: `PASSED`

**Formula or Contract**

$\forall n,\ \hat y_n=\Phi_nm_{\beta,n}+A_n\operatorname{vec}(M_{u,n})$ finite

**Code Snippet**

```python
for block in iter_time_blocks(...):
    state = model.update_block_ssgp_transfer(..., state=state)
    assert np.all(np.isfinite(mean))
```

**Line-by-Line Implementation Logic**

- 1. `for block in iter_time_blocks(...):`: Starts a loop; the indented lines below it run once per iteration.
- 2. `    state = model.update_block_ssgp_transfer(..., state=state)`: Processes one online block and updates the streaming state.
- 3. `    assert np.all(np.isfinite(mean))`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `for ... in ...:` repeats the indented block.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that multi-block online transfer remains numerically finite.

### 24. `test_baseline_imports_still_work`

- File: `tests/test_joint_ssgp_kron_model.py`
- Group: Model Sanity
- Result: `PASSED`

**Formula or Contract**

$\text{public API remains importable after Route B additions}$

**Code Snippet**

```python
import stvgp_kronecker.train_batch as train_batch
import stvgp_kronecker.train_online as train_online
assert hasattr(train_batch, 'main')
```

**Line-by-Line Implementation Logic**

- 1. `import stvgp_kronecker.train_batch as train_batch`: Runs this function call or check as one step of the verification.
- 2. `import stvgp_kronecker.train_online as train_online`: Runs this function call or check as one step of the verification.
- 3. `assert hasattr(train_batch, 'main')`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `assert condition` makes the test fail immediately if the condition is false.

**Verification Logic**

Checks that adding Route B did not break the public imports and legacy APIs.

### 25. `test_routeB_dense_vs_structured_new_block_likelihood`

- File: `tests/test_joint_ssgp_kron_routeB.py`
- Group: Route B Theory
- Result: `PASSED`

**Formula or Contract**

$R_{\beta\beta}=\Phi^\top\Phi/\sigma^2,\ R_{\beta u}=\Phi^\top A/\sigma^2,\ R_{uu}=A^\top A/\sigma^2=(T^\top T)\otimes(C^\top C)/\sigma^2$

**Code Snippet**

```python
stats = joint_likelihood_stats(y, Phi, T, C, sigma2)
assert allclose(stats['R_beta_beta'], Phi.T @ Phi / sigma2)
assert allclose(stats['R_beta_u'], Phi.T @ A / sigma2)
```

**Line-by-Line Implementation Logic**

- 1. `stats = joint_likelihood_stats(y, Phi, T, C, sigma2)`: Computes structured Route B likelihood natural-parameter blocks.
- 2. `assert allclose(stats['R_beta_beta'], Phi.T @ Phi / sigma2)`: Checks that two floating-point arrays are numerically close within tolerance.
- 3. `assert allclose(stats['R_beta_u'], Phi.T @ A / sigma2)`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `A @ B` means matrix multiplication.
- `.T` means matrix transpose.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.

**Verification Logic**

Checks that the structured new-block joint likelihood statistics match the dense joint likelihood statistics.

### 26. `test_routeB_dense_vs_structured_joint_old_likelihood_transfer`

- File: `tests/test_joint_ssgp_kron_routeB.py`
- Group: Route B Theory
- Result: `PASSED`

**Formula or Contract**

$R_{\beta u,o\to n}=R_{\beta u,o}L_{on},\quad R_{uu,o\to n}=L_{on}^{\top}R_{uu,o}L_{on}$

**Code Snippet**

```python
R_dense = T_joint.T @ R_old @ T_joint
assert allclose(R_dense[:d, d:], transfer_R_beta_u(R_beta_u, L_t, ms))
assert allclose(R_dense[d:, d:], kron(transfer_temporal_precision(B_old, L_t), G))
```

**Line-by-Line Implementation Logic**

- 1. `R_dense = T_joint.T @ R_old @ T_joint`: Uses a matrix transpose, usually to form a quadratic term or symmetry check.
- 2. `assert allclose(R_dense[:d, d:], transfer_R_beta_u(R_beta_u, L_t, ms))`: Checks that two floating-point arrays are numerically close within tolerance.
- 3. `assert allclose(R_dense[d:, d:], kron(transfer_temporal_precision(B_old, L_t), G))`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `A @ B` means matrix multiplication.
- `.T` means matrix transpose.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.

**Verification Logic**

Checks that joint old-likelihood transfer with the beta-u cross block matches the dense coordinate transform.

### 27. `test_routeB_schur_posterior_recovery_vs_dense_inverse`

- File: `tests/test_joint_ssgp_kron_routeB.py`
- Group: Route B Theory
- Result: `PASSED`

**Formula or Contract**

$S_{\beta|u}=(A_\beta-R_{\beta u}D_u^{-1}R_{u\beta})^{-1},\quad m_u=D_u^{-1}(h_u-R_{u\beta}m_\beta)$

**Code Snippet**

```python
schur = schur_recover_posterior(...)
_, cov, mean = dense_joint_posterior_reference(...)
assert allclose(schur['m_beta'], mean[:d])
assert allclose(schur['S_beta_beta'], cov[:d, :d])
```

**Line-by-Line Implementation Logic**

- 1. `schur = schur_recover_posterior(...)`: Recovers posterior moments using the Schur-complement implementation.
- 2. `_, cov, mean = dense_joint_posterior_reference(...)`: Computes `dense_joint_posterior_reference(...)` and stores the result in `_, cov, mean` for later checks.
- 3. `assert allclose(schur['m_beta'], mean[:d])`: Checks that two floating-point arrays are numerically close within tolerance.
- 4. `assert allclose(schur['S_beta_beta'], cov[:d, :d])`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that Schur-complement posterior recovery with structured solves matches a dense inverse.

### 28. `test_routeB_cross_covariance_matches_dense_reference`

- File: `tests/test_joint_ssgp_kron_routeB.py`
- Group: Route B Theory
- Result: `PASSED`

**Formula or Contract**

$S_{\beta u}=-S_{\beta\beta}R_{\beta u}D_u^{-1}$

**Code Snippet**

```python
routeB_cross_cov = -schur['S_beta_beta'] @ schur['W'].T
assert np.linalg.norm(cov[:d, d:]) > 1e-8
assert np.allclose(routeB_cross_cov, cov[:d, d:])
```

**Line-by-Line Implementation Logic**

- 1. `routeB_cross_cov = -schur['S_beta_beta'] @ schur['W'].T`: Uses a matrix transpose, usually to form a quadratic term or symmetry check.
- 2. `assert np.linalg.norm(cov[:d, d:]) > 1e-8`: Runs a test assertion; if this condition is false, the test fails.
- 3. `assert np.allclose(routeB_cross_cov, cov[:d, d:])`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `A @ B` means matrix multiplication.
- `.T` means matrix transpose.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `torch.linalg` and `np.linalg` provide linear-algebra routines such as solves and decompositions.

**Verification Logic**

Checks that Route B recovers the dense-reference beta-u posterior cross covariance.

### 29. `test_mean_field_has_zero_cross_covariance_and_differs_when_coupling_nonzero`

- File: `tests/test_joint_ssgp_kron_routeB.py`
- Group: Route B Theory
- Result: `PASSED`

**Formula or Contract**

$S_{\beta u}^{MF}=0,\quad S_{\beta u}^{dense}\ne 0$ under nonzero coupling

**Code Snippet**

```python
mean_field_cross_cov = np.zeros((d, ms * mt))
assert norm(mean_field_cross_cov - cov[:d, d:]) > 1e-8
assert abs(dense_predictive_variance - mean_field_predictive_variance) > 1e-8
```

**Line-by-Line Implementation Logic**

- 1. `mean_field_cross_cov = np.zeros((d, ms * mt))`: Builds the mean-field cross covariance, which should be exactly zero by construction.
- 2. `assert norm(mean_field_cross_cov - cov[:d, d:]) > 1e-8`: Runs a test assertion; if this condition is false, the test fails.
- 3. `assert abs(dense_predictive_variance - mean_field_predictive_variance) > 1e-8`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.

**Verification Logic**

Checks that mean-field has zero beta-u cross covariance and differs from dense posterior when coupling is nonzero.

### 30. `test_routeB_predictive_variance_matches_dense_joint_posterior_and_differs_from_mean_field`

- File: `tests/test_joint_ssgp_kron_routeB.py`
- Group: Route B Theory
- Result: `PASSED`

**Formula or Contract**

$\operatorname{Var}(y_*)=\sigma^2+\nu_*+[\phi_*,q_*]S[\phi_*,q_*]^\top$

**Code Snippet**

```python
pred = model.predict(phi_star=phi, c_proj_star=c, t_proj_star=t, state=state)
dense_var = sigma2 + nu + x @ cov @ x
assert allclose(pred.variance, dense_var)
assert abs(pred.variance - mean_field_var) > 1e-7
```

**Line-by-Line Implementation Logic**

- 1. `pred = model.predict(phi_star=phi, c_proj_star=c, t_proj_star=t, state=state)`: Computes `model.predict(phi_star=phi, c_proj_star=c, t_proj_star=t, state=state)` and stores the result in `pred` for later checks.
- 2. `dense_var = sigma2 + nu + x @ cov @ x`: Uses matrix multiplication to implement the corresponding linear-algebra formula.
- 3. `assert allclose(pred.variance, dense_var)`: Checks that two floating-point arrays are numerically close within tolerance.
- 4. `assert abs(pred.variance - mean_field_var) > 1e-7`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `A @ B` means matrix multiplication.

**Verification Logic**

Checks that Route B predictive variance matches the dense joint posterior and differs from mean-field when cross covariance matters.

### 31. `test_routeB_fixed_basis_streaming_equals_batch_joint_posterior`

- File: `tests/test_joint_ssgp_kron_routeB.py`
- Group: Route B Theory
- Result: `PASSED`

**Formula or Contract**

$\Lambda_N=\Lambda_0+\sum_n\sigma^{-2}[\Phi_n,A_n]^\top[\Phi_n,A_n]$

**Code Snippet**

```python
state = model.update_block_structured_joint_ssgp_transfer(...)
Lambda_batch = prior + H.T @ H / sigma2
assert allclose(state.routeB_dense_joint_precision(), Lambda_batch)
```

**Line-by-Line Implementation Logic**

- 1. `state = model.update_block_structured_joint_ssgp_transfer(...)`: Processes one online block and updates the streaming state.
- 2. `Lambda_batch = prior + H.T @ H / sigma2`: Uses a matrix transpose, usually to form a quadratic term or symmetry check.
- 3. `assert allclose(state.routeB_dense_joint_precision(), Lambda_batch)`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `A @ B` means matrix multiplication.
- `.T` means matrix transpose.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that fixed-basis Route B streaming equals the batch joint posterior.

### 32. `test_routeB_no_linear_mean_reduces_to_gp_only`

- File: `tests/test_joint_ssgp_kron_routeB.py`
- Group: Route B Theory
- Result: `PASSED`

**Formula or Contract**

$d_\beta=0\Rightarrow R_{\beta u}\ \text{empty and Route B}=GP\text{-only SSGP}$

**Code Snippet**

```python
Phi = np.zeros((y.size, 0))
routeB = model.update_block_structured_joint_ssgp_transfer(...)
gp_only = model.update_block_ssgp_transfer(...)
assert allclose(routeB.M_u, gp_only.M_u)
```

**Line-by-Line Implementation Logic**

- 1. `Phi = np.zeros((y.size, 0))`: Creates zero linear features to test GP-only or zero-coupling behavior.
- 2. `routeB = model.update_block_structured_joint_ssgp_transfer(...)`: Processes one online block and updates the streaming state.
- 3. `gp_only = model.update_block_ssgp_transfer(...)`: Processes one online block and updates the streaming state.
- 4. `assert allclose(routeB.M_u, gp_only.M_u)`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that Route B reduces to the GP-only SSGP update when there is no linear mean.

### 33. `test_routeB_zero_cross_feature_sanity`

- File: `tests/test_joint_ssgp_kron_routeB.py`
- Group: Route B Theory
- Result: `PASSED`

**Formula or Contract**

$\Phi=0\Rightarrow R_{\beta u}=0,\quad \operatorname{Var}(y_*)=\sigma^2+\nu_*+\phi_*^\top S_{\beta\beta}\phi_*+q_*^\top D_u^{-1}q_*$

**Code Snippet**

```python
Phi = np.zeros((y.size, d))
state = model.update_block_structured_joint_ssgp_transfer(...)
assert allclose(state.R_beta_u, 0.0)
assert allclose(pred.variance, separate)
```

**Line-by-Line Implementation Logic**

- 1. `Phi = np.zeros((y.size, d))`: Creates zero linear features to test GP-only or zero-coupling behavior.
- 2. `state = model.update_block_structured_joint_ssgp_transfer(...)`: Processes one online block and updates the streaming state.
- 3. `assert allclose(state.R_beta_u, 0.0)`: Checks that two floating-point arrays are numerically close within tolerance.
- 4. `assert allclose(pred.variance, separate)`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that when cross features are zero, the predictive variance decomposes into separate beta and GP terms.

### 34. `test_predictive_variance_respects_kernel_amplitude`

- File: `tests/test_joint_ssgp_kron_routeB.py`
- Group: Route B Theory
- Result: `PASSED`

**Formula or Contract**

$\nu_*=k(x_*,x_*)-k_{*u}K_{uu}^{-1}k_{u*}$ with $k(x_*,x_*)=\text{kernel variance}$

**Code Snippet**

```python
for kernel_variance in [2.0, 0.5]:
    model = JointSSGPKronHiPPOSVGP(..., prior_point_variance=dataset.gp_prior_variance)
    dense_var = sigma2 + nu + x @ cov @ x
    assert allclose(pred.variance, dense_var)
```

**Line-by-Line Implementation Logic**

- 1. `for kernel_variance in [2.0, 0.5]:`: Starts a loop; the indented lines below it run once per iteration.
- 2. `    model = JointSSGPKronHiPPOSVGP(..., prior_point_variance=dataset.gp_prior_variance)`: Computes `JointSSGPKronHiPPOSVGP(..., prior_point_variance=dataset.gp_prior_variance)` and stores the result in `model` for later checks.
- 3. `    dense_var = sigma2 + nu + x @ cov @ x`: Uses matrix multiplication to implement the corresponding linear-algebra formula.
- 4. `    assert allclose(pred.variance, dense_var)`: Checks that two floating-point arrays are numerically close within tolerance.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- `A @ B` means matrix multiplication.
- `for ... in ...:` repeats the indented block.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that sparse conditional residual variance explicitly respects non-unit kernel amplitude.

### 35. `test_loader_shapes_and_blocks`

- File: `tests/test_hipposvgp_era5_loader_baselines.py`
- Group: ERA5 Baseline Pipeline
- Result: `PASSED`

**Formula or Contract**

$Y\in\mathbb{R}^{T\times S},\quad \Phi\in\mathbb{R}^{TS\times p},\quad B_n=[t_n,t_n+b)$

**Code Snippet**

```python
dataset = load_hipposvgp_era5(..., first_n_locations=2, split='all')
assert dataset.Y.shape == (6, 2)
assert dataset.Phi.shape[0] == 12
assert [(b.start, b.stop) for b in blocks] == [(0, 2), (2, 4), (4, 6)]
```

**Line-by-Line Implementation Logic**

- 1. `dataset = load_hipposvgp_era5(..., first_n_locations=2, split='all')`: Computes `load_hipposvgp_era5(..., first_n_locations=2, split='all')` and stores the result in `dataset` for later checks.
- 2. `assert dataset.Y.shape == (6, 2)`: Runs a test assertion; if this condition is false, the test fails.
- 3. `assert dataset.Phi.shape[0] == 12`: Runs a test assertion; if this condition is false, the test fails.
- 4. `assert [(b.start, b.stop) for b in blocks] == [(0, 2), (2, 4), (4, 6)]`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `.shape` reports an array's dimensions, for example `(T, S)`.
- `for ... in ...:` repeats the indented block.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that the ERA5 loader returns \(Y\), coordinates, features, and online blocks with the shapes expected by baselines and Route B.

### 36. `test_loader_converts_to_routeb_factors`

- File: `tests/test_hipposvgp_era5_loader_baselines.py`
- Group: ERA5 Baseline Pipeline
- Result: `PASSED`

**Formula or Contract**

$Y_{loader}^{T\times S}\mapsto Y_{RouteB}^{S\times T},\quad (y_n,\Phi_n,T_n,K_t)$ match BlockFactors

**Code Snippet**

```python
factors = make_routeb_block_factors(dataset, block=slice(0, 2), ...)
assert factors.Y.shape == (2, 2)
assert factors.Phi.shape[0] == 4
assert np.isfinite(factors.y_vec).all()
```

**Line-by-Line Implementation Logic**

- 1. `factors = make_routeb_block_factors(dataset, block=slice(0, 2), ...)`: Computes `make_routeb_block_factors(dataset, block=slice(0, 2), ...)` and stores the result in `factors` for later checks.
- 2. `assert factors.Y.shape == (2, 2)`: Runs a test assertion; if this condition is false, the test fails.
- 3. `assert factors.Phi.shape[0] == 4`: Runs a test assertion; if this condition is false, the test fails.
- 4. `assert np.isfinite(factors.y_vec).all()`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `.shape` reports an array's dimensions, for example `(T, S)`.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.
- `...` in a snippet means nonessential arguments are omitted for readability.

**Verification Logic**

Checks that ERA5 loader outputs can be converted into Route B block factors without changing the structured joint model formulas.

### 37. `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[PersistenceBaseline]`

- File: `tests/test_hipposvgp_era5_loader_baselines.py`
- Group: ERA5 Baseline Pipeline
- Result: `PASSED`

**Formula or Contract**

$\hat y_{future}=g(\mathcal{D}_{seen},x_{future}),\quad \operatorname{Var}(\hat y)>0$

**Code Snippet**

```python
baseline.fit_initial_task(times[:3], coords, Y[:3], Phi[:3])
pred_before = baseline.predict(future_times, coords, future_phi)
Y_modified_future[3:5] += 1000.0
pred_after = baseline.predict(future_times, coords, future_phi)
assert np.allclose(pred_before.mean, pred_after.mean)
assert np.all(pred_before.variance > 0.0)
```

**Line-by-Line Implementation Logic**

- 1. `baseline.fit_initial_task(times[:3], coords, Y[:3], Phi[:3])`: Runs this function call or check as one step of the verification.
- 2. `pred_before = baseline.predict(future_times, coords, future_phi)`: Computes `baseline.predict(future_times, coords, future_phi)` and stores the result in `pred_before` for later checks.
- 3. `Y_modified_future[3:5] += 1000.0`: Computes `1000.0` and stores the result in `Y_modified_future[3:5] +` for later checks.
- 4. `pred_after = baseline.predict(future_times, coords, future_phi)`: Computes `baseline.predict(future_times, coords, future_phi)` and stores the result in `pred_after` for later checks.
- 5. `assert np.allclose(pred_before.mean, pred_after.mean)`: Checks that two floating-point arrays are numerically close within tolerance.
- 6. `assert np.all(pred_before.variance > 0.0)`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `allclose(a, b)` compares floating-point arrays with numerical tolerance.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.

**Verification Logic**

Checks that the persistence baseline predicts without reading future labels and returns finite positive variance.

### 38. `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[ClimatologyBaseline]`

- File: `tests/test_hipposvgp_era5_loader_baselines.py`
- Group: ERA5 Baseline Pipeline
- Result: `PASSED`

**Formula or Contract**

$\hat y_s=\frac{1}{|\mathcal{D}_{seen}|}\sum_{t\in seen}y_{t,s},\quad \sigma_s^2=\operatorname{Var}(y_{t,s}-\hat y_s)$

**Code Snippet**

```python
baseline = ClimatologyBaseline()
baseline.fit_initial_task(times[:3], coords, Y[:3], build_phi_features(times[:3], coords))
pred = baseline.predict(future_times, coords, future_phi)
assert pred.mean.shape == (2, 2)
assert np.all(pred.variance > 0.0)
```

**Line-by-Line Implementation Logic**

- 1. `baseline = ClimatologyBaseline()`: Computes `ClimatologyBaseline()` and stores the result in `baseline` for later checks.
- 2. `baseline.fit_initial_task(times[:3], coords, Y[:3], build_phi_features(times[:3], coords))`: Runs this function call or check as one step of the verification.
- 3. `pred = baseline.predict(future_times, coords, future_phi)`: Computes `baseline.predict(future_times, coords, future_phi)` and stores the result in `pred` for later checks.
- 4. `assert pred.mean.shape == (2, 2)`: Runs a test assertion; if this condition is false, the test fails.
- 5. `assert np.all(pred.variance > 0.0)`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `.shape` reports an array's dimensions, for example `(T, S)`.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.

**Verification Logic**

Checks that the climatology baseline estimates mean and variance from seen history only, with no future-label leakage.

### 39. `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[RidgeBaseline]`

- File: `tests/test_hipposvgp_era5_loader_baselines.py`
- Group: ERA5 Baseline Pipeline
- Result: `PASSED`

**Formula or Contract**

$\hat\beta=(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top y,\quad \sigma^2=\operatorname{Var}(y-\Phi\hat\beta)$

**Code Snippet**

```python
baseline = RidgeBaseline()
baseline.fit_initial_task(times[:3], coords, Y[:3], build_phi_features(times[:3], coords))
pred = baseline.predict(future_times, coords, future_phi)
assert pred.mean.shape == (2, 2)
assert np.all(pred.variance > 0.0)
```

**Line-by-Line Implementation Logic**

- 1. `baseline = RidgeBaseline()`: Computes `RidgeBaseline()` and stores the result in `baseline` for later checks.
- 2. `baseline.fit_initial_task(times[:3], coords, Y[:3], build_phi_features(times[:3], coords))`: Runs this function call or check as one step of the verification.
- 3. `pred = baseline.predict(future_times, coords, future_phi)`: Computes `baseline.predict(future_times, coords, future_phi)` and stores the result in `pred` for later checks.
- 4. `assert pred.mean.shape == (2, 2)`: Runs a test assertion; if this condition is false, the test fails.
- 5. `assert np.all(pred.variance > 0.0)`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `.shape` reports an array's dimensions, for example `(T, S)`.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.

**Verification Logic**

Checks that the ridge baseline has the expected closed-form fit, output shape, no-leakage future prediction, and residual variance.

### 40. `test_gpytorch_baselines_smoke_if_available`

- File: `tests/test_hipposvgp_era5_loader_baselines.py`
- Group: ERA5 Baseline Pipeline
- Result: `PASSED`

**Formula or Contract**

$p(y_*|\mathcal{D})=\mathcal{N}(\mu_*,\sigma_{*,likelihood}^2),\quad \sigma_{*,likelihood}^2>0$

**Code Snippet**

```python
for baseline in [IndependentTemporalGPBaseline, GPyTorchSGPRBaseline, GPyTorchSVGPBaseline]:
    baseline.fit_initial_task(times, coords, Y, Phi)
    pred = baseline.predict(times[:2], coords, build_phi_features(times[:2], coords))
    assert pred.mean.shape == (2, 1)
    assert np.all(pred.variance > 0.0)
```

**Line-by-Line Implementation Logic**

- 1. `for baseline in [IndependentTemporalGPBaseline, GPyTorchSGPRBaseline, GPyTorchSVGPBaseline]:`: Starts a loop; the indented lines below it run once per iteration.
- 2. `    baseline.fit_initial_task(times, coords, Y, Phi)`: Runs this function call or check as one step of the verification.
- 3. `    pred = baseline.predict(times[:2], coords, build_phi_features(times[:2], coords))`: Computes `baseline.predict(times[:2], coords, build_phi_features(times[:2], coords))` and stores the result in `pred` for later checks.
- 4. `    assert pred.mean.shape == (2, 1)`: Runs a test assertion; if this condition is false, the test fails.
- 5. `    assert np.all(pred.variance > 0.0)`: Runs a test assertion; if this condition is false, the test fails.

**Python Syntax Notes**

- `=` assigns the value on the right to the variable name on the left.
- `assert condition` makes the test fail immediately if the condition is false.
- `.shape` reports an array's dimensions, for example `(T, S)`.
- `for ... in ...:` repeats the indented block.
- Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.

**Verification Logic**

Checks that the GPyTorch independent GP, SGPR, and SVGP baselines can train and return finite positive likelihood predictive variance.
