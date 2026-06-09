# Route B 40 个数学理论与基线管线验证实验：公式-代码-逻辑对照

本文档面向代码基础较弱的读者，逐个解释当前项目中 40 个验证测试。前 34 个是 Route B/Stage-1 的核心数学与实现验证；后 6 个是新增 ERA5 loader 与 baseline 管线验证。每个块都说明：验证哪个数学公式或数据契约、对应代码片段、以及为什么这个测试能证明实现没有偏离理论或实验协议。

验证命令：

```bash
uv run --no-sync pytest -q
uv run --no-sync python scripts/verify_joint_ssgp_kron_derivations.py
```

结果：`40 passed, 1 warning`。warning 来自本机 CUDA driver 版本提示，不影响 CPU 数值验证。

## Python 语法速查

下面 40 个测试都只用到少量常见 Python/Numpy/PyTorch 语法。先掌握这些符号，后面每个代码块会更容易读：

- `=`：赋值，把右边的结果存到左边变量。
- `assert 条件`：测试断言，条件不成立时测试失败。
- `for ... in ...:`：循环，冒号后缩进的代码会重复执行。
- `if ...:`：条件判断，只有条件成立才执行缩进块。
- `A @ B`：矩阵乘法。
- `A.T`：矩阵转置。
- `x[:3]`：切片，取前 3 个元素；`x[3:5]` 取第 3 到第 4 个元素。
- `dict['key']`：从字典里按名字取值。
- `.shape`：数组形状，例如 `(T, S)`。
- `np.allclose(a,b)` / `torch.allclose(a,b)`：判断两个浮点数组是否近似相等。
- `np.isfinite(x)`：检查结果不是 NaN 或无穷大。
- `...`：文档中的省略号，表示省略了不影响理解的完整参数。

## 总览表

| # | 测试 | 分组 | 主要验证对象 | 结果 |
|---:|---|---|---|---|
| 1 | `test_temporal_and_spatial_shape_consistency` | Stage-1 Kronecker STGP | 验证时间核和空间核模块输出的矩阵维度与 Kronecker STGP 的理论对象一致。 | PASSED |
| 2 | `test_kronecker_projection_shapes` | Stage-1 Kronecker STGP | 验证 Kronecker 投影矩阵的行数等于所有时空观测点，列数等于时间诱导点乘空间诱导点。 | PASSED |
| 3 | `test_small_synthetic_batch_matches_dense_solution` | Stage-1 Kronecker STGP | 把模型的 batch posterior mean 与显式 dense GP 线性高斯后验公式逐元素比较。 | PASSED |
| 4 | `test_small_synthetic_training_reduces_loss` | Stage-1 Kronecker STGP | 验证训练目标可被优化器下降，排除 loss 符号、梯度和参数注册错误。 | PASSED |
| 5 | `test_blockwise_forward_returns_consistent_shapes` | Stage-1 Kronecker STGP | 验证 blockwise 前向传播在分块后仍返回与空间网格匹配的预测形状。 | PASSED |
| 6 | `test_online_recursion_matches_batch_solution` | Stage-1 Kronecker STGP | 固定 horizon 下，online precision 累加应与一次性 batch posterior 完全一致。 | PASSED |
| 7 | `test_temporal_cross_covariance_is_consistent` | Stage-1 Kronecker STGP | 验证不同 temporal horizons 之间的 cross covariance 满足核矩阵对称性。 | PASSED |
| 8 | `test_online_local_horizon_transfer_updates_state` | Stage-1 Kronecker STGP | 验证 local horizon 改变时 transfer matrix 存在、有限，并且 online 状态能连续更新。 | PASSED |
| 9 | `test_online_predictive_variance_matches_dense_precision_solver` | Stage-1 Kronecker STGP | 验证 Sylvester/precision solver 给出的 latent variance 与显式 dense precision correction 一致。 | PASSED |
| 10 | `test_load_processed_era5_task_aligns_locations` | ERA5 Data Contract | 验证 processed ERA5 每个 location 的乱序时间戳会被排序，并在共同时间轴上对齐。 | PASSED |
| 11 | `test_load_processed_era5_task_resplit_rebuilds_longer_validation` | ERA5 Data Contract | 验证 chronological resplit 不打乱时间顺序，避免 online/future evaluation 泄漏。 | PASSED |
| 12 | `test_load_processed_era5_tasks_concatenates_multiple_tasks` | ERA5 Data Contract | 验证多个 ERA5 task 可以按时间拼接，并保持空间位置一致。 | PASSED |
| 13 | `test_discover_and_count_processed_era5_tasks` | ERA5 Data Contract | 验证 task discovery 和 location counting 与磁盘上的 processed 数据结构一致。 | PASSED |
| 14 | `test_spatial_inducing_fps_spreads_across_domain` | Spatial Inducing Contract | 验证 farthest-point spatial inducing selection 能覆盖左右空间边界。 | PASSED |
| 15 | `test_Lon_kron_identity` | Kronecker Derivations | 验证 changing-basis transfer 的 dense 形式可化简为时间维 transfer 与空间单位矩阵的 Kronecker 积。 | PASSED |
| 16 | `test_old_likelihood_transfer_kron_identity` | Kronecker Derivations | 验证旧 likelihood precision 在新时间基上投影后仍保持 Kronecker 分解。 | PASSED |
| 17 | `test_fixed_basis_streaming_equals_batch` | Kronecker Derivations | 验证固定 basis 下 online natural parameters 累加与 batch Gaussian posterior 完全一致。 | PASSED |
| 18 | `test_no_linear_mean_reduces_to_gp_only` | Kronecker Derivations | 验证线性均值为零时模型退化为 GP-only SSGP update。 | PASSED |
| 19 | `test_no_old_data_transfer_zero` | Kronecker Derivations | 验证没有旧数据时 transfer 项不会凭空产生旧 likelihood 信息。 | PASSED |
| 20 | `test_projected_prior_dense_marginalization` | Kronecker Derivations | 验证旧 projected-prior dense marginalization 公式和 structured old-likelihood transfer 公式都能匹配 dense reference。 | PASSED |
| 21 | `test_old_likelihood_dense_vs_structured_information_vector` | Kronecker Derivations | 验证旧 information vector 的 dense transfer 与矩阵形式 $H_oL_t$ 一致。 | PASSED |
| 22 | `test_model_one_block_no_nan` | Model Sanity | 验证单个 block 更新不会产生 NaN/inf，保护数值稳定性。 | PASSED |
| 23 | `test_model_multi_block_no_nan` | Model Sanity | 验证多 block online transfer 后均值和状态仍有限。 | PASSED |
| 24 | `test_baseline_imports_still_work` | Model Sanity | 验证新增 Route B 代码没有破坏原 batch/online 入口和旧 API。 | PASSED |
| 25 | `test_routeB_dense_vs_structured_new_block_likelihood` | Route B Theory | 验证新 block joint likelihood 的 dense 统计量与 structured Kronecker 统计量一致。 | PASSED |
| 26 | `test_routeB_dense_vs_structured_joint_old_likelihood_transfer` | Route B Theory | 验证保留 beta-u cross block 后，旧 joint likelihood 的 basis transfer 仍与 dense 变换一致。 | PASSED |
| 27 | `test_routeB_schur_posterior_recovery_vs_dense_inverse` | Route B Theory | 验证 Schur complement + Sylvester solves 恢复的 posterior mean/covariance 与 dense inverse 一致。 | PASSED |
| 28 | `test_routeB_cross_covariance_matches_dense_reference` | Route B Theory | 验证 Route B 保留的 beta-u posterior cross covariance 与 dense reference 完全一致。 | PASSED |
| 29 | `test_mean_field_has_zero_cross_covariance_and_differs_when_coupling_nonzero` | Route B Theory | 验证 mean-field 在强 coupling 下确实丢失 cross covariance，并导致预测方差/均值偏离 dense posterior。 | PASSED |
| 30 | `test_routeB_predictive_variance_matches_dense_joint_posterior_and_differs_from_mean_field` | Route B Theory | 验证 Route B 预测方差包含 dense joint posterior 的 cross term，并区别于 mean-field 方差。 | PASSED |
| 31 | `test_routeB_fixed_basis_streaming_equals_batch_joint_posterior` | Route B Theory | 验证固定 basis 下 Route B streaming joint posterior 与 batch joint posterior 一致。 | PASSED |
| 32 | `test_routeB_no_linear_mean_reduces_to_gp_only` | Route B Theory | 验证没有线性均值时，Route B 不引入额外行为，退化为原 GP-only 更新。 | PASSED |
| 33 | `test_routeB_zero_cross_feature_sanity` | Route B Theory | 验证 cross block 为零时 Route B 方差自动分解为 beta 项和 GP 项的相加形式。 | PASSED |
| 34 | `test_predictive_variance_respects_kernel_amplitude` | Route B Theory | 验证 sparse conditional residual variance 显式尊重非单位 kernel amplitude，修复 coverage/NLL 风险点。 | PASSED |
| 35 | `test_loader_shapes_and_blocks` | ERA5 Baseline Pipeline | 验证新的 HiPPO-SVGP ERA5 loader 输出的 Y、coords、Phi 和 online block split 与 baseline/Route B 需要的形状一致。 | PASSED |
| 36 | `test_loader_converts_to_routeb_factors` | ERA5 Baseline Pipeline | 验证 ERA5 loader 可以转换成 Route B 的 BlockFactors，不改变主模型公式。 | PASSED |
| 37 | `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[PersistenceBaseline]` | ERA5 Baseline Pipeline | 验证 persistence baseline 的 future prediction 不读取 future labels，并且残差方差有限且为正。 | PASSED |
| 38 | `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[ClimatologyBaseline]` | ERA5 Baseline Pipeline | 验证 climatology baseline 的均值和 variance 只由 seen history 估计，future labels 修改不影响预测。 | PASSED |
| 39 | `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[RidgeBaseline]` | ERA5 Baseline Pipeline | 验证 ridge baseline 的闭式解、输出形状、no-leakage future prediction 和 residual variance 都可用。 | PASSED |
| 40 | `test_gpytorch_baselines_smoke_if_available` | ERA5 Baseline Pipeline | 验证 independent GP、SGPR 和 SVGP 三个 GPyTorch baseline 都能训练、返回 likelihood predictive variance，并且 variance 有限为正。 | PASSED |

## 逐项讲解

### 1. `test_temporal_and_spatial_shape_consistency`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：Stage-1 Kronecker STGP
- 结果：`PASSED`

**数学公式片段**

$K_{uu}^{(t)}\in\mathbb{R}^{M_t\times M_t},\ K_{fu}^{(t)}\in\mathbb{R}^{T\times M_t},\ K_{zz}^{(s)}\in\mathbb{R}^{M_s\times M_s},\ K_{xz}^{(s)}\in\mathbb{R}^{S\times M_s}$

**代码片段**

```python
temporal = model.build_temporal_covariances(times)
spatial_cov = model.build_spatial_covariances(spatial)
assert temporal.kuu_t.shape == (4, 4)
assert temporal.kfu_t.shape == (5, 4)
```

**代码逐行实现逻辑翻译**

- 1. `temporal = model.build_temporal_covariances(times)`：根据输入时间点构造时间方向的核矩阵和交叉核矩阵。
- 2. `spatial_cov = model.build_spatial_covariances(spatial)`：根据空间坐标构造空间方向的核矩阵和交叉核矩阵。
- 3. `assert temporal.kuu_t.shape == (4, 4)`：检查输出数组的维度是否符合理论设计的形状。
- 4. `assert temporal.kfu_t.shape == (5, 4)`：检查输出数组的维度是否符合理论设计的形状。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证时间核和空间核模块输出的矩阵维度与 Kronecker STGP 的理论对象一致。

### 2. `test_kronecker_projection_shapes`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：Stage-1 Kronecker STGP
- 结果：`PASSED`

**数学公式片段**

$A=A_t\otimes A_s,\quad A_t=K_{fu}^{(t)}K_{uu}^{(t)-1},\quad A_s=K_{xz}^{(s)}K_{zz}^{(s)-1}$

**代码片段**

```python
projection = model.build_projection(times, spatial)
assert projection.a_t.shape == (5, 4)
assert projection.a_s.shape == (4, 4)
assert torch.kron(projection.a_t, projection.a_s).shape == (20, 16)
```

**代码逐行实现逻辑翻译**

- 1. `projection = model.build_projection(times, spatial)`：计算时间投影和空间投影，后面可以通过 Kronecker 积组成完整时空投影。
- 2. `assert projection.a_t.shape == (5, 4)`：检查输出数组的维度是否符合理论设计的形状。
- 3. `assert projection.a_s.shape == (4, 4)`：检查输出数组的维度是否符合理论设计的形状。
- 4. `assert torch.kron(projection.a_t, projection.a_s).shape == (20, 16)`：检查输出数组的维度是否符合理论设计的形状。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证 Kronecker 投影矩阵的行数等于所有时空观测点，列数等于时间诱导点乘空间诱导点。

### 3. `test_small_synthetic_batch_matches_dense_solution`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：Stage-1 Kronecker STGP
- 结果：`PASSED`

**数学公式片段**

$\Lambda=K_{uu}^{-1}+\sigma^{-2}A^\top A,\quad m=\Lambda^{-1}\sigma^{-2}A^\top y$

**代码片段**

```python
precision = Kuu_inv + torch.reciprocal(sigma2) * (a_dense.T @ a_dense)
info = torch.reciprocal(sigma2) * (a_dense.T @ y.reshape(-1))
mean = torch.linalg.solve(precision, info)
assert torch.allclose(output['posterior_mean_u'], mean)
```

**代码逐行实现逻辑翻译**

- 1. `precision = Kuu_inv + torch.reciprocal(sigma2) * (a_dense.T @ a_dense)`：按理论公式构造 posterior precision 矩阵，也就是高斯后验的精度矩阵。
- 2. `info = torch.reciprocal(sigma2) * (a_dense.T @ y.reshape(-1))`：按理论公式构造 information vector，也就是 precision 形式里的右端项。
- 3. `mean = torch.linalg.solve(precision, info)`：解线性方程得到 posterior mean，作为模型输出要对照的 dense reference。
- 4. `assert torch.allclose(output['posterior_mean_u'], mean)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。
- `.T` 表示矩阵转置。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。
- `np.linalg` / `torch.linalg` 是线性代数工具箱，用于求解线性方程、Cholesky 分解、特征值等。

**验证逻辑**

把模型的 batch posterior mean 与显式 dense GP 线性高斯后验公式逐元素比较。

### 4. `test_small_synthetic_training_reduces_loss`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：Stage-1 Kronecker STGP
- 结果：`PASSED`

**数学公式片段**

$\mathcal{L}_{20}<\mathcal{L}_{0}$

**代码片段**

```python
for _ in range(20):
    output = model(times, spatial, y, cache_posterior=False)
    output['loss'].backward()
    optimizer.step()
assert losses[-1] < losses[0]
```

**代码逐行实现逻辑翻译**

- 1. `for _ in range(20):`：重复执行固定次数，用来模拟训练迭代或分块循环。
- 2. `    output = model(times, spatial, y, cache_posterior=False)`：计算右侧 `model(times, spatial, y, cache_posterior=False)`，并把结果保存到变量 `output`，供后续检查使用。
- 3. `    output['loss'].backward()`：根据当前 loss 反向传播，计算每个可训练参数应该如何调整。
- 4. `    optimizer.step()`：让优化器根据刚刚计算出的梯度更新模型参数。
- 5. `assert losses[-1] < losses[0]`：执行一个测试检查；只要这个条件不成立，该测试就会失败。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `for ... in ...:` 是循环；冒号后面缩进的多行代码会重复执行。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `range(n)` 生成从 0 到 n-1 的整数序列，常与 `for` 循环配合。

**验证逻辑**

验证训练目标可被优化器下降，排除 loss 符号、梯度和参数注册错误。

### 5. `test_blockwise_forward_returns_consistent_shapes`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：Stage-1 Kronecker STGP
- 结果：`PASSED`

**数学公式片段**

$\{B_1,\ldots,B_N\},\quad \hat{Y}_{B_n}\in\mathbb{R}^{|B_n|\times S}$

**代码片段**

```python
blockwise = model.forward_blockwise(..., block_size=2)
assert len(blockwise.block_outputs) == 3
assert blockwise.block_outputs[-1]['train_mean'].shape[1] == spatial.shape[0]
```

**代码逐行实现逻辑翻译**

- 1. `blockwise = model.forward_blockwise(..., block_size=2)`：计算右侧 `model.forward_blockwise(..., block_size=2)`，并把结果保存到变量 `blockwise`，供后续检查使用。
- 2. `assert len(blockwise.block_outputs) == 3`：检查实际结果是否等于预期值，通常用于验证 block 数量、shape 或字符串标记。
- 3. `assert blockwise.block_outputs[-1]['train_mean'].shape[1] == spatial.shape[0]`：检查输出数组的维度是否符合理论设计的形状。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证 blockwise 前向传播在分块后仍返回与空间网格匹配的预测形状。

### 6. `test_online_recursion_matches_batch_solution`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：Stage-1 Kronecker STGP
- 结果：`PASSED`

**数学公式片段**

$\Lambda_N=\Lambda_0+\sum_n\sigma^{-2}A_n^\top A_n,\quad h_N=h_0+\sum_n\sigma^{-2}A_n^\top y_n$

**代码片段**

```python
batch_output = batch_model(..., materialize_posterior_cov=True)
for block in blocks:
    online_model.update_block(...)
assert torch.allclose(batch_output['posterior_mean_u'], online_model.state.m)
```

**代码逐行实现逻辑翻译**

- 1. `batch_output = batch_model(..., materialize_posterior_cov=True)`：一次性用所有数据训练 batch 模型，作为 online 递推结果的 dense/batch 参考答案。
- 2. `for block in blocks:`：逐个遍历 online 时间块，模拟持续学习中一块一块接收数据的过程。
- 3. `    online_model.update_block(...)`：把一个 online block 送入旧 online 模型，检查递推更新是否和 batch 解一致。
- 4. `assert torch.allclose(batch_output['posterior_mean_u'], online_model.state.m)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `for ... in ...:` 是循环；冒号后面缩进的多行代码会重复执行。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

固定 horizon 下，online precision 累加应与一次性 batch posterior 完全一致。

### 7. `test_temporal_cross_covariance_is_consistent`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：Stage-1 Kronecker STGP
- 结果：`PASSED`

**数学公式片段**

$K_{ab}=K_{ba}^{\top}$

**代码片段**

```python
cross_ab = builder.compute_kuu_t_cross(horizon_a, horizon_b)
cross_ba = builder.compute_kuu_t_cross(horizon_b, horizon_a)
assert torch.allclose(cross_ab, cross_ba.T)
```

**代码逐行实现逻辑翻译**

- 1. `cross_ab = builder.compute_kuu_t_cross(horizon_a, horizon_b)`：计算右侧 `builder.compute_kuu_t_cross(horizon_a, horizon_b)`，并把结果保存到变量 `cross_ab`，供后续检查使用。
- 2. `cross_ba = builder.compute_kuu_t_cross(horizon_b, horizon_a)`：计算右侧 `builder.compute_kuu_t_cross(horizon_b, horizon_a)`，并把结果保存到变量 `cross_ba`，供后续检查使用。
- 3. `assert torch.allclose(cross_ab, cross_ba.T)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `.T` 表示矩阵转置。

**验证逻辑**

验证不同 temporal horizons 之间的 cross covariance 满足核矩阵对称性。

### 8. `test_online_local_horizon_transfer_updates_state`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：Stage-1 Kronecker STGP
- 结果：`PASSED`

**数学公式片段**

$u_o\mapsto u_n,\quad L_{on}=K_{on}K_{nn}^{-1}$

**代码片段**

```python
first = online_model.update_block(..., horizon=first_block_horizon)
second = online_model.update_block(..., horizon=second_block_horizon)
assert first['temporal_transfer'].shape == (4, 4)
```

**代码逐行实现逻辑翻译**

- 1. `first = online_model.update_block(..., horizon=first_block_horizon)`：计算右侧 `online_model.update_block(..., horizon=first_block_horizon)`，并把结果保存到变量 `first`，供后续检查使用。
- 2. `second = online_model.update_block(..., horizon=second_block_horizon)`：计算右侧 `online_model.update_block(..., horizon=second_block_horizon)`，并把结果保存到变量 `second`，供后续检查使用。
- 3. `assert first['temporal_transfer'].shape == (4, 4)`：检查输出数组的维度是否符合理论设计的形状。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证 local horizon 改变时 transfer matrix 存在、有限，并且 online 状态能连续更新。

### 9. `test_online_predictive_variance_matches_dense_precision_solver`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：Stage-1 Kronecker STGP
- 结果：`PASSED`

**数学公式片段**

$\operatorname{Var}(f_*)=k_{**}-a_*K_{uu}a_*^\top+a_*\Lambda^{-1}a_*^\top$

**代码片段**

```python
dense_latent_var = prior_diag - projected_prior_diag + dense_posterior_correction
assert torch.allclose(pred['latent_var'], dense_latent_var)
```

**代码逐行实现逻辑翻译**

- 1. `dense_latent_var = prior_diag - projected_prior_diag + dense_posterior_correction`：按 dense precision solver 的公式显式计算 latent predictive variance。
- 2. `assert torch.allclose(pred['latent_var'], dense_latent_var)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。

**验证逻辑**

验证 Sylvester/precision solver 给出的 latent variance 与显式 dense precision correction 一致。

### 10. `test_load_processed_era5_task_aligns_locations`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：ERA5 Data Contract
- 结果：`PASSED`

**数学公式片段**

$Y\in\mathbb{R}^{T\times S}$ with shared sorted times over all selected locations

**代码片段**

```python
task = load_processed_era5_task(...)
assert task.train.times.tolist() == [0.0, 1.0, 2.0]
assert task.train.observations.shape == (3, 2)
```

**代码逐行实现逻辑翻译**

- 1. `task = load_processed_era5_task(...)`：读取 processed ERA5 task，并检查它是否正确排序、对齐或拼接多个任务。
- 2. `assert task.train.times.tolist() == [0.0, 1.0, 2.0]`：检查实际结果是否等于预期值，通常用于验证 block 数量、shape 或字符串标记。
- 3. `assert task.train.observations.shape == (3, 2)`：检查输出数组的维度是否符合理论设计的形状。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。

**验证逻辑**

验证 processed ERA5 每个 location 的乱序时间戳会被排序，并在共同时间轴上对齐。

### 11. `test_load_processed_era5_task_resplit_rebuilds_longer_validation`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：ERA5 Data Contract
- 结果：`PASSED`

**数学公式片段**

$\{1,\ldots,T\}=\mathcal{T}_{train}\cup\mathcal{T}_{val}\cup\mathcal{T}_{test}$ chronologically

**代码片段**

```python
task = load_processed_era5_task(..., resplit=True)
assert task.train.times.tolist() == [0.0, 1.0, 2.0]
assert task.val.times.tolist() == [3.0, 4.0]
```

**代码逐行实现逻辑翻译**

- 1. `task = load_processed_era5_task(..., resplit=True)`：读取 processed ERA5 task，并检查它是否正确排序、对齐或拼接多个任务。
- 2. `assert task.train.times.tolist() == [0.0, 1.0, 2.0]`：检查实际结果是否等于预期值，通常用于验证 block 数量、shape 或字符串标记。
- 3. `assert task.val.times.tolist() == [3.0, 4.0]`：检查实际结果是否等于预期值，通常用于验证 block 数量、shape 或字符串标记。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。

**验证逻辑**

验证 chronological resplit 不打乱时间顺序，避免 online/future evaluation 泄漏。

### 12. `test_load_processed_era5_tasks_concatenates_multiple_tasks`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：ERA5 Data Contract
- 结果：`PASSED`

**数学公式片段**

$Y_{1:K}=[Y^{(1)};Y^{(2)};\ldots;Y^{(K)}]$ with shared spatial coordinates

**代码片段**

```python
task = load_processed_era5_tasks([task_1, task_2], resplit=True)
assert task.train.observations.shape == (6, 2)
assert task.test.times.tolist() == [9.0, 10.0, 11.0]
```

**代码逐行实现逻辑翻译**

- 1. `task = load_processed_era5_tasks([task_1, task_2], resplit=True)`：读取 processed ERA5 task，并检查它是否正确排序、对齐或拼接多个任务。
- 2. `assert task.train.observations.shape == (6, 2)`：检查输出数组的维度是否符合理论设计的形状。
- 3. `assert task.test.times.tolist() == [9.0, 10.0, 11.0]`：检查实际结果是否等于预期值，通常用于验证 block 数量、shape 或字符串标记。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。

**验证逻辑**

验证多个 ERA5 task 可以按时间拼接，并保持空间位置一致。

### 13. `test_discover_and_count_processed_era5_tasks`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：ERA5 Data Contract
- 结果：`PASSED`

**数学公式片段**

$|\mathcal{S}_{task}|=\#\{\text{selected location files}\}$

**代码片段**

```python
task_dirs = discover_processed_era5_task_dirs(root, ['task_1', 'task_2'])
assert count_processed_era5_locations(task_dirs[0]) == 2
```

**代码逐行实现逻辑翻译**

- 1. `task_dirs = discover_processed_era5_task_dirs(root, ['task_1', 'task_2'])`：发现指定的 ERA5 task 文件夹，确认 loader 能找到正确数据目录。
- 2. `assert count_processed_era5_locations(task_dirs[0]) == 2`：检查实际结果是否等于预期值，通常用于验证 block 数量、shape 或字符串标记。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。

**验证逻辑**

验证 task discovery 和 location counting 与磁盘上的 processed 数据结构一致。

### 14. `test_spatial_inducing_fps_spreads_across_domain`

- 文件：`stvgp_kronecker/tests/test_stage1.py`
- 分组：Spatial Inducing Contract
- 结果：`PASSED`

**数学公式片段**

$Z_s=\operatorname{FPS}(X_s)$ should cover the spatial domain better than first-N selection

**代码片段**

```python
first = select_spatial_inducing_points(..., selection_method='first')
fps = select_spatial_inducing_points(..., selection_method='fps')
assert float(fps[:, 0].min()) == -10.0
assert float(fps[:, 0].max()) == 2.0
```

**代码逐行实现逻辑翻译**

- 1. `first = select_spatial_inducing_points(..., selection_method='first')`：用简单 first-N 方式选择空间诱导点，作为对比基线。
- 2. `fps = select_spatial_inducing_points(..., selection_method='fps')`：用 farthest-point sampling 选择空间诱导点，检查它是否覆盖空间边界。
- 3. `assert float(fps[:, 0].min()) == -10.0`：检查实际结果是否等于预期值，通常用于验证 block 数量、shape 或字符串标记。
- 4. `assert float(fps[:, 0].max()) == 2.0`：检查实际结果是否等于预期值，通常用于验证 block 数量、shape 或字符串标记。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。

**验证逻辑**

验证 farthest-point spatial inducing selection 能覆盖左右空间边界。

### 15. `test_Lon_kron_identity`

- 文件：`tests/test_joint_ssgp_kron_derivations.py`
- 分组：Kronecker Derivations
- 结果：`PASSED`

**数学公式片段**

$L_{on}=K_{on}K_{nn}^{-1}=(K_{on}^{t}K_{nn}^{t-1})\otimes I_s$

**代码片段**

```python
L_t = compute_Lt(K_on_t, K_nn_t)
L_dense = kron(K_on_t, Ks) @ inv(kron(K_nn_t, Ks))
L_kron = kron(L_t, I_s)
assert err < 1e-8
```

**代码逐行实现逻辑翻译**

- 1. `L_t = compute_Lt(K_on_t, K_nn_t)`：计算右侧 `compute_Lt(K_on_t, K_nn_t)`，并把结果保存到变量 `L_t`，供后续检查使用。
- 2. `L_dense = kron(K_on_t, Ks) @ inv(kron(K_nn_t, Ks))`：计算右侧 `kron(K_on_t, Ks) @ inv(kron(K_nn_t, Ks))`，并把结果保存到变量 `L_dense`，供后续检查使用。
- 3. `L_kron = kron(L_t, I_s)`：计算右侧 `kron(L_t, I_s)`，并把结果保存到变量 `L_kron`，供后续检查使用。
- 4. `assert err < 1e-8`：执行一个测试检查；只要这个条件不成立，该测试就会失败。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。

**验证逻辑**

验证 changing-basis transfer 的 dense 形式可化简为时间维 transfer 与空间单位矩阵的 Kronecker 积。

### 16. `test_old_likelihood_transfer_kron_identity`

- 文件：`tests/test_joint_ssgp_kron_derivations.py`
- 分组：Kronecker Derivations
- 结果：`PASSED`

**数学公式片段**

$L_{on}^{\top}(B_o\otimes G)L_{on}=(L_t^{\top}B_oL_t)\otimes G$

**代码片段**

```python
Lambda_dense = L_dense.T @ kron(B_old, G) @ L_dense
Lambda_kron = kron(transfer_temporal_precision(B_old, L_t), G)
assert err < 1e-8
```

**代码逐行实现逻辑翻译**

- 1. `Lambda_dense = L_dense.T @ kron(B_old, G) @ L_dense`：计算右侧 `L_dense.T @ kron(B_old, G) @ L_dense`，并把结果保存到变量 `Lambda_dense`，供后续检查使用。
- 2. `Lambda_kron = kron(transfer_temporal_precision(B_old, L_t), G)`：计算右侧 `kron(transfer_temporal_precision(B_old, L_t), G)`，并把结果保存到变量 `Lambda_kron`，供后续检查使用。
- 3. `assert err < 1e-8`：执行一个测试检查；只要这个条件不成立，该测试就会失败。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。
- `.T` 表示矩阵转置。

**验证逻辑**

验证旧 likelihood precision 在新时间基上投影后仍保持 Kronecker 分解。

### 17. `test_fixed_basis_streaming_equals_batch`

- 文件：`tests/test_joint_ssgp_kron_derivations.py`
- 分组：Kronecker Derivations
- 结果：`PASSED`

**数学公式片段**

$\Lambda_{stream}=\Lambda_0+\sum_n H_n^\top H_n/\sigma^2=\Lambda_{batch}$

**代码片段**

```python
for H, y in blocks:
    Lambda_stream += H.T @ H / sigma2
    h_stream += H.T @ y / sigma2
assert mean_err < 1e-8 and prec_err < 1e-8
```

**代码逐行实现逻辑翻译**

- 1. `for H, y in blocks:`：逐个遍历 online 时间块，模拟持续学习中一块一块接收数据的过程。
- 2. `    Lambda_stream += H.T @ H / sigma2`：计算右侧 `H.T @ H / sigma2`，并把结果保存到变量 `Lambda_stream +`，供后续检查使用。
- 3. `    h_stream += H.T @ y / sigma2`：计算右侧 `H.T @ y / sigma2`，并把结果保存到变量 `h_stream +`，供后续检查使用。
- 4. `assert mean_err < 1e-8 and prec_err < 1e-8`：执行一个测试检查；只要这个条件不成立，该测试就会失败。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `for ... in ...:` 是循环；冒号后面缩进的多行代码会重复执行。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。
- `.T` 表示矩阵转置。

**验证逻辑**

验证固定 basis 下 online natural parameters 累加与 batch Gaussian posterior 完全一致。

### 18. `test_no_linear_mean_reduces_to_gp_only`

- 文件：`tests/test_joint_ssgp_kron_derivations.py`
- 分组：Kronecker Derivations
- 结果：`PASSED`

**数学公式片段**

$\Phi=0\Rightarrow \beta=0,\quad H_u=C^\top YT/\sigma^2,\quad B=T^\top T/\sigma^2$

**代码片段**

```python
state = model.update_block_ssgp_transfer(y_vec=y, Phi=zeros, ...)
assert allclose(state.beta_mean, 0)
assert allclose(state.B_temporal, B_gp)
assert allclose(state.H_info, H_gp)
```

**代码逐行实现逻辑翻译**

- 1. `state = model.update_block_ssgp_transfer(y_vec=y, Phi=zeros, ...)`：把当前 block 的观测数据送进模型，更新 posterior 状态；返回的 state 保存更新后的自然参数、均值和方差结构。
- 2. `assert allclose(state.beta_mean, 0)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。
- 3. `assert allclose(state.B_temporal, B_gp)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。
- 4. `assert allclose(state.H_info, H_gp)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证线性均值为零时模型退化为 GP-only SSGP update。

### 19. `test_no_old_data_transfer_zero`

- 文件：`tests/test_joint_ssgp_kron_derivations.py`
- 分组：Kronecker Derivations
- 结果：`PASSED`

**数学公式片段**

$B_o=0,\ H_o=0\Rightarrow B_{o\to n}=0,\ H_{o\to n}=0$

**代码片段**

```python
B_trans = transfer_temporal_precision(B_old_zero, L_t)
H_trans = transfer_information_matrix(H_old_zero, L_t)
assert allclose(B_trans, 0) and allclose(H_trans, 0)
```

**代码逐行实现逻辑翻译**

- 1. `B_trans = transfer_temporal_precision(B_old_zero, L_t)`：计算右侧 `transfer_temporal_precision(B_old_zero, L_t)`，并把结果保存到变量 `B_trans`，供后续检查使用。
- 2. `H_trans = transfer_information_matrix(H_old_zero, L_t)`：计算右侧 `transfer_information_matrix(H_old_zero, L_t)`，并把结果保存到变量 `H_trans`，供后续检查使用。
- 3. `assert allclose(B_trans, 0) and allclose(H_trans, 0)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。

**验证逻辑**

验证没有旧数据时 transfer 项不会凭空产生旧 likelihood 信息。

### 20. `test_projected_prior_dense_marginalization`

- 文件：`tests/test_joint_ssgp_kron_derivations.py`
- 分组：Kronecker Derivations
- 结果：`PASSED`

**数学公式片段**

$m_n=K_{no}K_{oo}^{-1}m_o,\quad S_n=K_{nn}+K_{no}K_{oo}^{-1}(S_o-K_{oo})K_{oo}^{-1}K_{on}$

**代码片段**

```python
m_proj, S_proj = projected_prior_transfer_dense(...)
assert projected_prior_error < 1e-8
assert structured_transfer_error < 1e-8
```

**代码逐行实现逻辑翻译**

- 1. `m_proj, S_proj = projected_prior_transfer_dense(...)`：计算右侧 `projected_prior_transfer_dense(...)`，并把结果保存到变量 `m_proj, S_proj`，供后续检查使用。
- 2. `assert projected_prior_error < 1e-8`：执行一个测试检查；只要这个条件不成立，该测试就会失败。
- 3. `assert structured_transfer_error < 1e-8`：执行一个测试检查；只要这个条件不成立，该测试就会失败。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。

**验证逻辑**

验证旧 projected-prior dense marginalization 公式和 structured old-likelihood transfer 公式都能匹配 dense reference。

### 21. `test_old_likelihood_dense_vs_structured_information_vector`

- 文件：`tests/test_joint_ssgp_kron_derivations.py`
- 分组：Kronecker Derivations
- 结果：`PASSED`

**数学公式片段**

$h_{u,o\to n}=L_{on}^{\top}h_{u,o},\quad \operatorname{vec}(H_oL_t)=L_{on}^{\top}\operatorname{vec}(H_o)$

**代码片段**

```python
h_dense = L_dense.T @ vec_f(H_old)
h_kron = vec_f(transfer_information_matrix(H_old, L_t))
assert norm(h_dense - h_kron) < 1e-8
```

**代码逐行实现逻辑翻译**

- 1. `h_dense = L_dense.T @ vec_f(H_old)`：计算右侧 `L_dense.T @ vec_f(H_old)`，并把结果保存到变量 `h_dense`，供后续检查使用。
- 2. `h_kron = vec_f(transfer_information_matrix(H_old, L_t))`：计算右侧 `vec_f(transfer_information_matrix(H_old, L_t))`，并把结果保存到变量 `h_kron`，供后续检查使用。
- 3. `assert norm(h_dense - h_kron) < 1e-8`：执行一个测试检查；只要这个条件不成立，该测试就会失败。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。
- `.T` 表示矩阵转置。

**验证逻辑**

验证旧 information vector 的 dense transfer 与矩阵形式 $H_oL_t$ 一致。

### 22. `test_model_one_block_no_nan`

- 文件：`tests/test_joint_ssgp_kron_model.py`
- 分组：Model Sanity
- 结果：`PASSED`

**数学公式片段**

$\hat y=\Phi m_\beta + A\,\operatorname{vec}(M_u)$ finite

**代码片段**

```python
state = model.update_block_ssgp_transfer(...)
mean = Phi @ state.beta_mean + A @ vec_f(state.M_u)
assert np.all(np.isfinite(mean))
```

**代码逐行实现逻辑翻译**

- 1. `state = model.update_block_ssgp_transfer(...)`：把当前 block 的观测数据送进模型，更新 posterior 状态；返回的 state 保存更新后的自然参数、均值和方差结构。
- 2. `mean = Phi @ state.beta_mean + A @ vec_f(state.M_u)`：解线性方程得到 posterior mean，作为模型输出要对照的 dense reference。
- 3. `assert np.all(np.isfinite(mean))`：检查计算结果中没有 NaN 或无穷大，确认数值过程稳定。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `isfinite` 检查结果不是 `NaN` 也不是无穷大，用来确认数值计算稳定。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证单个 block 更新不会产生 NaN/inf，保护数值稳定性。

### 23. `test_model_multi_block_no_nan`

- 文件：`tests/test_joint_ssgp_kron_model.py`
- 分组：Model Sanity
- 结果：`PASSED`

**数学公式片段**

$\forall n,\ \hat y_n=\Phi_nm_{\beta,n}+A_n\operatorname{vec}(M_{u,n})$ finite

**代码片段**

```python
for block in iter_time_blocks(...):
    state = model.update_block_ssgp_transfer(..., state=state)
    assert np.all(np.isfinite(mean))
```

**代码逐行实现逻辑翻译**

- 1. `for block in iter_time_blocks(...):`：开始一个循环，对集合里的每个元素重复执行缩进代码。
- 2. `    state = model.update_block_ssgp_transfer(..., state=state)`：把当前 block 的观测数据送进模型，更新 posterior 状态；返回的 state 保存更新后的自然参数、均值和方差结构。
- 3. `    assert np.all(np.isfinite(mean))`：检查计算结果中没有 NaN 或无穷大，确认数值过程稳定。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `isfinite` 检查结果不是 `NaN` 也不是无穷大，用来确认数值计算稳定。
- `for ... in ...:` 是循环；冒号后面缩进的多行代码会重复执行。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证多 block online transfer 后均值和状态仍有限。

### 24. `test_baseline_imports_still_work`

- 文件：`tests/test_joint_ssgp_kron_model.py`
- 分组：Model Sanity
- 结果：`PASSED`

**数学公式片段**

$\text{public API remains importable after Route B additions}$

**代码片段**

```python
import stvgp_kronecker.train_batch as train_batch
import stvgp_kronecker.train_online as train_online
assert hasattr(train_batch, 'main')
```

**代码逐行实现逻辑翻译**

- 1. `import stvgp_kronecker.train_batch as train_batch`：执行这一行代码对应的函数调用或检查；它是该验证步骤中的一个中间操作。
- 2. `import stvgp_kronecker.train_online as train_online`：执行这一行代码对应的函数调用或检查；它是该验证步骤中的一个中间操作。
- 3. `assert hasattr(train_batch, 'main')`：执行一个测试检查；只要这个条件不成立，该测试就会失败。

**涉及的基础语法提示**

- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。

**验证逻辑**

验证新增 Route B 代码没有破坏原 batch/online 入口和旧 API。

### 25. `test_routeB_dense_vs_structured_new_block_likelihood`

- 文件：`tests/test_joint_ssgp_kron_routeB.py`
- 分组：Route B Theory
- 结果：`PASSED`

**数学公式片段**

$R_{\beta\beta}=\Phi^\top\Phi/\sigma^2,\ R_{\beta u}=\Phi^\top A/\sigma^2,\ R_{uu}=A^\top A/\sigma^2=(T^\top T)\otimes(C^\top C)/\sigma^2$

**代码片段**

```python
stats = joint_likelihood_stats(y, Phi, T, C, sigma2)
assert allclose(stats['R_beta_beta'], Phi.T @ Phi / sigma2)
assert allclose(stats['R_beta_u'], Phi.T @ A / sigma2)
```

**代码逐行实现逻辑翻译**

- 1. `stats = joint_likelihood_stats(y, Phi, T, C, sigma2)`：调用 structured 统计量函数，计算 Route B 新 block likelihood 的各个自然参数块。
- 2. `assert allclose(stats['R_beta_beta'], Phi.T @ Phi / sigma2)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。
- 3. `assert allclose(stats['R_beta_u'], Phi.T @ A / sigma2)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。
- `.T` 表示矩阵转置。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。

**验证逻辑**

验证新 block joint likelihood 的 dense 统计量与 structured Kronecker 统计量一致。

### 26. `test_routeB_dense_vs_structured_joint_old_likelihood_transfer`

- 文件：`tests/test_joint_ssgp_kron_routeB.py`
- 分组：Route B Theory
- 结果：`PASSED`

**数学公式片段**

$R_{\beta u,o\to n}=R_{\beta u,o}L_{on},\quad R_{uu,o\to n}=L_{on}^{\top}R_{uu,o}L_{on}$

**代码片段**

```python
R_dense = T_joint.T @ R_old @ T_joint
assert allclose(R_dense[:d, d:], transfer_R_beta_u(R_beta_u, L_t, ms))
assert allclose(R_dense[d:, d:], kron(transfer_temporal_precision(B_old, L_t), G))
```

**代码逐行实现逻辑翻译**

- 1. `R_dense = T_joint.T @ R_old @ T_joint`：用完整 dense joint transform 显式计算旧 likelihood 转移后的 precision，作为 structured transfer 的参考答案。
- 2. `assert allclose(R_dense[:d, d:], transfer_R_beta_u(R_beta_u, L_t, ms))`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。
- 3. `assert allclose(R_dense[d:, d:], kron(transfer_temporal_precision(B_old, L_t), G))`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。
- `.T` 表示矩阵转置。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。

**验证逻辑**

验证保留 beta-u cross block 后，旧 joint likelihood 的 basis transfer 仍与 dense 变换一致。

### 27. `test_routeB_schur_posterior_recovery_vs_dense_inverse`

- 文件：`tests/test_joint_ssgp_kron_routeB.py`
- 分组：Route B Theory
- 结果：`PASSED`

**数学公式片段**

$S_{\beta|u}=(A_\beta-R_{\beta u}D_u^{-1}R_{u\beta})^{-1},\quad m_u=D_u^{-1}(h_u-R_{u\beta}m_\beta)$

**代码片段**

```python
schur = schur_recover_posterior(...)
_, cov, mean = dense_joint_posterior_reference(...)
assert allclose(schur['m_beta'], mean[:d])
assert allclose(schur['S_beta_beta'], cov[:d, :d])
```

**代码逐行实现逻辑翻译**

- 1. `schur = schur_recover_posterior(...)`：用 Schur complement 和 Sylvester solve 恢复 joint posterior 的均值与协方差块。
- 2. `_, cov, mean = dense_joint_posterior_reference(...)`：计算右侧 `dense_joint_posterior_reference(...)`，并把结果保存到变量 `_, cov, mean`，供后续检查使用。
- 3. `assert allclose(schur['m_beta'], mean[:d])`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。
- 4. `assert allclose(schur['S_beta_beta'], cov[:d, :d])`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。

**验证逻辑**

验证 Schur complement + Sylvester solves 恢复的 posterior mean/covariance 与 dense inverse 一致。

### 28. `test_routeB_cross_covariance_matches_dense_reference`

- 文件：`tests/test_joint_ssgp_kron_routeB.py`
- 分组：Route B Theory
- 结果：`PASSED`

**数学公式片段**

$S_{\beta u}=-S_{\beta\beta}R_{\beta u}D_u^{-1}$

**代码片段**

```python
routeB_cross_cov = -schur['S_beta_beta'] @ schur['W'].T
assert np.linalg.norm(cov[:d, d:]) > 1e-8
assert np.allclose(routeB_cross_cov, cov[:d, d:])
```

**代码逐行实现逻辑翻译**

- 1. `routeB_cross_cov = -schur['S_beta_beta'] @ schur['W'].T`：按 Route B 公式计算 beta 与 u 的 posterior cross covariance。
- 2. `assert np.linalg.norm(cov[:d, d:]) > 1e-8`：检查某个量确实大于阈值，常用于确认非零 coupling、loss 下降或方差为正。
- 3. `assert np.allclose(routeB_cross_cov, cov[:d, d:])`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。
- `.T` 表示矩阵转置。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `np.linalg` / `torch.linalg` 是线性代数工具箱，用于求解线性方程、Cholesky 分解、特征值等。

**验证逻辑**

验证 Route B 保留的 beta-u posterior cross covariance 与 dense reference 完全一致。

### 29. `test_mean_field_has_zero_cross_covariance_and_differs_when_coupling_nonzero`

- 文件：`tests/test_joint_ssgp_kron_routeB.py`
- 分组：Route B Theory
- 结果：`PASSED`

**数学公式片段**

$S_{\beta u}^{MF}=0,\quad S_{\beta u}^{dense}\ne 0$ under nonzero coupling

**代码片段**

```python
mean_field_cross_cov = np.zeros((d, ms * mt))
assert norm(mean_field_cross_cov - cov[:d, d:]) > 1e-8
assert abs(dense_predictive_variance - mean_field_predictive_variance) > 1e-8
```

**代码逐行实现逻辑翻译**

- 1. `mean_field_cross_cov = np.zeros((d, ms * mt))`：构造 mean-field 的 cross covariance；mean-field 假设下这一块被强制设为 0。
- 2. `assert norm(mean_field_cross_cov - cov[:d, d:]) > 1e-8`：检查某个量确实大于阈值，常用于确认非零 coupling、loss 下降或方差为正。
- 3. `assert abs(dense_predictive_variance - mean_field_predictive_variance) > 1e-8`：检查某个量确实大于阈值，常用于确认非零 coupling、loss 下降或方差为正。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `np.zeros(shape)` 创建全 0 数组，常用来构造零特征或零 cross-covariance。

**验证逻辑**

验证 mean-field 在强 coupling 下确实丢失 cross covariance，并导致预测方差/均值偏离 dense posterior。

### 30. `test_routeB_predictive_variance_matches_dense_joint_posterior_and_differs_from_mean_field`

- 文件：`tests/test_joint_ssgp_kron_routeB.py`
- 分组：Route B Theory
- 结果：`PASSED`

**数学公式片段**

$\operatorname{Var}(y_*)=\sigma^2+\nu_*+[\phi_*,q_*]S[\phi_*,q_*]^\top$

**代码片段**

```python
pred = model.predict(phi_star=phi, c_proj_star=c, t_proj_star=t, state=state)
dense_var = sigma2 + nu + x @ cov @ x
assert allclose(pred.variance, dense_var)
assert abs(pred.variance - mean_field_var) > 1e-7
```

**代码逐行实现逻辑翻译**

- 1. `pred = model.predict(phi_star=phi, c_proj_star=c, t_proj_star=t, state=state)`：调用模型的预测函数，得到后面要检查的预测均值和方差。
- 2. `dense_var = sigma2 + nu + x @ cov @ x`：按 dense joint posterior 公式计算预测方差参考值。
- 3. `assert allclose(pred.variance, dense_var)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。
- 4. `assert abs(pred.variance - mean_field_var) > 1e-7`：检查某个量确实大于阈值，常用于确认非零 coupling、loss 下降或方差为正。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证 Route B 预测方差包含 dense joint posterior 的 cross term，并区别于 mean-field 方差。

### 31. `test_routeB_fixed_basis_streaming_equals_batch_joint_posterior`

- 文件：`tests/test_joint_ssgp_kron_routeB.py`
- 分组：Route B Theory
- 结果：`PASSED`

**数学公式片段**

$\Lambda_N=\Lambda_0+\sum_n\sigma^{-2}[\Phi_n,A_n]^\top[\Phi_n,A_n]$

**代码片段**

```python
state = model.update_block_structured_joint_ssgp_transfer(...)
Lambda_batch = prior + H.T @ H / sigma2
assert allclose(state.routeB_dense_joint_precision(), Lambda_batch)
```

**代码逐行实现逻辑翻译**

- 1. `state = model.update_block_structured_joint_ssgp_transfer(...)`：把当前 block 的观测数据送进模型，更新 posterior 状态；返回的 state 保存更新后的自然参数、均值和方差结构。
- 2. `Lambda_batch = prior + H.T @ H / sigma2`：计算右侧 `prior + H.T @ H / sigma2`，并把结果保存到变量 `Lambda_batch`，供后续检查使用。
- 3. `assert allclose(state.routeB_dense_joint_precision(), Lambda_batch)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。
- `.T` 表示矩阵转置。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证固定 basis 下 Route B streaming joint posterior 与 batch joint posterior 一致。

### 32. `test_routeB_no_linear_mean_reduces_to_gp_only`

- 文件：`tests/test_joint_ssgp_kron_routeB.py`
- 分组：Route B Theory
- 结果：`PASSED`

**数学公式片段**

$d_\beta=0\Rightarrow R_{\beta u}\ \text{empty and Route B}=GP\text{-only SSGP}$

**代码片段**

```python
Phi = np.zeros((y.size, 0))
routeB = model.update_block_structured_joint_ssgp_transfer(...)
gp_only = model.update_block_ssgp_transfer(...)
assert allclose(routeB.M_u, gp_only.M_u)
```

**代码逐行实现逻辑翻译**

- 1. `Phi = np.zeros((y.size, 0))`：构造全 0 的线性特征，用来测试没有线性均值或没有 beta-u coupling 时模型是否正确退化。
- 2. `routeB = model.update_block_structured_joint_ssgp_transfer(...)`：把当前 block 的观测数据送进模型，更新 posterior 状态；返回的 state 保存更新后的自然参数、均值和方差结构。
- 3. `gp_only = model.update_block_ssgp_transfer(...)`：把当前 block 的观测数据送进模型，更新 posterior 状态；返回的 state 保存更新后的自然参数、均值和方差结构。
- 4. `assert allclose(routeB.M_u, gp_only.M_u)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- `np.zeros(shape)` 创建全 0 数组，常用来构造零特征或零 cross-covariance。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证没有线性均值时，Route B 不引入额外行为，退化为原 GP-only 更新。

### 33. `test_routeB_zero_cross_feature_sanity`

- 文件：`tests/test_joint_ssgp_kron_routeB.py`
- 分组：Route B Theory
- 结果：`PASSED`

**数学公式片段**

$\Phi=0\Rightarrow R_{\beta u}=0,\quad \operatorname{Var}(y_*)=\sigma^2+\nu_*+\phi_*^\top S_{\beta\beta}\phi_*+q_*^\top D_u^{-1}q_*$

**代码片段**

```python
Phi = np.zeros((y.size, d))
state = model.update_block_structured_joint_ssgp_transfer(...)
assert allclose(state.R_beta_u, 0.0)
assert allclose(pred.variance, separate)
```

**代码逐行实现逻辑翻译**

- 1. `Phi = np.zeros((y.size, d))`：构造全 0 的线性特征，用来测试没有线性均值或没有 beta-u coupling 时模型是否正确退化。
- 2. `state = model.update_block_structured_joint_ssgp_transfer(...)`：把当前 block 的观测数据送进模型，更新 posterior 状态；返回的 state 保存更新后的自然参数、均值和方差结构。
- 3. `assert allclose(state.R_beta_u, 0.0)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。
- 4. `assert allclose(pred.variance, separate)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- `np.zeros(shape)` 创建全 0 数组，常用来构造零特征或零 cross-covariance。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证 cross block 为零时 Route B 方差自动分解为 beta 项和 GP 项的相加形式。

### 34. `test_predictive_variance_respects_kernel_amplitude`

- 文件：`tests/test_joint_ssgp_kron_routeB.py`
- 分组：Route B Theory
- 结果：`PASSED`

**数学公式片段**

$\nu_*=k(x_*,x_*)-k_{*u}K_{uu}^{-1}k_{u*}$ with $k(x_*,x_*)=\text{kernel variance}$

**代码片段**

```python
for kernel_variance in [2.0, 0.5]:
    model = JointSSGPKronHiPPOSVGP(..., prior_point_variance=dataset.gp_prior_variance)
    dense_var = sigma2 + nu + x @ cov @ x
    assert allclose(pred.variance, dense_var)
```

**代码逐行实现逻辑翻译**

- 1. `for kernel_variance in [2.0, 0.5]:`：开始一个循环，对集合里的每个元素重复执行缩进代码。
- 2. `    model = JointSSGPKronHiPPOSVGP(..., prior_point_variance=dataset.gp_prior_variance)`：计算右侧 `JointSSGPKronHiPPOSVGP(..., prior_point_variance=dataset.gp_prior_variance)`，并把结果保存到变量 `model`，供后续检查使用。
- 3. `    dense_var = sigma2 + nu + x @ cov @ x`：按 dense joint posterior 公式计算预测方差参考值。
- 4. `    assert allclose(pred.variance, dense_var)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- `for ... in ...:` 是循环；冒号后面缩进的多行代码会重复执行。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- `@` 是矩阵乘法运算符，例如 `Phi.T @ Phi` 表示转置矩阵乘矩阵。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。

**验证逻辑**

验证 sparse conditional residual variance 显式尊重非单位 kernel amplitude，修复 coverage/NLL 风险点。

### 35. `test_loader_shapes_and_blocks`

- 文件：`tests/test_hipposvgp_era5_loader_baselines.py`
- 分组：ERA5 Baseline Pipeline
- 结果：`PASSED`

**数学公式片段**

$Y\in\mathbb{R}^{T\times S},\quad \Phi\in\mathbb{R}^{TS\times p},\quad B_n=[t_n,t_n+b)$

**代码片段**

```python
dataset = load_hipposvgp_era5(..., first_n_locations=2, split='all')
assert dataset.Y.shape == (6, 2)
assert dataset.Phi.shape[0] == 12
assert [(b.start, b.stop) for b in blocks] == [(0, 2), (2, 4), (4, 6)]
```

**代码逐行实现逻辑翻译**

- 1. `dataset = load_hipposvgp_era5(..., first_n_locations=2, split='all')`：从 processed ERA5 文件夹读取一个小数据集，并堆叠成时间 x 空间的矩阵。
- 2. `assert dataset.Y.shape == (6, 2)`：检查输出数组的维度是否符合理论设计的形状。
- 3. `assert dataset.Phi.shape[0] == 12`：检查输出数组的维度是否符合理论设计的形状。
- 4. `assert [(b.start, b.stop) for b in blocks] == [(0, 2), (2, 4), (4, 6)]`：检查实际结果是否等于预期值，通常用于验证 block 数量、shape 或字符串标记。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `for ... in ...:` 是循环；冒号后面缩进的多行代码会重复执行。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。

**验证逻辑**

验证新的 HiPPO-SVGP ERA5 loader 输出的 Y、coords、Phi 和 online block split 与 baseline/Route B 需要的形状一致。

### 36. `test_loader_converts_to_routeb_factors`

- 文件：`tests/test_hipposvgp_era5_loader_baselines.py`
- 分组：ERA5 Baseline Pipeline
- 结果：`PASSED`

**数学公式片段**

$Y_{loader}^{T\times S}\mapsto Y_{RouteB}^{S\times T},\quad (y_n,\Phi_n,T_n,K_t)$ match BlockFactors

**代码片段**

```python
factors = make_routeb_block_factors(dataset, block=slice(0, 2), ...)
assert factors.Y.shape == (2, 2)
assert factors.Phi.shape[0] == 4
assert np.isfinite(factors.y_vec).all()
```

**代码逐行实现逻辑翻译**

- 1. `factors = make_routeb_block_factors(dataset, block=slice(0, 2), ...)`：把一个时间 block 转换成 Route B 更新需要的因子：观测向量、线性特征、时间投影矩阵和核矩阵。
- 2. `assert factors.Y.shape == (2, 2)`：检查输出数组的维度是否符合理论设计的形状。
- 3. `assert factors.Phi.shape[0] == 4`：检查输出数组的维度是否符合理论设计的形状。
- 4. `assert np.isfinite(factors.y_vec).all()`：检查计算结果中没有 NaN 或无穷大，确认数值过程稳定。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `isfinite` 检查结果不是 `NaN` 也不是无穷大，用来确认数值计算稳定。
- `...` 在这里是省略号，表示文档只展示关键参数，完整代码在对应测试文件里。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。

**验证逻辑**

验证 ERA5 loader 可以转换成 Route B 的 BlockFactors，不改变主模型公式。

### 37. `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[PersistenceBaseline]`

- 文件：`tests/test_hipposvgp_era5_loader_baselines.py`
- 分组：ERA5 Baseline Pipeline
- 结果：`PASSED`

**数学公式片段**

$\hat y_{future}=g(\mathcal{D}_{seen},x_{future}),\quad \operatorname{Var}(\hat y)>0$

**代码片段**

```python
baseline.fit_initial_task(times[:3], coords, Y[:3], Phi[:3])
pred_before = baseline.predict(future_times, coords, future_phi)
Y_modified_future[3:5] += 1000.0
pred_after = baseline.predict(future_times, coords, future_phi)
assert np.allclose(pred_before.mean, pred_after.mean)
assert np.all(pred_before.variance > 0.0)
```

**代码逐行实现逻辑翻译**

- 1. `baseline.fit_initial_task(times[:3], coords, Y[:3], Phi[:3])`：用初始任务/已见数据训练 baseline；此时只允许使用 seen history。
- 2. `pred_before = baseline.predict(future_times, coords, future_phi)`：在修改 future 标签之前先做一次预测，作为 no-leakage 检查的基准。
- 3. `Y_modified_future[3:5] += 1000.0`：故意把 future 标签改得非常大，用来测试预测函数是否会错误读取 future ground truth。
- 4. `pred_after = baseline.predict(future_times, coords, future_phi)`：修改 future 标签后再预测一次；如果预测没变，说明模型没有偷看 future 标签。
- 5. `assert np.allclose(pred_before.mean, pred_after.mean)`：检查两个数组或矩阵的数值结果是否近似相等；如果不相等，就说明实现和理论公式不一致。
- 6. `assert np.all(pred_before.variance > 0.0)`：检查预测方差全部为正，避免 NLL 和 coverage 使用无效方差。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `allclose(a, b)` 表示数值近似相等；浮点数计算有微小误差，所以不用 `==` 做严格比较。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证 persistence baseline 的 future prediction 不读取 future labels，并且残差方差有限且为正。

### 38. `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[ClimatologyBaseline]`

- 文件：`tests/test_hipposvgp_era5_loader_baselines.py`
- 分组：ERA5 Baseline Pipeline
- 结果：`PASSED`

**数学公式片段**

$\hat y_s=\frac{1}{|\mathcal{D}_{seen}|}\sum_{t\in seen}y_{t,s},\quad \sigma_s^2=\operatorname{Var}(y_{t,s}-\hat y_s)$

**代码片段**

```python
baseline = ClimatologyBaseline()
baseline.fit_initial_task(times[:3], coords, Y[:3], build_phi_features(times[:3], coords))
pred = baseline.predict(future_times, coords, future_phi)
assert pred.mean.shape == (2, 2)
assert np.all(pred.variance > 0.0)
```

**代码逐行实现逻辑翻译**

- 1. `baseline = ClimatologyBaseline()`：计算右侧 `ClimatologyBaseline()`，并把结果保存到变量 `baseline`，供后续检查使用。
- 2. `baseline.fit_initial_task(times[:3], coords, Y[:3], build_phi_features(times[:3], coords))`：用初始任务/已见数据训练 baseline；此时只允许使用 seen history。
- 3. `pred = baseline.predict(future_times, coords, future_phi)`：调用模型的预测函数，得到后面要检查的预测均值和方差。
- 4. `assert pred.mean.shape == (2, 2)`：检查输出数组的维度是否符合理论设计的形状。
- 5. `assert np.all(pred.variance > 0.0)`：检查预测方差全部为正，避免 NLL 和 coverage 使用无效方差。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证 climatology baseline 的均值和 variance 只由 seen history 估计，future labels 修改不影响预测。

### 39. `test_deterministic_baselines_shapes_no_leakage_and_finite_variance[RidgeBaseline]`

- 文件：`tests/test_hipposvgp_era5_loader_baselines.py`
- 分组：ERA5 Baseline Pipeline
- 结果：`PASSED`

**数学公式片段**

$\hat\beta=(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top y,\quad \sigma^2=\operatorname{Var}(y-\Phi\hat\beta)$

**代码片段**

```python
baseline = RidgeBaseline()
baseline.fit_initial_task(times[:3], coords, Y[:3], build_phi_features(times[:3], coords))
pred = baseline.predict(future_times, coords, future_phi)
assert pred.mean.shape == (2, 2)
assert np.all(pred.variance > 0.0)
```

**代码逐行实现逻辑翻译**

- 1. `baseline = RidgeBaseline()`：计算右侧 `RidgeBaseline()`，并把结果保存到变量 `baseline`，供后续检查使用。
- 2. `baseline.fit_initial_task(times[:3], coords, Y[:3], build_phi_features(times[:3], coords))`：用初始任务/已见数据训练 baseline；此时只允许使用 seen history。
- 3. `pred = baseline.predict(future_times, coords, future_phi)`：调用模型的预测函数，得到后面要检查的预测均值和方差。
- 4. `assert pred.mean.shape == (2, 2)`：检查输出数组的维度是否符合理论设计的形状。
- 5. `assert np.all(pred.variance > 0.0)`：检查预测方差全部为正，避免 NLL 和 coverage 使用无效方差。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证 ridge baseline 的闭式解、输出形状、no-leakage future prediction 和 residual variance 都可用。

### 40. `test_gpytorch_baselines_smoke_if_available`

- 文件：`tests/test_hipposvgp_era5_loader_baselines.py`
- 分组：ERA5 Baseline Pipeline
- 结果：`PASSED`

**数学公式片段**

$p(y_*|\mathcal{D})=\mathcal{N}(\mu_*,\sigma_{*,likelihood}^2),\quad \sigma_{*,likelihood}^2>0$

**代码片段**

```python
for baseline in [IndependentTemporalGPBaseline, GPyTorchSGPRBaseline, GPyTorchSVGPBaseline]:
    baseline.fit_initial_task(times, coords, Y, Phi)
    pred = baseline.predict(times[:2], coords, build_phi_features(times[:2], coords))
    assert pred.mean.shape == (2, 1)
    assert np.all(pred.variance > 0.0)
```

**代码逐行实现逻辑翻译**

- 1. `for baseline in [IndependentTemporalGPBaseline, GPyTorchSGPRBaseline, GPyTorchSVGPBaseline]:`：依次取出每一个 GPyTorch baseline，让同一套训练和预测检查分别作用在 independent GP、SGPR 和 SVGP 上。
- 2. `    baseline.fit_initial_task(times, coords, Y, Phi)`：用初始任务/已见数据训练 baseline；此时只允许使用 seen history。
- 3. `    pred = baseline.predict(times[:2], coords, build_phi_features(times[:2], coords))`：调用模型的预测函数，得到后面要检查的预测均值和方差。
- 4. `    assert pred.mean.shape == (2, 1)`：检查输出数组的维度是否符合理论设计的形状。
- 5. `    assert np.all(pred.variance > 0.0)`：检查预测方差全部为正，避免 NLL 和 coverage 使用无效方差。

**涉及的基础语法提示**

- `=` 是赋值：把右边计算出的结果保存到左边的变量名里，例如 `state = ...`。
- `assert 条件` 是测试断言：条件为真就继续，条件为假就让测试失败。
- `for ... in ...:` 是循环；冒号后面缩进的多行代码会重复执行。
- 方括号 `[]` 用来索引数组/字典；例如 `state['m_beta']` 取字典中的 `m_beta` 项，`times[:3]` 取前 3 个元素。
- `.shape` 是数组维度；例如 `(6, 2)` 表示 6 行、2 列。
- 点号 `对象.方法(...)` 表示调用某个对象上的方法；例如 `baseline.predict(...)` 调用 baseline 的预测函数。

**验证逻辑**

验证 independent GP、SGPR 和 SVGP 三个 GPyTorch baseline 都能训练、返回 likelihood predictive variance，并且 variance 有限为正。
