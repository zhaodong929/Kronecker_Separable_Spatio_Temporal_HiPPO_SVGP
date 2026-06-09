# Route B Standard Experiment: Coupling Setting Note

本文档用于说明 standard synthetic experiment 中 `medium` / `strong` beta-u coupling 的实验含义、公式定义、参数设置，以及为什么这些设置与 Route B 的理论贡献相关。

## 1. 核心问题

Route B 的理论优势不是来自普通 GP transfer，而是来自保留线性回归系数 beta 与 sparse GP inducing variables u 之间的 posterior coupling。

模型可以写成：

```tex
y = \Phi \beta + f + \epsilon,
\qquad
f \sim \mathcal{GP}(0, k_t \otimes k_s),
\qquad
\epsilon \sim \mathcal{N}(0,\sigma^2 I).
```

在 sparse GP 近似中：

```tex
f \approx A u,
```

所以 likelihood 中实际出现的是：

```tex
y \approx \Phi \beta + A u + \epsilon.
```

其中：

- `Phi` 是线性回归特征矩阵；
- `A` 是 sparse GP inducing basis 对观测点的投影矩阵；
- `beta` 是线性部分参数；
- `u` 是 GP inducing variables。

Gaussian likelihood 给出的 beta-u cross natural precision block 是：

```tex
R_{\beta u}
=
\sigma^{-2}\Phi^\top A.
```

因此，`Phi` 和 `A` 的列空间越相似，`Phi^T A` 越大，beta-u posterior coupling 越强。

Route B 保留这个 cross block：

```tex
R_{\beta u} \neq 0,
\qquad
S_{\beta u} \neq 0.
```

Mean-field baseline 则相当于忽略 posterior cross covariance：

```tex
S_{\beta u}^{MF} = 0.
```

所以，如果实验中 beta-u coupling 很弱，那么 Route B 和 mean-field 的理论差异不会被充分触发；如果 coupling 为 medium 或 strong，则更能测试 Route B 的核心机制。

## 2. weak / medium / strong 是如何设置的

代码位置：

```text
scripts/run_joint_ssgp_kron_experiments.py
```

相关函数：

```text
augment_linear_signal(...)
```

设原始 standard 的线性特征为：

```tex
\Phi_{\text{base}}
=
\left[
1,\;
t,\;
s_c,\;
\sin(2\pi t)
\right],
```

其中：

- `t` 是时间坐标；
- `s_c = s - mean(s)` 是中心化后的空间坐标；
- `sin(2 pi t)` 是时间周期项。

为了增强 beta-u coupling，构造一个更接近 GP smooth basis 的特征组：

```tex
\Phi_{\text{smooth}}
=
\left[
\sin(2\pi t),\;
\cos(2\pi t),\;
s_c,\;
s_c\sin(2\pi t)
\right].
```

三种 coupling 设定是：

```tex
\Phi_{\text{weak}}
=
\Phi_{\text{base}},
```

```tex
\Phi_{\text{medium}}
=
0.5\Phi_{\text{base}} + 0.5\Phi_{\text{smooth}},
```

```tex
\Phi_{\text{strong}}
=
\Phi_{\text{smooth}}.
```

解释：

- `weak`：原始 standard 特征，beta-u overlap 较弱，现在不作为主实验条件；
- `medium`：一半原始特征、一半 smooth GP-like 特征，是更保守、更适合作为 main setting 的条件；
- `strong`：完全使用 smooth GP-like 特征，是机制验证或 stress-test，更直接放大 Route B 和 mean-field 的区别。

## 3. 为什么现在不主打 weak

Route B 理论要解决的是 beta 和 u posterior 相关性不能被 mean-field 丢掉的问题。这个问题只有在：

```tex
\Phi^\top A
```

非忽略不计时才明显。

如果使用 weak coupling，那么：

```tex
R_{\beta u} = \sigma^{-2}\Phi^\top A
```

本身较小，Route B 保留 cross covariance 的优势自然不明显。因此 weak 更像 negative control，而不是论文主要设定。

当前代码默认使用：

```text
--beta-u-correlation-design strong
```

`medium` 仍然保留为可选设置，用于更保守的补充实验：

```text
--beta-u-correlation-design medium
```

更合理的汇报方式是：

- `strong` 作为默认主机制验证设置；
- `medium` 作为 conservative retained setting；
- `weak` 不放入主 claim，最多作为补充诊断。

## 4. Standard strong experiment 参数设置

这次重新跑的 strong standard 实验使用以下设置：

| 参数 | 取值 |
|---|---:|
| dataset | synthetic |
| synthetic regime | standard |
| beta-u coupling | strong |
| num seeds | 10 |
| num_time | 100 |
| num_space | 6 |
| block_size | 5 |
| temporal inducing size M_t | 5 |
| spatial inducing size M_s | 4 |
| observation noise sigma | 0.08 |
| methods | no_transfer, mean_field_ssgp_transfer, structured_joint_ssgp_transfer |
| eval modes | current, seen_history |
| main eval mode | seen_history |
| ell_t fitting | initial-task full-GP MLL |
| initial task fraction | 0.2 |
| time normalization | expected_horizon |
| time scale | 1.0 |
| ell_t grid source | time_scale |

ell_t candidate grid comes from the general time-scale rule:

```tex
\ell_t
\in
\{0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.6\}.
```

The initial task is the first 20% of the time series. The selected ell_t is chosen by full-GP negative log marginal likelihood on the initial task, then frozen for all later online blocks and shared by all methods.

Selected model ell_t distribution in this strong standard run:

| selected ell_t | number of seeds |
|---:|---:|
| 0.2 | 7 |
| 0.4 | 2 |
| 0.8 | 1 |

This means the model ell_t is not manually chosen. It is fitted from the initial task using the same protocol for all methods.

## 5. Standard strong experiment result

Seen-history mean over 10 seeds:

| Method | RMSE | NLL | Cov90 | RMSE forgetting | NLL forgetting |
|---|---:|---:|---:|---:|---:|
| no_transfer | 0.4042 | 0.4613 | 0.9080 | 0.3196 | 1.6340 |
| mean-field | 0.4186 | 5.9408 | 0.8792 | 0.2795 | 7.1979 |
| Route B | 0.2391 | -0.0265 | 0.9257 | 0.1398 | 0.8954 |

Conclusion for this setting:

```text
Under strong beta-u coupling, Route B is better than mean-field on all five main seen-history metrics:
RMSE, NLL, coverage, RMSE forgetting, and NLL forgetting.
```

This is consistent with the Route B mechanism: when beta and u explain overlapping smooth structure, retaining beta-u posterior covariance improves prediction and continual retention.

## 6. Standard medium experiment result

For comparison, the medium standard setting used the same protocol, except:

```text
beta-u coupling = medium
```

Seen-history mean over 10 seeds:

| Method | RMSE | NLL | Cov90 | RMSE forgetting | NLL forgetting |
|---|---:|---:|---:|---:|---:|
| no_transfer | 0.3765 | 0.2545 | 0.9212 | 0.2938 | 1.4060 |
| mean-field | 0.2468 | 0.0816 | 0.8943 | 0.1392 | 0.9455 |
| Route B | 0.2403 | -0.1837 | 0.9323 | 0.1409 | 0.8310 |

Conclusion for medium:

```text
Route B improves RMSE, NLL, coverage, and NLL forgetting over mean-field.
RMSE forgetting is nearly tied but slightly worse than mean-field.
```

Therefore medium is a more conservative main setting, while strong provides a clearer mechanism demonstration.

## 7. Which setting should be used in the paper?

Recommended framing:

### Current default and main mechanism setting: strong coupling

Use strong if the purpose is to test the Route B theory directly:

```text
When beta-u posterior coupling is strong, mean-field is expected to lose information by setting S_beta_u = 0.
Route B is designed exactly for this case.
```

This is the current code default. It is the cleanest setting to demonstrate the value of retaining the beta-u cross covariance.

### Retained conservative setting: medium coupling

Use medium if the advisor wants a more conservative setup:

```text
The experiment does not force a fully aligned beta-u overlap, but still creates non-negligible coupling.
```

This is better for avoiding the criticism that the experiment was artificially designed to make Route B win.

### Not recommended as main claim: weak coupling

Weak coupling is not wrong, but it does not test the main theoretical contribution strongly. It can remain a negative control or be omitted from the main paper tables.

## 8. Suggested wording for advisor

Chinese:

```text
我们现在把 standard 实验中的 beta-u coupling 明确分成 medium 和 strong，并且代码默认使用 strong，medium 保留为更保守的补充设置。
Route B 的理论优势来自保留 R_{beta u}=sigma^{-2} Phi^T A 诱导的 beta-u posterior covariance。
如果 Phi 和 A 的 overlap 很弱，那么 Route B 和 mean-field 的差异本来就不明显。
所以 weak 不适合作为主实验设定。

strong 是当前默认主机制验证设定：它让线性特征更接近 smooth GP basis，从而放大 beta-u coupling。
medium 是较保守的保留设定：它只让线性特征和 GP smooth basis 有一半重合。

在 strong standard 实验中，使用 initial-task full-GP marginal likelihood 拟合 ell_t，并对所有方法共享固定 ell_t。
结果显示 Route B 在 seen-history 的 RMSE、NLL、coverage、RMSE forgetting 和 NLL forgetting 上都优于 mean-field。
```

English:

```text
We now separate the standard experiment into medium and strong beta-u coupling settings. The code default is strong, while medium remains available as a conservative retained setting.
The theoretical advantage of Route B comes from retaining the posterior beta-u covariance induced by
R_{beta u}=sigma^{-2} Phi^T A.
If the overlap between Phi and A is weak, then the difference between Route B and mean-field is not expected to be large.
Therefore weak coupling is not an appropriate main setting for testing the core Route B mechanism.

Strong coupling is the current main mechanism setting because it deliberately creates substantial beta-u posterior dependence.
Medium coupling is retained as a conservative setting because it only partially aligns the linear features with the smooth GP basis.

In the strong standard experiment, ell_t is selected by initial-task full-GP marginal likelihood and then frozen and shared by all methods.
Under this fair protocol, Route B outperforms mean-field on all main seen-history metrics:
RMSE, NLL, coverage, RMSE forgetting, and NLL forgetting.
```
