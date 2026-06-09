---
title: "Route B Structured Joint SSGP: 公式到代码学习手册"
subtitle: "对照 new_main_joint_training_ssgp_kron_routeB_refined2.tex 与项目实现"
author: "Generated from project source"
date: "2026-05-30"
toc: true
toc-depth: 3
mainfont: "Microsoft YaHei"
monofont: "Consolas"
geometry: margin=0.75in
colorlinks: true
header-includes:
  - \usepackage{amsmath}
  - \newcommand{\vecop}{\operatorname{vec}}
---

# 0. 先看整体：这个项目在做什么？

这个项目实现的是一个**在线/持续学习**的时空高斯过程模型。它把预测值拆成两部分：

$$
y(t,s)=\phi(t,s)^\top\beta+f(t,s)+\epsilon .
$$

可以用很直白的话理解：

- $\phi(t,s)^\top\beta$：线性回归部分，负责解释简单趋势，例如常数项、时间趋势、空间趋势、周期项。
- $f(t,s)$：GP residual 部分，负责解释线性回归解释不了的时空相关残差。
- $\epsilon$：观测噪声。

Route B 的核心不是“先做线性回归，再对残差做 GP”，而是**同时学习 $\beta$ 和 GP inducing variable $u$ 的联合 posterior**。这样做的目的，是保留 $\beta$ 与 $u$ 之间的 posterior coupling。

本项目中最重要的代码目录是：

```text
stvgp_kronecker/joint_ssgp_kron/
├── model.py             # Route B 主模型、更新、预测
├── structured_state.py  # 在线状态，保存 beta/u 的 posterior 和 natural stats
├── ssgp_transfer.py     # old likelihood changing-basis transfer
├── kron_utils.py        # Kronecker / Sylvester / Schur 数值工具
└── synthetic.py         # synthetic 数据和 block factor 构造
```

实验入口主要是：

```text
scripts/run_joint_ssgp_kron_experiments.py
scripts/verify_joint_ssgp_kron_derivations.py
scripts/generate_routeB_experiment_report.py
```

# 1. 代码调用总流程

一次 synthetic 实验大致按下面流程运行：

```text
main()
  -> parse_args()
  -> run_all_requested()
      -> make_dataset()
      -> fit_model_hyperparameters_from_initial_task()  # 可选 full-GP MLL
      -> run_structured_method()
          -> 按时间切 block
          -> 为当前 block 构造 Phi, T_n, Kt_new, K_on_t
          -> model.update_block_*()
          -> evaluate_state_on_factors()
              -> predictive variance decomposition
              -> RMSE / NLL / coverage
```

其中 Route B 主路径是：

```text
run_structured_method()
  -> JointSSGPKronHiPPOSVGP.update_block_structured_joint_ssgp_transfer()
      -> _routeB_transfer_old_stats()
      -> joint_likelihood_stats()
      -> recover_posterior_mean_structured()
          -> schur_recover_posterior()
              -> solve_Du_sylvester()
```

## 1.1 Route B 在线训练算法伪代码

下面这张“算法表”可以把代码调用流程和理论步骤连起来看。它对应的主要代码入口是：

- `scripts/run_joint_ssgp_kron_experiments.py::run_structured_method`
- `stvgp_kronecker/joint_ssgp_kron/model.py::update_block_structured_joint_ssgp_transfer`
- `stvgp_kronecker/joint_ssgp_kron/kron_utils.py::schur_recover_posterior`

| 阶段 | 伪代码步骤 | 对应代码 |
|---:|---|---|
| Input | 数据流 `D_1,...,D_N`，初始 prior `p(beta), p(u)`，kernel 参数，inducing basis 大小 `M_t, M_s` | `parse_args`, `make_dataset` |
| 1 | 初始化 `state = None`，表示还没有历史 old likelihood 信息 | `run_structured_method` |
| 2 | 对每个 online block `n = 1,...,N` 循环 | `for block_id, block in enumerate(blocks)` |
| 3 | 构造当前 block 的 `y_n, Phi_n, T_n, Kt_new, K_on_t` | `make_indexed_block_factors` |
| 4 | 如果已有旧状态，把 old likelihood 从旧 temporal basis 转到新 temporal basis | `_routeB_transfer_old_stats` |
| 5 | 计算当前 block 的 Gaussian likelihood natural stats | `joint_likelihood_stats` |
| 6 | 把 old stats 和 new stats 相加，得到 accumulated Route B stats | `R_beta_beta = old + new`, `R_beta_u = old + new`, `B_temporal = old + new` |
| 7 | 加入 beta prior，形成 joint precision 的 beta block | `A_beta = beta_prior_precision + R_beta_beta` |
| 8 | 用 Sylvester solver 计算 `D_u^{-1}B_beta_u^T` 和 `D_u^{-1}h_u` | `solve_Du_sylvester` |
| 9 | 用 Schur complement 恢复 `m_beta, S_beta_beta, m_u` | `schur_recover_posterior` |
| 10 | 把 posterior mean 和 natural stats 打包为新的 `state` | `StructuredKronState(...)` |
| 11 | 在 `current / seen_history / future` 模式下评估 RMSE, NLL, coverage | `evaluate_state_on_factors` |
| Output | 所有 block 的 metrics、variance decomposition、forgetting score、最终报告 | `write_csv`, `make_plots`, `generate_routeB_experiment_report.py` |

用更形象的流程图表示就是：

```text
┌──────────────────────────┐
│  Raw data / Synthetic D   │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│ Split into online blocks  │
│ D1, D2, ..., Dn           │
└─────────────┬────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ For current block n                         │
│ build y_n, Phi_n, T_n, Kt_new, K_on_t       │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ Transfer old Route B likelihood stats       │
│ R_bb, R_bu, B_temporal, h_beta, H_info      │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ Add new block likelihood stats              │
│ joint_likelihood_stats(y_n, Phi_n, T_n, C)  │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ Build structured joint precision            │
│ [ A_beta       R_beta_u ]                   │
│ [ R_beta_u.T   D_u      ]                   │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ Schur complement + Sylvester solves         │
│ recover m_beta, S_beta_beta, m_u            │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ Save new StructuredKronState                │
│ and evaluate current / seen_history / future│
└─────────────────────────────────────────────┘
```

这张图里最重要的是中间三步：

1. **Transfer old stats**：持续学习靠它保留历史信息。
2. **Add new likelihood stats**：当前 block 的新观测进入模型。
3. **Schur + Sylvester**：既保留 beta-u coupling，又避免大矩阵求逆。

# 2. 公式块 1：Joint Observation Model

理论文档第一部分写的是观测模型：

$$
y(t_n,s_i)
=
\phi(t_n,s_i)^\top \beta
+
f(t_n,s_i)
+
\epsilon_{n,i},
\qquad
\epsilon_{n,i}\sim\mathcal{N}(0,\sigma^2).
$$

按 block 堆叠后：

$$
y_n = \Phi_n\beta + f_n+\epsilon_n,
\qquad
\epsilon_n\sim\mathcal{N}(0,\sigma^2 I).
$$

## 2.1 小白解释

一个 online block 里会有很多观测点。代码中把这些观测点压成一个长向量 `y_vec`，把线性特征压成矩阵 `Phi`。

- `y_vec` 对应公式里的 $y_n$。
- `Phi` 对应公式里的 $\Phi_n$。
- `beta_mean` 是当前对 $\beta$ 的 posterior mean。
- GP 部分由 inducing variable `u` 近似，在代码里存成矩阵 `M_u`。

## 2.2 对应代码：block 数据进入模型

文件：`scripts/run_joint_ssgp_kron_experiments.py`

```python
factors = make_indexed_block_factors(
    dataset,
    block=block,
    z_t=z_t,
    z_t_old=old_z,
    spatial_idx=train_idx,
    lengthscale=model_ell_t,
)
```

这里 `factors` 里最重要的是：

```python
factors.y_vec  # y_n
factors.Phi    # Phi_n
factors.T      # temporal projection T_n
factors.Kt     # Kt_new
```

进入 Route B 更新：

```python
state = model.update_block_structured_joint_ssgp_transfer(
    y_vec=factors.y_vec,
    Phi=factors.Phi,
    T_n=factors.T,
    Kt_new=factors.Kt,
    K_on_t=factors.K_on_t,
    state=state,
)
```

### 2.2.1 这段代码具体在做什么？

1. `make_indexed_block_factors(...)` 把原始数据切成当前 online block 能使用的格式。
2. `block=block` 指当前训练的时间段，例如第 0 到第 4 个时间点。
3. `z_t=z_t` 是当前 block 使用的 temporal inducing basis / inducing times。
4. `z_t_old=old_z` 是上一轮的 temporal basis，用来计算 changing-basis transfer 的 `K_on_t`。
5. `spatial_idx=train_idx` 控制当前训练使用哪些空间位置；在 sparse-current 实验中它可能只是部分空间点。
6. `lengthscale=model_ell_t` 决定 temporal kernel 的平滑程度，也就是理论里的 $\ell_t$。
7. `state` 是上一轮训练后的压缩记忆。第一轮是 `None`，之后每轮都会把上一个 block 的返回 state 传进来。

所以这段代码不是把所有历史数据重新训练一遍，而是“当前 block 数据 + 上一轮压缩状态”一起进入 Route B 更新。

## 2.3 模型初始化

文件：`stvgp_kronecker/joint_ssgp_kron/model.py`

```python
class JointSSGPKronHiPPOSVGP:
    def __init__(
        self,
        *,
        Ks: np.ndarray,
        C: np.ndarray,
        sigma2: float,
        beta_prior_mean: np.ndarray,
        beta_prior_cov: np.ndarray,
        prior_point_variance: float = 1.0,
        jitter: float = 1e-6,
    ) -> None:
        self.Ks = symmetrize(np.asarray(Ks, dtype=float))
        self.C = np.asarray(C, dtype=float)
        self.sigma2 = float(sigma2)
        self.beta_prior_mean = np.asarray(beta_prior_mean, dtype=float)
        self.beta_prior_cov = symmetrize(np.asarray(beta_prior_cov, dtype=float))
        self.prior_point_variance = float(prior_point_variance)
        self.G = symmetrize(self.C.T @ self.C)
        self.Ks_inv = inv_spd(self.Ks, jitter=self.jitter)
```

代码含义：

- `sigma2` 就是 $\sigma^2$。
- `beta_prior_mean`, `beta_prior_cov` 是 $\beta$ 的 Gaussian prior。
- `Ks` 是 spatial inducing prior covariance。
- `C` 是空间投影矩阵。
- `G = C.T @ C` 对应理论里的 $G=C^\top C$。

### 2.3.1 初始化细节

- `np.asarray(..., dtype=float)`：把输入统一转成 NumPy 浮点数组，避免 list 或整数数组影响后续矩阵运算。
- `symmetrize(...)`：理论上的 covariance/precision 应该对称，但数值计算会产生很小的非对称误差，所以这里强制变成 $(A+A^\top)/2$。
- `self.G = C.T @ C`：这是后面 Kronecker/Sylvester 加速的关键。它让代码不需要显式构造巨大的 $A_n^\top A_n$。
- `self.Ks_inv = inv_spd(...)`：预先求空间 prior covariance 的逆。`jitter` 用来避免矩阵接近奇异时 Cholesky 分解失败。

# 3. 公式块 2：Separable Kernel 与 Kronecker 表示

理论文档写：

$$
k((s,t),(s',t'))=k_s(s,s')k_t(t,t')
$$

完整 grid 上：

$$
K_{ff}=K_t\otimes K_s.
$$

对 sparse inducing 表示，当前 block 的 GP 设计矩阵写成：

$$
A_n=T_n\otimes C.
$$

并且：

$$
A_n^\top A_n=(T_n^\top T_n)\otimes(C^\top C).
$$

## 3.1 小白解释

Kronecker product 是这个项目的加速核心。不要把它想太复杂：它就是利用“时间结构”和“空间结构”可以分开计算的事实。

如果直接构造 $A_n$，矩阵会很大。代码尽量不真的构造大矩阵，而是用矩阵乘法：

$$
(T\otimes C)\mathrm{vec}(U)=\mathrm{vec}(CUT^\top).
$$

## 3.2 对应代码：Kronecker 乘法

文件：`stvgp_kronecker/joint_ssgp_kron/kron_utils.py`

```python
def kron_mv(
    T: np.ndarray,
    C: np.ndarray,
    U: np.ndarray,
    *,
    output: Literal["vector", "matrix"] = "vector",
) -> np.ndarray:
    """Compute ``(T kron C) vec(U)`` without materializing the Kronecker matrix."""

    result = C @ U @ T.T
    if output == "matrix":
        return result
    return vec_f(result)
```

### 3.2.1 `kron_mv` 逐步解释

- `T` 的形状是 `(N_t, M_t)`，表示当前 block 中观测时间到 temporal inducing basis 的投影。
- `C` 的形状是 `(N_s, M_s)`，表示观测空间点到 spatial inducing basis 的投影。
- `U` 的形状是 `(M_s, M_t)`，是 GP inducing posterior mean 的矩阵形式。
- `C @ U @ T.T` 的输出形状是 `(N_s, N_t)`，正好对应当前 block 的空间-时间预测矩阵。
- `vec_f(result)` 把矩阵按列展开，得到和 `y_vec` 对齐的一维向量。

这就是为什么项目里经常把 `u` 存成矩阵 `M_u`，而不是一直存成长向量：矩阵形式可以直接利用 `C @ M_u @ T.T`。

对应关系：

| 理论符号 | 代码变量 |
|---|---|
| $T_n$ | `T` 或 `T_n` |
| $C$ | `C` |
| $u=\mathrm{vec}(U)$ | `vec_f(M_u)` |
| $A_nu=(T_n\otimes C)u$ | `kron_mv(T_n, C, M_u)` |

## 3.3 Fortran vectorization convention

项目用的是：

```python
def vec_f(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix).reshape(-1, order="F")

def unvec_f(vector: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(vector).reshape(shape, order="F")
```

这里的 `order="F"` 很重要。它表示按列展开矩阵。理论公式里的 $\vecop(\cdot)$ 在本项目中就是 `vec_f`。

如果这里的展开顺序弄错，`R_beta_u`、`H_info`、`M_u` 的维度表面上可能仍然对，但数学含义会错位，dense reference test 通常会失败。

# 4. 公式块 3：在线状态 StructuredKronState

理论上，Route B 要维护 joint likelihood natural parameters：

$$
R_n^z=
\begin{bmatrix}
R_{\beta\beta,n} & R_{\beta u,n}\\
R_{u\beta,n} & R_{uu,n}
\end{bmatrix},
\qquad
r_n^z=
\begin{bmatrix}
r_{\beta,n}\\r_{u,n}
\end{bmatrix}.
$$

其中：

$$
R_{uu,n}=B_n\otimes G.
$$

## 4.1 小白解释

`StructuredKronState` 可以理解为“模型训练到当前 block 后的记忆”。它既保存 posterior mean，也保存 old likelihood stats，供下一个 block 继续学习。

## 4.2 对应代码：状态对象

文件：`stvgp_kronecker/joint_ssgp_kron/structured_state.py`

```python
@dataclass(frozen=True)
class StructuredKronState:
    beta_mean: np.ndarray
    beta_cov: np.ndarray
    M_u: np.ndarray
    B_temporal: np.ndarray
    H_info: np.ndarray
    Kt_current: np.ndarray
    Ks: np.ndarray
    G: np.ndarray
    sigma2: float
    metadata: dict[str, Any] = field(default_factory=dict)
    R_beta_beta: np.ndarray | None = None
    R_beta_u: np.ndarray | None = None
    h_beta: np.ndarray | None = None
    beta_prior_precision: np.ndarray | None = None
    beta_prior_natural: np.ndarray | None = None
    Lambda_beta_given_u: np.ndarray | None = None
    S_beta_beta: np.ndarray | None = None
```

字段解释：

| 字段 | 理论含义 |
|---|---|
| `beta_mean` | $m_\beta$ |
| `beta_cov` / `S_beta_beta` | $S_{\beta\beta}$ |
| `M_u` | $m_u$ reshaped 成 $(M_s,M_t)$ |
| `B_temporal` | $B_n$，使 $R_{uu,n}=B_n\otimes G$ |
| `H_info` | $r_u$ 的矩阵形式 |
| `R_beta_beta` | $R_{\beta\beta,n}$ |
| `R_beta_u` | $R_{\beta u,n}$ |
| `h_beta` | $r_{\beta,n}$ |
| `Lambda_beta_given_u` | $\Lambda_{\beta\mid u}$ |

### 4.2.1 为什么要把这些量存在 state 里？

在线学习不能每次都重新读所有历史数据，否则计算量会随历史长度增长。Route B 把历史数据压缩成 natural statistics：

- `R_beta_beta`, `R_beta_u`, `B_temporal` 是历史 likelihood 的 precision 信息。
- `h_beta`, `H_info` 是历史 likelihood 的 information vector。
- 下一个 block 来时，先把这些 old stats transfer 到新 basis，然后加上 new stats。

这就是 streaming sparse GP 的思想：历史数据不直接保存，但历史对 posterior 的贡献被保存在少量统计量里。

# 5. 公式块 4：Changing-Basis Old-Likelihood Transfer

理论文档中，旧 block 的 inducing variable 是 $u_o$，新 block 的是 $u_n$。两者坐标不一样，所以要 transfer：

$$
p(u_o\mid u_n)=\mathcal{N}(L_{on}u_n,\Sigma_{o\mid n}),
\qquad
L_{on}=K_{on}K_{nn}^{-1}.
$$

Route B 的 old likelihood transfer 是：

$$
R_{\beta\beta,o\rightarrow n}=R_{\beta\beta,o},
$$

$$
R_{\beta u,o\rightarrow n}=R_{\beta u,o}L_{on},
$$

$$
R_{uu,o\rightarrow n}=L_{on}^\top R_{uu,o}L_{on},
$$

$$
r_{\beta,o\rightarrow n}=r_{\beta,o},
\qquad
r_{u,o\rightarrow n}=L_{on}^\top r_{u,o}.
$$

## 5.1 小白解释

可以把旧 posterior 看成“上一轮学到的信息”。下一轮 temporal basis 变了，所以旧信息要换坐标。Route B 和 mean-field 最大区别之一就是：Route B 不只转移 GP 的信息，也转移 $\beta$ 和 $u$ 的 cross block `R_beta_u`。

## 5.2 对应代码：计算 temporal transfer 矩阵

文件：`stvgp_kronecker/joint_ssgp_kron/ssgp_transfer.py`

```python
def compute_Lt(K_on_t: np.ndarray, K_nn_t: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Compute ``L_t = K_on_t @ inv(K_nn_t)`` using a Cholesky solve."""

    return solve_spd(K_nn_t, K_on_t.T, jitter=jitter).T
```

注意：代码里只算 temporal part $L_t$，因为空间 inducing set 固定时：

$$
L_{on}=L_{on}^{(t)}\otimes I_s.
$$

## 5.3 对应代码：转移 Route B 的全部 old likelihood stats

```python
def transfer_joint_old_likelihood(
    *,
    R_beta_beta: np.ndarray,
    R_beta_u: np.ndarray,
    h_beta: np.ndarray,
    B_temporal: np.ndarray,
    H_info: np.ndarray,
    L_t: np.ndarray,
    M_s: int,
) -> dict[str, np.ndarray]:
    """Transfer Route-B joint old-likelihood natural statistics."""

    return {
        "R_beta_beta": np.asarray(R_beta_beta, dtype=float).copy(),
        "R_beta_u": transfer_R_beta_u(R_beta_u, L_t, M_s),
        "h_beta": np.asarray(h_beta, dtype=float).copy(),
        "B_temporal": transfer_R_uu_kron(B_temporal, L_t),
        "H_info": transfer_h_u(H_info, L_t),
    }
```

公式对应关系：

| 理论公式 | 代码 |
|---|---|
| $R_{\beta\beta,o\to n}=R_{\beta\beta,o}$ | copy `R_beta_beta` |
| $R_{\beta u,o\to n}=R_{\beta u,o}L_{on}$ | `transfer_R_beta_u` |
| $R_{uu,o\to n}=L_{on}^\top R_{uu,o}L_{on}$ | `transfer_R_uu_kron` |
| $r_{\beta,o\to n}=r_{\beta,o}$ | copy `h_beta` |
| $r_{u,o\to n}=L_{on}^\top r_{u,o}$ | `transfer_h_u` |

### 5.3.1 transfer 细节

- `R_beta_beta` 不需要乘 `L_t`，因为 $\beta$ 是全局线性参数，不随 temporal inducing basis 改变。
- `h_beta` 也不需要乘 `L_t`，原因同上。
- `B_temporal` 必须做 `L_t.T @ B_old @ L_t`，因为它描述的是旧 $u_o$ 坐标里的 likelihood precision，需要换到新 $u_n$ 坐标。
- `H_info` 做 `H_old @ L_t`。由于项目用 `H_info` 的矩阵形式保存 $r_u$，这个右乘就对应 $L_{on}^\top r_{u,o}$。
- `R_beta_u` 是 Route B 关键。mean-field 路线没有显式保留它；Route B 保留它，所以 posterior 里 $\beta$ 和 $u$ 的相关性不会被强行丢掉。

## 5.4 对应代码：不显式构造大矩阵 $L_{on}$

```python
def apply_Lon_to_beta_u_cross_block(R_beta_u: np.ndarray, L_t: np.ndarray, M_s: int) -> np.ndarray:
    """Compute ``R_beta_u @ (L_t kron I_s)`` without materializing ``L_on``."""

    rows = []
    for row in R_beta_u:
        old = unvec_f(row, (M_s, L_t.shape[0]))
        rows.append(vec_f(apply_temporal_right(old, L_t)))
    return np.vstack(rows)
```

这段代码就是理论文档中“每一行 reshape 成 $M_s\times M_t$，再沿 temporal dimension 乘 $L_t$”的实现。

如果显式构造 `L_t kron I_s`，矩阵大小是 `(M_t_old*M_s, M_t_new*M_s)`。当 inducing 数量变大时，这会浪费内存。这里逐行 reshape 后做右乘，是同一个数学操作，但更省内存。

# 6. 公式块 5：New Block Gaussian Likelihood Natural Stats

理论文档写：

$$
H_n=\begin{bmatrix}\Phi_n&A_n\end{bmatrix}.
$$

当前 block 的 likelihood 贡献：

$$
R_{\mathrm{new},n}^z=\frac{1}{\sigma^2}H_n^\top H_n,
\qquad
r_{\mathrm{new},n}^z=\frac{1}{\sigma^2}H_n^\top y_n.
$$

展开为：

$$
R_{\beta\beta,\mathrm{new}}=\frac{1}{\sigma^2}\Phi_n^\top\Phi_n,
$$

$$
R_{\beta u,\mathrm{new}}=\frac{1}{\sigma^2}\Phi_n^\top A_n,
$$

$$
R_{uu,\mathrm{new}}=\frac{1}{\sigma^2}A_n^\top A_n,
$$

$$
r_{\beta,\mathrm{new}}=\frac{1}{\sigma^2}\Phi_n^\top y_n,
\qquad
r_{u,\mathrm{new}}=\frac{1}{\sigma^2}A_n^\top y_n.
$$

## 6.1 对应代码：joint_likelihood_stats

文件：`stvgp_kronecker/joint_ssgp_kron/ssgp_transfer.py`

```python
def joint_likelihood_stats(
    y_vec: np.ndarray,
    Phi: np.ndarray,
    T_n: np.ndarray,
    C: np.ndarray,
    sigma2: float,
) -> dict[str, np.ndarray]:
    ns, ms = C.shape
    nt, mt = T_n.shape
    Y = y_vec.reshape((ns, nt), order="F")
    R_beta_beta = (Phi.T @ Phi) / sigma2
    h_beta = Phi.T @ y_vec / sigma2
    R_beta_u_rows = []
    for j in range(Phi.shape[1]):
        Phi_j = Phi[:, j].reshape((ns, nt), order="F")
        R_beta_u_rows.append(vec_f(C.T @ Phi_j @ T_n) / sigma2)
    R_beta_u = np.vstack(R_beta_u_rows) if R_beta_u_rows else np.zeros((0, ms * mt))
    B_temporal = (T_n.T @ T_n) / sigma2
    H_info = (C.T @ Y @ T_n) / sigma2
```

## 6.2 小白解释

这里最容易混乱的是 `R_beta_u`。理论上它是：

$$
\frac{1}{\sigma^2}\Phi_n^\top(T_n\otimes C).
$$

代码没有显式构造 $T_n\otimes C$，而是对每个 beta 特征列做：

```python
Phi_j.reshape((ns, nt), order="F")
C.T @ Phi_j @ T_n
```

这正是 Kronecker 技巧：用小矩阵乘法替代大矩阵。

### 6.3 每个返回量后续怎么用？

`joint_likelihood_stats` 返回的字典会在 `update_block_structured_joint_ssgp_transfer` 中和 old stats 相加。

- `R_beta_beta` 后续进入 `A_beta = beta_prior_precision + R_beta_beta`。
- `R_beta_u` 后续进入 Schur complement 中的 $B_{\beta u}$。
- `B_temporal` 后续和 prior precision 一起形成 $D_u$。
- `h_beta` 后续进入 $h_\beta$。
- `H_info` 后续通过 `vec_f(H_info)` 变成 $h_u$。

所以这个函数就是“把当前 block 的 Gaussian likelihood 翻译成 Route B 需要的 natural parameters”。

# 7. 公式块 6：Accumulated Natural Parameters

理论文档把 old transfer 和 new block 相加：

$$
R_{\beta\beta,n}
=
R_{\beta\beta,o\to n}
+
\frac{1}{\sigma^2}\Phi_n^\top\Phi_n,
$$

$$
R_{\beta u,n}
=
R_{\beta u,o}(L_t\otimes I_s)
+
\frac{1}{\sigma^2}\Phi_n^\top A_n,
$$

$$
B_n
=
B_{o\to n}
+
\frac{1}{\sigma^2}T_n^\top T_n.
$$

## 7.1 对应代码：Route B 主更新

文件：`stvgp_kronecker/joint_ssgp_kron/model.py`

```python
old_stats = self._routeB_transfer_old_stats(
    state,
    Kt_new,
    K_on_t,
    no_transfer=no_transfer,
    beta_dim=d,
)
new_stats = joint_likelihood_stats(y_vec, Phi, T_n, self.C, self.sigma2)

R_beta_beta = symmetrize(old_stats["R_beta_beta"] + new_stats["R_beta_beta"])
R_beta_u = old_stats["R_beta_u"] + new_stats["R_beta_u"]
h_beta_lik = old_stats["h_beta"] + new_stats["h_beta"]
B_temporal = symmetrize(old_stats["B_temporal"] + new_stats["B_temporal"])
H_info = old_stats["H_info"] + new_stats["H_info"]
```

## 7.2 小白解释

这段就是“持续学习”的核心：

- `old_stats` 是过去 block 学到的东西，已经转到新 basis。
- `new_stats` 是当前 block 新学到的东西。
- 两者相加，得到当前时刻的 accumulated likelihood。

### 7.2.1 为什么这里直接相加？

Gaussian likelihood 的 natural parameters 有一个很好用的性质：独立数据块的 likelihood 相乘，对应 natural parameters 相加。也就是说：

$$
\log p(D_{old})+\log p(D_{new})
$$

对应到 precision/information 形式就是：

$$
R_{old}+R_{new},
\qquad
r_{old}+r_{new}.
$$

这就是代码中五个 `old_stats[...] + new_stats[...]` 的数学原因。

# 8. 公式块 7：加入 beta prior，形成 joint precision

理论中的 joint posterior precision：

$$
\Lambda_n=
\begin{bmatrix}
A_\beta & B_{\beta u}\\
B_{\beta u}^\top & D_u
\end{bmatrix},
\qquad
h_n=
\begin{bmatrix}
h_\beta\\h_u
\end{bmatrix}.
$$

其中：

$$
A_\beta=\Lambda_{\beta\beta,n},
\qquad
B_{\beta u}=\Lambda_{\beta u,n},
\qquad
D_u=\Lambda_{uu,n}.
$$

代码中 $A_\beta$ 由 beta prior precision 加 likelihood precision 得到：

```python
prior_mean, prior_cov = self._beta_prior_from_state(None, beta_drift, d)
beta_prior_precision = np.zeros((0, 0)) if d == 0 else inv_spd(prior_cov, jitter=self.jitter)
beta_prior_natural = np.zeros(0) if d == 0 else beta_prior_precision @ prior_mean
A_beta = symmetrize(beta_prior_precision + R_beta_beta)
h_beta_total = beta_prior_natural + h_beta_lik
```

解释：

- `R_beta_beta` 是 likelihood 给 $\beta$ 的 precision。
- `beta_prior_precision` 是 prior 给 $\beta$ 的 precision。
- 两者相加，就是 posterior precision 的 beta-beta block。

### 8.1 这里为什么用 `None` 而不是旧 state 的 beta posterior？

Route B 主公式维护的是 old likelihood ratio 加当前 prior，而不是把上一轮 posterior 直接当下一轮 prior。代码里：

```python
prior_mean, prior_cov = self._beta_prior_from_state(None, beta_drift, d)
```

这里传 `None` 表示使用原始 beta prior。历史数据对 $\beta$ 的贡献已经在 `old_stats["R_beta_beta"]` 和 `old_stats["h_beta"]` 里了。如果再用旧 posterior 当 prior，就会重复计算历史信息。

# 9. 公式块 8：Schur Complement Posterior Recovery

理论文档写：

$$
\Lambda_{\beta\mid u}
=
A_\beta-B_{\beta u}D_u^{-1}B_{\beta u}^\top.
$$

先解：

$$
D_uW=B_{\beta u}^\top,
\qquad
D_uv_h=h_u.
$$

然后：

$$
m_\beta
=
\Lambda_{\beta\mid u}^{-1}
(h_\beta-B_{\beta u}v_h),
$$

$$
m_u=v_h-Wm_\beta.
$$

## 9.1 对应代码：schur_recover_posterior

文件：`stvgp_kronecker/joint_ssgp_kron/kron_utils.py`

```python
W = solve_Du_sylvester(Kt_inv, Ks_inv, B_temporal, G, B_beta_u.T, jitter=jitter)
v_h = solve_Du_sylvester(Kt_inv, Ks_inv, B_temporal, G, h_u, jitter=jitter)
Lambda_beta_given_u = symmetrize(A_beta - B_beta_u @ W)
rhs_beta = h_beta - B_beta_u @ v_h
m_beta = solve_spd(Lambda_beta_given_u, rhs_beta, jitter=jitter)
S_beta_beta = inv_spd(Lambda_beta_given_u, jitter=jitter)
m_u = v_h - W @ m_beta
```

## 9.2 小白解释

如果直接求 joint precision 的 inverse，会非常大。Schur complement 的技巧是：

1. 先把大的 $u$ block 暂时“消掉”。
2. 得到一个很小的 $\beta$ precision：`Lambda_beta_given_u`。
3. 先求 `m_beta`。
4. 再用 `m_beta` 回代求 `m_u`。

因为 $\beta$ 维度通常很小，所以这一步很省。

### 9.3 变量名和公式的精确对应

| 代码变量 | 理论符号 | 作用 |
|---|---|---|
| `A_beta` | $A_\beta$ | beta-beta precision block |
| `B_beta_u` | $B_{\beta u}$ | beta-u cross precision block |
| `h_beta` | $h_\beta$ | beta information vector |
| `h_u` | $h_u$ | u information vector |
| `W` | $D_u^{-1}B_{\beta u}^\top$ | 构造 Schur complement |
| `v_h` | $D_u^{-1}h_u$ | 求 posterior mean |
| `Lambda_beta_given_u` | $\Lambda_{\beta\mid u}$ | Schur precision |
| `S_beta_beta` | $\Lambda_{\beta\mid u}^{-1}$ | beta posterior covariance block |
| `m_beta` | $m_\beta$ | beta posterior mean |
| `m_u` | $m_u$ | u posterior mean |

这里 `W` 和 `v_h` 都通过 Sylvester solver 得到，所以不会显式反转大矩阵 $D_u$。

# 10. 公式块 9：Sylvester Solve，避免大矩阵求逆

理论文档写：

$$
D_u=
(K_{nn}^{(t)})^{-1}\otimes K_s^{-1}
+
B_n\otimes G.
$$

要求解：

$$
D_uz=q.
$$

设：

$$
q=\vecop(Q),
\qquad
z=\vecop(Z).
$$

则等价于 Sylvester 型方程：

$$
K_s^{-1}Z(K_{nn}^{(t)})^{-1}+GZB_n=Q.
$$

## 10.1 对应代码：solve_sylvester_precision

文件：`stvgp_kronecker/joint_ssgp_kron/kron_utils.py`

```python
def solve_sylvester_precision(
    Kt_inv: np.ndarray,
    Ks_inv: np.ndarray,
    B: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    *,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Solve ``Ks_inv M Kt_inv + G M B = H``."""

    left_vals, P = eigh(G, Ks_inv, check_finite=False)
    right_vals, Q = eigh(B, Kt_inv, check_finite=False)
    rhs = P.T @ H @ Q
    denom = 1.0 + np.outer(left_vals, right_vals)
    X = rhs / denom
    return P @ X @ Q.T
```

### 10.1.1 Sylvester solver 的思路

直接解 $D_uz=q$ 会面对一个大小为 $(M_sM_t)\times(M_sM_t)$ 的矩阵。这个函数把问题改写成矩阵方程：

```text
Ks_inv M Kt_inv + G M B = H
```

然后用 generalized eigen decomposition 把左右两边同时对角化。对角化后，每个元素都可以单独除以：

```python
denom = 1.0 + np.outer(left_vals, right_vals)
X = rhs / denom
```

这一步就是把一个大线性系统变成很多个小的标量除法。

## 10.2 对应代码：solve_Du_sylvester

```python
def solve_Du_sylvester(
    Kt_inv: np.ndarray,
    Ks_inv: np.ndarray,
    B: np.ndarray,
    G: np.ndarray,
    rhs: np.ndarray,
    *,
    jitter: float = 1e-8,
) -> np.ndarray:
    if rhs.ndim == 1:
        Z = solve_sylvester_precision(
            Kt_inv,
            Ks_inv,
            B,
            G,
            unvec_f(rhs, (M_s, M_t)),
            jitter=jitter,
        )
        return vec_f(Z)
```

小白解释：`solve_Du_sylvester` 就是代码里所有 $D_u^{-1}(\cdot)$ 的统一入口。它没有真的构造巨大矩阵 $D_u$，而是把向量 reshape 成矩阵后用 Sylvester 方程解。

如果输入 `rhs` 是一维向量，函数返回一维向量；如果输入是多列矩阵，比如 $B_{\beta u}^\top$，函数会逐列解，然后用 `np.column_stack(cols)` 拼回来。

# 11. 公式块 10：Prediction Mean

理论文档写：

$$
\mathbb{E}[y_*]
=
\phi_*^\top m_\beta+a_*^\top m_u.
$$

## 11.1 对应代码：predict

文件：`stvgp_kronecker/joint_ssgp_kron/model.py`

```python
def predict(
    self,
    *,
    phi_star: np.ndarray,
    t_proj_star: np.ndarray,
    c_proj_star: np.ndarray,
    state: StructuredKronState,
    include_variance: bool = True,
) -> Prediction:
    beta_mean = 0.0 if phi_star.size == 0 else float(phi_star @ state.beta_mean)
    gp_mean = float(c_proj_star @ state.M_u @ t_proj_star)
    return Prediction(mean=beta_mean + gp_mean, variance=max(float(variance), self.jitter))
```

对应关系：

| 理论符号 | 代码 |
|---|---|
| $\phi_*^\top m_\beta$ | `phi_star @ state.beta_mean` |
| $a_*^\top m_u$ | `c_proj_star @ state.M_u @ t_proj_star` |

### 11.1.1 为什么 GP mean 写成三段乘法？

`state.M_u` 是 $m_u$ 的矩阵形式，形状是 `(M_s, M_t)`。

- `c_proj_star` 是测试空间点到 spatial inducing basis 的投影。
- `t_proj_star` 是测试时间点到 temporal inducing basis 的投影。
- `c_proj_star @ state.M_u @ t_proj_star` 等价于 $a_*^\top m_u$。

这样避免显式构造 $a_*=t_*\otimes c_*$。

# 12. 公式块 11：Predictive Variance

理论文档写：

$$
\mathrm{Var}(y_*)
=
\sigma^2
+
\nu_*
+
a_*^\top v_*
+
(\phi_*-B_{\beta u}v_*)^\top
\Lambda_{\beta\mid u}^{-1}
(\phi_*-B_{\beta u}v_*).
$$

其中：

$$
D_uv_*=a_*.
$$

## 12.1 对应代码：predictive_variance_decomposition

文件：`stvgp_kronecker/joint_ssgp_kron/model.py`

```python
projected_prior_var = float(
    (c_proj_star @ self.Ks @ c_proj_star)
    * (t_proj_star @ state.Kt_current @ t_proj_star)
)
nu_star = max(0.0, self.prior_point_variance - projected_prior_var)
Q = np.outer(c_proj_star, t_proj_star)
q = vec_f(Q)
```

这里对应：

$$
\nu_*=k(x_*,x_*)-K_{*u}K_{uu}^{-1}K_{u*}.
$$

接着是 Route B 方差主体：

```python
if state.R_beta_u is not None and state.S_beta_beta is not None:
    v_star = self.solve_Du(state, q)
    u_term = float(q @ v_star)
    if phi_star.size:
        adjusted_phi = phi_star - state.R_beta_u @ v_star
        beta_term = float(adjusted_phi @ state.S_beta_beta @ adjusted_phi)
```

对应：

$$
a_*^\top v_*
$$

和：

$$
(\phi_*-B_{\beta u}v_*)^\top
\Lambda_{\beta\mid u}^{-1}
(\phi_*-B_{\beta u}v_*).
$$

最后：

```python
total = self.sigma2 + nu_star + u_term + beta_term
```

对应：

$$
\sigma^2+\nu_*+\text{u posterior term}+\text{beta Schur term}.
$$

### 12.1.1 方差分解每一项是什么意思？

| 代码项 | 数学项 | 含义 |
|---|---|---|
| `self.sigma2` | $\sigma^2$ | observation noise，不确定性来自观测噪声 |
| `nu_star` | $\nu_*$ | sparse GP conditional residual variance |
| `u_term` | $a_*^\top D_u^{-1}a_*$ | GP inducing posterior uncertainty |
| `beta_term` | adjusted beta Schur quadratic | beta posterior uncertainty 以及 beta-u coupling 修正 |

`nu_star` 之前曾是重要 bug 来源：如果 kernel variance 不是 1，却仍默认 $k(x_*,x_*)=1$，NLL 和 coverage 会错。现在代码通过 `prior_point_variance` 显式带入 kernel amplitude。

## 12.2 为什么这里能体现 beta-u cross covariance？

看这一行：

```python
adjusted_phi = phi_star - state.R_beta_u @ v_star
```

如果没有 `R_beta_u`，那就是普通 mean-field 的：

$$
\phi_*^\top S_{\beta\beta}\phi_* + a_*^\top S_{uu}a_*.
$$

Route B 通过 `phi_star - R_beta_u @ v_star` 隐式保留了：

$$
2\phi_*^\top S_{\beta u}a_*.
$$

所以 Route B 的 predictive variance 不是简单地“beta variance + GP variance”，而是考虑了两者相关性。

如果 `state.R_beta_u is None`，代码会走 mean-field / projected-prior 的分支：

```python
beta_term = float(phi_star @ state.beta_cov @ phi_star)
Z = solve_sylvester_precision(...)
u_term = float(np.sum(Q * Z))
```

这时 beta 和 u 的 uncertainty 被分开计算，没有 `adjusted_phi`，因此没有 Route B 的 cross-covariance 修正。

# 13. 公式块 12：Mean-field 和 no-transfer 对照

理论文档中 mean-field ablation 是：

$$
q(\beta,u)\approx q(\beta)q(u).
$$

它不保留 $S_{\beta u}$。

## 13.1 对应代码：mean-field 走旧的 SSGP transfer

```python
def update_block_mean_field_ssgp_transfer(self, **kwargs: Any) -> StructuredKronState:
    return self.update_block_ssgp_transfer(**kwargs).copy_with(metadata={"method": "mean_field_ssgp_transfer"})
```

`update_block_ssgp_transfer` 会交替更新 beta 和 GP residual，但不存 Route B 的 `R_beta_u` cross natural block。

更具体地说，mean-field 路径里 beta 更新使用：

```python
beta_mean, beta_cov = self._update_beta(...)
```

GP 更新使用 residual：

```python
residual_vec = y_vec - Phi @ beta_mean
H_new = update_information_matrix(...)
M_u = solve_sylvester_precision(...)
```

这是一种交替/残差式实现。它有 GP，但没有 joint precision 里的 beta-u cross block。

## 13.2 no-transfer 是什么？

代码里：

```python
def update_block_no_transfer(self, **kwargs: Any) -> StructuredKronState:
    kwargs = dict(kwargs)
    kwargs["no_transfer"] = True
    return self.update_block_ssgp_transfer(**kwargs).copy_with(metadata={"method": "no_transfer"})
```

也就是说，`no_transfer` 不是“只有 linear 没有 GP”。它仍然有 linear + GP，只是不把旧 block 的 GP/likelihood 信息 transfer 到当前 block。它每个 block 更像重新从当前 block 学，持续学习 retention 会弱。

判断 no-transfer 的关键在 `_transfer_old_stats`：

```python
if state is None or no_transfer:
    return np.zeros((mt_new, mt_new)), np.zeros((self.Ks.shape[0], mt_new))
```

这里返回零矩阵，表示旧 likelihood stats 不进入当前 block。

# 14. 公式块 13：Initial-task full-GP MLL 选择 ell_t / noise / kernel variance

实验 runner 中还有一层超参数选择，不属于 Route B 核心公式，但影响实验：

```python
def full_gp_initial_task_marginal_nll(
    dataset: SyntheticDataset,
    stop_time: int,
    ell_t: float,
    *,
    noise: float | None = None,
    kernel_variance: float | None = None,
) -> float:
    """Exact GP marginal NLL on the initial task, integrating out beta."""
```

核心 covariance：

```python
Kt = rbf_kernel(times, lengthscale=float(ell_t), variance=variance)
Ks = rbf_kernel(spatial, lengthscale=dataset.spatial_lengthscale, variance=1.0)
cov = np.kron(Kt, Ks) + Phi @ beta_prior_cov @ Phi.T + noise2 * np.eye(y.shape[0])
```

选择方式：

```python
best_ell_t, best_noise, best_variance, best_score = min(scored, key=lambda item: item[3])
```

小白解释：

- 这不是 online 阶段偷偷看 future。
- 它只用 initial task/calibration task。
- 所有方法共享同一个选出来的 `ell_t` / noise / kernel variance。
- 这样比较 mean-field 和 Route B 才公平。

### 14.1 full-GP MLL 和实验指标 NLL 的区别

`full_gp_initial_task_marginal_nll` 用来选择超参数，它计算的是 initial task 上的 marginal negative log likelihood。这里会把 beta prior uncertainty 和 GP prior covariance 都放进 covariance：

```python
cov = np.kron(Kt, Ks) + Phi @ beta_prior_cov @ Phi.T + noise2 * I
```

而实验表里的 `nll` 是模型训练后对 evaluation data 的 predictive NLL。两者都叫 NLL，但用途不同：

- initial-task MLL/NLML：训练前或在线前选择共享超参数；
- predictive NLL：评估模型预测分布好不好。

# 15. 公式块 14：实验评估指标

评估函数位于 `scripts/run_joint_ssgp_kron_experiments.py`：

```python
def evaluate_state_on_factors(...):
    mean = factors.Phi @ state.beta_mean + dense_A_from_factors(factors.T, C_eval) @ vec_f(state.M_u)
    diag = block_variance_diagnostics(...)
    var = diag["total_variance"]
    return {
        "rmse": ...,
        "mae": ...,
        "nll": gaussian_nll(factors.y_vec, mean, var),
        "coverage90": coverage(factors.y_vec, mean, var, 0.90),
        ...
    }
```

重要点：

- NLL 和 coverage 覆盖的是 noisy observation $y_*$。
- 所以 predictive variance 中必须包含 $\sigma^2$。
- 代码中的 `total_variance` 已包含 `sigma2 + nu_star + posterior terms`。

# 16. 理论公式与代码函数速查表

| 理论模块 | 关键公式 | 代码函数 |
|---|---|---|
| observation model | $y=\Phi\beta+f+\epsilon$ | `make_indexed_block_factors`, `run_structured_method` |
| Kronecker design | $A=T\otimes C$ | `kron_mv`, `dense_A_from_factors` |
| old transfer matrix | $L_t=K_{on}K_{nn}^{-1}$ | `compute_Lt` |
| $u$ old precision transfer | $B_o\to L_t^\top B_oL_t$ | `transfer_temporal_precision` |
| beta-u cross transfer | $R_{\beta u} \to R_{\beta u}(L_t\otimes I_s)$ | `transfer_R_beta_u`, `apply_Lon_to_beta_u_cross_block` |
| new likelihood stats | $H^\top H/\sigma^2$, $H^\top y/\sigma^2$ | `joint_likelihood_stats` |
| accumulated stats | old + new | `update_block_structured_joint_ssgp_transfer` |
| Schur complement | $A-BD^{-1}B^\top$ | `schur_recover_posterior` |
| Sylvester solve | $K_s^{-1}ZK_t^{-1}+GZB=Q$ | `solve_sylvester_precision` |
| predictive mean | $\phi^\top m_\beta+a^\top m_u$ | `predict` |
| predictive variance | $\sigma^2+\nu+q^\top D^{-1}q+\cdots$ | `predictive_variance_decomposition` |
| dense validation | dense joint posterior | `dense_joint_posterior_reference`, tests |

# 17. 推荐阅读代码顺序

如果你代码基础不强，建议不要从实验脚本开始读。推荐顺序：

1. `structured_state.py`：先理解模型状态保存什么。
2. `kron_utils.py`：理解 `vec_f`, `kron_mv`, `solve_sylvester_precision`。
3. `ssgp_transfer.py`：理解 old likelihood transfer 和 new likelihood stats。
4. `model.py`：重点看 `update_block_structured_joint_ssgp_transfer` 和 `predictive_variance_decomposition`。
5. `run_joint_ssgp_kron_experiments.py`：最后看实验 runner 如何一块一块调用模型。
6. `verify_joint_ssgp_kron_derivations.py` 和 `tests/test_joint_ssgp_kron_routeB.py`：看理论公式如何被 dense reference 检查。

# 18. 最重要的一句话

Route B 的核心贡献是：

$$
\text{不把 }\beta\text{ 和 }u\text{ 拆成独立 posterior，而是保留 }R_{\beta u}
\text{ 并用 Schur complement 恢复 joint posterior effects。}
$$

代码中的核心体现就是：

```python
R_beta_u = old_stats["R_beta_u"] + new_stats["R_beta_u"]
schur = self.recover_posterior_mean_structured(...)
adjusted_phi = phi_star - state.R_beta_u @ v_star
```

这三处分别对应：

1. 保存 beta-u cross likelihood information；
2. 用 Schur complement 恢复 posterior mean/covariance；
3. 在 predictive variance 中隐式保留 beta-u posterior cross covariance。
