# Route B 理论公式逐行推导讲解 / Line-by-Line Formula Derivation Guide

源文档 / Source: `docs/new_main_joint_training_ssgp_kron_routeB_refined2_with_sylvester_appendix.tex`

这份讲解的目标不是替代理论文档，而是把文档中的主要公式按出现顺序翻译成“人能直接讲出来”的推导逻辑。每一块都包含：

- **公式 / Formula**：对应理论文档中的核心公式。
- **中文逐行解释**：每一行公式在做什么、为什么这样写。
- **English explanation**：同样逻辑的英文版本，方便和导师讨论。
- **讲给导师的要点 / Speaking point**：可以直接用于汇报的一句话。

The goal is not to replace the theory document. It is a teaching guide that explains the main formulas in the same order as the paper, with line-by-line derivations and bilingual speaking notes.

---

## 0. 符号总览 / Notation Overview

| 符号 / Symbol | 含义 / Meaning |
|---|---|
| \(t_n\) | time index or time coordinate |
| \(s_i\) | spatial location |
| \(y(t_n,s_i)\) | observed scalar value |
| \(\phi(t_n,s_i)\) | linear feature vector |
| \(\beta\) | linear-regression coefficient |
| \(f(t,s)\) | latent GP residual function |
| \(u_n\) | inducing variables under the current temporal basis |
| \(\Phi_n\) | block design matrix of all \(\phi\) rows |
| \(A_n\) | sparse GP projection from inducing variables to block observations |
| \(R^z,r^z\) | likelihood natural parameters |
| \(\Lambda,h\) | posterior natural parameters |
| \(B_{\beta u}\) | beta-u precision cross block |
| \(D_u\) | large inducing precision block |
| \(L_{on}\) | old-to-new inducing conditional mean operator |

---

## 1. Observation Model / 观测模型

### Formula 1.1 Scalar observation

$$
y(t_n,s_i)=\phi(t_n,s_i)^\top\beta+f(t_n,s_i)+\epsilon_{n,i},
\qquad
\epsilon_{n,i}\sim\mathcal N(0,\sigma^2).
$$

**中文逐行解释**

1. \(y(t_n,s_i)\) 是在时间 \(t_n\)、空间位置 \(s_i\) 看到的真实观测值。
2. \(\phi(t_n,s_i)^\top\beta\) 是线性趋势项。它负责解释可由人工特征描述的部分，例如常数项、时间趋势、经纬度或季节周期。
3. \(f(t_n,s_i)\) 是 GP residual。它负责解释线性特征解释不了的平滑时空残差。
4. \(\epsilon_{n,i}\) 是观测噪声，假设是均值为 0、方差为 \(\sigma^2\) 的高斯噪声。
5. 所以这个模型不是“先拟合 linear 再拟合 residual”，而是把 linear 和 GP residual 同时放进同一个观测方程。

**English explanation**

1. \(y(t_n,s_i)\) is the scalar observation at time \(t_n\) and spatial location \(s_i\).
2. \(\phi(t_n,s_i)^\top\beta\) is the parametric linear trend captured by engineered features.
3. \(f(t_n,s_i)\) is the nonparametric GP residual that captures smooth spatio-temporal structure not explained by the linear trend.
4. \(\epsilon_{n,i}\) is Gaussian observation noise with variance \(\sigma^2\).
5. The key modeling choice is joint observation: the linear trend and GP residual explain \(y\) together.

**讲给导师的要点 / Speaking point**

Route B starts from a joint likelihood for the raw observation \(y\), not from stale residuals produced by an old \(\hat\beta\).

### Formula 1.2 Block observation

$$
\mathbf y_n=\Phi_n\beta+\mathbf f_n+\epsilon_n,
\qquad
\epsilon_n\sim\mathcal N(0,\sigma^2 I_{B_n}).
$$

**中文逐行解释**

1. 一个 online block 里有很多观测点，把它们堆叠成向量 \(\mathbf y_n\)。
2. \(\Phi_n\) 是这个 block 的特征矩阵，每一行对应一个观测点的 \(\phi^\top\)。
3. \(\mathbf f_n\) 是 GP residual 在这个 block 所有观测点上的取值。
4. \(\epsilon_n\) 是同一 block 的噪声向量，协方差是 \(\sigma^2 I\)，表示各观测噪声独立同方差。
5. 这个形式方便后面写成矩阵高斯似然。

**English explanation**

1. All observations in online block \(n\) are stacked into \(\mathbf y_n\).
2. \(\Phi_n\) stacks the linear features row by row.
3. \(\mathbf f_n\) stacks the latent GP residual values at the same points.
4. The noise vector is independent Gaussian noise with covariance \(\sigma^2 I\).
5. This matrix form lets us write a Gaussian likelihood and derive natural-parameter updates.

---

## 2. Separable GP and Kronecker Prior / 可分离 GP 与 Kronecker 先验

### Formula 2.1 Separable spatio-temporal kernel

$$
f\sim\mathcal{GP}(0,k),
\qquad
k((s,t),(s',t'))=k_s(s,s')k_t(t,t').
$$

**中文逐行解释**

1. \(f\) 是均值为 0 的高斯过程。
2. 核函数 \(k\) 衡量两个时空点的相关性。
3. 这里假设时空核是可分离的：空间相关 \(k_s\) 乘以时间相关 \(k_t\)。
4. 这个假设非常关键，因为它让大矩阵可以写成 Kronecker 乘积，从而避免直接处理巨大 dense covariance。

**English explanation**

1. \(f\) is a zero-mean Gaussian process.
2. The kernel \(k\) measures covariance between two spatio-temporal inputs.
3. The separable assumption factorizes covariance into a spatial part and a temporal part.
4. This is the structural assumption that enables Kronecker algebra.

### Formula 2.2 Full-grid covariance

$$
K_{ff}=K_t\otimes K_s.
$$

**中文逐行解释**

1. 如果数据在完整的 time-by-space 网格上，时间协方差矩阵是 \(K_t\)，空间协方差矩阵是 \(K_s\)。
2. 因为核函数是 \(k_t k_s\)，完整网格上的协方差就是 \(K_t\otimes K_s\)。
3. \(\otimes\) 表示 Kronecker product，它把两个小矩阵组合成一个大矩阵。
4. 后面所有 Sylvester solver 的基础就是这个结构。

**English explanation**

1. On a full time-space grid, \(K_t\) stores temporal covariance and \(K_s\) stores spatial covariance.
2. The separable kernel makes the full covariance a Kronecker product.
3. The Kronecker product represents a large covariance using two smaller factors.
4. This structure is what makes the later Sylvester solver possible.

---

## 3. Temporal HiPPO-RFF Inducing Representation / 时间 HiPPO-RFF 诱导表示

### Formula 3.1 Scaled Legendre basis

$$
g_\ell^{(T)}(x)=\sqrt{\frac{2\ell+1}{T}}\,P_\ell\!\left(\frac{2x}{T}-1\right),
\qquad x\in[0,T].
$$

**中文逐行解释**

1. \(P_\ell\) 是第 \(\ell\) 阶 Legendre 多项式，它天然定义在 \([-1,1]\)。
2. \(\frac{2x}{T}-1\) 把时间区间 \([0,T]\) 线性映射到 \([-1,1]\)。
3. \(\sqrt{(2\ell+1)/T}\) 是归一化因子，使不同阶的 basis 在 \([0,T]\) 上具有合适的尺度。
4. \(T\) 会随着 online horizon 变化，所以 basis 本身也会变化，这就是 changing-basis 问题的来源。

**English explanation**

1. \(P_\ell\) is the \(\ell\)-th Legendre polynomial on \([-1,1]\).
2. The transformation \(2x/T-1\) maps the current time interval \([0,T]\) to \([-1,1]\).
3. The square-root factor normalizes the basis on the interval.
4. Since \(T\) changes online, the inducing coordinates also change.

### Formula 3.2 Temporal interdomain inducing variable

$$
u_\ell^{(t)}(T)=\int_0^T g_\ell^{(T)}(x)f_t(x)\,dx.
$$

**中文逐行解释**

1. 这不是在某个点上采样 GP，而是把 GP 函数 \(f_t(x)\) 投影到 basis \(g_\ell^{(T)}\) 上。
2. 积分表示“这一阶 basis 捕捉了时间函数中的多少成分”。
3. 所以 \(u_\ell^{(t)}(T)\) 是 interdomain inducing variable：它活在函数投影域，而不是原始输入点域。
4. 当 \(T\) 改变时，同一个 \(\ell\) 对应的投影方向也改变。

**English explanation**

1. This inducing variable is not a point evaluation. It is a projection of the temporal GP onto basis \(g_\ell^{(T)}\).
2. The integral measures how much of the temporal function lies in that basis direction.
3. Therefore it is an interdomain inducing variable.
4. Changing \(T\) changes the coordinate system.

### Formula 3.3 RFF approximation and basis matrix

$$
k_t(\tau,\tau')\approx
\sum_r w_r[
\cos(\omega_r\tau)\cos(\omega_r\tau')
+
\sin(\omega_r\tau)\sin(\omega_r\tau')
].
$$

$$
\big[B_t(T)\big]_{\ell,r}=\sqrt{w_r}I_\ell^{\cos}(\omega_r;T),
\qquad
\big[B_t(T)\big]_{\ell,R+r}=\sqrt{w_r}I_\ell^{\sin}(\omega_r;T).
$$

**中文逐行解释**

1. 第一行用随机傅里叶特征近似时间核。
2. 每个频率 \(\omega_r\) 贡献一对 cos/sin 特征。
3. \(w_r\) 是该频率的权重。
4. \(B_t(T)\) 把 HiPPO-Legendre basis 和 RFF basis 连接起来。
5. \(I_\ell^{\cos}\) 和 \(I_\ell^{\sin}\) 是把 Legendre basis 与 cos/sin 特征相乘后积分得到的系数。
6. 所以 \(B_t(T)\) 可以看成“时间 inducing basis 在 RFF 空间里的坐标”。

**English explanation**

1. The temporal kernel is approximated by random Fourier features.
2. Each frequency contributes a cosine and sine component.
3. \(B_t(T)\) maps the HiPPO-Legendre inducing basis into the RFF feature space.
4. The entries are oscillatory integrals between the Legendre basis and Fourier features.
5. This matrix lets us compute temporal inducing covariances efficiently.

### Formula 3.4 Temporal inducing covariance and cross-covariance

$$
K_{uu}^{(t)}(T)\approx B_t(T)B_t(T)^\top,
\qquad
k_{fu}^{(t)}(\tau;T)\approx \varphi_t(\tau)^\top B_t(T)^\top.
$$

**中文逐行解释**

1. \(K_{uu}^{(t)}\) 是 temporal inducing variables 之间的先验协方差。
2. 因为 inducing variables 被表示到 RFF 空间中，所以它近似为 \(B_tB_t^\top\)。
3. \(k_{fu}^{(t)}\) 是一个测试时间点 \(\tau\) 和 temporal inducing variables 的协方差。
4. \(\varphi_t(\tau)\) 是该时间点的 RFF 特征，乘上 \(B_t^\top\) 就得到它和 inducing basis 的相关性。

**English explanation**

1. \(K_{uu}^{(t)}\) is the prior covariance among temporal inducing variables.
2. In the RFF representation, it is approximated by \(B_tB_t^\top\).
3. \(k_{fu}^{(t)}\) is the covariance between a temporal input \(\tau\) and the inducing variables.
4. It is computed by projecting the RFF feature vector onto the inducing basis.

---

## 4. Mixed Spatio-Temporal Inducing Variables / 混合时空诱导变量

### Formula 4.1 Mixed inducing variable

$$
u_{\ell,j}(T)=\int_0^T g_\ell^{(T)}(x)f(x,z_j^s)\,dx.
$$

**中文逐行解释**

1. \(\ell\) 表示 temporal basis index。
2. \(j\) 表示 spatial inducing location index。
3. \(f(x,z_j^s)\) 是 GP 在固定空间 inducing 点 \(z_j^s\) 上随时间变化的函数。
4. 对时间积分后得到一个同时带有时间 basis 和空间 inducing 点的变量。

**English explanation**

1. \(\ell\) indexes the temporal basis.
2. \(j\) indexes the spatial inducing location.
3. The GP is evaluated at a fixed spatial inducing point and integrated over time.
4. The resulting variable combines temporal interdomain structure and spatial inducing points.

### Formula 4.2 Kronecker inducing covariance

$$
K_{uu}^{st}=K_{uu}^{(t)}\otimes K_{ZZ}^{(s)},
\qquad
K_{fu,n}^{st}=K_{fu,n}^{(t)}\otimes K_{XZ}^{(s)}.
$$

**中文逐行解释**

1. \(K_{uu}^{st}\) 是所有 mixed inducing variables 的先验协方差。
2. 时间部分由 \(K_{uu}^{(t)}\) 给出，空间部分由 \(K_{ZZ}^{(s)}\) 给出。
3. 因为核可分离，所以两者相乘变成 Kronecker product。
4. \(K_{fu,n}^{st}\) 是 block 观测点和 inducing variables 的 cross-covariance，也同样分解成时间 cross-cov 和空间 cross-cov。

**English explanation**

1. \(K_{uu}^{st}\) is the prior covariance of all mixed inducing variables.
2. The temporal factor is \(K_{uu}^{(t)}\), and the spatial factor is \(K_{ZZ}^{(s)}\).
3. Separability makes the covariance a Kronecker product.
4. The observation-inducing cross-covariance factorizes in the same way.

### Formula 4.3 Sparse projection matrix

$$
A_n=K_{fu,n}^{st}(K_{uu,n}^{st})^{-1}
=
\left[K_{fu,n}^{(t)}(K_{uu,n}^{(t)})^{-1}\right]
\otimes
\left[K_{XZ}^{(s)}(K_{ZZ}^{(s)})^{-1}\right].
$$

$$
T_n:=K_{fu,n}^{(t)}(K_{uu,n}^{(t)})^{-1},
\qquad
C:=K_{XZ}^{(s)}(K_{ZZ}^{(s)})^{-1},
\qquad
A_n=T_n\otimes C.
$$

**中文逐行解释**

1. \(A_n\) 是 sparse GP 的条件均值投影：它把 inducing variable \(u_n\) 映射到 block observation 位置。
2. 第一行是标准 sparse GP 条件均值系数 \(K_{fu}K_{uu}^{-1}\)。
3. 因为 \(K_{fu}\) 和 \(K_{uu}\) 都有 Kronecker 结构，所以 \(A_n\) 也分解成时间投影乘空间投影。
4. \(T_n\) 只管时间方向，\(C\) 只管空间方向。
5. 最后一行 \(A_n=T_n\otimes C\) 是后面高效计算 \(A_n^\top A_n\)、\(A_n^\top y_n\)、Sylvester solve 的核心。

**English explanation**

1. \(A_n\) is the sparse GP conditional-mean projection from inducing variables to observations.
2. It has the standard sparse GP form \(K_{fu}K_{uu}^{-1}\).
3. Since both covariance matrices are Kronecker-structured, the projection factorizes.
4. \(T_n\) handles temporal projection; \(C\) handles spatial projection.
5. The identity \(A_n=T_n\otimes C\) is the computational backbone of Route B.

---

## 5. Why Residual-Based Online Learning Fails / 为什么 residual-based 在线学习不一致

### Formula 5.1 Stale residual identity

$$
r_j(\hat\beta_n)
=y_j-\Phi_j\hat\beta_n
=y_j-\Phi_j\hat\beta_{n-1}-\Phi_j(\hat\beta_n-\hat\beta_{n-1})
=r_j(\hat\beta_{n-1})-\Phi_j(\hat\beta_n-\hat\beta_{n-1}).
$$

**中文逐行解释**

1. \(r_j(\hat\beta_n)\) 表示如果我们用最新的 \(\hat\beta_n\)，历史 block \(j\) 的 residual 应该是什么。
2. 第一等号只是 residual 的定义：观测减去 linear prediction。
3. 第二等号把 \(\hat\beta_n\) 写成旧参数 \(\hat\beta_{n-1}\) 加上变化量。
4. 第三等号说明：旧 residual \(r_j(\hat\beta_{n-1})\) 需要再减掉一项 \(\Phi_j(\hat\beta_n-\hat\beta_{n-1})\) 才是新 residual。
5. 如果历史数据已经丢掉，就无法修正旧 residual。这就是 stale residual target。

**English explanation**

1. \(r_j(\hat\beta_n)\) is the residual that historical block \(j\) would have under the latest linear coefficient.
2. The first equality is the residual definition.
3. The second equality decomposes the latest coefficient into the old coefficient plus a change.
4. The last equality shows that the old residual must be corrected when \(\beta\) changes.
5. If old raw observations are unavailable, residual-based online learning cannot repair the old targets.

### Formula 5.2 Joint likelihood avoids stale residuals

$$
y_j\mid\beta,u_j\sim
\mathcal N(\Phi_j\beta+A_j u_j,\sigma^2 I).
$$

**中文逐行解释**

1. Route B 不把 GP 训练目标设为 \(y-\Phi\hat\beta\)。
2. 它直接保留对 \(y_j\) 的联合解释：linear 部分 \(\Phi_j\beta\) 加 GP 部分 \(A_j u_j\)。
3. 因为 \(\beta\) 仍然是随机变量，历史信息可以继续约束 \(\beta\) 和 \(u\)，而不是固定在旧 residual 上。

**English explanation**

1. Route B does not train the GP on residuals computed from a point estimate of \(\beta\).
2. It keeps a joint likelihood for the raw observation \(y_j\).
3. Historical data constrain \(\beta\) and \(u\) jointly, avoiding stale residual targets.

---

## 6. Joint ELBO and the Missing Mean-Field Cross Term / 联合 ELBO 与 mean-field 丢失项

### Formula 6.1 Structured joint posterior

$$
q_n(\beta,u_n)
=
\mathcal N
\left(
\begin{bmatrix}\beta\\u_n\end{bmatrix}
\middle|
\begin{bmatrix}m_{\beta,n}\\m_{u,n}\end{bmatrix},
S_n
\right),
\qquad
S_n^{-1}=\Lambda_n.
$$

**中文逐行解释**

1. 近似后验不是 \(q(\beta)q(u)\)，而是 \(\beta\) 和 \(u_n\) 的联合高斯。
2. 均值由 \(m_{\beta,n}\) 和 \(m_{u,n}\) 组成。
3. 协方差 \(S_n\) 允许 \(\beta\) 和 \(u_n\) 有 cross covariance。
4. \(\Lambda_n=S_n^{-1}\) 是 precision matrix，后面所有更新都更方便写成 precision form。

**English explanation**

1. The approximate posterior is a joint Gaussian over \((\beta,u_n)\), not a product of marginals.
2. The mean contains both the linear coefficient mean and inducing mean.
3. The covariance allows beta-u posterior dependence.
4. The precision form \(\Lambda_n=S_n^{-1}\) is convenient for Gaussian updates.

### Formula 6.2 Expected log likelihood

$$
\begin{aligned}
\mathbb E_q[\log p(y_n\mid\beta,u_n)]
=&-\frac{B_n}{2}\log(2\pi\sigma^2)\\
&-\frac{1}{2\sigma^2}
\Big[
\|y_n-\Phi_n m_\beta-A_n m_u\|^2
+\operatorname{tr}(\Phi_nS_{\beta\beta}\Phi_n^\top)\\
&\qquad
+\operatorname{tr}(A_nS_{uu}A_n^\top)
+2\operatorname{tr}(\Phi_nS_{\beta u}A_n^\top)
\Big].
\end{aligned}
$$

**中文逐行解释**

1. 第一行是高斯 likelihood 的归一化常数，\(B_n\) 是 block 中观测数量。
2. 第二行的平方误差是用 posterior mean 预测 \(y_n\) 的误差。
3. \(\operatorname{tr}(\Phi S_{\beta\beta}\Phi^\top)\) 是 linear 参数不确定性带来的输出方差。
4. \(\operatorname{tr}(A S_{uu}A^\top)\) 是 GP inducing 参数不确定性带来的输出方差。
5. \(2\operatorname{tr}(\Phi S_{\beta u}A^\top)\) 是 linear 和 GP residual 的相关性项。
6. mean-field 假设 \(S_{\beta u}=0\)，所以会直接丢掉最后一项。

**English explanation**

1. The first term is the Gaussian likelihood normalization.
2. The squared residual uses the posterior means of \(\beta\) and \(u\).
3. The \(\beta\)-trace term is output uncertainty due to the linear coefficient.
4. The \(u\)-trace term is output uncertainty due to the GP inducing variables.
5. The cross trace is the covariance contribution between the linear trend and GP residual.
6. Mean-field removes this term by setting \(S_{\beta u}=0\).

**讲给导师的要点 / Speaking point**

The direct theoretical difference between Route B and mean-field is the retained beta-u covariance term.

---

## 7. Fixed-Basis Exact Gaussian Update / 固定 basis 下的精确高斯更新

### Formula 7.1 Joint state and likelihood matrix

$$
z=\begin{bmatrix}\beta\\u\end{bmatrix},
\qquad
H_n=\begin{bmatrix}\Phi_n&A_n\end{bmatrix},
\qquad
y_n\mid z\sim\mathcal N(H_nz,\sigma^2I).
$$

**中文逐行解释**

1. \(z\) 把 \(\beta\) 和 \(u\) 拼成一个大状态向量。
2. \(H_n\) 把 linear design matrix 和 GP projection matrix 横向拼接。
3. \(H_nz=\Phi_n\beta+A_nu\)，正好对应联合观测模型。
4. 在 fixed basis 情况下，上一轮和这一轮的 \(u\) 是同一组坐标，因此可以直接做标准 Gaussian update。

**English explanation**

1. \(z\) concatenates the linear coefficients and inducing variables.
2. \(H_n\) concatenates the linear and GP design matrices.
3. \(H_nz\) reproduces the joint observation mean.
4. With a fixed basis, the same state can be updated by standard Gaussian conditioning.

### Formula 7.2 Natural-parameter update

$$
\Lambda_n=\Lambda_{n-1}+\sigma^{-2}H_n^\top H_n,
\qquad
h_n=h_{n-1}+\sigma^{-2}H_n^\top y_n.
$$

$$
S_n=\Lambda_n^{-1},
\qquad
m_n=S_nh_n.
$$

**中文逐行解释**

1. 高斯 likelihood 对 posterior precision 的贡献是 \(\sigma^{-2}H^\top H\)。
2. 对 information vector 的贡献是 \(\sigma^{-2}H^\top y\)。
3. 所以新的 natural parameters 等于旧 posterior natural parameters 加上当前 block 的 likelihood natural parameters。
4. 得到 precision \(\Lambda_n\) 后，协方差是它的逆。
5. 均值由 \(m=S h=\Lambda^{-1}h\) 得到。

**English explanation**

1. A Gaussian likelihood contributes \(\sigma^{-2}H^\top H\) to precision.
2. It contributes \(\sigma^{-2}H^\top y\) to the information vector.
3. The new natural parameters are old natural parameters plus current likelihood contributions.
4. The covariance is the inverse precision.
5. The mean is recovered by \(m=\Lambda^{-1}h\).

### Formula 7.3 Block expansion

$$
H_n^\top H_n
=
\begin{bmatrix}
\Phi_n^\top\Phi_n & \Phi_n^\top A_n\\
A_n^\top\Phi_n & A_n^\top A_n
\end{bmatrix}.
$$

**中文逐行解释**

1. 左上角 \(\Phi^\top\Phi\) 是 linear-linear 信息。
2. 右下角 \(A^\top A\) 是 GP-GP 信息。
3. 右上角 \(\Phi^\top A\) 和左下角 \(A^\top\Phi\) 是 linear 与 GP 的耦合信息。
4. Route B 的核心就是不把这两个 cross blocks 扔掉。

**English explanation**

1. The top-left block is information about \(\beta\).
2. The bottom-right block is information about \(u\).
3. The off-diagonal blocks encode coupling between the linear trend and GP residual.
4. Route B keeps these cross blocks.

---

## 8. Changing-Basis Transfer / Changing basis 下的旧信息转移

### Formula 8.1 Old and new inducing variables are different

$$
u_\ell^{old}=\int_0^{T_{n-1}}g_\ell^{(T_{n-1})}(x)f(x)\,dx,
\qquad
u_\ell^{new}=\int_0^{T_n}g_\ell^{(T_n)}(x)f(x)\,dx.
$$

**中文逐行解释**

1. \(u_\ell^{old}\) 使用旧时间窗口 \(T_{n-1}\) 上的 basis。
2. \(u_\ell^{new}\) 使用新时间窗口 \(T_n\) 上的 basis。
3. 虽然都叫第 \(\ell\) 个 inducing variable，但投影函数不同、积分区间不同，所以它们不是同一个随机变量。
4. 因此不能把上一轮 \(q(\beta,u_o)\) 直接当作这一轮 \(q(\beta,u_n)\) 的 prior。

**English explanation**

1. The old inducing variable is defined using the old temporal basis and horizon.
2. The new inducing variable is defined using the new basis and horizon.
3. They are different random variables even if they share the same index.
4. Therefore old posterior information must be transferred to the new coordinates.

### Formula 8.2 Streaming old-likelihood ratio

$$
\frac{q_{n-1}(\beta,u_o)}{p(\beta)p(u_o)}.
$$

**中文逐行解释**

1. \(q_{n-1}\) 包含“旧数据之后的 posterior”。
2. \(p(\beta)p(u_o)\) 是旧坐标下的 base prior。
3. 两者相除后，留下的是旧数据带来的 likelihood information。
4. streaming sparse GP 的思想是：不要转移 posterior 本身，而是转移 old likelihood ratio。

**English explanation**

1. \(q_{n-1}\) is the posterior after old data.
2. \(p(\beta)p(u_o)\) is the base prior in the old coordinates.
3. Their ratio isolates the information contributed by old observations.
4. Streaming sparse GP transfers this likelihood ratio, not the posterior itself.

### Formula 8.3 Route B streaming VFE update

$$
q_n(\beta,u_n)\propto
p(\beta)p(u_n)
\exp\left\{
\mathbb E_{p(u_o\mid u_n)}
\left[
\log\frac{q_{n-1}(\beta,u_o)}{p(\beta)p(u_o)}
\right]
+\log p(y_n\mid\beta,u_n)
\right\}.
$$

**中文逐行解释**

1. 左边是当前 block 后的新 posterior。
2. \(p(\beta)p(u_n)\) 是当前坐标下的 base prior。
3. 指数里的第一项是“旧 likelihood 信息”，但它原本写在旧变量 \(u_o\) 上。
4. 用 \(p(u_o\mid u_n)\) 做期望，就是把旧坐标的信息转移到新坐标 \(u_n\)。
5. 指数里的第二项 \(\log p(y_n\mid\beta,u_n)\) 是当前 block 的新 likelihood。
6. 这条公式是 Route B changing-basis online update 的核心。

**English explanation**

1. The left side is the new posterior after block \(n\).
2. \(p(\beta)p(u_n)\) is the base prior in the new coordinates.
3. The first exponential term is historical likelihood information written in old coordinates.
4. The expectation under \(p(u_o\mid u_n)\) transfers that information to the new coordinates.
5. The second term is the current block likelihood.
6. This is the core changing-basis Route B update.

---

## 9. Old Likelihood Natural Parameters / 旧似然自然参数

### Formula 9.1 Natural-parameter representation

$$
\log\frac{q_{n-1}(\beta,u_o)}{p(\beta)p(u_o)}
=
-\frac12
\begin{bmatrix}\beta\\u_o\end{bmatrix}^\top
R_o^z
\begin{bmatrix}\beta\\u_o\end{bmatrix}
+
\begin{bmatrix}\beta\\u_o\end{bmatrix}^\top r_o^z
+c_o.
$$

**中文逐行解释**

1. old likelihood ratio 是一个高斯因子，所以 log 以后是二次函数。
2. \(-\frac12 z^\top Rz\) 是二次项，\(R_o^z\) 表示旧数据提供的 precision 信息。
3. \(z^\top r\) 是一次项，\(r_o^z\) 表示旧数据提供的 information vector。
4. \(c_o\) 是常数项，和 \((\beta,u)\) 无关，更新 posterior 时可以忽略。

**English explanation**

1. The old likelihood ratio is a Gaussian factor, so its log is quadratic.
2. \(R_o^z\) is the likelihood precision contributed by old data.
3. \(r_o^z\) is the corresponding information vector.
4. Constants can be ignored for posterior recovery.

### Formula 9.2 Block structure

$$
R_o^z=
\begin{bmatrix}
R_{\beta\beta,o}&R_{\beta u,o}\\
R_{u\beta,o}&R_{uu,o}
\end{bmatrix},
\qquad
r_o^z=
\begin{bmatrix}
r_{\beta,o}\\r_{u,o}
\end{bmatrix}.
$$

**中文逐行解释**

1. \(R_{\beta\beta,o}\) 是旧数据对 linear 系数的 precision。
2. \(R_{uu,o}\) 是旧数据对 inducing GP 变量的 precision。
3. \(R_{\beta u,o}\) 是旧数据对 \(\beta\) 和 \(u\) 之间耦合关系的 precision。
4. mean-field 或只转移 GP marginal 的方法通常丢掉第三项；Route B 保留它。

**English explanation**

1. \(R_{\beta\beta,o}\) is old information about \(\beta\).
2. \(R_{uu,o}\) is old information about \(u\).
3. \(R_{\beta u,o}\) stores beta-u coupling information.
4. Route B explicitly retains this block.

---

## 10. Closed-Form Transfer of Old Information / 旧信息的闭式转移

### Formula 10.1 GP conditional from new to old inducing variables

$$
p(u_o\mid u_n)=\mathcal N(L_{on}u_n,\Sigma_{o\mid n}),
\qquad
L_{on}=K_{on}K_{nn}^{-1}.
$$

**中文逐行解释**

1. GP prior 告诉我们旧 inducing variables 和新 inducing variables 的联合高斯关系。
2. 条件均值是 \(L_{on}u_n\)，即用新坐标 \(u_n\) 预测旧坐标 \(u_o\)。
3. \(L_{on}=K_{on}K_{nn}^{-1}\) 是高斯条件分布中的标准线性回归系数。
4. \(\Sigma_{o\mid n}\) 是条件不确定性，它只贡献常数项，不影响 posterior natural parameters。

**English explanation**

1. The GP prior defines a joint Gaussian between old and new inducing variables.
2. The conditional mean predicts old coordinates from new coordinates.
3. \(L_{on}=K_{on}K_{nn}^{-1}\) is the standard Gaussian conditioning operator.
4. The conditional covariance affects only constants in the likelihood transfer.

### Formula 10.2 Deterministic transfer matrix

$$
M_{on}:=
\begin{bmatrix}
I_d&0\\
0&L_{on}
\end{bmatrix}.
$$

**中文逐行解释**

1. \(\beta\) 是同一个全局 linear coefficient，不需要换坐标，所以对应 \(I_d\)。
2. \(u_o\) 需要由 \(u_n\) 预测，所以对应 \(L_{on}\)。
3. 这个 block diagonal matrix 把新状态 \((\beta,u_n)\) 映射成旧 likelihood 所需要的旧状态均值 \((\beta,L_{on}u_n)\)。

**English explanation**

1. \(\beta\) is unchanged across bases, so it uses the identity map.
2. \(u_o\) is predicted from \(u_n\), so it uses \(L_{on}\).
3. The block matrix maps new coordinates into the old likelihood coordinate system.

### Formula 10.3 Compact transfer

$$
R_{o\rightarrow n}^z=M_{on}^\top R_o^zM_{on},
\qquad
r_{o\rightarrow n}^z=M_{on}^\top r_o^z.
$$

**中文逐行解释**

1. 对二次项 \(z^\top Rz\)，变量替换 \(z_o=M_{on}z_n\) 会得到 \(z_n^\top M^\top R M z_n\)。
2. 所以 precision 按 \(M^\top R M\) 转移。
3. 对一次项 \(z^\top r\)，变量替换后得到 \(z_n^\top M^\top r\)。
4. 所以 information vector 按 \(M^\top r\) 转移。

**English explanation**

1. A quadratic form transforms as \(R\mapsto M^\top RM\).
2. A linear information vector transforms as \(r\mapsto M^\top r\).
3. This is simply Gaussian natural-parameter change of coordinates.

### Formula 10.4 Block transfer equations

$$
\begin{aligned}
R_{\beta\beta,o\rightarrow n}&=R_{\beta\beta,o},\\
R_{\beta u,o\rightarrow n}&=R_{\beta u,o}L_{on},\\
R_{u\beta,o\rightarrow n}&=L_{on}^\top R_{u\beta,o},\\
R_{uu,o\rightarrow n}&=L_{on}^\top R_{uu,o}L_{on},\\
r_{\beta,o\rightarrow n}&=r_{\beta,o},\\
r_{u,o\rightarrow n}&=L_{on}^\top r_{u,o}.
\end{aligned}
$$

**中文逐行解释**

1. 第一行：\(\beta\) 没有换坐标，所以 \(\beta\beta\) block 不变。
2. 第二行：\(\beta u\) block 的 \(u\) 侧需要从旧坐标转到新坐标，所以右乘 \(L_{on}\)。
3. 第三行：\(u\beta\) block 是对称位置，所以左乘 \(L_{on}^\top\)。
4. 第四行：\(uu\) block 两边都涉及 \(u\)，所以左右分别乘 \(L_{on}^\top\) 和 \(L_{on}\)。
5. 第五行：\(\beta\) information vector 不变。
6. 第六行：\(u\) information vector 转移到新坐标，所以乘 \(L_{on}^\top\)。

**English explanation**

1. The beta-beta block is unchanged because beta is unchanged.
2. The beta-u block transforms on the \(u\) side by right multiplication with \(L_{on}\).
3. The u-beta block transforms symmetrically by left multiplication with \(L_{on}^\top\).
4. The u-u block transforms on both sides.
5. The beta information vector is unchanged.
6. The u information vector is pulled into the new coordinates by \(L_{on}^\top\).

---

## 11. New Block Assimilation / 当前 block 的新信息吸收

### Formula 11.1 New likelihood natural parameters

$$
H_n=\begin{bmatrix}\Phi_n&A_n\end{bmatrix},
\qquad
R_{\mathrm{new},n}^z=\frac1{\sigma^2}H_n^\top H_n,
\qquad
r_{\mathrm{new},n}^z=\frac1{\sigma^2}H_n^\top y_n.
$$

**中文逐行解释**

1. \(H_n\) 把 linear 部分和 GP 部分拼到同一个设计矩阵里。
2. Gaussian likelihood 对 precision 的贡献是 \(H^\top H/\sigma^2\)。
3. Gaussian likelihood 对 information vector 的贡献是 \(H^\top y/\sigma^2\)。
4. 这里的 \(y_n\) 是原始观测，不是 residual。

**English explanation**

1. \(H_n\) combines the linear and GP design matrices.
2. A Gaussian likelihood contributes \(H^\top H/\sigma^2\) to precision.
3. It contributes \(H^\top y/\sigma^2\) to the information vector.
4. The target is raw \(y_n\), not a precomputed residual.

### Formula 11.2 Accumulated likelihood statistics

$$
\begin{aligned}
R_{\beta\beta,n}
&=R_{\beta\beta,o}+\frac1{\sigma^2}\Phi_n^\top\Phi_n,\\
R_{\beta u,n}
&=R_{\beta u,o}L_{on}+\frac1{\sigma^2}\Phi_n^\top A_n,\\
R_{u\beta,n}
&=L_{on}^\top R_{u\beta,o}+\frac1{\sigma^2}A_n^\top\Phi_n,\\
R_{uu,n}
&=L_{on}^\top R_{uu,o}L_{on}+\frac1{\sigma^2}A_n^\top A_n,\\
r_{\beta,n}
&=r_{\beta,o}+\frac1{\sigma^2}\Phi_n^\top y_n,\\
r_{u,n}
&=L_{on}^\top r_{u,o}+\frac1{\sigma^2}A_n^\top y_n.
\end{aligned}
$$

**中文逐行解释**

1. 每一行都有同样结构：旧信息转移到新坐标，加上当前 block 的新 likelihood 信息。
2. \(\beta\beta\) block：旧 \(\beta\) 信息不需要转移，只加当前 \(\Phi^\top\Phi\)。
3. \(\beta u\) block：旧 cross 信息先右乘 \(L_{on}\)，再加当前 \(\Phi^\top A\)。
4. \(u\beta\) block：和上一行转置对应。
5. \(uu\) block：旧 GP precision 需要 \(L^\top R L\) 转移，再加当前 \(A^\top A\)。
6. \(r_\beta\)：旧 linear information 加当前 \(\Phi^\top y\)。
7. \(r_u\)：旧 GP information 先转移，再加当前 \(A^\top y\)。

**English explanation**

1. Every line has the same pattern: transferred old information plus new block information.
2. The beta-beta statistic only accumulates \(\Phi^\top\Phi\).
3. The beta-u statistic transfers the old cross block and adds the new cross likelihood.
4. The u-beta block is the transpose-side counterpart.
5. The u-u statistic transfers the old GP precision and adds \(A^\top A\).
6. The beta information vector accumulates \(\Phi^\top y\).
7. The u information vector transfers old GP information and adds \(A^\top y\).

### Formula 11.3 Add priors to form posterior natural parameters

$$
\begin{aligned}
\Lambda_{\beta\beta,n}&=P_\beta+R_{\beta\beta,n},\\
\Lambda_{\beta u,n}&=R_{\beta u,n},\\
\Lambda_{uu,n}&=K_{nn}^{-1}+R_{uu,n},\\
h_{\beta,n}&=P_\beta m_{\beta,0}+r_{\beta,n},\\
h_{u,n}&=r_{u,n}.
\end{aligned}
$$

**中文逐行解释**

1. posterior precision = prior precision + likelihood precision。
2. \(\beta\) 的 prior precision 是 \(P_\beta=K_\beta^{-1}\)，所以加到 \(\Lambda_{\beta\beta}\)。
3. \(\beta u\) 没有独立 prior cross term，因为 prior 中 \(\beta\) 和 \(u\) 独立，所以 cross precision 只来自 likelihood。
4. \(u\) 的 prior precision 是 \(K_{nn}^{-1}\)，所以加到 \(\Lambda_{uu}\)。
5. \(h_\beta\) 的 prior information 是 \(P_\beta m_{\beta,0}\)。
6. \(u\) 的 prior mean 是 0，所以 \(h_u\) 只有 likelihood information。

**English explanation**

1. Posterior precision equals prior precision plus likelihood precision.
2. The beta prior contributes \(P_\beta\).
3. The beta-u precision has no prior cross term because the base prior factorizes.
4. The inducing prior contributes \(K_{nn}^{-1}\).
5. The beta information vector receives \(P_\beta m_{\beta,0}\).
6. The inducing prior mean is zero, so \(h_u\) only contains likelihood information.

---

## 12. Projected-Prior Baseline / Projected-prior 只是诊断基线

### Formula 12.1 Moment projection baseline

$$
p_n^{\mathrm{proj}}(\beta,u_n)
=
\int p(u_n\mid u_o)q_{n-1}(\beta,u_o)\,du_o.
$$

$$
\begin{aligned}
m_{\beta,n|n-1}^{proj}&=m_{\beta,o},\\
m_{u,n|n-1}^{proj}&=L_{no}m_{u,o},\\
S_{\beta u,n|n-1}^{proj}&=S_{\beta u,o}L_{no}^\top,\\
S_{uu,n|n-1}^{proj}
&=K_{nn}+L_{no}(S_{uu,o}-K_{oo})L_{no}^\top.
\end{aligned}
$$

**中文逐行解释**

1. projected-prior 直接把上一轮 posterior moment 投影成下一轮 prior。
2. \(\beta\) 不变，所以 \(m_\beta\) 不变。
3. \(u\) 的均值用 old-to-new 条件均值 \(L_{no}\) 转移。
4. cross covariance 也可以按 moment projection 转移。
5. \(S_{uu}\) 公式是 GP 条件传播中的 covariance projection。
6. 但它不是 Route B 的主理论，因为它转移 posterior moment，而不是 old likelihood ratio。

**English explanation**

1. Projected-prior transfer propagates posterior moments into the next prior.
2. Beta remains unchanged.
3. The inducing mean is projected using a conditional mean operator.
4. The cross covariance can also be projected.
5. The u-u covariance follows GP conditional moment propagation.
6. This is a diagnostic ablation, not the main Route B old-likelihood-ratio method.

---

## 13. Schur Complement Posterior Recovery / Schur 补恢复后验

### Formula 13.1 Precision block notation

$$
\Lambda_n=
\begin{bmatrix}
A_\beta&B_{\beta u}\\
B_{\beta u}^\top&D_u
\end{bmatrix},
\qquad
h_n=
\begin{bmatrix}
h_\beta\\h_u
\end{bmatrix}.
$$

**中文逐行解释**

1. \(A_\beta\) 是小的 \(\beta\)-precision block，维度只有 \(d\times d\)。
2. \(D_u\) 是大的 inducing precision block，维度是 \(M_tM_s\times M_tM_s\)。
3. \(B_{\beta u}\) 是 precision cross block，保留 linear 和 GP 的耦合。
4. 目标是不直接求整个 \(\Lambda^{-1}\)，而是利用 block 结构求需要的均值和二次型。

**English explanation**

1. \(A_\beta\) is the small beta precision block.
2. \(D_u\) is the large inducing precision block.
3. \(B_{\beta u}\) stores beta-u coupling in precision form.
4. We avoid inverting the full matrix by using block algebra.

### Formula 13.2 Schur complement precision

$$
\Lambda_{\beta|u}
:=
A_\beta-B_{\beta u}D_u^{-1}B_{\beta u}^\top.
$$

**中文逐行解释**

1. 这是把大的 \(u\) block 消去后，\(\beta\) 的有效 precision。
2. \(B_{\beta u}D_u^{-1}B_{\beta u}^\top\) 表示 GP block 通过 cross block 对 \(\beta\) precision 的修正。
3. 注意它是 Schur complement precision，不是 \(\beta\) covariance。
4. \(\beta\) covariance 是它的逆：\(S_{\beta\beta}=\Lambda_{\beta|u}^{-1}\)。

**English explanation**

1. This is the effective beta precision after eliminating the inducing block.
2. The correction term accounts for coupling through the GP block.
3. It is a precision matrix, not a covariance.
4. Its inverse gives \(S_{\beta\beta}\).

### Formula 13.3 Required Sylvester solves

$$
D_uW=B_{\beta u}^\top,
\qquad
D_uv_h=h_u,
\qquad
\Lambda_{\beta|u}=A_\beta-B_{\beta u}W.
$$

**中文逐行解释**

1. \(W=D_u^{-1}B_{\beta u}^\top\)，但不显式求 \(D_u^{-1}\)，而是解线性系统。
2. \(v_h=D_u^{-1}h_u\)，同样通过线性系统得到。
3. 因为 \(B_{\beta u}^\top\) 只有 \(d\) 列，所以只需要 \(d\) 次右端求解。
4. 加上 \(h_u\) 的一次求解，总共是 \(d+1\) 次 Sylvester solve。

**English explanation**

1. \(W\) is \(D_u^{-1}B_{\beta u}^\top\), computed by solving systems.
2. \(v_h\) is \(D_u^{-1}h_u\).
3. Since the beta dimension is small, only \(d\) right-hand sides are needed for \(W\).
4. Together with \(v_h\), posterior recovery requires \(d+1\) large-block solves.

### Formula 13.4 Posterior means

$$
m_\beta=
\Lambda_{\beta|u}^{-1}(h_\beta-B_{\beta u}v_h),
\qquad
m_u=v_h-Wm_\beta.
$$

**中文逐行解释**

1. \(h_\beta-B_{\beta u}v_h\) 是消去 \(u\) 后 \(\beta\) 的有效 information vector。
2. 乘上 \(\Lambda_{\beta|u}^{-1}\) 得到 \(\beta\) 的 posterior mean。
3. \(m_u=v_h-Wm_\beta\) 来自 block linear system 的第二行。
4. 它表示：先用 \(h_u\) 得到 \(u\) 的基准解，再减去 \(\beta\) mean 通过 cross block 对 \(u\) 的影响。

**English explanation**

1. \(h_\beta-B_{\beta u}v_h\) is the effective beta information vector after eliminating \(u\).
2. Multiplying by the inverse Schur complement gives \(m_\beta\).
3. The expression for \(m_u\) comes from the second block row of the precision system.
4. It adjusts the inducing mean by the beta-u coupling.

### Formula 13.5 Covariance blocks

$$
\begin{aligned}
S_{\beta\beta}&=\Lambda_{\beta|u}^{-1},\\
S_{\beta u}&=-\Lambda_{\beta|u}^{-1}B_{\beta u}D_u^{-1},\\
S_{uu}&=D_u^{-1}+D_u^{-1}B_{\beta u}^\top
\Lambda_{\beta|u}^{-1}B_{\beta u}D_u^{-1}.
\end{aligned}
$$

**中文逐行解释**

1. \(S_{\beta\beta}\) 是 Schur complement precision 的逆。
2. \(S_{\beta u}\) 是 posterior cross covariance，它不是直接存储的，而是由 Schur 公式隐式给出。
3. \(S_{uu}\) 是 GP block covariance：基础项 \(D_u^{-1}\) 加上通过 \(\beta\) block 回传的修正。
4. Route B 预测方差中真正用到的就是这些 block 的二次型，而不是完整 dense covariance。

**English explanation**

1. The beta covariance is the inverse Schur complement.
2. The beta-u covariance is recovered implicitly.
3. The u-u covariance includes the base inverse \(D_u^{-1}\) plus a coupling correction through beta.
4. Route B evaluates needed quadratic forms without materializing the full covariance.

---

## 14. Kronecker-Aware Implementation / Kronecker 结构实现

### Formula 14.1 \(A^\top A\) factorization

$$
A_n=T_n\otimes C,
\qquad
A_n^\top A_n=(T_n^\top T_n)\otimes(C^\top C),
\qquad
G:=C^\top C.
$$

**中文逐行解释**

1. \(A_n\) 已经分解成时间投影 \(T_n\) 和空间投影 \(C\)。
2. Kronecker product 的转置和乘法性质给出 \(A^\top A=(T^\top T)\otimes(C^\top C)\)。
3. \(G=C^\top C\) 只依赖空间 inducing geometry，可以提前算好。
4. 这让每个 block 只需要更新 temporal statistic。

**English explanation**

1. \(A_n\) factorizes into temporal and spatial projection matrices.
2. Kronecker algebra gives the factorization of \(A_n^\top A_n\).
3. \(G=C^\top C\) is a fixed spatial Gram factor.
4. Online updates only need to refresh temporal statistics.

### Formula 14.2 Temporal-only transfer operator

$$
K_{nn}=K_{nn}^{(t)}\otimes K_s,
\qquad
K_{on}=K_{on}^{(t)}\otimes K_s,
$$

$$
L_{on}=K_{on}K_{nn}^{-1}
=L_{on}^{(t)}\otimes I_s,
\qquad
L_{on}^{(t)}=K_{on}^{(t)}(K_{nn}^{(t)})^{-1}.
$$

**中文逐行解释**

1. 因为空间 inducing locations 固定，old-new covariance 和 new-new covariance 共享同一个空间因子 \(K_s\)。
2. 计算 \(K_{on}K_{nn}^{-1}\) 时，空间部分变成 \(K_sK_s^{-1}=I_s\)。
3. 所以 changing-basis transfer 只作用在 temporal inducing coordinates 上。
4. 这就是 \(L_{on}=L_{on}^{(t)}\otimes I_s\) 的含义。

**English explanation**

1. Fixed spatial inducing locations make old-new and new-new covariance share the same spatial factor.
2. The spatial factor cancels in the conditional mean operator.
3. Therefore the transfer acts only along the temporal inducing dimension.
4. This is why \(L_{on}=L_{on}^{(t)}\otimes I_s\).

### Formula 14.3 Kronecker-preserving \(u\)-precision transfer

$$
R_{uu,o}=B_o\otimes G,
$$

$$
R_{uu,o\rightarrow n}
=L_{on}^\top R_{uu,o}L_{on}
=\left[(L_{on}^{(t)})^\top B_oL_{on}^{(t)}\right]\otimes G.
$$

**中文逐行解释**

1. \(R_{uu,o}=B_o\otimes G\) 表示旧 likelihood precision 被维护成一个 temporal factor 乘固定 spatial factor。
2. \(L_{on}\) 只作用在时间维度，所以转移后仍然是同一个空间因子 \(G\)。
3. temporal factor 从 \(B_o\) 更新成 \((L_t)^\top B_oL_t\)。
4. 因此 old likelihood transfer 后，\(u\)-precision 仍保持单个 Kronecker product。

**English explanation**

1. The old u-u likelihood precision is stored as a temporal factor times a fixed spatial factor.
2. Since \(L_{on}\) acts only temporally, the spatial factor remains \(G\).
3. The temporal factor transforms as \(L_t^\top B_oL_t\).
4. The transfer preserves the single-Kronecker structure.

### Formula 14.4 Updated temporal likelihood statistic

$$
B_n=
B_{o\rightarrow n}
+
\frac1{\sigma^2}T_n^\top T_n.
$$

$$
\Lambda_{uu,n}
=
(K_{nn}^{(t)})^{-1}\otimes K_s^{-1}
+
B_n\otimes G.
$$

**中文逐行解释**

1. \(B_{o\rightarrow n}\) 是旧 temporal likelihood statistic 转移后的结果。
2. 当前 block 新增 \(\frac1{\sigma^2}T_n^\top T_n\)。
3. 两者相加得到新的 temporal statistic \(B_n\)。
4. posterior 的 \(u\)-precision 由 prior precision 加 likelihood precision 组成。
5. prior precision 是 \((K_t)^{-1}\otimes K_s^{-1}\)，likelihood precision 是 \(B_n\otimes G\)。

**English explanation**

1. \(B_{o\rightarrow n}\) is the transferred historical temporal statistic.
2. The current block contributes \(T_n^\top T_n/\sigma^2\).
3. Their sum gives the new temporal likelihood statistic.
4. The posterior u precision is prior precision plus likelihood precision.
5. This preserves the Sylvester-compatible structure.

---

## 15. Structured Beta-U Cross Block / 结构化 beta-u 交叉块

### Formula 15.1 Cross-block update

$$
R_{\beta u,o\rightarrow n}
=R_{\beta u,o}(L_{on}^{(t)}\otimes I_s),
$$

$$
R_{\beta u,n}
=
R_{\beta u,o}(L_{on}^{(t)}\otimes I_s)
+
\frac1{\sigma^2}\Phi_n^\top A_n.
$$

**中文逐行解释**

1. \(R_{\beta u,o}\) 是旧数据中 linear trend 和 GP inducing variables 的 coupling information。
2. 因为 \(u\) 的坐标改变了，旧 cross block 必须乘 \(L_{on}^{(t)}\otimes I_s\) 转到新坐标。
3. 当前 block 贡献新的 cross information \(\Phi^\top A/\sigma^2\)。
4. 二者相加得到当前的 \(R_{\beta u,n}\)。
5. 这个 block 是 precision cross block，不是 covariance cross block。

**English explanation**

1. \(R_{\beta u,o}\) stores historical coupling between beta and inducing variables.
2. It must be mapped into the new inducing coordinates.
3. The current block contributes new coupling information.
4. The updated cross block is the sum of transferred old coupling and new coupling.
5. This is a precision cross block, not the posterior covariance itself.

### Formula 15.2 Posterior cross covariance

$$
S_{\beta u}
=
-\Lambda_{\beta|u}^{-1}B_{\beta u}D_u^{-1}.
$$

**中文逐行解释**

1. \(R_{\beta u}\) 或 \(B_{\beta u}\) 是 posterior precision 里的交叉块。
2. 真正的 posterior covariance 交叉块是 precision matrix 取逆后得到的。
3. Schur 公式告诉我们它等于 \(-\Lambda_{\beta|u}^{-1}B_{\beta u}D_u^{-1}\)。
4. Route B 不需要把这个 dense matrix 完整存下来，只在预测方差里隐式使用。

**English explanation**

1. \(B_{\beta u}\) is a cross block in precision space.
2. The posterior cross covariance appears after inverting the full precision matrix.
3. The Schur identity gives its implicit form.
4. Route B uses it through solves and small matrix operations.

---

## 16. Sylvester Solves / Sylvester 求解

### Formula 16.1 Vector-to-matrix reshape

$$
q=\operatorname{vec}(Q),
\qquad
z=D_u^{-1}q=\operatorname{vec}(Z),
\qquad
Q,Z\in\mathbb R^{M_s\times M_t}.
$$

**中文逐行解释**

1. \(q\) 是一个长向量，长度是 \(M_tM_s\)。
2. 把它 reshape 成矩阵 \(Q\)，行对应空间 inducing index，列对应时间 inducing index。
3. 解 \(D_uz=q\) 后得到的向量 \(z\) 也 reshape 成矩阵 \(Z\)。
4. 这样 Kronecker 线性系统可以变成矩阵方程。

**English explanation**

1. The right-hand side \(q\) is a vector of length \(M_tM_s\).
2. It is reshaped into an \(M_s\times M_t\) matrix.
3. The solution vector is reshaped in the same way.
4. This converts a Kronecker linear system into a matrix equation.

### Formula 16.2 Sylvester equation

$$
D_u=(K_{nn}^{(t)})^{-1}\otimes K_s^{-1}+B_n\otimes G,
$$

$$
K_s^{-1}Z(K_{nn}^{(t)})^{-1}+GZB_n=Q.
$$

**中文逐行解释**

1. 第一行给出 \(D_u\) 的两个 Kronecker 项：prior precision 和 likelihood precision。
2. 使用恒等式 \((A\otimes B)\operatorname{vec}(Z)=\operatorname{vec}(BZA^\top)\)。
3. prior precision 项变成 \(K_s^{-1}Z(K_t)^{-1}\)。
4. likelihood precision 项变成 \(GZB_n\)。
5. 两项相加等于 \(Q\)，这就是 Sylvester-type matrix equation。

**English explanation**

1. \(D_u\) is a sum of a prior Kronecker term and a likelihood Kronecker term.
2. The vectorization identity converts each Kronecker product into left-right matrix multiplication.
3. The prior term becomes \(K_s^{-1}ZK_t^{-1}\).
4. The likelihood term becomes \(GZB_n\).
5. The original large linear system becomes a Sylvester-type equation.

---

## 17. Prediction and Uncertainty / 预测均值与不确定性

### Formula 17.1 Test-point features

$$
\phi_*:=\phi(t_*,s_*),
\qquad
a_*^\top:=K_{*u}K_{uu}^{-1}.
$$

**中文逐行解释**

1. \(\phi_*\) 是测试点的 linear feature vector。
2. \(a_*^\top\) 是 sparse GP 的测试点到 inducing variables 的条件均值投影。
3. 它和训练时的 \(A_n\) 是同类对象，只不过这里对应单个测试点。

**English explanation**

1. \(\phi_*\) is the linear feature vector for the test input.
2. \(a_*^\top\) is the sparse GP projection from inducing variables to the test point.
3. It is the single-point analogue of \(A_n\).

### Formula 17.2 Predictive mean

$$
\mathbb E[y_*]
=
\phi_*^\top m_\beta+a_*^\top m_u.
$$

**中文逐行解释**

1. 预测均值由 linear posterior mean 和 GP posterior mean 两部分相加。
2. \(\phi_*^\top m_\beta\) 是测试点的线性趋势预测。
3. \(a_*^\top m_u\) 是测试点的 GP residual 预测。
4. 这再次说明 Route B 不是 residual-only GP，而是 joint mean。

**English explanation**

1. The predictive mean is the sum of the linear mean and GP residual mean.
2. \(\phi_*^\top m_\beta\) is the linear trend prediction.
3. \(a_*^\top m_u\) is the GP residual prediction.
4. The prediction remains joint.

### Formula 17.3 Sparse conditional residual variance

$$
\nu_*:=k(x_*,x_*)-K_{*u}K_{uu}^{-1}K_{u*}.
$$

**中文逐行解释**

1. \(k(x_*,x_*)\) 是 GP prior 在测试点自己的方差。
2. \(K_{*u}K_{uu}^{-1}K_{u*}\) 是 inducing variables 能解释掉的那部分方差。
3. 两者相减得到 sparse GP 条件下仍然无法由 inducing variables 表示的 residual variance。
4. 如果 kernel amplitude 不是 1，这里的 \(k(x_*,x_*)\) 必须显式包含该 amplitude。

**English explanation**

1. \(k(x_*,x_*)\) is the prior variance at the test point.
2. \(K_{*u}K_{uu}^{-1}K_{u*}\) is the variance explained by inducing variables.
3. The difference is the residual conditional variance of the sparse GP approximation.
4. Non-unit kernel amplitude must be included here.

### Formula 17.4 Structured predictive quadratic

$$
D_uv_*=a_*,
$$

$$
\begin{bmatrix}\phi_*\\a_*\end{bmatrix}^\top
\Lambda^{-1}
\begin{bmatrix}\phi_*\\a_*\end{bmatrix}
=
a_*^\top v_*
+
\left(\phi_*-B_{\beta u}v_*\right)^\top
\Lambda_{\beta|u}^{-1}
\left(\phi_*-B_{\beta u}v_*\right).
$$

**中文逐行解释**

1. 第一行先解 \(v_*=D_u^{-1}a_*\)，仍然不显式求 \(D_u^{-1}\)。
2. 左边是 posterior parameter uncertainty 对测试输出的贡献。
3. 第一项 \(a_*^\top v_*\) 是 GP inducing uncertainty 的基础贡献。
4. \(\phi_*-B_{\beta u}v_*\) 是在考虑 beta-u precision coupling 后的有效 linear feature。
5. 中间乘 \(\Lambda_{\beta|u}^{-1}\) 表示 linear block 的 Schur covariance。
6. 这一整行把完整 covariance 二次型转成了一个 Sylvester solve 加一个小矩阵二次型。

**English explanation**

1. First solve \(v_*=D_u^{-1}a_*\) without forming \(D_u^{-1}\).
2. The left side is the posterior parameter uncertainty contribution.
3. \(a_*^\top v_*\) is the base inducing uncertainty contribution.
4. \(\phi_*-B_{\beta u}v_*\) is the effective beta feature after accounting for coupling.
5. \(\Lambda_{\beta|u}^{-1}\) is the Schur beta covariance.
6. The formula evaluates a full covariance quadratic using structured solves.

### Formula 17.5 Predictive variance

$$
\operatorname{Var}(y_*)
=
\sigma^2+\nu_*+a_*^\top v_*
+
\left(\phi_*-B_{\beta u}v_*\right)^\top
\Lambda_{\beta|u}^{-1}
\left(\phi_*-B_{\beta u}v_*\right).
$$

**中文逐行解释**

1. \(\sigma^2\) 是 observation noise。如果评估 noisy observation \(y_*\)，必须加它。
2. \(\nu_*\) 是 sparse GP 条件 residual variance。
3. \(a_*^\top v_*\) 是 inducing posterior uncertainty。
4. 最后一项是 linear uncertainty 加 beta-u coupling 修正后的贡献。
5. 这就是 Route B 的预测方差公式。

**English explanation**

1. \(\sigma^2\) is observation noise and is included for noisy predictive observations.
2. \(\nu_*\) is sparse conditional residual variance.
3. \(a_*^\top v_*\) is inducing posterior uncertainty.
4. The last term is the Schur beta uncertainty with beta-u coupling.
5. This is the Route B predictive variance.

### Formula 17.6 Mean-field comparison

$$
\operatorname{Var}_{MF}(y_*)
=
\sigma^2+\nu_*+
\phi_*^\top S_{\beta,MF}\phi_*
+
a_*^\top S_{u,MF}a_*.
$$

**中文逐行解释**

1. mean-field 把 \(\beta\) 和 \(u\) 当作后验独立。
2. 所以方差只剩 linear variance 加 GP variance。
3. 它没有 \(2\phi_*^\top S_{\beta u}a_*\) 这样的 cross covariance 项。
4. 因此 mean-field 是 calibration ablation，而不是 Route B 的完整理论版本。

**English explanation**

1. Mean-field assumes posterior independence between \(\beta\) and \(u\).
2. The variance is just the sum of linear uncertainty and GP uncertainty.
3. It drops the beta-u cross-covariance term.
4. It is a useful ablation but not the full structured Route B predictive distribution.

---

## 18. Main Logical Chain / 主理论链条

### Formula 18.1 Summary implication chain

$$
\begin{aligned}
&\text{two-stage learning has stale residuals}\\
\Rightarrow\;&\text{joint training fixes the observation target}\\
\Rightarrow\;&\text{changing HiPPO bases require old-information transfer}\\
\Rightarrow\;&\text{SSGP-style old-likelihood-ratio transfer gives a principled streaming update}\\
\Rightarrow\;&\text{Route B keeps structured }\beta\text{--}u\text{ covariance}\\
\Rightarrow\;&\text{Kronecker/Sylvester structure keeps the method scalable.}
\end{aligned}
$$

**中文逐行解释**

1. 两阶段 residual 方法的问题是 residual 会随着 \(\beta\) 改变而过期。
2. joint training 直接对 \(y\) 建模，所以没有旧 residual target。
3. HiPPO basis 随时间窗口变化，所以旧 inducing 坐标不能直接复用。
4. old-likelihood-ratio transfer 是 principled streaming update。
5. Route B 进一步保留 \(\beta-u\) cross precision/covariance。
6. Kronecker 和 Sylvester 保证这个联合模型仍然可扩展。

**English explanation**

1. Two-stage residual learning suffers from stale residuals.
2. Joint training models raw observations and fixes the target.
3. Changing HiPPO bases require transferring old information across coordinates.
4. Old-likelihood-ratio transfer gives a principled streaming update.
5. Route B retains beta-u structured covariance.
6. Kronecker/Sylvester algebra keeps the method scalable.

---

## 19. Appendix Proof: \(L_{on}=L_{on}^{(t)}\otimes I_s\)

### Formula 19.1 Old-new covariance factorization

$$
\begin{aligned}
\operatorname{cov}(u_{\ell,j}^{old},u_{m,j'}^{new})
&=
\int_0^{T_o}\int_0^{T_n}
g_\ell^{(T_o)}(x)g_m^{(T_n)}(x')
k((x,z_j),(x',z_{j'}))\,dx'\,dx\\
&=
\left[
\int_0^{T_o}\int_0^{T_n}
g_\ell^{(T_o)}(x)g_m^{(T_n)}(x')k_t(x,x')\,dx'\,dx
\right]
k_s(z_j,z_{j'}).
\end{aligned}
$$

**中文逐行解释**

1. 第一行从定义出发：old inducing variable 和 new inducing variable 都是对 \(f\) 的积分，所以 covariance 是双重积分。
2. 积分内部的 covariance 是 GP kernel \(k((x,z_j),(x',z_{j'}))\)。
3. 使用 separable kernel 后，kernel 分成 \(k_t(x,x')k_s(z_j,z_{j'})\)。
4. 空间项和积分变量无关，可以提出积分外。
5. 剩下的积分只依赖时间 basis 和时间核，所以得到 temporal factor 乘 spatial factor。

**English explanation**

1. The covariance between old and new inducing variables is a double integral of the GP kernel.
2. The kernel inside the integral is the covariance of the latent function values.
3. Separability splits it into temporal and spatial factors.
4. The spatial factor is constant with respect to the time integrals.
5. Thus old-new covariance factorizes into temporal covariance times spatial covariance.

### Formula 19.2 Conditional operator proof

$$
K_{on}=K_{on}^{(t)}\otimes K_s,
\qquad
K_{nn}=K_{nn}^{(t)}\otimes K_s.
$$

$$
\begin{aligned}
L_{on}
&=K_{on}K_{nn}^{-1}\\
&=(K_{on}^{(t)}\otimes K_s)
\left[(K_{nn}^{(t)})^{-1}\otimes K_s^{-1}\right]\\
&=K_{on}^{(t)}(K_{nn}^{(t)})^{-1}\otimes K_sK_s^{-1}\\
&=L_{on}^{(t)}\otimes I_s.
\end{aligned}
$$

**中文逐行解释**

1. 前两项说明 old-new 和 new-new inducing covariance 都有同一个空间因子 \(K_s\)。
2. \(L_{on}\) 是条件均值 operator 的定义。
3. Kronecker inverse identity 给出 \((A\otimes B)^{-1}=A^{-1}\otimes B^{-1}\)。
4. Kronecker product multiplication identity 给出 \((A\otimes B)(C\otimes D)=AC\otimes BD\)。
5. 空间部分 \(K_sK_s^{-1}\) 消掉成 \(I_s\)。
6. 所以 transfer 只在时间维度上发生。

**English explanation**

1. Both covariance matrices share the same spatial factor.
2. The conditional operator is \(K_{on}K_{nn}^{-1}\).
3. Use the Kronecker inverse identity.
4. Use the Kronecker multiplication identity.
5. The spatial covariance cancels to identity.
6. Therefore the transfer is temporal-only.

---

## 20. Appendix Proof: Why \(R_{uu,o}=B_o\otimes G\)

### Formula 20.1 Likelihood precision from one block

$$
\Lambda_{\mathrm{like},j}
=
\frac1{\sigma^2}A_j^\top A_j,
\qquad
A_j=T_j\otimes C.
$$

$$
\Lambda_{\mathrm{like},j}
=
\frac1{\sigma^2}(T_j^\top T_j)\otimes(C^\top C).
$$

**中文逐行解释**

1. 对 Gaussian likelihood，一个 block 对 \(u\)-precision 的贡献是 \(A_j^\top A_j/\sigma^2\)。
2. 因为 \(A_j=T_j\otimes C\)，代入后使用 Kronecker 转置和乘法规则。
3. 时间部分变成 \(T_j^\top T_j\)，空间部分变成 \(C^\top C\)。
4. 所以每个 block 的 likelihood precision 都共享同一个空间因子 \(G=C^\top C\)。

**English explanation**

1. A Gaussian likelihood contributes \(A_j^\top A_j/\sigma^2\) to inducing precision.
2. Substitute \(A_j=T_j\otimes C\).
3. Kronecker algebra separates temporal and spatial factors.
4. Every block shares the same spatial factor \(G=C^\top C\).

### Formula 20.2 Historical sum

$$
R_{uu,o}
:=
\sum_{j<n}\Lambda_{\mathrm{like},j}
=
\left[
\sum_{j<n}\frac1{\sigma^2}T_j^\top T_j
\right]\otimes G.
$$

$$
B_o:=
\sum_{j<n}\frac1{\sigma^2}T_j^\top T_j,
\qquad
R_{uu,o}=B_o\otimes G.
$$

**中文逐行解释**

1. \(R_{uu,o}\) 是所有历史 block 对 \(u\)-precision 的 likelihood contribution 之和。
2. 因为每项都有同一个空间因子 \(G\)，可以把 \(G\) 提出来。
3. 方括号里的历史时间统计定义为 \(B_o\)。
4. 于是旧 \(u-u\) likelihood precision 保持为 \(B_o\otimes G\)。
5. 这不是任意 posterior covariance 的性质，而是算法在 natural-parameter form 中维护的 invariant。

**English explanation**

1. \(R_{uu,o}\) is the sum of historical likelihood precision contributions.
2. Since every term shares \(G\), the spatial factor can be factored out.
3. The temporal sum is called \(B_o\).
4. Hence \(R_{uu,o}=B_o\otimes G\).
5. This is an algorithmic natural-parameter invariant, not a property of arbitrary dense covariance.

---

## 21. Appendix: Explicit Sylvester Solver / 显式 Sylvester 求解器

### Formula 21.1 General and symmetric Sylvester forms

$$
K_s^{-1}ZK_t^{-T}+GZB_n^\top=Q.
$$

$$
K_s^{-1}ZK_t^{-1}+GZB_n=Q
\quad\text{when matrices are symmetric.}
$$

**中文逐行解释**

1. 第一行是一般情况，右乘的是 \(K_t^{-T}\) 和 \(B_n^\top\)。
2. 如果 \(K_t\) 和 \(B_n\) 对称，那么转置可以去掉。
3. 实际实现中通常使用对称化和 jitter，目标就是稳定地使用第二种形式。

**English explanation**

1. The first equation is the general matrix equation.
2. Symmetry removes the transposes.
3. The implementation uses symmetrization and jitter to make this stable.

### Formula 21.2 Whitening

$$
\widetilde Z=K_s^{-1/2}ZK_t^{-1/2},
\qquad
\widetilde Q=K_s^{1/2}QK_t^{1/2},
$$

$$
\widetilde G=K_s^{1/2}GK_s^{1/2},
\qquad
\widetilde B=K_t^{1/2}B_nK_t^{1/2}.
$$

$$
\widetilde Z+\widetilde G\widetilde Z\widetilde B=\widetilde Q.
$$

**中文逐行解释**

1. whitening 的目的是把 prior precision 部分变成 identity。
2. \(\widetilde Z\) 是把解 \(Z\) 同时在空间和时间方向白化。
3. \(\widetilde Q\) 是对应右端项的变换。
4. \(\widetilde G\) 和 \(\widetilde B\) 是白化后的 likelihood factors。
5. 变换后方程变成 \(\widetilde Z+\widetilde G\widetilde Z\widetilde B=\widetilde Q\)，比原式更容易 diagonalize。

**English explanation**

1. Whitening turns the prior precision part into an identity term.
2. \(\widetilde Z\) is the whitened solution matrix.
3. \(\widetilde Q\) is the transformed right-hand side.
4. \(\widetilde G\) and \(\widetilde B\) are whitened likelihood factors.
5. The resulting equation is easier to diagonalize.

### Formula 21.3 Diagonal solution

$$
\widetilde G=U_s\Gamma U_s^\top,
\qquad
\widetilde B=U_tHU_t^\top.
$$

$$
\widehat Z=U_s^\top\widetilde ZU_t,
\qquad
\widehat Q=U_s^\top\widetilde QU_t.
$$

$$
(1+\gamma_i\eta_j)\widehat Z_{ij}=\widehat Q_{ij},
\qquad
\widehat Z_{ij}=\frac{\widehat Q_{ij}}{1+\gamma_i\eta_j}.
$$

**中文逐行解释**

1. 先对白化后的空间因子 \(\widetilde G\) 和时间因子 \(\widetilde B\) 做特征分解。
2. 用 \(U_s\) 和 \(U_t\) 把未知量和右端项旋转到 eigenbasis。
3. 在这个坐标系中，\(\Gamma\) 和 \(H\) 都是对角矩阵。
4. 原本的矩阵方程变成每个 entry 独立的标量方程。
5. 每个 \(\widehat Z_{ij}\) 只需要除以 \(1+\gamma_i\eta_j\)。

**English explanation**

1. Eigendecompose the whitened spatial and temporal likelihood factors.
2. Rotate the solution and right-hand side into the eigenbasis.
3. Both factors become diagonal.
4. The matrix equation decouples entry by entry.
5. Each entry is solved by a scalar division.

### Formula 21.4 Fast quadratic form

$$
q^\top D_u^{-1}q
=
\sum_{i=1}^{M_s}\sum_{j=1}^{M_t}
\frac{\widehat Q_{ij}^2}{1+\gamma_i\eta_j}.
$$

**中文逐行解释**

1. 预测方差经常只需要标量 \(q^\top D_u^{-1}q\)，不一定需要完整解向量。
2. 在 diagonalized coordinates 中，解是 \(\widehat Q_{ij}/(1+\gamma_i\eta_j)\)。
3. inner product \(q^\top z\) 变成每个 entry 的 \(\widehat Q_{ij}\widehat Z_{ij}\) 之和。
4. 代入 \(\widehat Z_{ij}\) 后得到这个快速求和公式。
5. 这样可以更快地算 predictive uncertainty。

**English explanation**

1. Prediction often needs only the scalar quadratic form.
2. In diagonal coordinates, the solution entry is \(\widehat Q_{ij}/(1+\gamma_i\eta_j)\).
3. The inner product is the sum of \(\widehat Q_{ij}\widehat Z_{ij}\).
4. Substitution gives the fast quadratic formula.
5. This avoids reconstructing the full solution when only uncertainty is needed.

### Formula 21.5 Complexity

$$
\mathcal O(M_t^3+M_s^3+M_t^2M_s+M_tM_s^2).
$$

**中文逐行解释**

1. \(M_t^3\) 来自时间方向的矩阵分解。
2. \(M_s^3\) 来自空间方向的矩阵分解。
3. \(M_t^2M_s\) 和 \(M_tM_s^2\) 来自左右两侧 basis transform。
4. 相比 dense solve 的 \(\mathcal O((M_tM_s)^3)\)，这个复杂度利用了 Kronecker/Sylvester 结构。
5. 如果空间 inducing geometry 固定，空间分解还可以缓存。

**English explanation**

1. \(M_t^3\) comes from temporal eigendecomposition.
2. \(M_s^3\) comes from spatial eigendecomposition.
3. The mixed terms come from two-sided matrix transforms.
4. This is much cheaper than a dense solve over \(M_tM_s\) variables.
5. Fixed spatial geometry allows caching spatial factors.

---

## 22. When the Single-Kronecker Assumption Can Fail / 单 Kronecker 假设何时会失败

### Formula 22.1 Changing spatial pattern

$$
C_j\neq C,
\qquad
G_j=C_j^\top C_j,
\qquad
R_{uu,o}=\sum_{j<n}B_j\otimes G_j.
$$

**中文逐行解释**

1. 如果每个 block 观察到的空间位置不同，空间 projection matrix 可能变成 \(C_j\)。
2. 那么 \(G_j=C_j^\top C_j\) 也会随 block 改变。
3. 历史 precision 就不能再写成单个 \(B_o\otimes G\)。
4. 它变成多个 Kronecker products 的和。

**English explanation**

1. If spatial observation patterns change, the spatial projection can become block-dependent.
2. Then the spatial Gram factor also changes.
3. The historical precision is no longer a single Kronecker product.
4. It becomes a sum of Kronecker products.

### Formula 22.2 Sum-of-Kronecker extension

$$
R_{uu,o}=\sum_{r=1}^{R_o}B_r\otimes G_r.
$$

$$
D_u=K_t^{-1}\otimes K_s^{-1}+\sum_{r=1}^R B_r\otimes G_r.
$$

$$
K_s^{-1}MK_t^{-1}+\sum_{r=1}^RG_rMB_r=H.
$$

**中文逐行解释**

1. 第一行是更一般的历史 likelihood precision 表示。
2. 第二行把这个更一般的 likelihood precision 加到 prior precision 上。
3. 第三行是对应的矩阵方程。
4. 它不再是简单的单项 Sylvester equation，因此需要 generalized Sylvester solver 或 conjugate gradient。
5. 这是未来扩展，不是当前主 Route B 方法。

**English explanation**

1. The first equation is a more general historical likelihood representation.
2. The second equation adds it to the prior precision.
3. The third equation is the corresponding matrix equation.
4. It is no longer the simple single-term Sylvester equation.
5. This is a future extension rather than the current main Route B implementation.

---

## 23. One-Slide Summary for Supervisor / 给导师汇报的一页总结

**中文版本**

Route B 的核心是：不再用 \(y-\Phi\hat\beta\) 这种会过期的 residual 作为 GP 目标，而是直接对 \(y=\Phi\beta+A u+\epsilon\) 做联合高斯更新。由于 HiPPO temporal basis 会随 online horizon 改变，旧 posterior 不能直接复用，所以我们转移的是旧似然比 \(q_{n-1}/p\)，并通过 \(p(u_o\mid u_n)\) 把旧坐标映射到新坐标。与 mean-field 不同，Route B 保留 \(\beta-u\) precision cross block，因此预测方差中隐式包含 \(S_{\beta u}\) 的影响。为了可扩展，\(u-u\) precision 维护为 \(B_n\otimes G\) 加 Kronecker prior，从而所有 \(D_u^{-1}\) 操作都可以用 Sylvester solve 完成。

**English version**

Route B replaces stale residual learning with a joint Gaussian likelihood for \(y=\Phi\beta+Au+\epsilon\). Because the HiPPO temporal basis changes online, the old posterior cannot be reused directly; instead, Route B transfers the old likelihood ratio \(q_{n-1}/p\) through the GP conditional \(p(u_o\mid u_n)\). Unlike mean-field, it retains the beta-u precision cross block, so predictive uncertainty implicitly includes the beta-u posterior covariance. Scalability comes from preserving the u-u precision as a Kronecker/Sylvester structure, enabling all \(D_u^{-1}\) operations through structured solves.

---

## 24. Suggested Explanation Order / 建议汇报顺序

1. **Start with the failure mode**: residual targets become stale when \(\beta\) changes.
2. **Introduce the fix**: use the joint likelihood \(y=\Phi\beta+Au+\epsilon\).
3. **Explain changing basis**: old and new inducing variables are different random variables.
4. **Explain old-likelihood transfer**: transfer \(q_{old}/p_{old}\), not posterior moments.
5. **Explain Route B's key difference**: keep \(R_{\beta u}\) and recover \(S_{\beta u}\) through Schur complement.
6. **Explain scalability**: \(A=T\otimes C\), \(R_{uu}=B\otimes G\), and \(D_u^{-1}\) is solved by Sylvester equations.
7. **Explain uncertainty**: predictive variance contains noise, sparse residual variance, inducing uncertainty, and beta-u coupling correction.

