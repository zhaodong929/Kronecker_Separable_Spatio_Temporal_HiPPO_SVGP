---
title: "Route B Structured Joint SSGP: Formula-to-Code Walkthrough"
subtitle: "English version aligned with the Route B theory and implementation"
author: "Generated from project source"
date: "2026-06-01"
toc: true
toc-depth: 3
mainfont: "Times New Roman"
monofont: "Consolas"
geometry: margin=0.75in
colorlinks: true
header-includes:
  - \usepackage{amsmath}
  - \newcommand{\vecop}{\operatorname{vec}}
---

# 0. Project Goal

This project implements an online continual-learning spatio-temporal Gaussian process model with a linear mean:

$$
y(t,s)=\phi(t,s)^\top\beta+f(t,s)+\epsilon.
$$

The key Route B idea is to learn the linear coefficient \(\beta\) and the GP inducing variables \(u\) jointly. The method does not first fit a linear model and then train a GP on fixed residuals. Instead, it keeps a structured joint posterior over \((\beta,u)\), including the beta-u posterior coupling that a mean-field approximation would remove.

Main implementation directory:

```text
stvgp_kronecker/joint_ssgp_kron/
├── model.py             # Main Route B model, online updates, prediction
├── structured_state.py  # Online state and natural statistics
├── ssgp_transfer.py     # Old-likelihood changing-basis transfer
├── kron_utils.py        # Kronecker, Sylvester, and Schur utilities
└── synthetic.py         # Synthetic data and block-factor construction
```

Main experiment and verification scripts:

```text
scripts/run_joint_ssgp_kron_experiments.py
scripts/verify_joint_ssgp_kron_derivations.py
scripts/generate_routeB_experiment_report.py
```

# 1. Overall Call Flow

A synthetic Route B experiment follows this high-level flow:

```text
main()
  -> parse_args()
  -> run_all_requested()
      -> make_dataset()
      -> fit_model_hyperparameters_from_initial_task()  # optional full-GP MLL
      -> run_structured_method()
          -> split the time axis into online blocks
          -> build Phi, T_n, Kt_new, K_on_t for the current block
          -> model.update_block_*()
          -> evaluate_state_on_factors()
              -> predictive variance decomposition
              -> RMSE / NLL / coverage
```

The main Route B path is:

```text
run_structured_method()
  -> JointSSGPKronHiPPOSVGP.update_block_structured_joint_ssgp_transfer()
      -> _routeB_transfer_old_stats()
      -> joint_likelihood_stats()
      -> recover_posterior_mean_structured()
          -> schur_recover_posterior()
              -> solve_Du_sylvester()
```

## 1.1 Online Route B Algorithm

| Stage | Algorithm step | Main code location |
|---:|---|---|
| Input | Data stream \(D_1,\ldots,D_N\), priors \(p(\beta)\), \(p(u)\), kernel parameters, inducing sizes \(M_t,M_s\) | `parse_args`, `make_dataset` |
| 1 | Initialize `state = None` because no old likelihood information exists yet | `run_structured_method` |
| 2 | Loop over online blocks \(n=1,\ldots,N\) | `for block_id, block in enumerate(blocks)` |
| 3 | Build current block factors \(y_n,\Phi_n,T_n,K_{t,new},K_{on}^{(t)}\) | `make_indexed_block_factors` |
| 4 | Transfer old likelihood statistics to the new temporal basis | `_routeB_transfer_old_stats` |
| 5 | Compute current block Gaussian likelihood natural statistics | `joint_likelihood_stats` |
| 6 | Add transferred old statistics and current likelihood statistics | `R_beta_beta`, `R_beta_u`, `B_temporal` |
| 7 | Add the beta prior to form the beta precision block | `A_beta = beta_prior_precision + R_beta_beta` |
| 8 | Solve \(D_u^{-1}B_{\beta u}^\top\) and \(D_u^{-1}h_u\) | `solve_Du_sylvester` |
| 9 | Recover \(m_\beta,S_{\beta\beta},m_u\) by Schur complement | `schur_recover_posterior` |
| 10 | Store posterior means and natural statistics in a new online state | `StructuredKronState(...)` |
| 11 | Evaluate current, seen-history, and future modes | `evaluate_state_on_factors` |

The central operations are old-statistic transfer, new-block likelihood assimilation, and Schur/Sylvester posterior recovery.

# 2. Formula Block 1: Joint Observation Model

Theory:

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

For one online block:

$$
y_n = \Phi_n\beta + f_n+\epsilon_n,
\qquad
\epsilon_n\sim\mathcal{N}(0,\sigma^2 I).
$$

## 2.1 Code: block data entering the model

File: `scripts/run_joint_ssgp_kron_experiments.py`

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

Important fields:

```python
factors.y_vec  # y_n
factors.Phi    # Phi_n
factors.T      # temporal projection T_n
factors.Kt     # Kt_new
```

Route B update:

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

Implementation mapping:

| Theory symbol | Code object |
|---|---|
| \(y_n\) | `factors.y_vec` |
| \(\Phi_n\) | `factors.Phi` |
| \(T_n\) | `factors.T` |
| \(K_{nn}^{(t)}\) | `factors.Kt` |
| \(K_{on}^{(t)}\) | `factors.K_on_t` |
| previous posterior/natural statistics | `state` |

## 2.2 Code: model initialization

File: `stvgp_kronecker/joint_ssgp_kron/model.py`

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

Implementation mapping:

| Theory symbol | Code object |
|---|---|
| \(\sigma^2\) | `sigma2` |
| \(m_{\beta,0}\) | `beta_prior_mean` |
| \(K_\beta\) | `beta_prior_cov` |
| \(K_s\) | `Ks` |
| \(C\) | `C` |
| \(G=C^\top C\) | `self.G = symmetrize(self.C.T @ self.C)` |
| \(K_s^{-1}\) | `self.Ks_inv` |

# 3. Formula Block 2: Separable Kernel and Kronecker Representation

Theory:

$$
k((s,t),(s',t'))=k_s(s,s')k_t(t,t').
$$

On a full grid:

$$
K_{ff}=K_t\otimes K_s.
$$

For sparse inducing projection:

$$
A_n=T_n\otimes C,
\qquad
A_n^\top A_n=(T_n^\top T_n)\otimes(C^\top C).
$$

## 3.1 Code: Kronecker matrix-vector multiplication

File: `stvgp_kronecker/joint_ssgp_kron/kron_utils.py`

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

Formula-code mapping:

$$
(T\otimes C)\operatorname{vec}(U)=\operatorname{vec}(CUT^\top).
$$

| Theory symbol | Code object |
|---|---|
| \(T_n\) | `T` |
| \(C\) | `C` |
| \(u=\operatorname{vec}(U)\) | `vec_f(U)` |
| \(A_nu=(T_n\otimes C)u\) | `kron_mv(T_n, C, U)` |

## 3.2 Vectorization convention

The project uses Fortran-style vectorization:

```python
def vec_f(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix).reshape(-1, order="F")
```

This convention must remain consistent across `kron_mv`, transfer, Schur recovery, and prediction.

# 4. Formula Block 3: Online State

Route B maintains likelihood natural statistics rather than storing all previous raw data.

The important state components are:

$$
R_{\beta\beta},\quad R_{\beta u},\quad B,\quad h_\beta,\quad H_u.
$$

## 4.1 Code: state object

File: `stvgp_kronecker/joint_ssgp_kron/structured_state.py`

```python
@dataclass
class StructuredKronState:
    beta_mean: np.ndarray
    beta_cov: np.ndarray
    M_u: np.ndarray
    R_beta_beta: np.ndarray
    R_beta_u: np.ndarray
    B_temporal: np.ndarray
    h_beta: np.ndarray
    H_info: np.ndarray
    Kt: np.ndarray
    z_t: np.ndarray
```

State meaning:

| Field | Meaning |
|---|---|
| `beta_mean` | posterior mean of \(\beta\) |
| `beta_cov` | posterior covariance block \(S_{\beta\beta}\) |
| `M_u` | matrix form of inducing posterior mean \(u\) |
| `R_beta_beta` | accumulated likelihood precision for beta |
| `R_beta_u` | accumulated beta-u likelihood precision cross block |
| `B_temporal` | temporal likelihood statistic for the \(u-u\) block |
| `h_beta` | beta information vector |
| `H_info` | matrix form of the inducing information vector |
| `Kt` | current temporal inducing prior covariance |
| `z_t` | current temporal inducing coordinates |

# 5. Formula Block 4: Changing-Basis Old-Likelihood Transfer

Because the temporal HiPPO basis changes with the online horizon, old and new inducing variables are different:

$$
u_\ell^{old}=\int_0^{T_{n-1}}g_\ell^{(T_{n-1})}(x)f(x)\,dx,
\qquad
u_\ell^{new}=\int_0^{T_n}g_\ell^{(T_n)}(x)f(x)\,dx.
$$

The GP conditional mean operator is:

$$
L_{on}=K_{on}K_{nn}^{-1}.
$$

Under fixed spatial inducing locations:

$$
L_{on}=L_{on}^{(t)}\otimes I_s.
$$

## 5.1 Code: temporal transfer matrix

File: `stvgp_kronecker/joint_ssgp_kron/ssgp_transfer.py`

```python
def temporal_transfer_matrix(K_on_t: np.ndarray, Kt_new: np.ndarray) -> np.ndarray:
    return solve_spd(Kt_new, K_on_t.T).T
```

This implements:

$$
L_{on}^{(t)} = K_{on}^{(t)}(K_{nn}^{(t)})^{-1}.
$$

## 5.2 Code: transfer Route B old likelihood statistics

The old likelihood natural parameters transfer as:

$$
\begin{aligned}
R_{\beta\beta,o\to n} &= R_{\beta\beta,o},\\
R_{\beta u,o\to n} &= R_{\beta u,o}L_{on},\\
R_{uu,o\to n} &= L_{on}^\top R_{uu,o}L_{on},\\
r_{\beta,o\to n} &= r_{\beta,o},\\
r_{u,o\to n} &= L_{on}^\top r_{u,o}.
\end{aligned}
$$

Structured code path:

```python
L_t = temporal_transfer_matrix(K_on_t, Kt_new)
B_old_trans = L_t.T @ state.B_temporal @ L_t
R_beta_u_old_trans = transfer_cross_block_temporal(state.R_beta_u, L_t, ms)
H_old_trans = transfer_information_matrix(state.H_info, L_t)
```

Implementation mapping:

| Formula | Code |
|---|---|
| \(B_{o\to n}=L_t^\top B_oL_t\) | `B_old_trans = L_t.T @ state.B_temporal @ L_t` |
| \(R_{\beta u,o\to n}=R_{\beta u,o}(L_t\otimes I_s)\) | `transfer_cross_block_temporal(...)` |
| \(H_{o\to n}=H_oL_t\) | `transfer_information_matrix(state.H_info, L_t)` |

# 6. Formula Block 5: New Block Gaussian Likelihood Natural Statistics

Current block likelihood:

$$
p(y_n\mid\beta,u_n)
=
\mathcal N(y_n\mid \Phi_n\beta + A_nu_n,\sigma^2 I).
$$

With \(A_n=T_n\otimes C\), the likelihood natural statistics are:

$$
\begin{aligned}
R_{\beta\beta,new} &= \sigma^{-2}\Phi_n^\top\Phi_n,\\
R_{\beta u,new} &= \sigma^{-2}\Phi_n^\top A_n,\\
R_{uu,new} &= \sigma^{-2}A_n^\top A_n,\\
r_{\beta,new} &= \sigma^{-2}\Phi_n^\top y_n,\\
r_{u,new} &= \sigma^{-2}A_n^\top y_n.
\end{aligned}
$$

## 6.1 Code: `joint_likelihood_stats`

File: `stvgp_kronecker/joint_ssgp_kron/structured_state.py`

```python
stats = joint_likelihood_stats(
    y_vec=y_vec,
    Phi=Phi,
    T=T_n,
    C=C,
    sigma2=sigma2,
)
```

Typical returned fields:

```python
stats.R_beta_beta
stats.R_beta_u
stats.B_temporal
stats.h_beta
stats.H_info
```

Mapping:

| Formula term | Code field |
|---|---|
| \(R_{\beta\beta,new}\) | `stats.R_beta_beta` |
| \(R_{\beta u,new}\) | `stats.R_beta_u` |
| temporal factor of \(R_{uu,new}\) | `stats.B_temporal` |
| \(r_{\beta,new}\) | `stats.h_beta` |
| matrix form of \(r_{u,new}\) | `stats.H_info` |

# 7. Formula Block 6: Accumulated Natural Parameters

Route B accumulates transferred old likelihood statistics and new likelihood statistics:

$$
\begin{aligned}
R_{\beta\beta,n}
&=
R_{\beta\beta,o\to n}
+
\sigma^{-2}\Phi_n^\top\Phi_n,\\
R_{\beta u,n}
&=
R_{\beta u,o\to n}
+
\sigma^{-2}\Phi_n^\top A_n,\\
B_n
&=
B_{o\to n}
+
\sigma^{-2}T_n^\top T_n.
\end{aligned}
$$

## 7.1 Code: main Route B update

File: `stvgp_kronecker/joint_ssgp_kron/model.py`

```python
if state is None:
    R_beta_beta_old = np.zeros((p, p))
    R_beta_u_old = np.zeros((p, mt * ms))
    B_old = np.zeros((mt, mt))
    h_beta_old = np.zeros(p)
    H_old = np.zeros((ms, mt))
else:
    transferred = self._routeB_transfer_old_stats(state, K_on_t, Kt_new)

new_stats = joint_likelihood_stats(y_vec, Phi, T_n, self.C, self.sigma2)

R_beta_beta = R_beta_beta_old + new_stats.R_beta_beta
R_beta_u = R_beta_u_old + new_stats.R_beta_u
B_temporal = B_old + new_stats.B_temporal
h_beta = h_beta_old + new_stats.h_beta
H_info = H_old + new_stats.H_info
```

The update is an additive natural-parameter update. This is why old statistics and current likelihood statistics can be combined without replaying all historical data.

# 8. Formula Block 7: Add Beta Prior and Build Joint Precision

After likelihood accumulation:

$$
\begin{aligned}
\Lambda_{\beta\beta,n} &= P_\beta + R_{\beta\beta,n},\\
\Lambda_{\beta u,n} &= R_{\beta u,n},\\
\Lambda_{uu,n} &= K_{nn}^{-1}+R_{uu,n},\\
h_{\beta,n} &= P_\beta m_{\beta,0}+r_{\beta,n},\\
h_{u,n} &= r_{u,n}.
\end{aligned}
$$

## 8.1 Code mapping

```python
P_beta = inv_spd(self.beta_prior_cov, jitter=self.jitter)
A_beta = P_beta + R_beta_beta
h_beta_total = P_beta @ self.beta_prior_mean + h_beta
```

The beta prior is static in the clean Route B derivation. The old posterior over beta is not directly used as the next beta prior; historical beta information is already represented in the old likelihood natural statistics.

# 9. Formula Block 8: Schur Complement Posterior Recovery

The joint precision is:

$$
\Lambda=
\begin{bmatrix}
A_\beta & B_{\beta u}\\
B_{\beta u}^\top & D_u
\end{bmatrix}.
$$

Schur complement:

$$
\Lambda_{\beta|u}=A_\beta-B_{\beta u}D_u^{-1}B_{\beta u}^\top.
$$

Posterior means:

$$
m_\beta
=
\Lambda_{\beta|u}^{-1}
\left(h_\beta-B_{\beta u}D_u^{-1}h_u\right),
\qquad
m_u=D_u^{-1}h_u-D_u^{-1}B_{\beta u}^\top m_\beta.
$$

## 9.1 Code: `schur_recover_posterior`

File: `stvgp_kronecker/joint_ssgp_kron/kron_utils.py`

```python
schur = schur_recover_posterior(
    A_beta=A_beta,
    B_beta_u=R_beta_u,
    h_beta=h_beta_total,
    h_u=h_u,
    solve_Du=solve_Du_sylvester,
)
```

Variable mapping:

| Formula | Code |
|---|---|
| \(A_\beta\) | `A_beta` |
| \(B_{\beta u}\) | `R_beta_u` |
| \(h_\beta\) | `h_beta_total` |
| \(h_u\) | `h_u` |
| \(D_u^{-1}(\cdot)\) | `solve_Du_sylvester(...)` |
| \(m_\beta\) | `schur.m_beta` |
| \(S_{\beta\beta}\) | `schur.S_beta_beta` |
| \(m_u\) | `schur.m_u` |

# 10. Formula Block 9: Sylvester Solve Instead of Dense Inversion

The \(u\)-block precision has the form:

$$
D_u
=
K_t^{-1}\otimes K_s^{-1}
+
B_n\otimes G.
$$

For \(q=\operatorname{vec}(Q)\) and \(z=D_u^{-1}q=\operatorname{vec}(Z)\):

$$
K_s^{-1}ZK_t^{-1}+GZB_n=Q.
$$

## 10.1 Code: Sylvester precision solve

File: `stvgp_kronecker/joint_ssgp_kron/kron_utils.py`

```python
def solve_sylvester_precision(
    Kt: np.ndarray,
    Ks: np.ndarray,
    B: np.ndarray,
    G: np.ndarray,
    Q: np.ndarray,
    *,
    jitter: float = 1e-6,
) -> np.ndarray:
    ...
```

## 10.2 Code: Route B wrapper

File: `stvgp_kronecker/joint_ssgp_kron/model.py`

```python
def solve_Du_sylvester(q: np.ndarray) -> np.ndarray:
    Q = unvec_f(q, rows=ms, cols=mt)
    Z = solve_sylvester_precision(
        Kt=Kt_new,
        Ks=self.Ks,
        B=B_temporal,
        G=self.G,
        Q=Q,
        jitter=self.jitter,
    )
    return vec_f(Z)
```

This is the single entry point for all operations of the form \(D_u^{-1}q\). It avoids constructing and inverting the full \((M_tM_s)\times(M_tM_s)\) dense matrix.

# 11. Formula Block 10: Predictive Mean

For a test point \(x_*=(t_*,s_*)\):

$$
\mathbb{E}[y_*]
=
\phi_*^\top m_\beta+a_*^\top m_u.
$$

## 11.1 Code: prediction

File: `stvgp_kronecker/joint_ssgp_kron/model.py`

```python
linear_mean = Phi_star @ state.beta_mean
gp_mean_matrix = C_star @ state.M_u @ T_star.T
gp_mean = vec_f(gp_mean_matrix)
mean = linear_mean + gp_mean
```

Mapping:

| Formula | Code |
|---|---|
| \(\phi_*^\top m_\beta\) | `Phi_star @ state.beta_mean` |
| \(a_*^\top m_u\) | `vec_f(C_star @ state.M_u @ T_star.T)` |
| predictive mean | `linear_mean + gp_mean` |

# 12. Formula Block 11: Predictive Variance

Route B predictive variance:

$$
\operatorname{Var}(y_*)
=
\sigma^2+\nu_*+a_*^\top v_*
+
(\phi_*-B_{\beta u}v_*)^\top
\Lambda_{\beta|u}^{-1}
(\phi_*-B_{\beta u}v_*),
\qquad
D_uv_*=a_*.
$$

## 12.1 Code: predictive variance decomposition

File: `stvgp_kronecker/joint_ssgp_kron/model.py`

```python
v_star = solve_Du_sylvester(a_star)
u_posterior_term = float(a_star.T @ v_star)
adjusted_phi = phi_star - R_beta_u @ v_star
beta_schur_term = float(adjusted_phi.T @ S_beta_beta @ adjusted_phi)
total_var = sigma2 + nu_star + u_posterior_term + beta_schur_term
```

Variance terms:

| Term | Meaning |
|---|---|
| `sigma2` | observation noise variance |
| `nu_star` | sparse conditional residual variance |
| `u_posterior_term` | inducing posterior uncertainty |
| `beta_schur_term` | beta uncertainty corrected by beta-u coupling |
| `total_var` | final predictive variance for noisy observations |

## 12.2 Where beta-u covariance enters

The dense covariance expansion would include:

$$
\phi_*^\top S_{\beta\beta}\phi_*
+
a_*^\top S_{uu}a_*
+
2\phi_*^\top S_{\beta u}a_*.
$$

Route B does not explicitly materialize \(S_{\beta u}\). Instead, it appears through:

```python
adjusted_phi = phi_star - R_beta_u @ v_star
```

This is the Schur-complement way to retain beta-u posterior coupling while still using structured solves for the large \(u\)-block.

# 13. Formula Block 12: Mean-Field and No-Transfer Baselines

Mean-field approximation:

$$
q(\beta,u)=q(\beta)q(u),
\qquad
S_{\beta u}=0.
$$

Mean-field predictive variance:

$$
\operatorname{Var}_{MF}(y_*)
=
\sigma^2+\nu_*+
\phi_*^\top S_{\beta,MF}\phi_*
+
a_*^\top S_{u,MF}a_*.
$$

## 13.1 Code path

The mean-field baseline follows the older SSGP transfer path and does not explicitly retain `R_beta_u`:

```python
if method == "mean_field_ssgp_transfer":
    state = model.update_block_ssgp_transfer(...)
elif method == "structured_joint_ssgp_transfer":
    state = model.update_block_structured_joint_ssgp_transfer(...)
```

## 13.2 Meaning of no-transfer

The no-transfer baseline trains on the current block without carrying old GP likelihood information forward:

```python
if method == "no_transfer":
    state = None
    state = model.update_block_structured_joint_ssgp_transfer(..., state=state)
```

It may still use the linear component and current-block GP information, but it does not preserve historical inducing likelihood statistics.

# 14. Formula Block 13: Initial-Task Full-GP MLL Hyperparameter Selection

The runner supports a method-independent initial-task fitting protocol. It selects temporal lengthscale, and optionally noise/kernel variance, using full-GP marginal likelihood on an initial calibration task.

This is different from test-set NLL:

| Quantity | Used for | Timing |
|---|---|---|
| Full-GP marginal likelihood / NLML | hyperparameter selection | initial task only |
| Experimental NLL | model evaluation | online evaluation windows |

## 14.1 Code path

File: `scripts/run_joint_ssgp_kron_experiments.py`

```python
if args.ell_t_fit_mode == "initial_task_fullgp":
    fitted = fit_model_hyperparameters_from_initial_task(...)
    model_ell_t = fitted.ell_t
    sigma2 = fitted.noise
    kernel_variance = fitted.kernel_variance
```

Protocol rules:

- Fit hyperparameters once on the initial task.
- Freeze selected values for all later online blocks.
- Use the same selected values for all methods.
- Do not tune separately for Route B, mean-field, or no-transfer.
- Do not use future online blocks or test labels.

# 15. Formula Block 14: Evaluation Metrics

Main metrics:

$$
\operatorname{RMSE}
=
\sqrt{\frac{1}{N}\sum_i(\hat y_i-y_i)^2},
\qquad
\operatorname{MAE}
=
\frac{1}{N}\sum_i|\hat y_i-y_i|.
$$

Gaussian negative log likelihood:

$$
\operatorname{NLL}
=
\frac{1}{2}\log(2\pi v_i)
+
\frac{(y_i-\mu_i)^2}{2v_i}.
$$

90 percent coverage:

$$
y_i\in[\mu_i-1.645\sqrt{v_i},\ \mu_i+1.645\sqrt{v_i}].
$$

Continual-learning forgetting:

$$
F_n(M)
=
\frac{1}{n-1}\sum_{j<n}
\left[
M_{\text{after training }n\text{ on block }j}
-
M_{\text{after training }j\text{ on block }j}
\right].
$$

Evaluation modes:

| Mode | Meaning |
|---|---|
| `current` | evaluate block \(n\) after training on block \(n\) |
| `seen_history` | evaluate all seen blocks \(1,\ldots,n\) after training on block \(n\) |
| `future` | evaluate block \(n+1\) after training on block \(n\), if available |

# 16. Formula-to-Code Quick Reference

| Theory formula | Main code |
|---|---|
| \(y_n=\Phi_n\beta+A_nu_n+\epsilon_n\) | `make_indexed_block_factors`, `update_block_structured_joint_ssgp_transfer` |
| \(A_n=T_n\otimes C\) | `kron_mv`, `joint_likelihood_stats` |
| \(G=C^\top C\) | `self.G = self.C.T @ self.C` |
| \(L_{on}=K_{on}K_{nn}^{-1}\) | `temporal_transfer_matrix` |
| \(R_{\beta u,o\to n}=R_{\beta u,o}L_{on}\) | `transfer_cross_block_temporal` |
| \(B_n=L_t^\top B_oL_t+\sigma^{-2}T_n^\top T_n\) | `_routeB_transfer_old_stats`, `joint_likelihood_stats` |
| \(\Lambda_{\beta|u}=A_\beta-B_{\beta u}D_u^{-1}B_{\beta u}^\top\) | `schur_recover_posterior` |
| \(D_u^{-1}q\) by Sylvester equation | `solve_Du_sylvester`, `solve_sylvester_precision` |
| \(\mathbb E[y_*]=\phi_*^\top m_\beta+a_*^\top m_u\) | `predict` |
| Route B predictive variance | `predictive_variance_decomposition` |
| seen-history retention / forgetting | `evaluate_state_on_factors`, experiment runner summaries |

# 17. Recommended Code Reading Order

1. `scripts/run_joint_ssgp_kron_experiments.py`
2. `stvgp_kronecker/joint_ssgp_kron/synthetic.py`
3. `stvgp_kronecker/joint_ssgp_kron/model.py`
4. `stvgp_kronecker/joint_ssgp_kron/structured_state.py`
5. `stvgp_kronecker/joint_ssgp_kron/ssgp_transfer.py`
6. `stvgp_kronecker/joint_ssgp_kron/kron_utils.py`
7. `scripts/verify_joint_ssgp_kron_derivations.py`
8. `tests/test_joint_ssgp_kron_routeB.py`

# 18. Core Takeaway

Route B's main contribution is structured joint old-likelihood transfer. It preserves the beta-u cross natural block, recovers posterior moments with a Schur complement, and keeps the large GP block scalable through Kronecker/Sylvester solves.

