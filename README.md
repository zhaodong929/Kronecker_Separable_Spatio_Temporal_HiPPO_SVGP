# Kronecker-Separable Spatio-Temporal HiPPO-SVGP

This repository contains the LaTeX source and compiled PDF for a joint online Gaussian process model with a linear mean component and a Kronecker-separable spatio-temporal HiPPO-SVGP inducing representation.

## Main Files

- `new_main_joint_training_ssgp_kron_with_appendix.tex`: LaTeX source with the full derivation and appendix.
- `new_main_joint_training_ssgp_kron_with_appendix.pdf`: compiled formula document.

## Core Model

The document studies online spatio-temporal regression at spatial locations `s_i` and time points `t_n`. The observation model is

```math
y(t_n, s_i) = \phi(t_n, s_i)^T \beta + f(t_n, s_i) + \epsilon_{n,i},
\qquad \epsilon_{n,i} \sim \mathcal{N}(0, \sigma^2).
```

At block level,

```math
y_n = \Phi_n \beta + f_n + \epsilon_n,
\qquad \epsilon_n \sim \mathcal{N}(0, \sigma^2 I).
```

The latent process is a separable spatio-temporal Gaussian process:

```math
k((s,t),(s',t')) = k_s(s,s') k_t(t,t'),
\qquad K_{ff} = K_t \otimes K_s.
```

The main modeling decision is to train the linear mean parameter `beta` and the sparse GP inducing variables `u` jointly, instead of estimating `beta` first and fitting the GP to residuals.

## Why Joint Training

A two-stage residual method uses

```math
r_n(\hat{\beta}_n) = y_n - \Phi_n \hat{\beta}_n.
```

For an old block `j < n`, if the estimate changes from `\hat{\beta}_{n-1}` to `\hat{\beta}_n`, then

```math
r_j(\hat{\beta}_n)
= r_j(\hat{\beta}_{n-1}) - \Phi_j(\hat{\beta}_n - \hat{\beta}_{n-1}).
```

Historical residual targets therefore become stale whenever `beta` is updated. The joint model avoids this by keeping the likelihood on `y_n`:

```math
p(y_n | \beta, u_n)
= \mathcal{N}(y_n | \Phi_n \beta + A_n u_n, \sigma^2 I).
```

## Kronecker HiPPO-SVGP Representation

Temporal inducing variables use HiPPO-LegS interdomain features over a horizon `T`:

```math
u_\ell^{(t)}(T) = \int_0^T g_\ell^{(T)}(x) f_t(x) dx.
```

A stationary temporal kernel is approximated with random Fourier features. This gives analytic temporal inducing covariances and point-to-inducing cross-covariances through Legendre oscillatory integrals and spherical Bessel identities.

Spatial inducing variables use fixed inducing locations

```math
Z_s = \{z_1^{(s)}, \ldots, z_{M_s}^{(s)}\}.
```

The mixed spatio-temporal inducing covariance is

```math
K_{uu}^{(st)}(T) = K_{uu}^{(t)}(T) \otimes K_{ZZ}^{(s)}.
```

For block `D_n`, the sparse GP projection is Kronecker structured:

```math
A_n = T_n \otimes C,
```

where

```math
T_n = K_{fu,n}^{(t)} (K_{uu,n}^{(t)})^{-1},
\qquad
C = K_{XZ}^{(s)} (K_{ZZ}^{(s)})^{-1}.
```

## Online Joint ELBO

The online joint variational objective is

```math
\mathcal{L}_n
= \mathbb{E}_{q_n(\beta,u_n)}[\log p(y_n | \beta,u_n)]
- \mathrm{KL}[q_n(\beta,u_n) \Vert p_n(\beta,u_n)].
```

A practical mean-field version uses

```math
q_n(\beta,u_n) = q_n(\beta) q_n(u_n),
```

with Gaussian factors for both the linear coefficient and the GP inducing state.

## Changing-Basis Transfer

Because the HiPPO temporal basis may change between online blocks, old inducing variables `u_o` and new inducing variables `u_n` are not in the same coordinate system. The main proposed update transfers old information through an SSGP-style old-likelihood ratio.

For

```math
q_{n-1}(u_o)=\mathcal{N}(m_o,S_o),
\qquad p(u_o)=\mathcal{N}(0,K_{oo}),
```

define old likelihood natural statistics

```math
R_o = S_o^{-1} - K_{oo}^{-1},
\qquad r_o = S_o^{-1} m_o.
```

With

```math
L_{on}=K_{on}K_{nn}^{-1},
```

the old likelihood information transfers as

```math
\Lambda_{o\to n}=L_{on}^T R_o L_{on},
\qquad h_{o\to n}=L_{on}^T r_o.
```

The GP coordinate update is

```math
S_{u,n}^{-1}
= K_{nn}^{-1} + \Lambda_{o\to n} + \sigma^{-2}A_n^T A_n,
```

```math
S_{u,n}^{-1}m_{u,n}
= h_{o\to n} + \sigma^{-2}A_n^T(y_n - \Phi_n m_{\beta,n}).
```

## Kronecker-Preserving Old-Likelihood Transfer

The appendix proves that when the spatial inducing set is fixed and the changing basis acts only along time,

```math
L_{on}=L_{on}^{(t)} \otimes I_s.
```

If the old likelihood precision is maintained in structured form

```math
R_o = B_o \otimes G,
\qquad G=C^T C,
```

then the transferred precision remains Kronecker separable:

```math
\Lambda_{o\to n}
= [(L_{on}^{(t)})^T B_o L_{on}^{(t)}] \otimes G.
```

The temporal likelihood statistic is updated by

```math
B_n = (L_{on}^{(t)})^T B_o L_{on}^{(t)} + \sigma^{-2}T_n^T T_n.
```

Thus the inducing precision keeps the form

```math
\Lambda_{u,n}
= (K_{nn}^{(t)})^{-1} \otimes K_s^{-1} + B_n \otimes G.
```

## Sylvester Computation

The Kronecker precision allows posterior mean and uncertainty solves to be reduced to Sylvester equations. For `z = vec(Z)` and `q = vec(Q)`,

```math
\Lambda_{u,n} z = q
```

is equivalent to

```math
K_s^{-1} Z (K_{nn}^{(t)})^{-1} + G Z B_n = Q.
```

This avoids materializing dense `M_t M_s` by `M_t M_s` precision matrices and is the main scalability mechanism in the formulation.

## Recommended Comparisons

The LaTeX document recommends evaluating:

1. two-stage future-only residual baseline;
2. two-stage residual GP with buffer or sketch correction;
3. fixed-basis exact joint Gaussian update;
4. joint mean-field without changing-basis transfer;
5. joint mean-field with projected-prior transfer;
6. joint mean-field with SSGP-style old-likelihood-ratio transfer;
7. structured joint posterior with SSGP-style transfer.

Suggested metrics include RMSE, MAE, test negative log likelihood, predictive interval coverage, runtime per online block, peak memory, memory-matched RMSE/NLL, calibration error, and continual evaluation windows.

## Repository Layout

```text
.
├── README.md
├── .gitignore
├── new_main_joint_training_ssgp_kron_with_appendix.tex
└── new_main_joint_training_ssgp_kron_with_appendix.pdf
```

## Status

This repository currently contains the theoretical derivation, appendix, and compiled formula document. It is intended as the reference material for implementing joint online Kronecker-separable spatio-temporal HiPPO-SVGP training.
