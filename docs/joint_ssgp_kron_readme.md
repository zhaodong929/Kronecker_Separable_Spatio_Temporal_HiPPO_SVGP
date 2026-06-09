# Joint SSGP Kronecker Route B Implementation

Route B is now the main implementation under `stvgp_kronecker/joint_ssgp_kron/`.
It keeps a structured joint Gaussian posterior over `z = [beta; u]` rather than
using the previous residual/plug-in mean-field update as the main method.

## Shape Convention

- `u = vec_F(M_u)` with `M_u.shape == (M_s, M_t)`.
- `A = T_n kron C`.
- `A @ vec_F(M_u) = vec_F(C @ M_u @ T_n.T)`.
- `H_info.shape == (M_s, M_t)` stores the likelihood natural vector `h_u`.
- `D_u x = q` is solved as `K_s^{-1} Z K_t^{-1} + G Z B = Q`, where
  `x = vec_F(Z)` and `q = vec_F(Q)`.
- A temporal basis transfer uses `L_on = L_t kron I_s`; therefore
  `h_u -> L_on.T h_u` is implemented as `H_info @ L_t`.

## Route B Natural Statistics

For `y = Phi beta + A u + eps`, the structured joint likelihood update stores:

```text
R_beta_beta = Phi.T Phi / sigma2
R_beta_u    = Phi.T A / sigma2
R_uu        = (T.T T / sigma2) kron (C.T C)
h_beta      = Phi.T y / sigma2
h_u         = A.T y / sigma2
```

`R_beta_u` is a likelihood natural-precision cross block, not posterior
covariance. The posterior beta-u covariance is recovered implicitly by Schur
complement.

## Changing-Basis Transfer

The old joint likelihood ratio is transferred as:

```text
R_beta_beta -> R_beta_beta
R_beta_u    -> R_beta_u @ (L_t kron I_s)
R_uu        -> (L_t.T B_old L_t) kron G
h_beta      -> h_beta
h_u         -> (L_t kron I_s).T h_u
```

The Kronecker invariant applies only to the `u-u` block:

```text
R_uu = B_temporal kron G
G = C.T C
```

The full joint likelihood block is not claimed to be a single Kronecker product.

## Posterior Recovery

With

```text
Lambda = [[A_beta, B_beta_u],
          [B_beta_u.T, D_u]]
```

Route B forms:

```text
Lambda_beta_given_u = A_beta - B_beta_u D_u^{-1} B_beta_u.T
m_beta = Lambda_beta_given_u^{-1}(h_beta - B_beta_u D_u^{-1} h_u)
m_u = D_u^{-1} h_u - D_u^{-1} B_beta_u.T m_beta
```

`Lambda_beta_given_u` is a precision. It is intentionally not named `S_beta`.

## Prediction

The structured predictive variance is:

```text
sigma2 + nu_star + a_star.T v_star
+ (phi_star - B_beta_u v_star).T
   Lambda_beta_given_u^{-1}
  (phi_star - B_beta_u v_star)
```

where `D_u v_star = a_star`. The conditional term uses explicit kernel
amplitude:

```text
nu_star = max(0, prior_point_variance - a_star.T K_uu a_star)
```

This keeps the beta-u covariance effect that the mean-field ablation drops.

## Main Files

- `kron_utils.py`: vectorization helpers, dense references, `solve_Du_sylvester`,
  and Schur recovery.
- `ssgp_transfer.py`: Route B likelihood statistics and joint transfer helpers.
- `structured_state.py`: old fields plus Route B natural statistics.
- `model.py`: `update_block_structured_joint_ssgp_transfer` as the main Route B
  method; old methods remain as ablations.
- `scripts/verify_joint_ssgp_kron_derivations.py`: old checks plus Route B JSON.
- `scripts/run_joint_ssgp_kron_experiments.py`: includes
  `structured_joint_ssgp_transfer` and `mean_field_ssgp_transfer`.

## Limitations

- If spatial observation pattern changes per block, the old likelihood may become
  a sum of Kronecker products.
- If spatial inducing locations move online, `L_on` may not equal `L_t kron I_s`.
- Non-Gaussian likelihoods break the simple Gaussian natural-statistic update.
- Dense unrestricted posterior covariance should not be used to infer a single
  Kronecker `R_uu`.
