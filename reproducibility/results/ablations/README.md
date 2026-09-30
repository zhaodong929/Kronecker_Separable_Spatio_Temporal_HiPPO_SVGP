# Ablation and mechanism evidence

This directory contains the compact, paper-facing data behind the ablation evidence shown in the current manuscript.

| Artifact | Evidence represented |
|---|---|
| `matched_controls.csv` | Joint/changing, zero-cross/changing, and joint/fixed-global controls on ERA5-Land and PEMS-BAY |
| `coordinate_transfer.csv` | Conditional transport versus identity reuse at `M_t=128`, including predictive KL |
| `structured_solver.csv` | Dense versus Schur--Sylvester posterior recovery on matched systems |
| `cross_domain_main_table.csv` | Compact copy of the cross-domain main table values |
| `figures/figure3_main_mechanism.*` | Main three-stage mechanism figure |
| `figures/mechanism_diagnostics_two_panel.*` | Coordinate-transfer and predictive-KL diagnostic figure |
| `figures/structured_dense_solver_scaling.pdf` | Structured-solver scaling figure |

The authoritative prose and LaTeX tables remain in `reproducibility/paper/main.tex` and `reproducibility/paper/appendix.tex`. These CSVs are reporting artifacts copied from that paper snapshot; they are not new experiments and were not selected using formal-stream outcomes.

The matched controls keep the same split, calibration, inducing budgets, hyperparameters, and information boundary within each dataset. `zero-cross` removes the accumulated trend--residual likelihood cross block. `identity reuse` carries historical natural parameters forward without the conditional coordinate transport. The solver comparison uses identical positive-definite systems and right-hand sides.
