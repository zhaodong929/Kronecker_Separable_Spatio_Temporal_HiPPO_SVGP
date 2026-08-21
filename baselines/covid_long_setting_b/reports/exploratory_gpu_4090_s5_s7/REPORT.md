# Accelerated 4090 Exploratory Results

All rows below use the common Gaussian metric evaluator on restored `log1p(admissions per 100k)`.
They are not admitted to the strict main table: the original capacity/convergence gates were relaxed for this cloud-budget run.

| Method | Qualification | Seeds | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |
|---|---|---:|---:|---:|---:|---:|---:|
| LMC-SVGP | exploratory_budget_limited | 3 | 0.3376 +/- 0.0529 | 0.1965 +/- 0.0281 | 1.0809 +/- 0.4905 | 0.1806 +/- 0.0140 | 0.6403 +/- 0.0362 |
| IMC-SVGP | exploratory_budget_relaxed | 3 | 0.3435 +/- 0.0447 | 0.2002 +/- 0.0219 | 1.0005 +/- 0.3588 | 0.1797 +/- 0.0016 | 0.6485 +/- 0.0224 |
| FSDE-SVI | exploratory_budget_relaxed | 3 | 0.4705 +/- 0.0478 | 0.2580 +/- 0.0158 | 0.6693 +/- 0.1044 | 0.0238 +/- 0.0066 | 0.9119 +/- 0.0158 |

## Missing Or Invalid Archives

- `st_svgp` seed 5: `FileNotFoundError(2, 'No such file or directory')`
- `st_svgp` seed 6: `FileNotFoundError(2, 'No such file or directory')`
- `st_svgp` seed 7: `FileNotFoundError(2, 'No such file or directory')`
