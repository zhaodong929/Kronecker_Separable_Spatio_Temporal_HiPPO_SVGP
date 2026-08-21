# COVID 4090 Exploratory Upload

This upload preserves the completed no-leakage archives from the expedited
three-seed AutoDL RTX 4090 run. It is not a replacement for the locked formal
benchmark or `BASELINE_FAIRNESS_PROTOCOL.json`.

| Method | Seeds | Status | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |
|---|---:|---|---:|---:|---:|---:|---:|
| LMC-SVGP | 5, 6, 7 | exploratory budget-limited | 0.3376 | 0.1965 | 1.0809 | 0.1806 | 0.6403 |
| IMC-SVGP | 5, 6, 7 | exploratory budget-relaxed | 0.3435 | 0.2002 | 1.0005 | 0.1797 | 0.6485 |
| FSDE-SVI | 5, 6, 7 | exploratory budget-relaxed | 0.4705 | 0.2580 | 0.6693 | 0.0238 | 0.9119 |
| ST-SVGP causal refit | 5, 6, 7 | failed before archive | n/a | n/a | n/a | n/a | n/a |

The three ST-SVGP long-stream jobs used the development-selected `Ms=52`
configuration. Each exhausted the 24 GB GPU during the growing causal refit,
so no `predictions.npz` was written. Their `run.log` files are included as
failure evidence.

The report includes nine valid `(143, 10)` prediction archives with zero
current-hidden reads and one delayed-label absorption per target. OVC remains
validation pending because its two exact-fantasy memory audits did not pass.
