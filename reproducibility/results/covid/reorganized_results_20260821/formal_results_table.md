| Method | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |
|---|---:|---:|---:|---:|---:|
| Last-value persistence | 0.1955 +/- 0.0095 | 0.1087 +/- 0.0051 | -0.1894 +/- 0.0321 | 0.0656 +/- 0.0316 | 0.9408 +/- 0.0187 |
| Kron-STGP (point-inducing temporal control) | 0.1600 +/- 0.0106 | 0.0931 +/- 0.0044 | -0.2811 +/- 0.0326 | 0.1586 +/- 0.0208 | 0.9709 +/- 0.0123 |
| KronHiPPO-STGP (cumulative HiPPO) | 0.1565 +/- 0.0127 | 0.0878 +/- 0.0055 | -0.3652 +/- 0.0441 | 0.1210 +/- 0.0316 | 0.9601 +/- 0.0230 |
| Streaming sparse GP (Bui et al.; controlled transfer) | 0.5514 +/- 0.0502 | 0.3278 +/- 0.0365 | 1.0158 +/- 0.2164 | 0.1578 +/- 0.0380 | 0.6820 +/- 0.0647 |
| Streaming sparse GP (Bui et al.; adaptive update, CPU) | 0.1742 +/- 0.0290 | 0.0990 +/- 0.0167 | -0.2603 +/- 0.1757 | 0.1267 +/- 0.0435 | 0.9520 +/- 0.0134 |
| ST-SVGP | 0.2094 +/- 0.0136 | 0.1192 +/- 0.0067 | -0.0840 +/- 0.0390 | 0.0900 +/- 0.0239 | 0.9526 +/- 0.0136 |
| LMC-SVGP | 0.3376 +/- 0.0529 | 0.1965 +/- 0.0281 | 1.0809 +/- 0.4905 | 0.1806 +/- 0.0140 | 0.6403 +/- 0.0362 |
| ICM-SVGP | 0.3435 +/- 0.0447 | 0.2002 +/- 0.0219 | 1.0005 +/- 0.3588 | 0.1797 +/- 0.0016 | 0.6485 +/- 0.0224 |
| FSDE-SVI | 0.4705 +/- 0.0478 | 0.2580 +/- 0.0158 | 0.6693 +/- 0.1044 | 0.0238 +/- 0.0066 | 0.9119 +/- 0.0158 |

Values are mean +/- sample SD over the retained repetitions for each method. The provenance and replication count for every row are recorded in `source_manifest.json`.
OHSVGP, Task-1 lag ridge and the previous five-seed LMC-SVGP, ICM-SVGP and FSDE-SVI values are excluded from this table.
