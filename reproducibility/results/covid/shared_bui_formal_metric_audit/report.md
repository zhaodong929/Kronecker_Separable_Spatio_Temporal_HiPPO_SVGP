# COVID Long-Stream Setting B Gaussian Benchmark

All metric rows use the same delayed-observation information set and the common log1p(per-100k) target scale.

## Capacity Selection

Seed 0 only. Every method receives the predeclared Task-1 candidate budget and is selected by Gaussian NLPD with RMSE as the tie-breaker; formal spatial splits 5-9 are not opened. ST-SVGP, Bui OSGPR and OVC-SVGP use the 38-fit/4-validation spatial split. Bui OSGPR and OVC-SVGP share one final time-by-space inducing grid selected from their common three-grid budget by macro-averaged Task-1 validation Gaussian NLPD across OVC, controlled Bui and adaptive Bui. The official complete-output LMC/IMC/FSDE family shares one M,Q choice using a 48-week history plus four chronological Setting B validation weeks, because these models cannot fit a 38-output subset and then represent the held-out output dimensions without changing their official likelihood. This validation-geometry exception is explicit.

Route B ordinary and cumulative HiPPO are exactly matched at `Mt=32, Ms=32`. ST-SVGP tunes spatial inducing locations only. Bui OSGPR and OVC-SVGP receive the same three temporal-by-spatial candidate grids under the same Task-1 selection budget; their selected final grids need not have identical raw counts.
LMC-SVGP and IMC-SVGP share the selected temporal inducing count and latent rank from the explicit chronological Task-1 validation record, because their official complete-output trainers cannot fit a 38-output Task-1 spatial subset without changing the model core.

| Method | Splits | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 5 | 0.1955 +/- 0.0095 | 0.1087 +/- 0.0051 | -0.1894 +/- 0.0321 | 0.0656 +/- 0.0316 | 0.9408 +/- 0.0187 |
| Task-1 lag ridge | 5 | 0.6877 +/- 0.0555 | 0.4910 +/- 0.0510 | 3.8552 +/- 1.0231 | 0.3699 +/- 0.0210 | 0.3014 +/- 0.0384 |
| OHSVGP (RBF) | 5 | 0.6198 +/- 0.0774 | 0.3631 +/- 0.0502 | 0.9388 +/- 0.1421 | 0.1011 +/- 0.0353 | 0.9119 +/- 0.0680 |
| Route B ordinary inducing | 5 | 0.1600 +/- 0.0106 | 0.0931 +/- 0.0044 | -0.2811 +/- 0.0326 | 0.1586 +/- 0.0208 | 0.9709 +/- 0.0123 |
| Route B cumulative HiPPO | 5 | 0.1565 +/- 0.0127 | 0.0878 +/- 0.0055 | -0.3652 +/- 0.0441 | 0.1210 +/- 0.0316 | 0.9601 +/- 0.0230 |
| Bui OSGPR (controlled) | 5 | 0.5514 +/- 0.0502 | 0.3278 +/- 0.0365 | 1.0158 +/- 0.2164 | 0.1578 +/- 0.0380 | 0.6820 +/- 0.0647 |
| Bui OSGPR (adaptive, CPU) | 5 | 0.1742 +/- 0.0290 | 0.0990 +/- 0.0167 | -0.2603 +/- 0.1757 | 0.1267 +/- 0.0435 | 0.9520 +/- 0.0134 |
| LMC-SVGP | 5 | 0.8082 +/- 0.0246 | 0.4577 +/- 0.0143 | 1.2100 +/- 0.0281 | 0.0198 +/- 0.0099 | 0.9024 +/- 0.0056 |
| IMC-SVGP | 5 | 0.8082 +/- 0.0246 | 0.4577 +/- 0.0143 | 1.2100 +/- 0.0281 | 0.0218 +/- 0.0102 | 0.9024 +/- 0.0056 |

## Excluded Candidates

| Method | Status | Reason |
|---|---|---|
| ST-SVGP | formal_run_in_progress_isolated_workers | Task-1-only capacity selected at spatial_inducing=32; seed-0 five-week smoke and 39-week causal-refit diagnostic passed after official-reproduction gate |
| OVC-SVGP | formal_run_resource_limited_shared_grid | The shared 8x32 exact-fantasy formal seed-5 run reached 10.46 GiB RSS after 1,731 seconds without writing a complete archive. It was stopped under the host-memory guard; no partial result is reported and the earlier 4x32 archive is not substituted. |
| FSDE-SVI | formal_seed0_gate_in_progress_isolated_workers | thin official-core adapter: trains only on complete arrived histories and analytically conditions the official 52-output Gaussian predictive distribution on current visible labels |
| EARTH | protocol_incompatible_without_core_rewrite | The official source imports an undeclared vmamba module absent from the pinned snapshot. Independently, its loader constructs full-node targets Y for every forecast week and the model accepts only complete historical windows. It exposes neither a current-week node-observation mask nor predictive variance, so a Setting B adapter would require changing the official data, loss, and output interfaces. |

## Pending Archives

- OVC-SVGP seed 5: archive_missing
- OVC-SVGP seed 6: archive_missing
- OVC-SVGP seed 7: archive_missing
- OVC-SVGP seed 8: archive_missing
- OVC-SVGP seed 9: archive_missing
- ST-SVGP seed 7: archive_missing
- ST-SVGP seed 8: archive_missing
- ST-SVGP seed 9: archive_missing
- FSDE-SVI seed 5: archive_missing
- FSDE-SVI seed 6: archive_missing
- FSDE-SVI seed 7: archive_missing
- FSDE-SVI seed 8: archive_missing
- FSDE-SVI seed 9: archive_missing
