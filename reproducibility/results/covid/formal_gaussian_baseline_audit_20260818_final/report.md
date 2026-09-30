# COVID Long-Stream Setting B Gaussian Benchmark

All metric rows use the same delayed-observation information set and the common log1p(per-100k) target scale.

## Capacity Selection

Seed 0 only. Train on the 38 Task-1 fit jurisdictions across all 52 Task-1 weeks; select by validation Gaussian NLPD on the four predeclared Task-1 validation jurisdictions. RMSE is the tie-breaker. Every eligible method receives three Task-1 candidates under this same selection budget. Formal spatial splits 5-9 are not opened during selection.

Route B ordinary and cumulative HiPPO are exactly matched at `Mt=32, Ms=32`. ST-SVGP tunes spatial inducing locations only. Bui OSGPR and OVC-SVGP receive the same three temporal-by-spatial candidate grids under the same Task-1 selection budget; their selected final grids need not have identical raw counts.

| Method | Splits | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 5 | 0.1955 +/- 0.0095 | 0.1087 +/- 0.0051 | -0.1894 +/- 0.0321 | 0.0656 +/- 0.0316 | 0.9408 +/- 0.0187 |
| Task-1 lag ridge | 5 | 0.6877 +/- 0.0555 | 0.4910 +/- 0.0510 | 3.8552 +/- 1.0231 | 0.3699 +/- 0.0210 | 0.3014 +/- 0.0384 |
| OHSVGP (RBF) | 5 | 0.6198 +/- 0.0774 | 0.3631 +/- 0.0502 | 0.9388 +/- 0.1421 | 0.1011 +/- 0.0353 | 0.9119 +/- 0.0680 |
| Route B ordinary inducing | 5 | 0.1600 +/- 0.0106 | 0.0931 +/- 0.0044 | -0.2811 +/- 0.0326 | 0.1586 +/- 0.0208 | 0.9709 +/- 0.0123 |
| Route B cumulative HiPPO | 5 | 0.1565 +/- 0.0127 | 0.0878 +/- 0.0055 | -0.3652 +/- 0.0441 | 0.1210 +/- 0.0316 | 0.9601 +/- 0.0230 |
| Bui OSGPR (controlled) | 5 | 0.4854 +/- 0.0452 | 0.2827 +/- 0.0306 | 0.8068 +/- 0.1687 | 0.1246 +/- 0.0363 | 0.7459 +/- 0.0628 |
| Bui OSGPR (adaptive, CPU) | 5 | 0.1669 +/- 0.0182 | 0.0925 +/- 0.0081 | -0.3274 +/- 0.0848 | 0.1017 +/- 0.0409 | 0.9345 +/- 0.0313 |

## Excluded Candidates

| Method | Status | Reason |
|---|---|---|
| ST-SVGP | formal_run_resource_limited_local_cpu | The full 143-week causal-refit adapter exceeded the local CPU memory guard before a formal archive could be written. Independent 16-week causal segments also failed before their first archive under the selected Ms=32 configuration; a fixed-grid missing-label attempt reached the 15 GB host limit at the seed-0 39-week diagnostic. Both are documented; no formal metric row is reported. |
| OVC-SVGP | formal_run_resource_limited_local_cpu | Official exact-fantasy conditioning reached about 15 GB RSS on formal seed 5 before producing a complete 143-week archive, so the local CPU run was stopped under a resource guard rather than risking host OOM. |
| LMC-SVGP | protocol_incompatible_without_core_rewrite | The official LMC trainer creates a tf.data.Dataset from complete time-output vectors Y_train.T and exposes no causal current-week output mask. A Setting B adapter would require changing its upstream data and likelihood interface. |
| IMC-SVGP | protocol_incompatible_without_core_rewrite | The official IMC trainer creates a tf.data.Dataset from complete time-output vectors Y_train.T and exposes no causal current-week output mask. A Setting B adapter would require changing its upstream data and likelihood interface. |
| FSDE-SVI | protocol_incompatible_without_core_rewrite | The official FSDE-SVI Dataset stores a dense [outputs, time] target matrix and minibatches complete output columns. A Setting B adapter would require rewriting the upstream objective to represent ten unobserved current outputs. |
| EARTH | protocol_incompatible_without_core_rewrite | The official source imports an undeclared vmamba module absent from the pinned snapshot. Independently, its loader constructs full-node targets Y for every forecast week and the model accepts only complete historical windows. It exposes neither a current-week node-observation mask nor predictive variance, so a Setting B adapter would require changing the official data, loss, and output interfaces. |

## Pending Archives

- OVC-SVGP seed 5: archive_missing
- OVC-SVGP seed 6: archive_missing
- OVC-SVGP seed 7: archive_missing
- OVC-SVGP seed 8: archive_missing
- OVC-SVGP seed 9: archive_missing
- ST-SVGP seed 5: archive_missing
- ST-SVGP seed 6: archive_missing
- ST-SVGP seed 7: archive_missing
- ST-SVGP seed 8: archive_missing
- ST-SVGP seed 9: archive_missing
