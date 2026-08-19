# COVID Long-Stream Setting B Gaussian Benchmark

All metric rows use the same delayed-observation information set and the common log1p(per-100k) target scale.

## Capacity Selection

Seed 0 only. Every method receives the predeclared Task-1 candidate budget and is selected by Gaussian NLPD with RMSE as the tie-breaker; formal spatial splits 5-9 are not opened for capacity selection. ST-SVGP, Bui OSGPR and OVC-SVGP use the 38-fit/4-validation spatial split. Bui OSGPR and OVC-SVGP share a three-grid Task-1 budget whose macro-average winner is 8x32. If a method cannot complete the full formal stream at its Task-1-selected capacity, it is reported in the failure table; it is not replaced in the primary benchmark by a lower-capacity result. The official complete-output LMC/IMC/FSDE family shares one M,Q choice using a 48-week history plus four chronological Setting B validation weeks, because these models cannot fit a 38-output subset and then represent the held-out output dimensions without changing their official likelihood. This validation-geometry exception is explicit.

Route B ordinary and cumulative HiPPO are exactly matched at `Mt=32, Ms=32`. ST-SVGP tunes spatial inducing locations only. Bui OSGPR and OVC-SVGP receive the same three temporal-by-spatial candidate grids under the same Task-1 selection budget. The Task-1 winner is 8x32. The completed Bui rows use that selected capacity; OVC-SVGP is listed in the failure table because its exact-fantasy continuation cannot complete at that capacity. The lower-capacity 4x32 OVC archive is retained only as a non-primary feasibility study.
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
| ST-SVGP | 5 | 0.2094 +/- 0.0136 | 0.1192 +/- 0.0067 | -0.0840 +/- 0.0390 | 0.0900 +/- 0.0239 | 0.9526 +/- 0.0136 |
| LMC-SVGP | 5 | 0.8082 +/- 0.0246 | 0.4577 +/- 0.0143 | 1.2100 +/- 0.0281 | 0.0201 +/- 0.0096 | 0.9024 +/- 0.0056 |
| IMC-SVGP | 5 | 0.8082 +/- 0.0246 | 0.4577 +/- 0.0143 | 1.2100 +/- 0.0281 | 0.0202 +/- 0.0091 | 0.9024 +/- 0.0056 |
| FSDE-SVI | 5 | 0.8179 +/- 0.0680 | 0.5325 +/- 0.0497 | 4.0649 +/- 0.7098 | 0.2873 +/- 0.0190 | 0.4379 +/- 0.0361 |

## Excluded Candidates

| Method | Status | Reason |
|---|---|---|
| OVC-SVGP | formal_resource_limited | The Task-1-selected 8x32 exact-fantasy formal seed-5 run reached 10.46 GiB RSS after 1,731 seconds without writing an archive. The primary benchmark does not replace it with the lower-capacity 4x32 archive. |
| EARTH | protocol_incompatible_without_core_rewrite | The official source imports an undeclared vmamba module absent from the pinned snapshot. Independently, its loader constructs full-node targets Y for every forecast week and the model accepts only complete historical windows. It exposes neither a current-week node-observation mask nor predictive variance, so a Setting B adapter would require changing the official data, loss, and output interfaces. |

## FactorialSDE Provenance

LMC-SVGP, IMC-SVGP and FSDE-SVI are completed Setting B adapters that call the pinned upstream model cores. Their unmodified upstream county-level COVID entrypoints use a different protocol and are not claimed as completed reproductions or scored in this table; see `baselines/covid_long_setting_b/reproduction/factorial_sde_official_entrypoint_status.json`.

## Resource-Bounded OVC Study

The completed OVC `4x32` archives are retained as a non-primary feasibility study and are not substituted for the Task-1-selected `8x32` primary row; see `ovc_resource_bounded_feasibility.md` in this report directory.

## Completion Audit

Archive coverage, causal-update counters, capacity selection and external-source status are itemized in `completion_audit.md` in this report directory.
