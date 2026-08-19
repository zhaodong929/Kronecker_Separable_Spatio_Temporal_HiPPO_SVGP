# OVC Resource-Bounded Feasibility Study

This supplementary table is **not** part of the Task-1-selected primary
benchmark. The primary shared Bui/OVC Task-1 validation selected `Mt=8,
Ms=32`. Official OVC exact-fantasy conditioning could not complete that grid
on formal seed 5, so it has no primary numeric row.

The table below answers a separate implementation question: whether the
official OVC adapter can complete the same 143-week Setting B stream on a
smaller common `Mt=4, Ms=32` point-inducing grid. Every value is recomputed
from complete formal seed 5--9 archives with the common Gaussian evaluator.

| Method | Grid | RMSE | CRPS | Gaussian NLPD | ECE | Coverage90 |
|---|---|---:|---:|---:|---:|---:|
| OVC-SVGP | `4x32` | 0.3106 +/- 0.0503 | 0.1768 +/- 0.0219 | 0.3257 +/- 0.0942 | 0.1149 +/- 0.0399 | 0.9561 +/- 0.0300 |
| Bui OSGPR (controlled) | `4x32` | 0.6242 +/- 0.0517 | 0.3861 +/- 0.0408 | 1.2972 +/- 0.2584 | 0.2105 +/- 0.0375 | 0.6074 +/- 0.0684 |
| Bui OSGPR (adaptive, CPU) | `4x32` | 0.2398 +/- 0.0409 | 0.1487 +/- 0.0405 | 0.1376 +/- 0.3042 | 0.1681 +/- 0.0721 | 0.9631 +/- 0.0373 |

Source archives and the original common-evaluator output are preserved in
`baselines/covid_long_setting_b/results/formal_selected_ovc_float32`,
`baselines/covid_long_setting_b/results/formal_selected_bui_ovc_shared_m4_feasible`,
and `baselines/covid_long_setting_b/reports/formal_gaussian_final_common_m4`.
