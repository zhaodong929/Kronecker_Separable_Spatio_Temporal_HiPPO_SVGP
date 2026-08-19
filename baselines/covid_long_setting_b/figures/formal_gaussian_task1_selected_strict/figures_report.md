# Formal COVID Long-Stream Paper Materials

This package uses the audited CDC NHSN mandatory-period weekly stream, 52 jurisdictions, 52 Task-1 weeks, 143 strict-online weeks, and formal spatial split seeds 5-9. All completed rows use the Gaussian likelihood on the standardized form of `log1p(weekly COVID admissions per 100k)`; trajectory figures restore the target scale for display.

## Main Table

The complete table is in `formal_results_table.csv` and `formal_results_table.tex`. Metrics are RMSE, CRPS, native Gaussian NLPD, ECE and Coverage90. OVC-SVGP is not numerically ranked because the Task-1-selected 8x32 exact-fantasy formal run is resource-limited; its lower-capacity feasibility study remains outside this table.

Route B cumulative HiPPO has RMSE 0.1565 +/- 0.0127, compared with 0.1600 +/- 0.0106 for ordinary inducing. Its CRPS and Gaussian NLPD are also lower, while Coverage90 is closer to the nominal 0.90 than ordinary inducing.

## Figure Reading Guide

`fig_covid_prediction_trajectories` shows four predeclared seed-5 held-out jurisdictions and separates point-trajectory comparison from the uncertainty bands of the two controlled Route B variants.
`fig_covid_error_heatmap` averages only seeds in which a jurisdiction is held out and shows where the paired ordinary-minus-HiPPO error difference occurs.
`fig_covid_memory_gap` uses paired bootstrap resampling over spatial split seeds, so the confidence band does not treat jurisdiction-week cells as independent observations.
`fig_covid_calibration_curve` uses the archived predictive variances directly; no intervals are reconstructed from aggregate metrics.
`fig_covid_metric_summary` is a compact visual copy of the formal table, with the two Route B variants highlighted.

All figures are newly written under this package directory and do not overwrite the previous COVID dataset overview figures.
