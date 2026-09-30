# Formal COVID Long-Stream Paper Materials

This package uses the audited CDC NHSN mandatory-period weekly stream, 52 jurisdictions, 52 Task-1 weeks and 143 strict-online weeks. All completed rows use the Gaussian likelihood on the standardized form of `log1p(weekly COVID admissions per 100k)`; trajectory figures restore the target scale for display.

## Main Table

The complete table is in `../formal_results_table.csv` and `../formal_results_table.tex`. Metrics are RMSE, CRPS, native Gaussian NLPD, ECE and Coverage90. OHSVGP, Task-1 lag ridge and the superseded five-seed LMC-SVGP, ICM-SVGP and FSDE-SVI rows are not included. Exact per-row provenance and repetition counts are in `../source_manifest.json`.

KronHiPPO-STGP has RMSE 0.1565 +/- 0.0127, compared with 0.1600 +/- 0.0106 for Kron-STGP. Its CRPS and Gaussian NLPD are also lower, while Coverage90 is closer to the nominal 0.90 than Kron-STGP.

## Figure Reading Guide

`fig_covid_prediction_trajectories` shows four predeclared seed-5 held-out jurisdictions and separates point-trajectory comparison from the uncertainty bands of the two controlled Route B variants.
`fig_covid_error_heatmap` averages only seeds in which a jurisdiction is held out and shows where the paired ordinary-minus-HiPPO error difference occurs.
`fig_covid_memory_gap` uses paired bootstrap resampling over spatial split seeds, so the confidence band does not treat jurisdiction-week cells as independent observations.
`fig_covid_calibration_curve` uses the retained predictive variances directly; no intervals are reconstructed from aggregate metrics.
`fig_covid_metric_summary` is a compact visual copy of the current table, with Kron-STGP and KronHiPPO-STGP highlighted.
LMC-SVGP, ICM-SVGP and FSDE-SVI have only a current cloud aggregate locally, not corresponding prediction archives. Their values appear in the metric panel but no trajectory, interval or error heatmap is fabricated for them.

All figures are newly written under this package directory and do not overwrite the previous COVID dataset overview figures.
