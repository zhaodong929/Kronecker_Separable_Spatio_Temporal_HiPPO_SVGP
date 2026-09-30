# Experiment Index

This index identifies the evidence used by the current paper snapshot.

| Dataset | Formal protocol | Main result evidence | Status |
|---|---|---|---|
| ERA5-Land | `protocols/era5/dataset_manifest.json`, `protocols/era5/long_streaming_manifest.json` | `results/era5/summary.csv`, `results/era5/final_audit.json` | Complete long-stream archive; five reported splits |
| COVID-19 | `protocols/covid/seed*/protocol.json` and `protocol.audit.json` | `results/covid/formal_gaussian_st_bui_complete/`, `results/covid/reorganized_results_20260821/` | Formal tables and per-seed reports retained |
| PEMS-BAY | `protocols/pems/traffic_paired_spatial_v1.json`, `protocols/pems/traffic_external_gp_a100_v1.json` | `results/pems/pems_matched_mechanism_20260914/`, `results/pems/external_gp_paper_ready/` | Three matched mechanism splits and external-GP audit retained |

## Paper result labels

- `formal`: included in the current paper tables or their cited appendix tables.
- `diagnostic`: retained for interpretation, capacity checks, or implementation auditing; it is not a formal main-table row unless the corresponding report says so.
- `excluded`: incomplete or resource-limited attempts retained with their evidence and reason.

The official ERA5 ST-SVGP Task 1--10 resource-limited attempts are recorded in `results/excluded/OFFICIAL_ST_SVGP_ALL_VERSIONS_AUDIT_20260828.md`; no metrics are inferred from those attempts.

## Reproduction entry points

- ERA5 data generation: `code/scripts_snapshot/regenerate_era5_land_tasks.py`
- COVID protocol generation: `code/scripts_snapshot/build_covid_long_stream_protocol.py`
- PEMS protocol: `protocols/pems/traffic_paired_spatial_protocol.md`
- Paper source: `paper/main.tex`, `paper/appendix.tex`, `paper/references.bib`
