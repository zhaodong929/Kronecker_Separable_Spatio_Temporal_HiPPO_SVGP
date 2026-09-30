# Reproducibility Archive

This directory is the code-first publication archive for the current KronHiPPO-STGP paper. It records the three formal experiments, their protocols, the paper source, compact result tables, audit evidence, and the code snapshot used to regenerate the archived protocols.

## Contents

- `data/`: intentionally not included in this code-first commit. The ERA5 Task 1--10 data will be uploaded in a later data archive; its manifests and regeneration script are already present.
- `protocols/`: dataset and strict-online protocol manifests for ERA5-Land, COVID-19, and PEMS-BAY.
- `results/`: compact formal paper-ready tables, figures, reports, resource records, and audit evidence. Large prediction archives and raw run logs are intentionally excluded from this code-first commit.
- `paper/`: the current LaTeX source, bibliography, figures, style files, and compiled PDF used for the paper snapshot.
- `code/`: a source snapshot of the scripts and model package used by the archived worktree.

## Reproduction order

1. Install the environment from `code/pyproject.toml` and the files under `code/requirements/`.
2. Read the protocol manifests before running any model. Do not use formal-stream results for configuration selection.
3. For ERA5, use `protocols/era5/dataset_manifest.json` and `protocols/era5/long_streaming_manifest.json`. The long dataset is being uploaded separately; the original NetCDF download is described by `download_manifest.json` and can be reconstructed with `code/scripts_snapshot/regenerate_era5_land_tasks.py`.
4. For COVID-19, use the seed-specific manifests under `protocols/covid/` and the reports under `results/covid/`.
5. For PEMS-BAY, use `protocols/pems/traffic_paired_spatial_v1.json`, `protocols/pems/traffic_external_gp_a100_v1.json`, and `protocols/pems/traffic_paired_spatial_protocol.md`.
6. Use the result archives and audit files under `results/` as the authoritative paper evidence. Temporary checkpoints, virtual environments, raw run logs, and incomplete prediction archives are intentionally excluded.

## Important provenance boundaries

The three studies share the strict-online information boundary but use domain-specific time scales, spatial splits, covariates, and computational budgets. The main paper table must therefore be read together with the dataset-specific protocol manifests.

The ERA5 long stream is a processed project artifact, not a file that was originally present in the public repository. This code-first commit includes its raw-data manifest, validation record, task boundaries, split hashes, and scaler provenance so that the extension can be audited or regenerated when the later data archive is available.

The compact result tables are reporting artifacts. They do not contain model checkpoints, training environments, or large prediction archives. Re-running the experiments requires this code snapshot, the protocol manifests, the later ERA5 data archive, and the corresponding external source datasets for COVID-19 and PEMS-BAY.

## Integrity

`SOURCE_MANIFEST.sha256` records SHA-256 hashes for every tracked file in this archive. The archive was prepared from the project worktree at commit `61ec3ed` and from the current paper source snapshot. This is the code-first commit; the large data archive will have its own manifest and commit.
