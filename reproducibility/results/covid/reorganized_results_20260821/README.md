# Current COVID Long-Stream Results

`formal_results_table.*` is the current paper table. It excludes OHSVGP and
Task-1 lag ridge, and replaces the previous five-seed LMC-SVGP, ICM-SVGP and FSDE-SVI rows with the
latest cloud aggregate. The old rows remain only in the immutable archival
report tree and in `superseded_rows_audit.*`; they are not a source for any
current table or figure.

`source_manifest.json` is the authoritative per-row provenance record. It
retains the actual number of repetitions used to compute each mean and sample
standard deviation. The table intentionally has no seed-count column.

`figures/` contains regenerated metric, trajectory, Route-B error, long-memory
and calibration figures. The three cloud methods have aggregate metrics but no
local prediction archives, so no trajectory or uncertainty plot is fabricated
for them.

`reproducibility/` contains the two generators used to build this package.
