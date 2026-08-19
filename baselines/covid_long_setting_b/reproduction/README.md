# Official-reproduction gates

An external method moves through the following states only in this order:

1. `source_locked`: official URL, commit, license, environment and entrypoint
   are recorded in `../catalog.json`.
2. `official_reproduction_passed`: an unmodified official example or test has
   run in its isolated environment, with its command and output recorded here.
3. `covid_adapter_smoke_passed`: seed 0, five weeks, with the common archive
   audit reporting zero current-hidden reads.
4. `covid_adapter_diagnostic_passed`: seed 0, 39 weeks, finite predictions
   and positive predictive variances.
5. `formal_result_available`: seeds 5--9, all 143 weekly updates, scored by
   the common evaluator.

No baseline is called a completed COVID comparison before the fifth state.
