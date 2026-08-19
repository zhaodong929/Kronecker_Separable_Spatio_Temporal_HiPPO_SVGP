# COVID long-stream Setting B baselines

This directory is the single provenance and execution root for the strict-online
COVID baseline study. Existing seed-5--9 prediction archives are preserved as
preliminary audit evidence. They are never overwritten and do not determine a
new configuration.

## Fixed protocol

The first 52 weekly observations initialise every method. The following 143
weeks are genuine weekly online updates. At week `t`, every method may use all
arrived labels through `t-1`, then the 42 current visible jurisdictions, and
must predict the ten current hidden jurisdictions. A hidden label is absorbed
once, at the following week. The Gaussian comparison uses the common
`log1p(weekly admissions per 100,000)` scale and reports:

```text
RMSE, CRPS, Gaussian NLPD, ECE, Coverage90
```

Methods share an information set, not a hand-imposed feature map. Only Route B
ordinary versus cumulative HiPPO is exactly capacity-matched (`Mt=32, Ms=32`).

## Current status

| Status | Methods | Use in the current main table |
| --- | --- | --- |
| Frozen prior formal archive | Persistence, Task-1 lag ridge, controlled/adaptive Bui OSGPR, Route B ordinary, Route B cumulative HiPPO | Retained accuracy rows |
| Validation pending | OHSVGP RBF, OVC-SVGP, ST-SVGP, LMC-SVGP, IMC-SVGP, FSDE-SVI | Excluded until repair gates and new seed-5--9 archives pass |
| Protocol incompatible | EARTH | Excluded; its official interface has neither a legal current-week node mask nor predictive variance |

The old OHSVGP, ST-SVGP, LMC/IMC and FSDE archives remain valuable diagnostic
records, but are not formal results. In particular, the old LMC/IMC/FSDE
`M=4,Q=2,50-step` runs are not converged baseline rows.

## Seed-0 development

All new selection uses seed 0 only, with three chronological strict-online
windows: `1--28 -> 29--36`, `1--36 -> 37--44`, and `1--44 -> 45--52`. Every
fold is standardised only from its own visible training prefix, then scored
after restoration to the original target scale. Mean Gaussian NLPD selects;
mean RMSE breaks ties. CRPS, ECE and Coverage90 remain diagnostics.

Task-1 objective convergence is separate from model selection: check every
250 updates, require at least 2,500 updates, compare two 10-check moving
medians, and accept a plateau only below 0.1% relative change. The 50,000-step
limit is explicitly `max_budget_not_converged`, never a convergence claim.
All Task-1 checkpoints are retained for audit; only terminal plateau runs are
eligible for a configuration comparison.

```bash
python baselines/covid_long_setting_b/build_blocked_development_protocols.py
python baselines/covid_long_setting_b/run_blocked_development.py --phase capacity
# On the validated 4090 node only:
python baselines/covid_long_setting_b/run_blocked_development.py --phase capacity --execute --resume
python baselines/covid_long_setting_b/run_blocked_development.py --phase online_steps --methods lmc imc fsde --execute
```

The capacity policy is recorded in [capacity_policy.json](capacity_policy.json).
Factorial LMC/IMC/FSDE compare `(M,Q)` in `(16,4)`, `(32,4)`, `(50,4)`,
`(32,8)`, `(32,16)`, then select online posterior steps from `5,25,100`.
OHSVGP tests `M={32,64}` and `RFF={64,128,256}` after two unmodified official
gates. OVC selects its own `4x32`, `8x32`, `12x32` grid and must pass two
clean-process retention audits. ST-SVGP tests `Ms={16,32,52}` and is described
as a causal refit adaptation, not online posterior transfer.

## Formal lock and cloud run

Before a repaired method enters formal seed 5--9 execution, create a frozen
hash manifest of the old archives, capture the isolated environments, and run
the 4090 plus 120 GiB RAM preflight. High-RSS methods are intentionally run
serially.

```bash
python baselines/covid_long_setting_b/snapshot_frozen_archives.py
python baselines/covid_long_setting_b/preflight_4090_node.py
python baselines/covid_long_setting_b/capture_environment_locks.py --environment ohsvgp=/env/ohsvgp/bin/python --environment ovc=/env/ovc/bin/python --environment st_svgp=/env/st/bin/python --environment factorial_gpflow=/env/gpflow/bin/python --environment factorial_fsde=/env/fsde/bin/python
python baselines/covid_long_setting_b/assess_ovc_memory_audits.py --audit-root <selected-ovc-audits> --output <assessment.json>
python baselines/covid_long_setting_b/generate_baseline_fairness_protocol.py --ovc-memory-assessment <assessment.json> --environment-lock <environment-lock.json> --hardware-fingerprint <cloud_4090_preflight.json>
python baselines/covid_long_setting_b/run_locked_formal.py
python baselines/covid_long_setting_b/run_locked_formal.py --execute
```

`BASELINE_FAIRNESS_PROTOCOL.json` contains the selected configurations, source
commits, development records, precision, causal requirements, environment
locks and hardware fingerprint. The locked runner rejects altered locks,
changed frozen archives, non-passing gates, non-4090 hardware records, unknown
output roots and archive overwrites. It writes all new archives into a fresh
`formal_repaired_4090_v1` root.

## Layout

```text
catalog.json                       method provenance and status
protocol.py                        read-only causal observations
archive.py                         common prediction archive and causal audit
development.py                     blocked fold construction and scoring helpers
run_blocked_development.py         seed-0 capacity/online-step selection
preflight_4090_node.py             cloud hardware gate
generate_baseline_fairness_protocol.py
run_locked_formal.py               formal seeds only after the lock
```

The catalog is the machine-readable source of truth. A downloaded official
repository, a passing official reproduction, a strict-online adaptation and a
final comparable formal result are deliberately distinct states.
