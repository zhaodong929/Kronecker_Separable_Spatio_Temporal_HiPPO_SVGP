# Setting B Baseline Completion Audit

This audit distinguishes completed comparable COVID results from explicitly
excluded or resource-limited methods. It does not reinterpret an unavailable
method as a poor-performing one.

## Fixed protocol

- Data: 52 jurisdictions, 195 weekly observations.
- Task 1: first 52 weeks only for standardisation and capacity selection.
- Formal splits: seeds 5--9; 143 online weeks per split.
- Legal week `t` information: delayed hidden labels from `t-1`, then the 42
  current visible labels; the 10 current hidden labels are queried only after
  prediction.
- Formal metrics: RMSE, CRPS, Gaussian NLPD, ECE and Coverage90, all evaluated
  from the same `predictions.npz` archive contract.

The protocol implementation and archive audit are in
`baselines/covid_long_setting_b/protocol.py` and
`baselines/covid_long_setting_b/archive.py`.

## Capacity selection

All capacity decisions use seed 0 and Task 1 only. Formal seeds are not used
to select a capacity.

| Family | Selection rule | Selected configuration | Evidence |
|---|---|---|---|
| Route B ordinary vs cumulative HiPPO | Exact semantic match | `Mt=32, Ms=32` for both | `capacity_policy.json` |
| ST-SVGP | Spatial inducing count only | `Ms=32` | `results/task1_capacity_selection/task1_capacity_selection.json` |
| Bui OSGPR and OVC-SVGP | Shared Cartesian point-inducing grid | `Mt=8, Ms=32` | `results/task1_capacity_selection/shared_bui_ovc_grid_selection.json` |
| LMC-SVGP, IMC-SVGP and FSDE-SVI | Shared temporal inducing count and latent rank | `M=4, Q=2` | `results/task1_capacity_selection_factorial/task1_capacity_selection.json` |

The lower-capacity `4x32` OVC archive is a non-primary feasibility result. It
is not substituted for OVC in the strict primary table; its actual Gaussian
metrics are retained separately in `ovc_resource_bounded_feasibility.md`.

## Formal results

The following methods each have five finite `(143, 10)` prediction archives
and a completed row in `aggregate_metrics.csv`:

1. Persistence
2. Task-1 lag ridge
3. OHSVGP (RBF)
4. Route B ordinary inducing
5. Route B cumulative HiPPO
6. Bui OSGPR (controlled)
7. Bui OSGPR (adaptive, CPU)
8. ST-SVGP
9. LMC-SVGP
10. IMC-SVGP
11. FSDE-SVI

The common evaluator recorded no missing archive in
`incomplete_archives.csv`.

## Non-numeric rows

| Method | Status | Evidence | Interpretation |
|---|---|---|---|
| OVC-SVGP | Resource-limited at its Task-1-selected `8x32` capacity | `reproduction/ovc_shared_m8_formal_resource_guard.json` | The official exact-fantasy seed-5 run reached 10.46 GiB RSS after 1,731 seconds without an archive. The completed `4x32` archive is not used as a primary result. |
| EARTH | Protocol incompatible without a model-core rewrite | `reproduction/earth_setting_b_incompatibility.json` | Official EARTH has neither a current-week node mask nor predictive variance. |

## Official-source provenance

| Family | Upstream gate | Evidence |
|---|---|---|
| ST-SVGP | Unmodified official Air Quality example completed, then causal Setting B adapter completed | `reproduction/st_svgp_official_run.json` |
| OVC-SVGP | Official fantasy-conditioning test passed; causal adapter passed seed-0 gates | `reproduction/ovc_official_fantasy_test.log`, `reproduction/ovc_adapter_seed0_diagnostic39.json` |
| LMC/IMC/FSDE | Causal adapters call pinned FactorialSDE cores and completed formal archives | `catalog.json`, `results/formal_selected_factorial`, `results/formal_selected_fsde_svi_isolated` |
| FactorialSDE upstream county-level COVID entries | Attempted but not represented as a completed reproduction because of independently recorded environment/snapshot failures | `reproduction/factorial_sde_official_entrypoint_status.json` |

The FactorialSDE Setting B rows must therefore be described as **official-core
adapters**, not as scores reproduced from the repository's separate
county-level COVID script.
