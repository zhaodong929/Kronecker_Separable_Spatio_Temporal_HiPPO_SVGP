# Cloud Handoff

## Source

- Prepared cloud source commit: `57b0b2d862acd25b09cb0ef676da6899a7fba246`.
- The local worktree contains unrelated changes. The cloud checkout must remain on `codex/covid-formal-paper-materials` with this commit in its history; verify it with `git rev-parse HEAD` before any experiment.

## Completed

- Three seed-0 blocked validation windows and Gaussian-NLPD selection protocol are implemented.
- Pre-repair seed-5--9 archives are frozen in `baselines/covid_long_setting_b/reproduction/convergence_repair_v1/frozen_pre_repair_archives.json`.
- Registry tests pass. OHSVGP, OVC, ST-SVGP, LMC, IMC, and FSDE-SVI are `validation_pending`.

## Not Completed

- Cloud hardware preflight; official OHSVGP gates; seed-0 capacity/convergence and online-step selection; OVC memory audit; fairness lock; fresh formal seeds 5--9.

## First Command

```bash
python baselines/covid_long_setting_b/preflight_4090_node.py
```

PASS: one RTX 4090, at least 120 GiB host RAM, and no active compute process. Output: `baselines/covid_long_setting_b/results/convergence_repair_v1/cloud_4090_preflight.json`.

## Next Commands

```bash
# OHSVGP: official gate, then seed-0 capacity selection
python baselines/covid_long_setting_b/run_ohsvgp_reproduction_gates.py --python <ohsvgp-python> --execute
python baselines/covid_long_setting_b/run_blocked_development.py --phase capacity --methods ohsvgp --execute --resume

# Factorial family: capacity first, then online posterior steps
python baselines/covid_long_setting_b/run_blocked_development.py --phase capacity --methods lmc imc fsde --execute --resume
python baselines/covid_long_setting_b/run_blocked_development.py --phase online_steps --methods lmc imc fsde --execute

# Causal-refit ST-SVGP and OVC capacity selection
python baselines/covid_long_setting_b/run_blocked_development.py --phase capacity --methods st_svgp --execute --resume
python baselines/covid_long_setting_b/run_blocked_development.py --phase capacity --methods ovc --execute --resume

# OVC: replace <Mt> with the selected temporal-inducing count
python baselines/covid_long_setting_b/run_ovc_memory_audit.py --output-dir <ovc-audit-root> --replicate-id 1 --temporal-inducing <Mt>
python baselines/covid_long_setting_b/run_ovc_memory_audit.py --output-dir <ovc-audit-root> --replicate-id 2 --temporal-inducing <Mt>
python baselines/covid_long_setting_b/assess_ovc_memory_audits.py --audit-root <ovc-audit-root> --output <ovc-assessment.json>

# Lock environments and the selected protocol
python baselines/covid_long_setting_b/capture_environment_locks.py --environment ohsvgp=<path> --environment ovc=<path> --environment st_svgp=<path> --environment factorial_gpflow=<path> --environment factorial_fsde=<path>
python baselines/covid_long_setting_b/generate_baseline_fairness_protocol.py --ovc-memory-assessment <ovc-assessment.json> --environment-lock <environment-lock.json> --hardware-fingerprint baselines/covid_long_setting_b/results/convergence_repair_v1/cloud_4090_preflight.json

# Locked methods only; this runner fixes seeds 5 6 7
python baselines/covid_long_setting_b/run_locked_formal.py --execute
```

## Outputs And Gates

- Development: `baselines/covid_long_setting_b/results/convergence_repair_v1/blocked_development/`.
- OHSVGP gate: `baselines/covid_long_setting_b/reproduction/convergence_repair_v1/ohsvgp/`.
- Fairness lock: `baselines/covid_long_setting_b/BASELINE_FAIRNESS_PROTOCOL.json`.
- Formal results: `baselines/covid_long_setting_b/results/formal_repaired_4090_v1/`.
- PASS: objective plateau; finite positive variances; zero current-hidden reads; exactly one delayed-label absorption. OHSVGP requires both official examples; OVC requires both clean-process memory audits; each admitted formal method requires three complete `(143, 10)` archives for seeds 5, 6, 7.
- FAIL/PENDING: do not run formal seeds; retain the method and its evidence in the appendix only.
