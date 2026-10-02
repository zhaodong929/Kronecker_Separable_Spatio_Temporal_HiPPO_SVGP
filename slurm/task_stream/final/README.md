# Fixed-work final comparison

This is the 65-run ERA5/COVID/PEMS comparison, with five methods and 13 spatial
splits. The numerical release remains `ea54966e3790bdf33114605048065ac0e98f68c8`,
exactly as in selection; orchestration lives outside that immutable snapshot.

`campaign.py build` requires every group and all three declared candidates,
verifies the selected result, configuration, feature identity, input hashes and
fixed refit plan against the actual prepared geometry. It also binds the existing
CPU/GPU numerical and full-size shape qualification evidence. These checks do
not certify convergence or admit results to a paper table.

`run.sbatch` submits three A30 workers, with 8 CPUs and 60GB RAM each, for 48h.
All 65 cases reside in a frozen manifest. Each has independent attempts, W&B
logging, selected-work provenance, outputs and failure records. A failed child
does not omit other cases. A restarted worker reuses only artifacts verified
against the same selection, data, code and fixed refit plan. Numerical workers
use `--stage final --selection ...`, with no task truncation or new time cap.

Deployment: `/vol/bitbucket/nk523/hipposvgp-fair-20260929/studies/final65-20261002`.
W&B: `harrisonzhu/KronHiPPO-STGP`.

The first submission, 294854, was rejected by the W&B supervisor before fitting:
the launcher omitted the qualification-record field. It was stopped and repaired
by linking the actual prior qualification evidence, not by bypassing the guard.
Attempt logs and `state-startup-294854` preserve that history. Submission IDs and
current progress live in the campaign record and Obsidian notes.
