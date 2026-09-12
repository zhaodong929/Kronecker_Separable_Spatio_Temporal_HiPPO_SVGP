# PEMS-BAY Existing-GP A100 Run

This array adds OHSVGP, Maddox StreamingSGPR, Bui OSGPR and ST-SVGP to the
locked three-split Protocol-N comparison. Each task receives the same 2016-step
Task 1, 260/65 spatial split and delayed-hidden/current-visible information
order. Hyperparameters are never selected from the 50,100-step formal stream.

OHSVGP, Maddox and Bui carry a posterior state. The available official
ST-SVGP API instead performs a causal refit; the traffic adapter predeclares a
12-step history window and one inference update so it can finish on this long
5-minute stream. It must be reported as `ST-SVGP (causal refit)`, not as online
posterior transfer.

`submit_external_gp.sh` launches a three-task, 24-hour A100 array for OHSVGP,
Maddox and Bui, plus a separate 72-hour ST-SVGP task. This keeps the fast jobs
from inheriting ST-SVGP's larger queue request while remaining within the
cluster QOS limit. Each task smoke-tests all three seeds and then runs those
seeds formally in sequence. A CPU summary job starts only after all four
methods succeed.
