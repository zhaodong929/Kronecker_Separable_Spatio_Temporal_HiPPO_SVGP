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

`submit_external_gp.sh` launches a 12-task smoke array and a matching formal
array with `aftercorr`, so each seed/method begins as soon as its own smoke
passes. A CPU summary job runs only after all formal tasks succeed.
