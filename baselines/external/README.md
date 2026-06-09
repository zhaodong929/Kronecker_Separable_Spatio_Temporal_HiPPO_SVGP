# External online GP baseline sources

This folder contains reference implementations collected for the ERA5 Route B
baseline plan. These repositories are kept as provenance/reference material until
they are adapted to the local `OnlineBaseline` interface and ERA5 held-out
seen-history protocol.

| Folder | Source | Intended role | Current integration status |
| --- | --- | --- | --- |
| `thangbui_streaming_sparse_gp` | https://github.com/thangbui/streaming_sparse_gp | OSVGP / OSGPR streaming sparse GP reference | Downloaded; not yet wrapped in the unified ERA5 runner |
| `wjmaddox_online_vargp` | https://github.com/wjmaddox/online_vargp | Online variational conditioning, OVC fixed/optimized inducing point reference | Downloaded; not yet wrapped in the unified ERA5 runner |
| `wjmaddox_online_gp` | https://github.com/wjmaddox/online_gp | Online GP / WISKI / online SGPR-SVGP reference | Downloaded; not yet wrapped in the unified ERA5 runner |
| `nkiyohara_HIPPOSVGP` | https://github.com/nkiyohara/HIPPOSVGP | OHSVGP-style HiPPO temporal baseline reference | Clone attempted; local folder exists but may need integrity check before use |

The immediately runnable fair baselines remain the local GPyTorch SGPR/SVGP
wrappers in `baselines/online_baselines.py`. Those wrappers now support:

- Matérn-3/2 kernels;
- Rich-Φ and lag-Φ covariates via the ERA5 loader/runner;
- task-1 full-GP MLL grid hyperparameter selection;
- frozen kernel/noise hyperparameters during task-2 online evaluation.

External online GP repositories should only be reported as **fully integrated**
after they implement the same local interface:

```python
fit_initial_task(...)
update_block(...)
predict(...)
name(...)
```

and are evaluated with the same task split, location subset, standardization,
block split, metrics, and held-out seen-history forgetting definition.
