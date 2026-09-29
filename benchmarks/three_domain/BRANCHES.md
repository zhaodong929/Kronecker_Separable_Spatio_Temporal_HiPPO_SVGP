# Branch provenance

Active work: `codex/fair-three-domain-comparison`.
Integration base: `codex/pems-a100-existing-gp` at
`61ec3edecbc6a671a664ea1ec4026bdcc7c69798`.

The historical ERA5 benchmark/results, COVID results/materials, and profiler
branches remain unchanged. Their experiments are historical evidence and are
not implicitly admitted to this campaign. The base already contains imported
COVID exploratory results and the RTX4090 ERA5 archive.

Keep official model repositories outside the tracked tree, pinned by the
existing `cloud/autodl_era5/clone_official_baselines.sh`. MGPVAE is pinned in
`baselines/mgpvae/official.py`. Do not merge generated data, environments, old
result bundles, or authentication configuration into this branch.
