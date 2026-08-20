# AutoDL ERA5 GPU-only RTX 4090 comparison

- Run label: rtx4090_gpu_only_published_20260820T145100Z
- Code commit: 071484f832b49b01c74742a2881aa69d711861b5
- Hardware: NVIDIA GeForce RTX 4090
- Verified scope: shared batch and short/long online GPU rows.
- Markovflow is excluded because the official TensorFlow 2.4 stack failed at cusolverDnCreate on this RTX 4090.
- Official long-batch ST-VGP/ST-SVGP rows are explicit RTX 4090 OOM exclusions: Ms=32 alone requested 52.11 GiB.
- CPU preparation, CPU X-lag, CPU postprocessing, and GPflow capacity preflights are excluded.
- No CPU-to-GPU or cross-device FLOP ratio is asserted.
