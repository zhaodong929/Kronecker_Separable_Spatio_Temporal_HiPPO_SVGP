# 2026-09-30 task-pipeline implementation and verification

The user approved task-level evaluation while retaining observation-wise baseline algorithms, requested all debug fixes, fair timing/FLOP accounting, contribution ablation code and thorough W&B logging. The latter request supersedes the earlier no-ablation scope; the five-method baseline policy itself remains unchanged.

Implemented code and exact contracts: [task_stream README](../../../benchmarks/task_stream/README.md).

## Verified before device deployment

- Final combined CPU suite: **85 passed**, including task/data/Torch adapter/factory/checkpoint/logging/reporting/selection, structured algebra and existing temporal gradient/empirical-Bayes/backend/lineage regressions.
- 5 ST/MGP official/independent-prefix task tests and 2 Markov initial-fit factory tests passed on CPU.
- 1 Bui OSGPR test passed on DoC corgi, actual TensorFlow/GPflow environment, two CPU threads: initial fit plus three online updates and kernel round-trip.
- 22 initial-period selection/provenance guard tests passed.
- Real preparation passed for ERA5 seed0 (64 tasks/133 features), COVID seed5 (143/13), PEMS seed1 (672/28). All 13 persistent prepared splits are being generated; PEMS source is now the original raw HDF, independently SHA256 matched on local and DoC copies.
- End-to-end tiny Kron CLI validation wrote predictions/checkpoints and selected its candidate successfully. No test result is admitted to the paper table.

Found and fixed while testing: Matérn chi-square draws were using global RNG despite a model seed; OH needed its own isolated official ELBO MC RNG; ST factory had to apply declared initial parameters; frozen selection needed source-content and feature hashes rather than only a commit ID; ablation fit identity needed actual fitted parameters/data/random features rather than only a config hash.

## Operational state

Authorized cancellation removed old jobs `294499_3`, `294532_1`, `294549`, `294547`, `294543`, `294542`, `294535`. Their outputs remain preserved. A subsequent DoC squeue query returned no jobs for nk523. This is a reset for the new protocol, not completion of the old 65-run experiment campaign.

New GPU qualification uses an immutable source release, one A30 GPU per method and at most three simultaneous tasks. CPU/GPU independently fitted stochastic OH trajectories are not an equivalence criterion; its fixed-state prediction is compared separately. Two fitting iterations plus two tasks test execution/shape/numerics only. They do not establish convergence or the full experiment duration.

## Remaining scientific admission requirements

Device qualification and real-size validation/refit resource/convergence checks; a frozen declared candidate budget and selected configurations for all splits; all 65 main runs; paired native-scale summaries; and independently measured computation/profiling. No full-run completion or performance ranking is claimed. Checkpoint files are inspectable but automatic restoration has not been qualified. The existing full-size MGP GPU concerns remain a gate until tested under the new source and geometry.

## Device results and follow-up

All five methods completed the A30 qualification array **294564** on source `fcff705`. Four independently fitted deterministic/reference-compatible trajectories matched CPU; OH matched a common fixed learned posterior (its independent stochastic optimizer paths are not an equivalence criterion). Every W&B supervisor exited zero. Exact errors and links: [device-qualification.json](device-qualification.json).

Subsequent changes: ERA calendar columns use physical 24/168-hour cycles (same 133 features, weather transforms unchanged); explicit NVTX acquisition/counter ingestion and profiled-latency exclusion; compact MGP training RTS avoids time-indexed full spatial covariance while matching the corrected objective and every parameter gradient. CPU combined suite **91 passed**; compact-MGP plus Markov factories **7 passed**, in addition to the five earlier Markov reference tests and remote OSGPR test. The changed MGP training path requires a fresh tiny GPU check before real-size domain qualification.

`qualify_domains.sbatch` runs the three representative domains sequentially per method, keeping at most five queued array elements and three GPUs globally. All 15 domain/method shape pilots use two initial fitting steps and two tasks; this is not convergence or a new final result.

### 全13分割の準備・監査完了

`outputs/task-stream-20260930/preparation-summary.json` に全13分割・6,291,293,687 bytesの成果物を記録。SHA256: `a58ba603c203bb9c78162fedbea1b7ab26f77e22e746fff41921e2fe7f577682`。全特徴の有限性、bias=1、fit-only標準化、内部selectionとの完全なprefix一致、ERAの24/168時間周期を検証済み。ERA5は5×64タスク、COVIDは5×143、PEMSは3×672。旧位相版は別ディレクトリへ保存し不使用。全分割をDoCへ転送中。

### 実規模pilotで判明したメモリ問題

`294570` のコンパクトMGP CPU/GPU再照合は成功。`294573` はCOVID全5方式、PEMSのKron/OH/MGP、ERAのKron/OHが実規模2-step/2-task検証を完了。PEMS OSGPRは初期SGPRの128×1,048,320行の中間行列、STは4032×260×260行列の複数保持でA30メモリ不足。両方とも初期学習中で、オンラインタスク開始前。失敗成果物は保存し、成功扱いにしない。ERAの当該2方式は元のworkerが停止したため未実行。独立した後続domain検証は継続し、workerの最終終了コードは失敗を保つようdispatcherを修正。

目的関数・データ量を変えずに、OSGPRの加法的十分統計を行チャンクで計算し、STの反復する空間行列を保持しない実装を検証中。小規模dense参照の損失・全勾配・予測との照合を通すまでは実規模再投入しない。本番65本は依然未開始。

### Exact bounded-memory initialization repairs

Initial SGPR now accumulates full-data sufficient statistics in 2048-row chunks, with a recomputing custom VJP. Dense GPflow objective/all-gradient/prediction/two-Adam-step comparisons passed **7 tests** in the actual DoC TensorFlow environment. ST Gaussian training now stores scalar observation precision and a single fixed-geometry spatial projection instead of time-indexed observation-space covariance matrices. Official posterior/objective/all-gradient and two-Adam-step plus delayed-task prediction comparisons passed **4 tests**, and the switched ST factory passed its integration test. Both preserve the full initial observations and original objective; upstream source is unchanged. GPU requalification is required next.

The earlier compact MGP completed all three real-size shape pilots. All13prepared splits/39artifacts are now on DoC with transfer verification. NCU was not found on the accessible scheduler/corgi environments; corgi restricts hardware counters. A30-specific availability is recorded inside allocated qualification workers. Actual FLOP counter acquisition remains unqualified and absent counters remain NA.
