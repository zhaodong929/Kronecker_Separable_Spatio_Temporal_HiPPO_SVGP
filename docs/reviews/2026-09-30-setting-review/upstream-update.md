# GitHub追加資料の確認（2026-09-30）

## 取得した版

リモート全branchを取得し、`origin/main` の `7da38bf`（paper reproducibility code and experiment protocols）、`1bf0a4d68d4279022c843ed1c65cadb3c749ed88`（cross-domain ablation evidence）を確認した。更新は `REPRODUCIBILITY.md` と `reproducibility/` の追加であり、main直下の既存実装を置き換えるcommitではない。

元workspaceの追跡branch `codex/autodl-era5-benchmark` は `git pull --ff-only` で0398b28→3910b14へ更新。ここでの18commitは8月までの既存更新であり、今回のmain追加とは別。従来fetch refspecがこの1branchに限定されていたため、今回は明示的に全headsをfetchした。

最新mainは独立worktree `/data/nk523/projects/hipposvgp-upstream-review-20260930` にcheckout。比較worktree `/data/nk523/projects/hipposvgp-fair-comparison` の科学コード・DoCジョブは変更していない。取得時点で比較branchのリモートに2fb69aaより新しいcommitはなかった。

ユーザー補足：今回のレビュー結果は実装者に伝えていない。したがって、この公開をこちらの指摘への修正として評価しない。目的は、未入手だった掲載実験のコード・設定・結果対応を補うこと。

## 新しく分かったこと

| 新資料 | 内容とレビューへの影響 |
|---|---|
| `reproducibility/EXPERIMENT_INDEX.md` | 3設定の設定manifest、掲載集計、診断・除外の位置付けを提示。掲載結果の所在が以前より追いやすくなった |
| `protocols/era5/`、`code/scripts_snapshot/regenerate_era5_land_tasks.py` | 原本CDSの35変数・2020年1〜3月・1860時間、Task3〜10再生成、Task1/2保存、初期scalerと分割seedを記録。ERA5長期データ自体は未同梱 |
| `protocols/pems/traffic_external_gp_a100_v1.json` | 掲載PEMSは2016初期＋50100逐次、seeds1/2/3、1サンプル遅延。旧Buiは固定kernel、旧STは12時点窓の因果的refit・zero mean。今回のadaptive Bui／共通平均付き状態継続STとは異なる |
| `results/pems/external_gp_paper_ready/complete_subset/` | OHSVGP・Bui各3分割の集計とshape/truth/delay等の保存済み監査。新たにraw予測を再監査したわけではない |
| `results/ablations/` | joint/zero-cross/fixed-global、coordinate transfer、dense/structured solverの既存証拠を集約。追加実験ではなく、本文由来の報告CSVという注記。今回新たにablationを走らせる必要を意味しない |
| `paper/main.tex` / `references.bib` | 抽象の広いO(M6)→O(M3)記述が外れ、evolving memory中心へ変更。平均＋GPにuniversal krigingの説明とCressie1993/Stein1999引用を追加。本文の定理適用と実験対応は依然別に確認が必要 |

READMEは61ec3ed由来のsnapshotと説明しているが、完全に同一ではない。Pythonソースを比較するとscripts_snapshotは61ファイル同一・14変更・163新規、stvgp_kroneckerは24同一・6変更・4新規。既存commit名だけで同じ実装と認定せず、追加snapshotを独立のsourceとして扱う。MGPVAE実装は追加資料に見当たらない。

## 掲載版の観測配置処理に数値的な問題を確認

新snapshotの `code/stvgp_kronecker/joint_ssgp_kron/torch_backend.py:496`〜532 は、履歴と現在の時間因子Bを加算してから、**全履歴に今回の `C_observed.T @ C_observed` を掛ける**。観測地点が変わる場合、必要な `B_old⊗G_old + B_new⊗G_new` と一般に一致しない。

`code/scripts_snapshot/run_traffic_routeb.py:597`/`:609` は、同じstateにhidden配置とvisible配置を順に渡す。`run_iclr_era5_routeb_strict_online.py:877`付近にもこの経路がある。固定配置では成立する構造を、異なる配置にそのまま使う箇所がある。

取得したsnapshotそのものをCPUで実行した最小再現（CUDA無効、2threads、float64）：事前分散1、観測noise1、時間因子1、平均次元0、観測係数C=1 then2、両観測y=1、jitter1e-12。

- 正しい精度は `1 + 1² + 2² = 6`、自然平均は3、posterior meanは **0.5**。
- snapshotの出力は **0.33333333333333337**。B=2、最新G=4で、精度を約9としている。

したがって、**公開されたコードの異なる観測配置を跨ぐ更新は、同じ有限Gaussianモデルの正しい更新になっていない**。これは単なる理論の言い換えでは解消しない。今回の比較branchは `multi_geometry.py` で配置ごとの因子を保持する別実装であり、この再現で現campaignを無効と判断しない。

ただし、掲載スカラー全てがこのsnapshotの全く同じコードから生成されたことまでは、raw予測・checkpoint・runごとのsource hashが不足しているため認定できない。「掲載版のこの経路に問題を確認、掲載数値への範囲はrun provenance照合が必要」とする。旧PEMS数値を正しい比較結果としてそのまま移植しない。

再現方法はsnapshotを最優先importした上で `TorchJointSSGPKronHiPPOSVGP` を上記条件で構築し、`update_block_structured_joint_ssgp_transfer(..., C_observed=[[c]])` をc=1,2の順に呼ぶ。最初の試行はrepo直下の古い同名packageが優先されimport失敗した。再試行では `sys.path.insert(0, 'reproducibility/code')` を明示し、追加snapshotを実行した。

## 前回指摘との対応・追加の不整合

1. **OSGPR振幅**：snapshotの `make_kernel`（62〜79行）は3振幅をadaptive時にtrainableとし、`theta_from_model`（105〜112行）は時間側のみ保存。同じ不具合が残る。ただしmanifestにある旧PEMSのfrozen設定にはこのadaptive振幅消失を自動適用しない。旧COVID adaptiveへの適用は個別run source対応を確認する。
2. **PEMS正規化**：追加 `data/traffic.py:400`〜402は初期全センサーでmean/stdをfitする。今回のfit地点限定の前処理とは異なり、旧スコアをそのまま新表へ持ち込めない。naive時刻差分の使用も406行に残る。
3. **ERA5時刻**：Task1/2検証記録は `Dataset.to_dataframe().dropna()` に合わせた地点ごとの行詰めを明記し、372時間中2地点のcompactionを認める。この意味でrawと最大差0であり、正規UTCの全地点同時刻一致を意味しない。以前確認した全期間6地点のずれと矛盾しない。新資料は原処理の再現であり、今回のUTC修復の代替ではない。
4. **COVID反復数**：`results/ablations/cross_domain_main_table.csv` はLMC/ICM/FSDEをruns=5とする一方、`results/covid/reorganized_results_20260821/formal_results_table.csv` とsource_manifestは同じ数値を3反復・exploratory_gpu_4090_s5_s7由来と明記。掲載表の5反復表記を裏付ける追加証拠にはならない。これらの方式は今回の再実行範囲外のまま。
5. **紙面版の区別**：追加paperのappendix.texは以前取得したOverleaf版とbyte単位で一致。main.texとreferences.bibは異なる。抽象の変更は一部の過大な読み方を弱めるが、実験節の旧条件を更新した証拠ではない。
6. **配布整合性**：SOURCE_MANIFESTの935件中573件が存在し全てhash一致。欠落362件は329個のpycと33個のZone.Identifier。科学ソース破損を示すものではないが、配布物だけではmanifest全件検証が通らない。またEXPERIMENT_INDEXが参照する `results/excluded/OFFICIAL_ST_SVGP_ALL_VERSIONS_AUDIT_20260828.md` は未同梱。

## 結論と既存レビューへの反映

掲載実験の再構築に役立つ新情報が多数ある。とくに旧PEMSの固定Bui／窓付きST、ERA5の生成・行詰め手順、原稿更新、既存機構実験の対応が具体化した。一方、これはDoCで進めている5方式65本の完成結果・修正版ではない。

前回レビューの修正優先順位は維持する。加えて旧掲載版の観測配置処理に数値不一致を確認したため、旧COVID/PEMSの提案法・機構実験はsource/run対応とこの点を確認するまで正しさの参照結果に使わない。新しい平均＋GPの文献説明はモデル選択の背景として使えるが、比較実験の代わりにはならない。

本調査では新モデル学習・既存ジョブの再開・原稿変更は行っていない。保存済み集計の読解、ソースdiff、hash、有限のCPU数値例を確認した範囲であり、追加snapshot全体の完全な再現試験ではない。
