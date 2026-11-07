# 复现指南（README）

本目录是一次完整实验的可复现交付物：**代码 + 训练好的权重 + 逐图分数 + 结果表**。
别人拿到后只需自备 4 份数据集，就能复算报告里的每一个数字；即使不重训，也能用附带权重直接复现评测与打分。

---

## 1. 目录结构

```
deepfake多生成器实验报告/
├── README.md                  ← 本文件（复现步骤）
├── 实验报告.md                 ← 实验结论与分析
├── requirements.txt            ← 依赖（含版本配对说明）
├── 代码/                       ← 30 个脚本，全部可独立运行
│   ├── paths.py               ← 数据集路径解析（环境变量驱动，不用改源码）
│   ├── verify_env.py          ← 环境自检 + 数值复现校验（先跑这个）
│   ├── build_manifest.py      ← 扫描四数据集 → manifest.csv
│   ├── make_split.py          ← group-aware 切分 → split_{train,val,test}.csv
│   ├── train_multi.py         ← 6 个训练配置（--domains gan|swap|diff）
│   ├── train_own.py           ← 单数据集 120k 训练（own_resnet50）
│   ├── run_experiments.sh     ← 一次跑完 m1~m6
│   ├── eval_cross.py          ← 6 模型 × 6 格评测（--dump-scores 出逐图分数）
│   ├── match_eval.py          ← 画质（锐度）匹配后重算 AUC
│   ├── bootstrap_ci.py        ← bootstrap 95% CI
│   ├── format_ablation.py     ← 格式消融（原始编码 vs 统一重编码）
│   ├── gap_report.py          ← 同域上限 vs 跨域最佳
│   ├── summarize_cross.py     ← 汇总表
│   ├── score_files.py         ← 任意图片打分（中文路径安全）
│   ├── audit_manifest.py / dup_scan*.py / near_dup_verify.py / leak_audit.py
│   │                          ← 查重、近重复、泄漏审计
│   └── fetch_models.py / batch_detect.py / run_batch.py / calibrate.py / make_label_page.py
│                              ← 早期 HuggingFace 通用模型流水线（本实验未采用，留档）
├── models/                    ← 权重 + SHA256SUMS.txt
│   ├── cnndetection/blur_jpg_prob0.5.pth   ← 所有训练的起点（CNNDetection 官方权重）
│   ├── m1_gan … m6_all/best.pt             ← 6 个训练配置
│   └── own_resnet50/best.pt                ← 120k 单数据集模型
├── 结果数据/                   ← 归档结果（可逐项对照复算）
│   ├── manifest_stats.csv      ← 四数据集实际规模（复现时先比这个）
│   ├── split_counts.csv        ← train/val/test 三份划分的条目数
│   ├── cross_eval_v2.csv       ← 6 模型 × 6 格原始指标
│   ├── matched_auc.csv         ← 画质匹配后 AUC
│   ├── auc_ci.csv              ← bootstrap CI（auc_full 与 CI 同源）
│   ├── generalization_gap.csv  ← 同域上限 vs 跨域最佳
│   ├── cross_summary.csv       ← 同域/跨域均值、最差格
│   ├── format_ablation_m6_all.csv
│   └── scores_*.csv            ← 36 份逐图分数（每格每模型的原始预测）
└── 样例/                       ← 5 张图 + expected.csv + smoke_result.csv
```

## 2. 环境

```bash
python -m venv .venv
# torch 与 torchvision 必须成对：2.9.x 只配 0.24.x，装 0.26 会在 torchvision.ops.nms 报 symbol 缺失
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

- 实测环境：Python 3.13 / torch 2.9.1+cu128 / torchvision 0.24.1+cu128 / RTX 5060 Laptop 8GB (sm_120)。
- 显存：训练 batch 48 + 256px + ResNet50 ≈ 4.5GB，8GB 卡够用；推理 batch 1 即可。
- 只用 CPU 也能跑推理（`score_files.py`、`verify_env.py`），训练不建议。
- 国内装 cu128 轮子：阿里云 `pytorch-wheels` 镜像需给 whl 直链，`-i` 不认 `+cu128` 后缀。

## 3. 数据准备

脚本不写死路径，读环境变量（`代码/paths.py`）：

| 环境变量 | 数据集 | 需要的目录结构 | 本机条数（伪/真） |
|---|---|---|---|
| `DATA_RVF` | real-vs-fake 120k | `real/*.jpg`、`fake/*.jpg`，256×256 对齐脸 | 60000 / 60000 |
| `DATA_HF140K` | 140k-Real-and-Fake | `{train,validation,test}/{real,fake}/*.jpg` | 70000 / 70000 |
| `DATA_DF40` | DF40 | `deepfacelab/{real,fake}/frames/<clip>/*`、`faceswap/{ff,cdf}/frames/<clip>/*`、`simswap/…`、`DiT/…` | 见 manifest_stats |
| `DATA_CELEBDF` | Celeb-DF（aligned） | `real/*.png`、`fake/*.png` | 1984 / 1216 |

```bash
# Windows PowerShell
$env:DATA_RVF     = "D:/data/real-vs-fake"
$env:DATA_HF140K  = "D:/data/140k-Real-and-Fake-Faces/images"
$env:DATA_DF40    = "D:/data/DF40"
$env:DATA_CELEBDF = "D:/data/CelebDF_test_aligned"

# 检查是否都被识别
python 代码/paths.py
```

获取渠道：

- **140k Real and Fake Faces** — Kaggle `xhlulu/140k-real-and-fake-faces`。注意官方包目录是 `train/valid/test`，本脚本按 `train/validation/test` 读，需把 `valid` 改名或建软链。
- **DF40** — https://github.com/YKYH/DF40 （faceswap / simswap / deepfacelab / DiT 四类，`frames/<clip>/` 组织）。
- **Celeb-DF (v2)** — https://github.com/yuezunli/celeb-deepfakeforensics ，取对齐帧版本（`CelebDF_test_aligned`）。
- **real-vs-fake 120k** — 本机来自竞赛/主办方提供（real 为 FFHQ 重编号 `00000.jpg`，fake 为 GAN 随机串命名）。公开等价物可用 FFHQ + StyleGAN 自建，或 Kaggle 上同名 `real-vs-fake` 数据集；**换源后指标会变**，此时只能对照趋势，不能对照具体数值。

数据规模校验：跑完 `build_manifest.py` 后，与 `结果数据/manifest_stats.csv` 对比每行 count。数字对不上说明数据版本不同，后续 AUC 会有偏移。

## 4. 四步验证

**① 不碰数据，先验环境与权重（1 分钟）**

```bash
cd <本目录>
python 代码/verify_env.py
#   期望末行：通过。   且 “漂移文件数 0/5”
```

它会重算 `样例/` 五张图（m6_all / m2_swap / own_resnet50，raw 与统一重编码两路平均），
与 `样例/expected.csv` 的参考分数逐项比对，容差 0.02。通过即说明权重、依赖、后处理管线都对。

**② 建索引与切分（需要数据，350k 张扫描约 10~20 分钟）**

```bash
python 代码/build_manifest.py --out manifest.csv     # → manifest.csv
python 代码/make_split.py --manifest manifest.csv --outdir .   # → split_{train,val,test}.csv
```

切分是确定性哈希，同数据同版本必然复现同一份划分。跑完先对照：
`结果数据/manifest_stats.csv`（数据集规模）与 `结果数据/split_counts.csv`（三份划分的条目数）。
数字不一致说明数据版本不同，后面的 AUC 会有偏移——此时只能比趋势，不能比绝对值。

**③ 用附带权重复算 6 模型 × 6 格评测**

```bash
python 代码/eval_cross.py --models m1_gan,m2_swap,m3_diff,m4_gan_swap,m5_gan_diff,m6_all \
       --cells all --per-class 1000 --dump-scores --out cross_eval_v2.csv
```

**④ 逐项生成对照表**

```bash
python 代码/match_eval.py        # → matched_auc.csv
python 代码/bootstrap_ci.py      # → auc_ci.csv
python 代码/gap_report.py        # → generalization_gap.csv
python 代码/summarize_cross.py   # → cross_summary.csv
diff matched_auc.csv 结果数据/matched_auc.csv     # 应逐行一致
```

`结果数据/scores/` 里是 36 份逐图分数（我当时跑出来的），可直接和你的 `scores_*.csv` 逐行 diff。

## 5. 完整重训（可选）

前置：已完成 §4②（`manifest.csv` 与 `split_*.csv` 已生成）。

```bash
bash 代码/run_experiments.sh 4000 2      # per_class=4000, epochs=2, seed=42，跑 m1~m6
python 代码/train_own.py --epochs 2      # 单数据集 120k 模型 → models/own_resnet50
```

- 6 个配置 ×（4000/类，2 epoch）在 8GB 卡上约 30~40 分钟/个，共 3.5~4 小时；`run_experiments.sh` 会写 `train_<name>.log`。
- 切分是确定性哈希（`make_split.py` 里 `clip_in_train`），同数据同版本必然得到同一份切分：
  train 170,495（真 51,920 / 伪 118,575）、val 20,000（1:1）、test 38,683（真 10,843 / 伪 27,840）。
- 训练已固定 `--seed 42`（`seed_all()`），但 cuDNN 非确定性仍在，跨卡重训会有小幅漂移。

## 6. 复现时应看到的结果

画质（锐度）1:1 匹配后 AUC（`结果数据/matched_auc.csv`，每格匹配对数 389~1023）：

| 模型 | gan | swap_faceswap | swap_simswap | swap_dfl | diffusion | celebdf | 六格均值 |
|---|---|---|---|---|---|---|---|
| m1_gan | 0.984 | 0.607 | 0.606 | 0.604 | 0.786 | 0.600 | 0.698 |
| m2_swap | 0.793 | 0.911 | 0.907 | 0.681 | 0.862 | 0.486 | 0.773 |
| m3_diff | 0.856 | 0.741 | 0.875 | 0.610 | 0.983 | 0.554 | 0.770 |
| m4_gan_swap | 0.946 | 0.895 | 0.888 | 0.667 | 0.864 | 0.583 | 0.807 |
| m5_gan_diff | 0.974 | 0.746 | 0.806 | 0.632 | 0.962 | 0.617 | 0.790 |
| m6_all | 0.944 | 0.908 | 0.906 | 0.636 | 0.906 | 0.531 | 0.805 |

（看对角就能判断"是否在自己训练域上"：m1 的 gan 0.984、m2 的三个换脸格 0.911/0.907/0.681、m3 的 diffusion 0.983、m4 覆盖四格、m6 覆盖五格。）

要复现的三条结构性结论（数值可漂移，方向不应变）：

1. **同域 ≫ 跨域**：每个模型在自己训练域上最高。m1 同域 0.984 vs 跨域均值 0.641；m2 同域 0.833 vs 跨域 0.714；m6 同域 0.860 vs 跨域 0.531。跨域丢 4~34 个 AUC 点，换脸格最惨（m1 从 0.984 掉到 0.607）。
2. **加域是零和**：m1（0.698）→ m4 加换脸（0.807，+0.109）明显赚；但 m4 → m6 再加扩散，总分 0.807→0.805 原地踏步，扩散格 +0.042、swap_dfl −0.031、celebdf −0.052。覆盖面扩张，能力没提升。
3. **跨库必崩**：所有模型在 Celeb-DF 上 0.486~0.617，最差的 m2_swap 0.486（低于随机）。换一个数据集就不再可信。

复现容差：同机同版本重算评测，AUC 应完全一致（<1e-3）；重训后 AUC 允许 ±0.01~0.02，超过说明数据版本或依赖版本不同。

## 7. 复现边界（务必看）

| 项目 | 可复现性 |
|---|---|
| 附带权重的推理打分、评测表 | **100%**：同数据同版本逐位一致；不同 GPU/库版本误差 < 0.02（`verify_env.py` 自动卡这个阈值） |
| 切分（哪些图在 train/test） | **100%**：确定性哈希，只要路径与数据版本一致 |
| 重训后的绝对 AUC | 趋势可复现，数值 ±0.01~0.02；归档权重来自未固定种子的原始训练 |
| `scores_*.csv` 里每张图的具体分数 | 100%（用附带权重） |

## 8. 已知偏差与坑（诚实清单）

1. **报告 §1.1 的原始计数表**是按官方全量发布写的，与本机实际扫描不一致（例：DF40/faceswap 表里写 79996，本机 manifest 只有 24634；Celeb-DF fake 5639 vs 1984）。**以 `结果数据/manifest_stats.csv` 为准**，切分规模（170495/20000/38683）与报告一致。
2. **`auc_ci.csv` 早期版本有缺陷**：点估计取 matched AUC、CI 却在全量格上 bootstrap，出现"AUC 落在 CI 外"（如 m1_gan 换脸 0.607 但 CI 0.522~0.579）。已修：现同时给 `auc_full`（与 CI 同源）、`auc_matched`、`pairs`。
3. **`cross_summary.csv` 是旧版**（同域/跨域均值比现在低 0.02~0.06），已用现行 `summarize_cross.py` 重算覆盖，重算后 m4=0.807 / m6=0.805，与 `实验报告.md` §2 的"0.807→0.805"一致。`matched_auc.csv`、`generalization_gap.csv` 重算后与归档完全相同。
4. **A594952758.jpg 是已知误报**：m6/m2/m3 给 0.98+，仅 own_resnet50 给 0.43。换脸域 TPR@FPR1% 只有 0.12~0.13，这类样本必须靠分域阈值 + 人工复核，别信单模型 0.5 阈值。
5. **`score_files.py` 只接受显式文件参数**，且 7 模型 × 2 路 = 14 次前向，4 万张要跑 9~10 小时。全量请改用 2 模型 + 批推理（本包未附全量 runner）。
6. **中文路径**：OpenCV `imread` 读不了，所有脚本统一 `np.fromfile + cv2.imdecode`，自己写新脚本时别退回去。
7. **DataLoader 崩 `OpenBLAS: Memory allocation failed`**：每个 worker 必须把 BLAS 线程钉成 1（`train_*.py` 里 `OPENBLAS_NUM_THREADS=1` + `cv2.setNumThreads(0)`），并关 pinned memory。
8. **权重校验**：`models/SHA256SUMS.txt` 八份全在；`cnndetection/blur_jpg_prob0.5.pth` 是重训必需，缺了只能从 CNNDetection 官方仓库重下。
9. 早期 HuggingFace 通用模型（ViT/ONNX 三个，约 1GB）**未随包提供**，`fetch_models.py` 可重下；它们在本任务上 AUC 仅 0.52~0.79，已被自训模型取代。
10. **不做训练、只想复算表格**：先 `cp 结果数据/scores/*.csv .` 再跑 `match_eval.py`，否则它找不到 `scores_{model}_{cell}.csv`。

## 9. 一条命令清单

```bash
python 代码/verify_env.py                                        # 自检+数值校验
python 代码/score_files.py 样例/A634139568.png 样例/A870383032.jpg  # 单图打分
python 代码/build_manifest.py --out manifest.csv                  # 建索引
python 代码/make_split.py --manifest manifest.csv --outdir .       # 切分
bash 代码/run_experiments.sh 4000 2                               # 重训 6 配置
python 代码/eval_cross.py --models m6_all --cells all --dump-scores --out cross_eval.csv
```
