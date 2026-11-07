#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用你自己标注过的样本校准阈值与模型权重。

输入:
  --results  batch_detect.py 产出的 results.csv
  --labels   标注文件 labels.csv，两列: path,label  （label: 1=伪造 / 0=真实）
             也接受 label 写成 fake/real、true/false

输出:
  各模型单独 AUC、最优权重组合、FPR=1%/5%/10% 下的阈值与召回、混淆矩阵

为什么要做这一步：公开预训练模型的"99% 准确率"是在其自身测试集上的，
换到你的数据分布（特定换脸工具、特定压缩、特定分辨率）会显著下降。
用 200~500 张你确认过的样本跑一遍，比换任何模型都管用。
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

COLS = ["m_vit", "m_v2", "m_ai"]


def norm_label(v) -> int:
    s = str(v).strip().lower()
    if s in {"1", "fake", "true", "yes", "伪造", "假"}:
        return 1
    if s in {"0", "real", "false", "no", "真实", "真"}:
        return 0
    try:
        return 1 if float(s) >= 0.5 else 0
    except ValueError:
        raise ValueError(f"无法解析标签: {v!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="calibration_report.txt")
    args = ap.parse_args()

    res = pd.read_csv(args.results)
    lab = pd.read_csv(args.labels)

    res = res[res["status"] == "ok"].copy()
    for c in COLS:
        res[c] = pd.to_numeric(res[c], errors="coerce")

    lab.columns = [c.strip().lower() for c in lab.columns]
    lab["label"] = lab["label"].map(norm_label)
    res["_key"] = res["path"].map(lambda p: os.path.basename(str(p)))
    lab["_key"] = lab["path"].map(lambda p: os.path.basename(str(p)))

    df = res.merge(lab[["_key", "label"]], on="_key", how="inner").drop_duplicates("_key")
    if len(df) < 50:
        raise SystemExit(f"匹配到 {len(df)} 条样本，太少（建议 ≥200）。检查 path 是否对得上。")

    y = df["label"].to_numpy()
    print(f"匹配样本 {len(df)} 条：伪造 {int(y.sum())} / 真实 {int((1-y).sum())}\n")
    lines = [f"samples={len(df)} fake={int(y.sum())} real={int((1-y).sum())}", ""]

    # 单模型
    singles = {}
    for c in cols:
        m = df[c].notna()
        if m.sum() < 50 or len(np.unique(y[m.to_numpy()])) < 2:
            continue
        auc = roc_auc_score(y[m.to_numpy()], df.loc[m, c])
        singles[c] = auc
        lines.append(f"{c:8s} AUC={auc:.4f}  (n={int(m.sum())})")
    print("\n".join(lines[-len(singles):]))

    # 权重网格：两位 deepfake 模型，步长 0.1
    best = (None, -1.0)
    grid = []
    for w in np.arange(0.0, 1.01, 0.1):
        s = w * df["m_vit"].fillna(df["m_vit"].median()) + \
            (1 - w) * df["m_v2"].fillna(df["m_v2"].median())
        auc = roc_auc_score(y, s)
        grid.append((round(float(w), 2), auc))
        if auc > best[1]:
            best = (round(float(w), 2), auc)

    lines.append("")
    lines.append(f"最优权重 m_vit={best[0]:.1f} / m_v2={1-best[0]:.1f}  AUC={best[1]:.4f}")
    print(f"最优权重 m_vit={best[0]:.1f} / m_v2={1-best[0]:.1f}  AUC={best[1]:.4f}")

    w = best[0]
    ens = w * df["m_vit"].fillna(df["m_vit"].median()) + \
        (1 - w) * df["m_v2"].fillna(df["m_v2"].median())

    fpr, tpr, thr = roc_curve(y, ens)
    lines.append("")
    lines.append("阈值建议（按目标误报率）:")
    print("\n阈值建议（按目标误报率）:")
    for target in (0.005, 0.01, 0.05, 0.10):
        ok = np.where(fpr <= target)[0]
        if not len(ok):
            continue
        i = ok[np.argmax(tpr[ok])]
        t = float(thr[i])
        pred = (ens >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        recall = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        row = (f"  FPR≈{target*100:4.1f}%  阈值={t:.4f}  召回={recall:.4f}  "
               f"精确率={prec:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")
        lines.append(row)
        print(row)

    # 一致性：两模型分歧度
    diff = (df["m_vit"] - df["m_v2"]).abs()
    lines.append("")
    lines.append(f"两模型分歧: 均值={diff.mean():.4f} p95={diff.quantile(0.95):.4f} "
                 f"不一致率(跨阈值)={float(((df['m_vit'] >= 0.5) != (df['m_v2'] >= 0.5)).mean()):.4f}")
    print(lines[-1].strip())

    lines.append("")
    lines.append(f"建议命令: python batch_detect.py --input <DIR> --out results.csv "
                 f"--threshold {float(thr[np.argmax(tpr - fpr)]):.3f}")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
