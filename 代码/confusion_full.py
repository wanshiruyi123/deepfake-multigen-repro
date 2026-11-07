# -*- coding: utf-8 -*-
"""按目录标签统计全量推理结果的混淆矩阵。"""
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "full120k_results.csv")
df["label"] = df["path"].str.replace("\\", "/").str.contains("/fake/").astype(int)
y = df["label"].to_numpy()
s = df["score"].astype(float).to_numpy()

print(f"总数 {len(df)}  真图 {(y == 0).sum()}  伪造 {(y == 1).sum()}\n")
for thr in (0.5, 0.5876, 0.8131, 0.9687):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / len(y)
    print(f"thr={thr:.4f}  acc={acc * 100:.3f}%  "
          f"伪造检出 {tp}/{tp + fn} ({tp / (tp + fn) * 100:.2f}%)  "
          f"真图误报 {fp}/{fp + tn} ({fp / (fp + tn) * 100:.2f}%)")

# AUC
n1, n0 = int((y == 1).sum()), int((y == 0).sum())
order = np.argsort(s, kind="mergesort")
ranks = np.empty(len(s))
ranks[order] = np.arange(1, len(s) + 1)
print(f"\n全量 AUC = {(ranks[y == 1].mean() - (n1 + 1) / 2) / n0:.6f}")
print(f"分数分布: 真图 中位数={np.median(s[y == 0]):.6f} p99={np.percentile(s[y == 0], 99):.4f} | "
      f"伪造 中位数={np.median(s[y == 1]):.6f} p1={np.percentile(s[y == 1], 1):.4f}")

# 最难的两类
hard_fake = df[(y == 1) & (s < 0.5876)]
hard_real = df[(y == 0) & (s >= 0.5876)]
print(f"\n漏检伪造 {len(hard_fake)} 张，最高分 {s[hard_fake.index].max():.4f}" if len(hard_fake) else "\n无漏检")
print(f"误报真图 {len(hard_real)} 张，最低分 {s[hard_real.index].min():.4f}" if len(hard_real) else "无误报")
