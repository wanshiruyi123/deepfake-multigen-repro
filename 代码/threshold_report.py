# -*- coding: utf-8 -*-
"""从评测 CSV 计算 FPR=1% 阈值下的判定指标，并给出最终推荐阈值。"""
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "eval_own4k.csv")
df = df[df["status"] == "ok"]
col = "own"
y = df["label"].to_numpy()
s = df[col].astype(float).to_numpy()

neg = np.sort(s[y == 0])[::-1]
for fpr in (0.001, 0.005, 0.01, 0.05):
    k = max(1, int(round(fpr * len(neg))))
    thr = float(neg[k - 1])
    tpr = float((s[y == 1] >= thr).mean())
    actual_fpr = float((s[y == 0] >= thr).mean())
    acc = float(((s >= thr).astype(int) == y).mean())
    print(f"FPR<={fpr * 100:.1f}%  thr={thr:.4f}  实际FPR={actual_fpr * 100:.2f}%  "
          f"TPR={tpr * 100:.2f}%  acc={acc * 100:.2f}%")

acc05 = float(((s >= 0.5).astype(int) == y).mean())
print(f"\nthr=0.5  acc={acc05 * 100:.2f}%  真图误报={float((s[y == 0] >= 0.5).mean()) * 100:.2f}%  "
      f"伪造漏检={float((s[y == 1] < 0.5).mean()) * 100:.2f}%")
