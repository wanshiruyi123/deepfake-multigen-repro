# -*- coding: utf-8 -*-
"""对每个 (model, cell) 的匹配后 AUC 做 bootstrap 置信区间"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "matched_auc.csv")
rng = np.random.RandomState(0)


def auc(y, s):
    """并列取平均秩的 Mann-Whitney AUC（无 scipy 依赖）"""
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    r = np.empty(len(s))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rows = []
for _, r in df.iterrows():
    d = pd.read_csv(ROOT / f"scores_{r.model}_{r.cell}.csv")
    y, s = d.label.values, d.score.values
    point = auc(y, s)                 # 点估计必须与 bootstrap 用同一批样本
    bs = []
    n = len(y)
    for _ in range(600):
        idx = rng.randint(0, n, n)
        yy, ss = y[idx], s[idx]
        if yy.min() == yy.max():
            continue
        bs.append(auc(yy, ss))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append({"model": r.model, "cell": r.cell,
                 "auc_full": round(float(point), 3),
                 "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                 "width": round(float(hi - lo), 3),
                 "auc_matched": round(float(r.auc_matched), 3),
                 "pairs": int(r.pairs)})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "auc_ci.csv", index=False, encoding="utf-8")
print(out.pivot(index="model", columns="cell", values="auc_full").to_string())
print("\n平均 95% CI 宽度:", round(float(out.width.mean()), 3))
print("CI 宽度最大:", round(float(out.width.max()), 3))
