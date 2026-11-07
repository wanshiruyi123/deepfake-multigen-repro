# -*- coding: utf-8 -*-
"""量化评测集与训练集的重叠，并只用完全隔离的 held-out 集重测。"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from train_own import build_split  # noqa: E402


def main() -> None:
    root = paths.RVF
    train, val = build_split(root, 30000)
    tr = {p for p, _ in train}
    va = {p for p, _ in val}
    print(f"训练脚本切分：train={len(tr)}  val={len(va)}")

    # 1) 上一轮 4000 张评测集与训练集的重叠率
    ev = pd.read_csv("eval_own4k.csv")
    evp = set(ev["path"])
    in_tr = len(evp & tr)
    print(f"\n[重叠审计] eval_own4k 共 {len(evp)} 张：")
    print(f"  落在训练集: {in_tr} ({in_tr / len(evp) * 100:.1f}%)")
    print(f"  落在 held-out: {len(evp & va)} ({len(evp & va) / len(evp) * 100:.1f}%)")

    # 2) 只用 held-out 分数重算，衡量真实泛化
    y = ev["label"].to_numpy()
    s = ev["own"].astype(float).to_numpy()
    mask = np.array([p in va for p in ev["path"]])
    if mask.sum() > 100:
        yh, sh = y[mask], s[mask]
        n1, n0 = int((yh == 1).sum()), int((yh == 0).sum())
        order = np.argsort(sh, kind="mergesort")
        ranks = np.empty(len(sh))
        ranks[order] = np.arange(1, len(sh) + 1)
        auc_h = (ranks[yh == 1].mean() - (n1 + 1) / 2) / n0
        acc_h = ((sh >= 0.5).astype(int) == yh).mean()
        print(f"\n[仅 held-out 子集] n={mask.sum()} "
              f"(real={n0}, fake={n1})")
        print(f"  AUC={auc_h:.6f}  acc@0.5={acc_h * 100:.2f}%")
        neg = np.sort(sh[yh == 0])[::-1]
        for fpr in (0.001, 0.005, 0.01, 0.05):
            k = max(1, int(round(fpr * len(neg))))
            thr = float(neg[k - 1])
            print(f"  FPR<={fpr * 100:.1f}%  thr={thr:.4f} "
                  f"TPR={float((sh[yh == 1] >= thr).mean()) * 100:.2f}%")

    # 3) 全量结果里训练图占比（说明 full120k 的 99.0% 为何偏乐观）
    full = pd.read_csv("full120k_results.csv")
    fp = set(full["path"])
    print(f"\n[全量 120k] 其中训练集图片 {len(fp & tr)} 张 "
          f"({len(fp & tr) / len(fp) * 100:.1f}%)，held-out {len(fp & va)} 张")


if __name__ == "__main__":
    main()
