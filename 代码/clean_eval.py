# -*- coding: utf-8 -*-
"""严格无泄漏评测 + 捷径学习检验。

1) 测试集只从"从未进入训练"的图片里抽（按训练脚本同一套确定性切分反推）
2) 三个变体，检验模型依赖的是生成痕迹还是数据管线差异：
   orig    : 原始图
   recomp  : 全部重编码为同一 JPEG 质量(q=88)，抹掉原始压缩指纹
   down128 : 先降到 128 再升回 256，抹掉高频细节
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from eval_dataset import auc, load_own, tpr_at_fpr  # noqa: E402
from train_own import build_split  # noqa: E402

cv2.setNumThreads(0)


def pick_test(root: Path, per_class: int, seed: int = 123):
    train, _ = build_split(root, 30000)
    tr = {p for p, _ in train}
    out = []
    for cls, lab in (("real", 0), ("fake", 1)):
        files = [str(p) for p in sorted((root / cls).glob("*.jpg"))]
        unseen = [p for p in files if p not in tr]
        random.Random(seed).shuffle(unseen)
        out += [(p, lab) for p in unseen[:per_class]]
        print(f"  {cls}: 候选未见图 {len(unseen)}，抽取 {min(per_class, len(unseen))}")
    return out


def transform(im: np.ndarray, variant: str) -> np.ndarray:
    if variant == "orig":
        return im
    if variant == "recomp":
        ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else im
    if variant == "down128":
        h, w = im.shape[:2]
        small = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(variant)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(paths.RVF))
    ap.add_argument("--per-class", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--variants", default="orig,recomp,down128")
    ap.add_argument("--out", default="clean_eval.csv")
    args = ap.parse_args()

    items = pick_test(Path(args.root), args.per_class)
    print(f"测试集 {len(items)} 张（全部与训练集互斥）")
    run = load_own()
    y = np.array([l for _, l in items])

    rows = {"path": [p for p, _ in items], "label": y}
    for variant in args.variants.split(","):
        t0 = time.time()
        scores = []
        for st in range(0, len(items), args.batch):
            chunk = items[st : st + args.batch]
            rgb = []
            for p, _ in chunk:
                im = cv2.imread(p)
                if im is None:
                    im = np.zeros((256, 256, 3), np.uint8)
                im = cv2.resize(im, (256, 256))
                rgb.append(cv2.cvtColor(transform(im, variant), cv2.COLOR_BGR2RGB))
            scores += run(rgb)
        s = np.array(scores)
        rows[variant] = np.round(s, 6)
        a = auc(y, s)
        acc = ((s >= 0.5).astype(int) == y).mean()
        t1 = tpr_at_fpr(y, s, 0.01)
        t5 = tpr_at_fpr(y, s, 0.05)
        print(f"\n[{variant}] {time.time() - t0:.0f}s  n={len(y)}")
        print(f"  AUC={a:.6f}  acc@0.5={acc * 100:.2f}%  "
              f"TPR@FPR1%={t1[1] * 100:.2f}% (thr={t1[0]:.4f})  "
              f"TPR@FPR5%={t5[1] * 100:.2f}%")
        print(f"  真图误报 {float((s[y == 0] >= 0.5).mean()) * 100:.2f}%  "
              f"伪造漏检 {float((s[y == 1] < 0.5).mean()) * 100:.2f}%")

    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n明细写入 {args.out}")


if __name__ == "__main__":
    main()
