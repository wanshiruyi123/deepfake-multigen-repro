# -*- coding: utf-8 -*-
"""对疑似近重复对做像素级验证，并导出对照图。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

import cv2
import numpy as np
import pandas as pd

ROOT = paths.RVF


def dhash(im, size=8):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def main() -> None:
    df = pd.read_csv("dup_hashes.csv")
    dh = df["dhash"].to_numpy(np.uint64)
    idx = {}
    for i, v in enumerate(dh):
        for b in range(4):
            idx.setdefault((b, int((v >> (b * 16)) & 0xFFFF)), []).append(i)

    pairs = set()
    for lst in idx.values():
        if len(lst) < 2 or len(lst) > 400:
            continue
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                i, j = lst[a], lst[b]
                if bin(int(dh[i] ^ dh[j])).count("1") <= 6:
                    pairs.add((i, j) if i < j else (j, i))
    print(f"候选对 {len(pairs)}")

    paths = df["path"].tolist()
    clss = df["cls"].tolist()

    # 像素级：MSE（对齐后直接比）
    def mse(p, q):
        a = cv2.imread(p)
        b = cv2.imread(q)
        if a is None or b is None:
            return 255.0
        a = cv2.resize(a, (128, 128)).astype(np.float32)
        b = cv2.resize(b, (128, 128)).astype(np.float32)
        return float(((a - b) ** 2).mean())

    # 抽样 400 对做像素验证，看分布
    import random
    rnd = random.Random(0)
    samp = rnd.sample(sorted(pairs), min(400, len(pairs)))
    ms = [(i, j, mse(paths[i], paths[j])) for i, j in samp]
    arr = np.array([m for _, _, m in ms])
    print(f"抽样 {len(ms)} 对像素 MSE 分布：min={arr.min():.1f} "
          f"p1={np.percentile(arr, 1):.1f} 中位={np.median(arr):.1f} "
          f"p99={np.percentile(arr, 99):.1f} max={arr.max():.1f}")
    truly_dup = [(i, j, m) for i, j, m in ms if m < 50]
    print(f"像素 MSE<50（极可能真重复）的对数：{len(truly_dup)}")

    # 导出对照图：前 5 对（按 MSE 升序）
    rows = []
    for i, j, m in sorted(ms, key=lambda x: x[2])[:5]:
        a = cv2.resize(cv2.imread(paths[i]), (140, 140))
        b = cv2.resize(cv2.imread(paths[j]), (140, 140))
        gap = np.zeros((140, 8, 3), np.uint8)
        row = np.hstack([a, gap, b, gap, np.zeros((140, 120, 3), np.uint8)])
        cv2.putText(row, f"{m:.0f}", (300, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        rows.append(row)
        print(f"  {clss[i]}/{Path(paths[i]).name} <-> {clss[j]}/{Path(paths[j]).name}  MSE={m:.1f}")
    cv2.imwrite("peek/near_dup_check.jpg", np.vstack(rows),
                [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    print("对照图 peek/near_dup_check.jpg（左右为疑似重复对，数字为 MSE）")


if __name__ == "__main__":
    main()
