# -*- coding: utf-8 -*-
"""生成测试格样本对照图：每格 上排=real 下排=fake

用途：判断 AUC 异常（<0.5）是否由真实样本与伪造样本之间的
画质/对齐/裁剪管线差异主导，而非模型判别力问题。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_cross import build_cells  # noqa: E402

cv2.setNumThreads(0)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "peek"
OUT.mkdir(exist_ok=True)

K = 5
cells = build_cells(60)
for cell, items in cells.items():
    reals = [p for p, l in items if l == 0][:K]
    fakes = [p for p, l in items if l == 1][:K]
    rows = []
    for group in (reals, fakes):
        tiles = []
        for p in group:
            im = cv2.imread(p)
            if im is None:
                im = np.zeros((256, 256, 3), np.uint8)
            tiles.append(cv2.resize(im, (170, 170)))
        if tiles:
            rows.append(np.hstack(tiles))
    if len(rows) < 2:
        print(f"{cell}: 缺 real 或 fake，跳过")
        continue
    img = np.vstack(rows)
    cv2.putText(img, "REAL (top) / FAKE (bottom)", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    p = OUT / f"cell_{cell}.jpg"
    cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"{cell:16s} -> {p.name}")

# 同时统计各格的锐度（拉普拉斯方差）与平均亮度，量化画质差异
print("\n画质统计（拉普拉斯方差 = 锐度，越大越锐利）:")
print(f"{'cell':16s} {'real_sharp':>11s} {'fake_sharp':>11s} {'real_L':>7s} {'fake_L':>7s}")
for cell, items in cells.items():
    st = {}
    for lab in (0, 1):
        vals, lum = [], []
        for p, l in items:
            if l != lab:
                continue
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            vals.append(cv2.Laplacian(cv2.resize(im, (256, 256)), cv2.CV_64F).var())
            lum.append(float(im.mean()))
        st[lab] = (float(np.mean(vals)) if vals else 0.0,
                   float(np.mean(lum)) if lum else 0.0)
    print(f"{cell:16s} {st[0][0]:11.1f} {st[1][0]:11.1f} "
          f"{st[0][1]:7.1f} {st[1][1]:7.1f}")
