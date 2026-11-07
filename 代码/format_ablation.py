# -*- coding: utf-8 -*-
"""验证「格式捷径」是否真实存在、统一重编码是否必要

对照实验：同一个模型、同一批测试图，
  A) 原始管线：直接 cv2.imread 送入（保留 png/jpg 原有编码特征）
  B) 统一管线：decode -> JPEG q95 重编码 -> decode

如果 A 明显高于 B，说明模型能从"无损 vs 有损编码"这一纯格式线索中获益，
那么任何跨数据集实验若不统一编码，结论都不可信。
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_cross import (IMAGESIZE, MEAN, STD, build_cells, load_model,  # noqa: E402
                        metrics)

cv2.setNumThreads(0)
ROOT = Path(__file__).resolve().parents[1]


class RawDS(Dataset):
    """不重编码"""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            im = np.zeros((IMAGESIZE, IMAGESIZE, 3), np.uint8)
        im = cv2.cvtColor(cv2.resize(im, (IMAGESIZE, IMAGESIZE)), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(im).permute(2, 0, 1), lab


class UniDS(RawDS):
    """统一 JPEG q95 重编码"""

    def __getitem__(self, i):
        p, lab = self.items[i]
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            im = np.zeros((IMAGESIZE, IMAGESIZE, 3), np.uint8)
        else:
            ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(cv2.resize(im, (IMAGESIZE, IMAGESIZE)), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(im).permute(2, 0, 1), lab


def run(net, ds, dev, bs=96):
    dl = DataLoader(ds, batch_size=bs, num_workers=2)
    mg = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    sg = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    ys, ss = [], []
    with torch.inference_mode():
        for x, y in dl:
            x = x.to(dev).float().div_(255.0)
            x = ((x - mg) / sg).to(memory_format=torch.channels_last)
            ss.append(torch.sigmoid(net(x)).squeeze(1).float().cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ss)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "m1_gan"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pth = ROOT / "models" / model / "best.pt"
    net, _ = load_model(pth, dev)
    cells = build_cells(n)

    print(f"\n模型 {model}   每格 {n}/类\n")
    print(f"{'cell':16s} {'原始编码':>10s} {'统一重编码':>12s} {'差值':>8s}   格式构成")
    print("-" * 78)
    rows = []
    for cell, items in cells.items():
        exts = pd.Series([Path(p).suffix.lower() for p, _ in items])
        lab = np.array([l for _, l in items])
        comp = (f"real={exts[lab==0].mode().iloc[0]} "
                f"fake={exts[lab==1].mode().iloc[0]}")
        y1, s1 = run(net, RawDS(items), dev)
        y2, s2 = run(net, UniDS(items), dev)
        m1, m2 = metrics(y1, s1), metrics(y2, s2)
        d = m1["auc"] - m2["auc"]
        print(f"{cell:16s} {m1['auc']:10.4f} {m2['auc']:12.4f} {d:+8.4f}   {comp}")
        rows.append({"cell": cell, "auc_raw": m1["auc"], "auc_unified": m2["auc"],
                     "delta": round(d, 4), "format": comp})

    pd.DataFrame(rows).to_csv(ROOT / f"format_ablation_{model}.csv", index=False)
    print("\n差值 > 0 表示「靠原始编码特征」比「统一重编码」更容易分对，"
          "\n即存在格式捷径。跨域格若差值显著为正，说明不统一编码会严重虚高。")


if __name__ == "__main__":
    main()
