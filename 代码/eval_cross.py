# -*- coding: utf-8 -*-
"""跨数据集评测矩阵：任意模型 × 任意测试格

测试格设计:
  gan           140k test (GAN 域，真图来自高清人脸)
  swap_faceswap DF40 FaceSwap，真图来自 Celeb-DF 视频帧
  swap_simswap  DF40 SimSwap，同上
  swap_dfl      DF40 DeepFaceLab，真图用 dfl real + CelebDF real
  diffusion     DF40 DiT（扩散换脸），真图来自 Celeb-DF 视频帧
  celebdf       官方 Celeb-DF 测试集原始构成

统一重编码与训练一致，避免格式差异影响对比。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

cv2.setNumThreads(0)

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGESIZE = 256
JPEG_Q = 95


def unify_decode(path: str):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else im


class DS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        im = unify_decode(p)
        if im is None:
            im = np.zeros((IMAGESIZE, IMAGESIZE, 3), np.uint8)
        im = cv2.cvtColor(cv2.resize(im, (IMAGESIZE, IMAGESIZE)), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(im).permute(2, 0, 1), lab


def build_cells(n_per_class: int = 1000, seed: int = 2024):
    """组装各测试格，返回 {cell: [(path,label), ...]}"""
    te = pd.read_csv(ROOT / "split_test.csv", encoding="utf-8")
    rng = np.random.RandomState(seed)

    def take(sub, n):
        ps = sub.path.tolist()
        if len(ps) <= n:
            return ps
        idx = rng.choice(len(ps), n, replace=False)
        return [ps[i] for i in sorted(idx)]

    # 真实样本池
    gan_real = te[(te.testset == "gan") & (te.label == 0)]
    cdf_real = te[(te.testset == "celebdfofficial") & (te.label == 0)]
    dfl_real = te[(te.testset == "deepfacelab") & (te.label == 0)]

    cells = {}

    # 1. GAN
    cells["gan"] = ([(p, 0) for p in take(gan_real, n_per_class)] +
                    [(p, 1) for p in take(te[(te.testset == "gan") & (te.label == 1)],
                                          n_per_class)])

    # 2/3. 换脸（faceswap / simswap），真图 = Celeb-DF real
    for m in ("faceswap", "simswap"):
        sub = te[te.testset.str.startswith(f"swap_{m}")]
        cells[f"swap_{m}"] = ([(p, 0) for p in take(cdf_real, n_per_class)] +
                              [(p, 1) for p in take(sub, n_per_class)])

    # 4. DeepFaceLab，真图 = dfl real 优先，不足用 CelebDF real 补
    dsub = te[te.testset == "deepfacelab"]
    reals = take(dfl_real, n_per_class)
    if len(reals) < n_per_class:
        reals += take(cdf_real, n_per_class - len(reals))
    cells["swap_dfl"] = ([(p, 0) for p in reals] +
                         [(p, 1) for p in take(dsub[dsub.label == 1], 800)])

    # 5. 扩散
    dsub = te[te.testset == "diffusion"]
    cells["diffusion"] = ([(p, 0) for p in take(cdf_real, n_per_class)] +
                          [(p, 1) for p in take(dsub, n_per_class)])

    # 6. Celeb-DF 官方构成（原样，不抽样）
    cells["celebdf"] = ([(p, 0) for p in cdf_real.path] +
                        [(p, 1) for p in te[(te.testset == "celebdfofficial") &
                                            (te.label == 1)].path])

    for k, v in cells.items():
        nr = sum(1 for _, l in v if l == 0)
        print(f"  {k:15s} real={nr:5d} fake={len(v)-nr:5d}")
    return cells


def metrics(y, s):
    y, s = np.asarray(y), np.asarray(s)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return {}
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    acc = float(((s >= 0.5).astype(int) == y).mean())
    # FPR 分位阈值（真实样本分数分布）
    s0 = np.sort(s[y == 0])
    out = {"auc": round(auc, 5), "acc": round(acc, 4)}
    for fpr in (0.001, 0.01, 0.05):
        thr = float(np.quantile(s0, 1 - fpr))
        out[f"thr@{fpr}"] = round(thr, 5)
        out[f"tpr@{fpr}"] = round(float((s[y == 1] >= thr).mean()), 4)
    return out


def load_model(pth: Path, dev):
    net = resnet50(num_classes=1)
    ck = torch.load(pth, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.eval().to(dev).to(memory_format=torch.channels_last)
    return net, ck.get("image_size", IMAGESIZE)


def score_items(net, items, dev, bs=128, workers=6):
    dl = DataLoader(DS(items), batch_size=bs, num_workers=workers)
    ys, ss = [], []
    MEAN_G = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    STD_G = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    with torch.inference_mode():
        for x, y in dl:
            x = x.to(dev).float().div_(255.0)
            x = ((x - MEAN_G) / STD_G).to(memory_format=torch.channels_last)
            s = torch.sigmoid(net(x)).squeeze(1).float().cpu().numpy()
            ys.append(y.numpy())
            ss.append(s)
    return np.concatenate(ys), np.concatenate(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="逗号分隔的模型目录名，或 'all' 扫描 models/m*_*/best.pt")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--out", default="cross_eval.csv")
    ap.add_argument("--dump-scores", action="store_true",
                    help="额外保存逐图分数，便于画 ROC / 复查")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}\n组装测试格:")
    cells = build_cells(args.per_class)
    if args.cells != "all":
        want = [c.strip() for c in args.cells.split(",")]
        cells = {k: v for k, v in cells.items() if k in want}
        print("  仅评测:", list(cells))

    if args.models == "all":
        mdls = sorted(p.parent.name for p in (ROOT / "models").glob("m*/best.pt"))
    else:
        mdls = [m.strip() for m in args.models.split(",")]
    print(f"\n模型: {mdls}\n")

    rows = []
    for name in mdls:
        pth = ROOT / "models" / name / "best.pt"
        if not pth.exists():
            print(f"跳过 {name}: 无 best.pt")
            continue
        net, _ = load_model(pth, dev)
        print(f"== {name}")
        for cell, items in cells.items():
            y, s = score_items(net, items, dev)
            m = metrics(y, s)
            m.update({"model": name, "cell": cell, "n": len(items)})
            rows.append(m)
            print(f"   {cell:15s} AUC={m['auc']:.4f} acc={m['acc']:.4f} "
                  f"TPR@FPR1%={m['tpr@0.01']:.4f}")
            if args.dump_scores:
                d = pd.DataFrame({"path": [p for p, _ in items],
                                  "label": y, "score": s})
                d.to_csv(ROOT / f"scores_{name}_{cell}.csv", index=False)
        del net
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / args.out, index=False, encoding="utf-8")

    print("\n" + "=" * 78)
    print("AUC 矩阵（行=模型，列=测试格）")
    print("=" * 78)
    piv = df.pivot(index="model", columns="cell", values="auc")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nTPR@FPR=1% 矩阵")
    piv2 = df.pivot(index="model", columns="cell", values="tpr@0.01")
    print(piv2.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n已保存 -> {args.out}")


if __name__ == "__main__":
    main()
