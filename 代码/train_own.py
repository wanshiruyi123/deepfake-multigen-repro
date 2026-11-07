# -*- coding: utf-8 -*-
"""在 real-vs-fake 数据集上微调专属检测器。

以 CNNDetection(ProGAN 检测器, ResNet50 单 logit) 为起点，在本数据上微调。
数据集自带标签: real/=0, fake/=1。

用法:
  .venv/Scripts/python.exe batch/train_own.py --epochs 2 --subset-per-class 20000
产出:
  models/own_resnet50/best.pt   (验证集 AUC 最优的权重)
  train_log.csv                 每 200 step 的训练记录
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

# 必须在 numpy/torch 导入前钉死 BLAS 线程数，否则每个 DataLoader worker
# 都会开满核心数个线程，内存与调度双重爆炸（实测导致 OpenBLAS 分配失败崩溃）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

cv2.setNumThreads(0)  # DataLoader worker 内禁用 OpenCV 内部线程，避免争抢

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGESIZE = 256


def stable_hash(s: str) -> int:
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def build_split(root: Path, subset_per_class: int | None):
    """文件名哈希切分 85/15，保证可复现且跨 subset 稳定。"""
    train, val = [], []
    for cls, lab in (("real", 0), ("fake", 1)):
        files = sorted((root / cls).glob("*.jpg"))
        if subset_per_class:
            files = random.Random(7).sample(files, min(subset_per_class, len(files)))
        for f in files:
            (train if stable_hash(f.name) % 100 < 85 else val).append((str(f), lab))
    random.Random(11).shuffle(train)
    return train, val


class FaceDS(Dataset):
    def __init__(self, items: list, train: bool):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        im = cv2.imread(p)
        if im is None:
            im = np.zeros((IMAGESIZE, IMAGESIZE, 3), np.uint8)
        im = cv2.cvtColor(cv2.resize(im, (IMAGESIZE, IMAGESIZE)), cv2.COLOR_BGR2RGB)
        if self.train and random.random() < 0.5:
            im = im[:, ::-1].copy()
        # 返回 uint8(HWC)，归一化放到 GPU 上做，节省 4 倍主机内存
        return torch.from_numpy(im).permute(2, 0, 1), lab


def seed_all(seed: int) -> None:
    """固定所有随机源，保证同机同环境下可复现（跨卡/跨版本仍有微小差异）。"""
    import os as _os
    import random as _random
    _os.environ.setdefault("PYTHONHASHSEED", str(seed))
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(paths.RVF))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--subset-per-class", type=int, default=None,
                    help="每类只用 N 张训练（省时间；默认全量）")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                        / "models" / "own_resnet50"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_all(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  torch={torch.__version__}")
    if dev == "cpu":
        print("警告: 无 GPU，训练会非常慢，建议先装 cu128 torch")

    train_items, val_items = build_split(Path(args.root), args.subset_per_class)
    print(f"train={len(train_items)}  val={len(val_items)}")

    # CNNDetection 权重做初始化
    pth = Path(__file__).resolve().parents[1] / "models" / "cnndetection" \
        / "blur_jpg_prob0.5.pth"
    net = resnet50(num_classes=1)
    if pth.exists():
        sd = torch.load(pth, map_location="cpu", weights_only=False)["model"]
        net.load_state_dict(sd)
        print(f"初始化: {pth.name}")
    else:
        print("警告: 未找到 CNNDetection 权重，随机初始化")
    net = net.to(dev).to(memory_format=torch.channels_last)

    head_params = [p for n, p in net.named_parameters() if n.startswith("fc.")]
    back_params = [p for n, p in net.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([
        {"params": back_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, len(train_items) // args.batch))
    lossfn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(dev)

    nw = min(6, os.cpu_count() or 4)
    tl = DataLoader(FaceDS(train_items, True), batch_size=args.batch,
                    shuffle=True, num_workers=nw, pin_memory=False,
                    persistent_workers=False, drop_last=True)
    vl = DataLoader(FaceDS(val_items, False), batch_size=args.batch * 2,
                    num_workers=nw, pin_memory=False)

    MEAN_G = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    STD_G = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    def norm(x: torch.Tensor) -> torch.Tensor:
        x = x.to(dev, non_blocking=True).float().div_(255.0)
        return (x - MEAN_G) / STD_G

    def evaluate():
        net.eval()
        ys, ss = [], []
        with torch.inference_mode():
            for x, y in vl:
                s = torch.sigmoid(
                    net(norm(x).to(memory_format=torch.channels_last))
                ).squeeze(1).float().cpu().numpy()
                ys.append(y.numpy())
                ss.append(s)
        y = np.concatenate(ys)
        s = np.concatenate(ss)
        n1 = int((y == 1).sum())
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(s))
        ranks[order] = np.arange(1, len(s) + 1)
        # 简化并列处理（连续分数几乎无并列）
        return float((ranks[y == 1].mean() - (n1 + 1) / 2) / (len(y) - n1))

    best = 0.0
    logf = open(out / "train_log.csv", "w", newline="", encoding="utf-8")
    logw = csv.writer(logf)
    logw.writerow(["epoch", "step", "loss", "val_auc", "elapsed_s"])

    step = 0
    for ep in range(1, args.epochs + 1):
        net.train()
        t0, running, nrun = time.time(), 0.0, 0
        for x, y in tl:
            x = norm(x).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True).float()
            if dev == "cuda":
                with torch.autocast(dev, dtype=torch.float16):
                    loss = lossfn(net(x).squeeze(1), y)
            else:
                loss = lossfn(net(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running += float(loss.detach())
            nrun += 1
            if step % 200 == 0:
                el = time.time() - t0
                print(f"ep{ep} step{step} loss={running / nrun:.4f} "
                      f"({nrun * args.batch / el:.0f} img/s)", flush=True)
                torch.save({"state_dict": net.state_dict(), "step": step,
                            "image_size": IMAGESIZE}, out / "last.pt")
                running, nrun = 0.0, 0
        a = evaluate()
        print(f"== epoch {ep} 完成  val AUC={a:.4f}  用时 {time.time() - t0:.0f}s")
        logw.writerow([ep, step, running / max(nrun, 1), a, round(time.time() - t0)])
        logf.flush()
        if a > best:
            best = a
            torch.save({"state_dict": net.state_dict(), "val_auc": a,
                        "image_size": IMAGESIZE}, out / "best.pt")
            print(f"   保存 best.pt (AUC={a:.4f})")
    logf.close()
    print(f"\n完成。best val AUC={best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
