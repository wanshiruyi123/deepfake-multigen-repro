# -*- coding: utf-8 -*-
"""多域混合训练：研究「多生成器混训能否提升跨生成器泛化」

设计:
  1. real 池固定 = GAN 域真图 + deepfacelab 真帧。所有实验组共用同一 real 池，
     因此组间差异只来自 fake 的生成器多样性 —— 这是干净的单变量对照。
  2. 训练与评测都走统一 JPEG 重编码管线 (decode -> encode q95 -> decode)，
     抹掉 png/jpg 格式与首次压缩质量的差异，杜绝"靠文件格式作弊"。
  3. 以 CNNDetection 权重为共同初始化，超参一致，仅训练数据不同。

用法:
  # 单域基线
  python train_multi.py --name m1_gan --domains gan --per-class 12000
  # 换脸 + 扩散 全混合
  python train_multi.py --name m6_all --domains gan,swap,diff --per-class 12000
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

cv2.setNumThreads(0)

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGESIZE = 256
JPEG_Q = 95          # 统一重编码质量


def unify_decode(path: str) -> np.ndarray | None:
    """解码 -> 统一 JPEG q95 重编码 -> 再解码。返回 BGR uint8。

    这一步是实验可信度的关键：数据集中 GAN 域全是 .jpg、DF40 全是 .png，
    若不统一，模型可以直接凭"无损/有损"这一个线索做到接近 100% 准确，
    完全不看内容。重编码后所有样本共享同一编码管线。
    """
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
    if not ok:
        return im
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class FaceDS(Dataset):
    def __init__(self, items, train: bool):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        im = unify_decode(p)
        if im is None:
            im = np.zeros((IMAGESIZE, IMAGESIZE, 3), np.uint8)
        im = cv2.cvtColor(cv2.resize(im, (IMAGESIZE, IMAGESIZE)), cv2.COLOR_BGR2RGB)
        if self.train and random.random() < 0.5:
            im = im[:, ::-1].copy()
        return torch.from_numpy(im).permute(2, 0, 1), lab


def pick(rows, n, seed):
    """按 n 抽样，行数不足则全取"""
    if len(rows) <= n:
        return list(rows)
    return random.Random(seed).sample(list(rows), n)


def build_train(domains: list[str], per_class: int, seed: int = 7):
    """构造训练集：real 池三源覆盖，fake 按选定域均分

    real 构成（第一轮实验的教训：real 分布必须覆盖测试 real 分布，
    否则模型把"高清=真"学进去，测试时视频帧真图全判假）：
      GAN 真图（高清）        ~40%
      deepfacelab 真帧        ~30%
      Celeb-DF 真帧(训练份)   ~30%
    各源不足时全取，GAN 补足。
    """
    tr = pd.read_csv(ROOT / "split_train.csv", encoding="utf-8")
    real_pool = tr[tr.label == 0]

    real_gan = real_pool[real_pool.domain == "gan"]
    real_dfl = real_pool[(real_pool.domain == "swap")
                         & (real_pool.method == "deepfacelab")]
    real_cdf = real_pool[real_pool.method == "celebreal"]

    n_gan = int(per_class * 0.40)
    n_dfl = int(per_class * 0.30)
    n_cdf = per_class - n_gan - n_dfl

    real_items = [(p, 0) for p in pick(real_gan.path, n_gan, seed)]
    real_items += [(p, 0) for p in pick(real_dfl.path, n_dfl, seed + 1)]
    real_items += [(p, 0) for p in pick(real_cdf.path, n_cdf, seed + 2)]
    if len(real_items) < per_class:                # 视频帧真图不足时用 GAN 补
        have = {p for p, _ in real_items}
        extra = [p for p in real_gan.path if p not in have]
        real_items += [(p, 0) for p in pick(extra, per_class - len(real_items), seed + 3)]
    print(f"  real: gan={sum(1 for p,_ in real_items if p in set(real_gan.path))} "
          f"dfl={sum(1 for p,_ in real_items if p in set(real_dfl.path))} "
          f"cdf={sum(1 for p,_ in real_items if p in set(real_cdf.path))}")

    fake_pool = tr[tr.label == 1]
    DMAP = {"gan": "gan", "swap": "swap", "diff": "diffusion",
            "diffusion": "diffusion"}
    groups = []
    for d in domains:
        sub = fake_pool[fake_pool.domain == DMAP.get(d, d)]
        if d == "swap":
            for m, g in sub.groupby("method"):
                groups.append((f"swap/{m}", g))
        else:
            groups.append((d, sub))
    if not groups:
        sys.exit("没有可用的 fake 数据")

    per_group = per_class // len(groups)
    fake_items = []
    for i, (name, g) in enumerate(groups):
        n = per_group if i < len(groups) - 1 else per_class - per_group * (len(groups) - 1)
        ps = pick(g.path, n, seed + 10 + i)
        fake_items += [(p, 1) for p in ps]
        print(f"  fake[{name:16s}] {len(ps):6d} 张")

    items = real_items + fake_items
    random.Random(seed + 99).shuffle(items)
    return items


def build_val(n_per_class: int = 1500):
    va = pd.read_csv(ROOT / "split_val.csv", encoding="utf-8")
    real = pick(va[va.label == 0].path, n_per_class, 3)
    fake = pick(va[va.label == 1].path, n_per_class, 4)
    return [(p, 0) for p in real] + [(p, 1) for p in fake]


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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
    ap.add_argument("--name", required=True, help="实验名，决定输出目录")
    ap.add_argument("--domains", default="gan",
                    help="训练用 fake 域，逗号分隔: gan,swap,diff")
    ap.add_argument("--per-class", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_all(args.seed)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    out = ROOT / "models" / args.name
    out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"== {args.name}  domains={domains}  per_class={args.per_class}  device={dev}")

    train_items = build_train(domains, args.per_class)
    val_items = build_val()
    n_fake = sum(1 for _, l in train_items if l == 1)
    print(f"train={len(train_items)} (real={len(train_items)-n_fake} fake={n_fake})"
          f"  val={len(val_items)}")

    pth = ROOT / "models" / "cnndetection" / "blur_jpg_prob0.5.pth"
    net = resnet50(num_classes=1)
    sd = torch.load(pth, map_location="cpu", weights_only=False)["model"]
    net.load_state_dict(sd)
    print(f"初始化: {pth.name}")
    net = net.to(dev).to(memory_format=torch.channels_last)

    head = [p for n, p in net.named_parameters() if n.startswith("fc.")]
    back = [p for n, p in net.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([
        {"params": back, "lr": args.lr_backbone},
        {"params": head, "lr": args.lr_head},
    ], weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_items) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch)
    lossfn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(dev)

    tl = DataLoader(FaceDS(train_items, True), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, pin_memory=False,
                    persistent_workers=False, drop_last=True)
    vl = DataLoader(FaceDS(val_items, False), batch_size=args.batch * 2,
                    num_workers=args.workers, pin_memory=False)

    MEAN_G = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    STD_G = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    def norm(x):
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
        return auc_score(np.concatenate(ys), np.concatenate(ss))

    best = 0.0
    logf = open(out / "train_log.csv", "w", newline="", encoding="utf-8")
    logw = csv.writer(logf)
    logw.writerow(["epoch", "step", "loss", "val_auc", "elapsed_s"])

    # 先用未训练权重测一次，作为"起点 baseline"，能看出训练带来多少增益
    a0 = evaluate()
    print(f"起点 val AUC={a0:.4f}")

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
            if step % 150 == 0:
                el = time.time() - t0
                print(f"ep{ep} step{step}/{steps_per_epoch} loss={running/nrun:.4f} "
                      f"({nrun*args.batch/el:.0f} img/s)", flush=True)
                torch.save({"state_dict": net.state_dict(), "step": step,
                            "image_size": IMAGESIZE, "domains": domains},
                           out / "last.pt")
                logw.writerow([ep, step, round(running / nrun, 5), "", round(el)])
                logf.flush()
                running, nrun = 0.0, 0
        a = evaluate()
        print(f"== {args.name} ep{ep}  val AUC={a:.4f}  用时 {time.time()-t0:.0f}s",
              flush=True)
        logw.writerow([ep, step, "", round(a, 6), round(time.time() - t0)])
        logf.flush()
        if a >= best:
            best = a
            torch.save({"state_dict": net.state_dict(), "val_auc": a,
                        "image_size": IMAGESIZE, "domains": domains,
                        "name": args.name}, out / "best.pt")
            print(f"   保存 best.pt (AUC={a:.4f})")
    logf.close()
    print(f"\n{args.name} 完成。起点 {a0:.4f} -> best {best:.4f}  ({out/'best.pt'})")


if __name__ == "__main__":
    main()
