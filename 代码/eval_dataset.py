# -*- coding: utf-8 -*-
"""在自带标签的 real-vs-fake 数据集上评测模型组合。

用法:
  .venv/Scripts/python.exe batch/eval_dataset.py --per-class 1000 --seed 42 \
      --out eval_sample.csv

输出:
  1) eval_sample.csv  每张图的逐模型分数与真实标签
  2) stdout 指标: 每模型 AUC / acc@0.5 / TPR@FPR1% / TPR@FPR5%，
     以及校准/验证两半划分后的最优双模型权重组合
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet50
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from batch_detect import MODELS, MODEL_ROOT, resolve_fake_idx  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_hf(repo_key: str):
    repo, local = next((r, l) for k, r, l, _ in MODELS if k == repo_key)
    d = str(MODEL_ROOT / local)
    proc = AutoImageProcessor.from_pretrained(d, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(d, local_files_only=True)
    model.eval()
    fake_idx = resolve_fake_idx(model.config.id2label)

    def run(rgb_list):
        inp = proc(images=rgb_list, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inp).logits
        return torch.softmax(logits, dim=-1)[:, fake_idx].tolist()

    return run


def load_cnndet(size: int):
    pth = MODEL_ROOT / "cnndetection" / "blur_jpg_prob0.5.pth"
    sd = torch.load(pth, map_location="cpu", weights_only=False)["model"]
    net = resnet50(num_classes=1)
    net.load_state_dict(sd)
    net.eval()

    def run(rgb_list):
        arr = np.stack([
            cv2.resize(im, (size, size)).astype(np.float32) / 255.0
            for im in rgb_list
        ])
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(arr).permute(0, 3, 1, 2)
        with torch.inference_mode():
            logits = net(x).squeeze(1)
        return torch.sigmoid(logits).tolist()

    return run


def load_own(size: int = 256):
    """自训模型 models/own_resnet50/best.pt"""
    pth = MODEL_ROOT / "own_resnet50" / "best.pt"
    ck = torch.load(pth, map_location="cpu", weights_only=False)
    size = int(ck.get("image_size", size))
    net = resnet50(num_classes=1)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()

    def run(rgb_list):
        arr = np.stack([
            cv2.resize(im, (size, size)).astype(np.float32) / 255.0
            for im in rgb_list
        ])
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(arr).permute(0, 3, 1, 2)
        if torch.cuda.is_available():
            x = x.cuda()
        with torch.inference_mode():
            logits = net(x).squeeze(1)
        return torch.sigmoid(logits.float()).cpu().tolist()

    return run


def auc(y: np.ndarray, s: np.ndarray) -> float:
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float):
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0:
        return float("nan"), float("nan")
    k = max(1, int(round(fpr_target * len(neg))))
    thr = float(neg[k - 1])
    return thr, float((s[y == 1] >= thr).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(paths.RVF))
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--out", default="eval_sample.csv")
    ap.add_argument("--models", default="cnn128,cnn256,m_vit,m_v2",
                    help="逗号分隔的模型名")
    args = ap.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)
    items = []
    for cls, lab in (("real", 0), ("fake", 1)):
        files = sorted((root / cls).glob("*.jpg"))
        rng.shuffle(files)
        items += [(str(f), lab) for f in files[: args.per_class]]
    print(f"抽样 {len(items)} 张 (real={sum(1 for _, l in items if l == 0)}, "
          f"fake={sum(1 for _, l in items if l == 1)})")

    torch.set_num_threads(max(1, (torch.get_num_threads() or 8) // 2))
    runners = {}
    for key in [k.strip() for k in args.models.split(",") if k.strip()]:
        t0 = time.time()
        if key.startswith("cnn"):
            runners[key] = load_cnndet(int(key[3:]))
        elif key == "own":
            runners[key] = load_own()
        else:
            runners[key] = load_hf(key)
        print(f"  {key} 加载 {time.time() - t0:.1f}s")

    rows, bs = [], args.batch
    t0 = time.time()
    for st in range(0, len(items), bs):
        chunk = items[st : st + bs]
        imgs = [cv2.imread(p) for p, _ in chunk]
        ok_idx = [i for i, im in enumerate(imgs) if im is not None]
        rgb = [cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB) for i in ok_idx]
        per_model = {k: [None] * len(chunk) for k in runners}
        for key, run in runners.items():
            if not rgb:
                continue
            for i, p in zip(ok_idx, run(rgb)):
                per_model[key][i] = round(float(p), 6)
        for i, (p, lab) in enumerate(chunk):
            if imgs[i] is None:
                rows.append({"path": p, "label": lab, "status": "error"})
                continue
            row = {"path": p, "label": lab, "status": "ok"}
            row.update({k: per_model[k][i] for k in runners})
            rows.append(row)
        done = st + len(chunk)
        if done % (bs * 10) == 0 or done == len(items):
            el = time.time() - t0
            print(f"  {done}/{len(items)}  {done / el:.1f} img/s")

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    ok = df[df["status"] == "ok"].copy()
    y = ok["label"].to_numpy()
    print(f"\n有效 {len(ok)} / {len(df)} 张，推理耗时 {time.time() - t0:.0f}s")
    cols = [c for c in ok.columns if c not in ("path", "label", "status")]

    half = len(ok) // 2
    cal, val = ok.iloc[:half], ok.iloc[half:]
    yc, yv = cal["label"].to_numpy(), val["label"].to_numpy()

    print(f"\n{'模型':8s} {'AUC':>7s} {'acc@0.5':>9s} {'TPR@FPR1%':>11s} {'TPR@FPR5%':>11s}   (全量)")
    for c in cols:
        s = ok[c].astype(float).to_numpy()
        a = auc(y, s)
        acc = ((s >= 0.5).astype(int) == y).mean()
        t1 = tpr_at_fpr(y, s, 0.01)
        t5 = tpr_at_fpr(y, s, 0.05)
        print(f"{c:8s} {a:7.4f} {acc:9.4f} {t1[1]:11.4f} {t5[1]:11.4f}")

    if len(cols) >= 2:
        best = None
        for w in np.arange(0.0, 1.0001, 0.025):
            sc = (w * cal[cols[0]].astype(float)
                  + (1 - w) * cal[cols[1]].astype(float)).to_numpy()
            a = auc(yc, sc)
            if best is None or a > best[1]:
                best = (w, a)
        w = best[0]
        sv = (w * val[cols[0]].astype(float)
              + (1 - w) * val[cols[1]].astype(float)).to_numpy()
        acc = ((sv >= 0.5).astype(int) == yv).mean()
        t1 = tpr_at_fpr(yv, sv, 0.01)
        print(f"\n最优双模型组合: {cols[0]}={w:.3f} + {cols[1]}={1 - w:.3f}")
        print(f"验证集: AUC={auc(yv, sv):.4f}  acc@0.5={acc:.4f}  "
              f"TPR@FPR1%={t1[1]:.4f} (thr={t1[0]:.4f})")


if __name__ == "__main__":
    main()
