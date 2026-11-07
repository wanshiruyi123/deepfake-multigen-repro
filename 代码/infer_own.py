# -*- coding: utf-8 -*-
"""用自训模型 (models/own_resnet50/best.pt) 批量推理任意图片目录。

用法:
  .venv/Scripts/python.exe batch/infer_own.py --input D:/images --out results.csv
特性:
  - GPU 优先，自动降级 CPU
  - CSV 逐批落盘，可中断后 --resume 续跑
  - 输出列: path, score, decision, ms
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import resnet50

IMGSZ = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scan(root: Path) -> list[str]:
    if root.is_file():
        return [str(root)]
    out = []
    for p in root.rglob("*"):
        if p.suffix.lower() in EXTS:
            out.append(str(p))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="图片目录或单文件")
    ap.add_argument("--out", default="own_results.csv")
    ap.add_argument("--weights", default=str(Path(__file__).resolve().parents[1]
                                            / "models" / "own_resnet50" / "best.pt"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    sz = int(ck.get("image_size", IMGSZ))
    net = resnet50(num_classes=1)
    net.load_state_dict(ck["state_dict"])
    net.eval().to(dev).to(memory_format=torch.channels_last)
    print(f"device={dev}  weights={args.weights}  (训练时 val AUC={ck.get('val_auc')})")

    m_g = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    s_g = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    paths = scan(Path(args.input))
    done = set()
    if args.resume and Path(args.out).exists():
        with open(args.out, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add(row["path"])
        print(f"续跑：已有 {len(done)} 条完成记录")
    todo = [p for p in paths if p not in done]
    print(f"总计 {len(paths)} 张，本次处理 {len(todo)} 张")

    new = not (args.resume and Path(args.out).exists())
    f = open(args.out, "a" if not new else "w", newline="", encoding="utf-8-sig")
    w = csv.writer(f)
    if new:
        w.writerow(["path", "score", "decision", "ms"])

    bs = args.batch
    t0 = time.time()
    n_ok = 0
    for st in range(0, len(todo), bs):
        chunk = todo[st : st + bs]
        arr = np.empty((len(chunk), sz, sz, 3), dtype=np.float32)
        ms_list = []
        for i, p in enumerate(chunk):
            t1 = time.time()
            im = cv2.imread(p)
            if im is None:
                arr[i] = 0.0
            else:
                im = cv2.cvtColor(cv2.resize(im, (sz, sz)), cv2.COLOR_BGR2RGB)
                arr[i] = im.astype(np.float32) / 255.0
            ms_list.append(round((time.time() - t1) * 1000, 1))
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(dev)
        x = (x - m_g) / s_g
        with torch.inference_mode():
            s = torch.sigmoid(net(x.to(memory_format=torch.channels_last))
                              .squeeze(1)).float().cpu().numpy()
        for i, p in enumerate(chunk):
            sc = round(float(s[i]), 6)
            w.writerow([p, sc,
                        "fake" if sc >= args.threshold else "real", ms_list[i]])
        f.flush()
        n_ok += len(chunk)
        done_n = st + len(chunk)
        if done_n % (bs * 20) == 0 or done_n == len(todo):
            el = time.time() - t0
            eta = (len(todo) - done_n) / max(n_ok / el, 0.1)
            print(f"  {done_n}/{len(todo)}  {n_ok / el:.0f} img/s  ETA {eta / 60:.0f} min")
    f.close()
    print(f"完成: {args.out}")


if __name__ == "__main__":
    main()
