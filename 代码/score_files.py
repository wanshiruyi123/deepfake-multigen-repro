# -*- coding: utf-8 -*-
"""对任意图片文件列表，用全部 7 个已训模型打分（中文路径安全）"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50

cv2.setNumThreads(0)
ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SZ = 256
JPEG_Q = 95
MODELS = ["m1_gan", "m2_swap", "m3_diff", "m4_gan_swap", "m5_gan_diff",
          "m6_all", "own_resnet50"]


def imread_cn(path: str):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def reencode(im, q=JPEG_Q):
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else im


def prep(im):
    im = cv2.cvtColor(cv2.resize(im, (SZ, SZ)), cv2.COLOR_BGR2RGB)
    x = (im.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def load(net_dir: str, dev: str):
    ck = torch.load(ROOT / "models" / net_dir / "best.pt",
                    map_location="cpu", weights_only=False)
    net = resnet50(num_classes=1)
    net.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    net.eval().to(dev)
    return net


def main():
    files = sys.argv[1:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = {m: load(m, dev) for m in MODELS}
    print(f"{'文件':22s} " + " ".join(f"{m[:9]:>10s}" for m in MODELS))
    for f in files:
        im = imread_cn(f)
        if im is None:
            print(f"{Path(f).name:22s} UNREADABLE")
            continue
        scores = {}
        for variant, v in (("raw", im), ("uni", reencode(im))):
            x = prep(v).to(dev)
            with torch.inference_mode():
                for m, net in nets.items():
                    scores[(m, variant)] = float(
                        torch.sigmoid(net(x)).item())
        cells = []
        for m in MODELS:
            r, u = scores[(m, "raw")], scores[(m, "uni")]
            cells.append(f"{(r + u) / 2:10.4f}")
        print(f"{Path(f).name:22s} " + " ".join(cells))
    print("\n分数 = 伪造概率（0=真实 1=伪造），raw 与 unified 重编码两路取平均")


if __name__ == "__main__":
    main()
