#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量 Deepfake / AI 生成人脸检测（本地推理，无网络依赖，可断点续跑）

三个预训练模型集成：
  deepfake_vit  Wvolf/ViT_Deepfake_Detection              ViT-base  脸 vs 伪造
  deepfake_v2   prithivMLmods/Deep-Fake-Detector-v2-Model SigLIP/ViT 伪造检测 v2
  ai_image      umm-maybe/AI-image-detector               ViT-base  AI 生成图

用法：
  # 首次：下载模型 + 小样本自检
  python batch_detect.py --input D:/imgs --limit 50 --out sample.csv

  # 全量（4 万张，支持中断后续跑）
  python batch_detect.py --input D:/imgs --out results.csv

  # 关掉人脸裁剪 / 改阈值 / 加进程
  python batch_detect.py --input D:/imgs --out r.csv --no-face-crop --threshold 0.65 --workers 8

输出 results.csv 列：
  path,status,face,m_vit,m_v2,m_ai,score,decision,ms
  score      = 两个 deepfake 模型的加权均值（0-1，越高越可能伪造）
  decision   = fake / real / error
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 400_000_000

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# (key, repo_id, 本地目录名, 默认权重) —— 权重只作用于 deepfake 两个模型；ai_image 单独列
# 注意：umm-maybe/AI-image-detector 已移除。作者在模型卡中明确说明它
# 「not a deepfake photo detector」，只针对 2022 年前的 AI 艺术图，
# 对人脸照片会产生大量误报（实测真实照片被骗到 0.865）。要覆盖扩散生成的
# 整脸图像，请改用 Organika/sdxl-detector 或 NPR，别用这个。
MODELS = [
    ("m_vit", "Wvolf/ViT_Deepfake_Detection", "wvolf_vit_deepfake", 0.5),
    ("m_v2", "prithivMLmods/Deep-Fake-Detector-v2-Model", "prithiv_v2", 0.5),
]
MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"


# --------------------------------------------------------------------------- #
# 人脸裁剪（YuNet 优先，缺失则 Haar）
# --------------------------------------------------------------------------- #
class FaceCropper:
    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self._det = None
        self._kind = None
        if mode == "off":
            return
        try:
            import cv2

            model_path = Path(__file__).with_name("face_detection_yunet_2023mar.onnx")
            if model_path.exists():
                self._det = cv2.FaceDetectorYN.create(
                    str(model_path), "", (320, 320), 0.6, 0.3, 5000
                )
                self._kind = "yunet"
            else:
                xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._cascade = cv2.CascadeClassifier(xml)
                self._kind = "haar"
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] 人脸检测不可用，跳过裁剪: {e}", file=sys.stderr)
            self._kind = None

    def crop(self, img: Image.Image) -> tuple[Image.Image, int]:
        """返回 (裁剪后图像, 是否检出人脸 0/1)。"""
        if self._kind is None:
            return img, 0
        import cv2

        w, h = img.size
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        box = None
        if self._kind == "yunet":
            self._det.setInputSize((w, h))
            _, faces = self._det.detect(bgr)
            if faces is not None and len(faces):
                f = max(faces, key=lambda r: r[2] * r[3])
                box = (f[0], f[1], f[0] + f[2], f[1] + f[3])
        else:
            import cv2 as _cv2

            gray = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)
            rects = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(rects):
                x, y, ww, hh = max(rects, key=lambda r: r[2] * r[3])
                box = (x, y, x + ww, y + hh)

        if box is None:
            return img, 0

        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        ratio = (bw * bh) / float(w * h)
        if ratio < 0.01 or ratio > 0.85:             # 太小/太大都不可信
            return img, 0
        mx, my = bw * 0.35, bh * 0.35
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return img, 0
        return img.crop((int(x1), int(y1), int(x2), int(y2))), 1


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
_CROPPER: FaceCropper | None = None


def _worker_init(mode: str, _wid: int = 0) -> None:
    global _CROPPER
    _CROPPER = FaceCropper(mode)
    torch.set_num_threads(1)


class ImageFolder(Dataset):
    def __init__(self, paths: list[str], face_mode: str):
        self.paths = paths
        self.face_mode = face_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        try:
            img = Image.open(p)
            img = img.convert("RGB")
            try:
                from PIL import ImageOps

                img = ImageOps.exif_transpose(img)
            except Exception:                        # noqa: BLE001
                pass
            if max(img.size) > 2048:                 # 超大图先降采样，省解码时间
                img.thumbnail((2048, 2048))
            face = 0
            global _CROPPER
            if _CROPPER is None:
                _CROPPER = FaceCropper(self.face_mode)
            if _CROPPER.mode != "off":
                img, face = _CROPPER.crop(img)
            return p, img, face, ""
        except Exception as e:                       # noqa: BLE001
            return p, None, 0, f"{type(e).__name__}: {e}"


def collate(batch):
    paths = [b[0] for b in batch]
    imgs = [b[1] for b in batch]
    faces = [b[2] for b in batch]
    errs = [b[3] for b in batch]
    return paths, imgs, faces, errs


# --------------------------------------------------------------------------- #
def resolve_fake_idx(id2label: dict) -> int:
    labs = {int(k): str(v).lower() for k, v in id2label.items()}
    for i, l in labs.items():
        if "real" in l or "authentic" in l or "nature" in l:
            others = [k for k in labs if k != i]
            return others[0] if others else i
    for i, l in labs.items():
        if any(k in l for k in ("fake", "deepfake", "artificial", "synthetic",
                                "manipulated", "ai", "generated")):
            return i
    return max(labs)


def load_model(repo: str, local_slug: str):
    """优先用 models/<slug>/ 本地目录（离线可用），否则回落到 Hub。"""
    local = MODEL_ROOT / local_slug
    src = str(local) if (local / "config.json").exists() else repo
    last = None
    for kwargs in ({}, {"subfolder": "checkpoint-5252"}):
        try:
            proc = AutoImageProcessor.from_pretrained(src, **kwargs)
            model = AutoModelForImageClassification.from_pretrained(src, **kwargs)
            model.eval()
            return proc, model, resolve_fake_idx(model.config.id2label), str(src)
        except Exception as e:                       # noqa: BLE001
            last = e
    raise RuntimeError(f"加载 {repo} 失败: {last}")


def scan(root: str) -> list[str]:
    out = []
    if os.path.isfile(root):
        return [root]
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def load_done(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("path") and row.get("status") == "ok":
                done.add(row["path"])
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="图片目录（递归）")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--high", type=float, default=0.8, help="高危阈值")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--face-crop", default="auto", choices=["auto", "off"])
    ap.add_argument("--resume", action="store_true", help="跳过已完成的图片")
    ap.add_argument("--shard", default="0/1", help="分片 i/n，开多进程并行时用，如 0/4 1/4 ...")
    args = ap.parse_args()

    paths = scan(args.input)
    if args.limit:
        paths = paths[: args.limit]

    try:
        si, sn = (int(x) for x in args.shard.split("/"))
        assert 0 <= si < sn
    except Exception:                                # noqa: BLE001
        raise SystemExit("--shard 格式应为 i/n，例如 0/4")
    if sn > 1:
        paths = [p for i, p in enumerate(paths) if i % sn == si]
        if args.out == "results.csv":
            root, ext = os.path.splitext("results.csv")
            args.out = f"{root}.shard{si}{ext}"
    if not paths:
        raise SystemExit(f"未找到图片: {args.input}")

    done = load_done(args.out) if args.resume else set()
    todo = [p for p in paths if p not in done]
    print(f"总计 {len(paths)} 张，已完成 {len(done)}，待处理 {len(todo)}")

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // max(1, args.workers)))

    print("加载模型…")
    loaded = []
    for key, repo, slug, w in MODELS:
        t0 = time.time()
        proc, model, fake_idx, src = load_model(repo, slug)
        labels = {int(k): str(v) for k, v in model.config.id2label.items()}
        print(f"  {key:12s} fake={labels[fake_idx]!r:14s} {time.time()-t0:5.1f}s  {src}")
        loaded.append((key, proc, model, fake_idx, w))

    ds = ImageFolder(todo, args.face_crop)
    dl = DataLoader(
        ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
        worker_init_fn=partial(_worker_init, args.face_crop),
        prefetch_factor=2 if args.workers else None,
        persistent_workers=args.workers > 0,
    )

    keys = [k for k, *_ in loaded]
    header = ["path", "status", "face", *keys, "score", "decision", "ms"]
    new_file = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=header)
    if new_file:
        writer.writeheader()

    t_start = time.time()
    n_ok = n_err = 0
    fake_hits = 0
    try:
        for paths_b, imgs, faces, errs in dl:
            idx_ok = [i for i, im in enumerate(imgs) if im is not None]
            scores = {key: [None] * len(paths_b) for key, *_ in loaded}

            if idx_ok:
                pil = [imgs[i] for i in idx_ok]
                for key, proc, model, fake_idx, _w in loaded:
                    with torch.no_grad():
                        inputs = proc(images=pil, return_tensors="pt")
                        logits = model(**inputs).logits
                        probs = F.softmax(logits, dim=-1)[:, fake_idx].tolist()
                    for j, p in zip(idx_ok, probs):
                        scores[key][j] = round(float(p), 6)

            ms = int(1000 * (time.time() - t_start) / max(1, n_ok + n_err + 1))
            for i, p in enumerate(paths_b):
                if imgs[i] is None:
                    row = {"path": p, "status": "error", "face": 0,
                           "score": "", "decision": "error", "ms": ms}
                    row.update({k: "" for k in keys})
                    n_err += 1
                    writer.writerow(row)
                    continue
                per = {k: scores.get(k, [None])[i] for k in keys}
                vals = [x for x in per.values() if x is not None]
                score = round(float(np.mean(vals)), 6) if vals else ""
                dec = "fake" if (score != "" and score >= args.threshold) else "real"
                if dec == "fake":
                    fake_hits += 1
                row = {"path": p, "status": "ok", "face": faces[i],
                       "score": score, "decision": dec, "ms": ms}
                row.update(per)
                writer.writerow(row)
                n_ok += 1

            fh.flush()
            total = n_ok + n_err
            rate = total / max(1e-9, time.time() - t_start)
            eta = (len(todo) - total) / rate if rate > 0 else 0
            print(f"\r{n_ok + n_err}/{len(todo)}  ok={n_ok} err={n_err} "
                  f"fake={fake_hits}  {rate:.1f} img/s  ETA {eta/60:.1f} min   ",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\n中断，已写入的结果可直接用 --resume 续跑")
    finally:
        fh.close()

    dur = time.time() - t_start
    print(f"\n完成 {n_ok} 张（失败 {n_err}），用时 {dur/60:.1f} min，"
          f"均速 {n_ok/max(1e-9,dur):.1f} img/s")
    print(f"判定伪造 {fake_hits} 张（阈值 {args.threshold}）-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
