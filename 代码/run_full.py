# -*- coding: utf-8 -*-
"""全量批量判定：4~5 模型打分 + 分档融合 + 双阈值三分区

用法:
    python 代码/run_full.py --data "E:/初赛验证集（A题）/初赛验证集" --out full_result.csv
    python 代码/run_full.py ... --resume            # 断点续跑
    python 代码/run_full.py ... --limit 500         # 只跑前 500 张（冒烟）

输出列:
    file,rel_dir,w,h,ext,bytes,tier,
    s_m1_gan,s_m2_swap,s_m3_diff,s_m6_all,s_own,
    fused,decision,confidence,pred

    decision : fake / real / uncertain
    pred     : 0/1，灰区按 0.5 强行二分（给需要单一标签的提交用）
    confidence: high(fake|real) / low(uncertain)
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
from torchvision.models import resnet50

cv2.setNumThreads(0)

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SZ = 256
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# ---- 决策配置（先验起点，见 docs：需用 300~500 张人工标注重标定） ----
# tier A: 对齐人脸小图（<=300px），训练分布主项
# tier B: 大方图（>=512 且方形），高分辨率生成图
# tier C: 非方/其它，训练分布外
TIER_WEIGHTS = {
    "A": {"m6_all": 0.50, "m2_swap": 0.30, "m3_diff": 0.20},
    "B": {"m1_gan": 0.50, "m6_all": 0.30, "m2_swap": 0.20},
    "C": {"m6_all": 1.00},
}
TIER_HI = {"A": 0.85, "B": 0.85, "C": 0.85}
TIER_LO = {"A": 0.15, "B": 0.15, "C": 0.15}
TIER_TRUST = {"A": "normal", "B": "normal", "C": "low"}   # C 档整档降级


def imread_cn(path):
    """中文路径安全读图"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def prep(im):
    im = cv2.resize(im, (SZ, SZ), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((im - MEAN) / STD).transpose(2, 0, 1)


def tier_of(w, h, small=300, big=512):
    if max(w, h) <= small:
        return "A"
    if w == h and w >= big:
        return "B"
    return "C"


def load_net(name, dev):
    ck = torch.load(ROOT / "models" / name / "best.pt", map_location="cpu",
                    weights_only=False)
    net = resnet50(num_classes=1)
    net.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    return net.eval().to(dev)


# ---- YuNet（C 档场景照的人脸裁剪路径） ----
_YUNET = None


def yunet():
    global _YUNET
    if _YUNET is None:
        p = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
        _YUNET = (cv2.FaceDetectorYN_create(str(p), "", (320, 320), 0.6)
                  if p.exists() else False)
    return _YUNET


def crop_face(im, margin=0.3):
    """返回裁出的人脸区域；检不到返回 None"""
    det = yunet()
    if det is False:
        return None
    h, w = im.shape[:2]
    det.setInputSize((w, h))
    ok, faces = det.detect(im)
    if faces is None or len(faces) == 0:
        return None
    x, y, fw, fh = faces[int(np.argmax(faces[:, 14]))][:4].astype(int)
    mx, my = int(fw * margin), int(fh * margin)
    return im[max(0, y - my):min(h, y + fh + my),
              max(0, x - mx):min(w, x + fw + mx)]


def decide(tier, sc):
    w = TIER_WEIGHTS[tier]
    tot = sum(w.values())
    fused = sum(sc.get(k, 0.0) * v for k, v in w.items()) / tot
    hi, lo = TIER_HI[tier], TIER_LO[tier]
    if fused >= hi:
        dec, conf = "fake", "high"
    elif fused <= lo:
        dec, conf = "real", "high"
    else:
        dec, conf = "uncertain", "low"
    if TIER_TRUST[tier] == "low":          # C 档：整档降级，进复核队列
        conf = "low"
    return round(fused, 4), dec, conf, int(fused >= 0.5)


def collect(data_dir):
    files = []
    for p in Path(data_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            files.append(p)
    files.sort(key=lambda p: str(p))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="full_result.csv")
    ap.add_argument("--models", default="m1_gan,m2_swap,m3_diff,m6_all")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hi", type=float, default=None, help="覆盖所有档位的判伪阈值")
    ap.add_argument("--lo", type=float, default=None, help="覆盖所有档位的判真阈值")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config", default=None,
                    help="thresholds.json（calibrate_tiers.py 产出），覆盖各档位 hi/lo")
    args = ap.parse_args()

    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for t, v in (cfg.get("tiers") or {}).items():
            if t in TIER_HI:
                TIER_HI[t] = float(v["hi"])
                TIER_LO[t] = float(v["lo"])
        print(f"已载入标定阈值 {args.config}: {TIER_HI}", flush=True)

    if args.hi is not None:
        for k in TIER_HI:
            TIER_HI[k] = args.hi
    if args.lo is not None:
        for k in TIER_LO:
            TIER_LO[k] = args.lo

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = collect(args.data)
    if args.limit:
        files = files[: args.limit]
    print(f"待处理 {len(files)} 张 | 模型 {models} | 设备 {dev}", flush=True)

    out_path = Path(args.out)
    done = set()
    if args.resume and out_path.exists():
        with out_path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                done.add((r.get("file") or "").strip())
        files = [f for f in files if f.name not in done]
        print(f"续跑：已完成 {len(done)}，剩余 {len(files)}", flush=True)

    nets = {m: load_net(m, dev) for m in models}
    print("模型加载完成", flush=True)

    header = (["file", "rel_dir", "w", "h", "ext", "bytes", "tier"]
              + [f"s_{m}" for m in models]
              + ["s_whole_fused", "fused", "decision", "confidence", "pred"])
    new_file = not out_path.exists() or not args.resume
    fh = out_path.open("w" if new_file else "a", encoding="utf-8-sig", newline="")
    wr = csv.writer(fh)
    if new_file:
        wr.writerow(header)

    t0 = time.time()
    buf, meta = [], []
    stats = {"A": 0, "B": 0, "C": 0}
    dec_stats = {"fake": 0, "real": 0, "uncertain": 0}

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(dev)
        scores = {}
        with torch.inference_mode():
            for m, net in nets.items():
                scores[m] = torch.sigmoid(net(x)).squeeze(1).float().cpu().numpy()
            # C 档：额外前向一次人脸裁剪图（与整图同 batch）
            idx_c = [i for i, md in enumerate(meta)
                     if md["tier"] == "C" and md.get("crop") is not None]
            scores_c = {}
            if idx_c:
                xc = torch.from_numpy(np.stack([meta[i]["crop"] for i in idx_c])).to(dev)
                for m, net in nets.items():
                    scores_c[m] = torch.sigmoid(net(xc)).squeeze(1).float().cpu().numpy()
        pos_c = {k: i for i, k in enumerate(idx_c)}
        for i, md in enumerate(meta):
            sc = {m: float(scores[m][i]) for m in models}
            tier = md["tier"]
            whole_fused, _, _, _ = decide(tier, sc)
            if tier == "C" and i in pos_c:
                scc = {m: float(scores_c[m][pos_c[i]]) for m in models}
                fused, dec, conf, pred = decide(tier, scc)   # C 档用人脸裁剪分
            else:
                fused, dec, conf, pred = decide(tier, sc)
            stats[tier] += 1
            dec_stats[dec] += 1
            wr.writerow([md["file"], md["rel_dir"], md["w"], md["h"], md["ext"],
                         md["bytes"], tier]
                        + [round(sc.get(m, 0.0), 4) for m in models]
                        + [whole_fused, fused, dec, conf, pred])
        buf.clear()
        meta.clear()

    n_err = 0
    for i, p in enumerate(files, 1):
        im = imread_cn(p)
        if im is None:
            n_err += 1
            continue
        h, w = im.shape[:2]
        tier = tier_of(w, h)
        try:
            rel = str(p.parent.relative_to(args.data))
        except ValueError:
            rel = ""
        buf.append(prep(im))
        meta.append({"file": p.name, "rel_dir": rel, "w": w, "h": h,
                     "ext": p.suffix.lower(), "bytes": p.stat().st_size,
                     "tier": tier, "crop": None})
        if tier == "C":                       # 场景照：额外走人脸裁剪路径
            fc = crop_face(im)
            if fc is not None and fc.size > 0 and min(fc.shape[:2]) >= 32:
                meta[-1]["crop"] = prep(fc)
        if len(buf) >= args.batch:
            flush()
        if i % 2000 == 0:
            el = time.time() - t0
            eta = el / i * (len(files) - i)
            print(f"  {i}/{len(files)}  已用 {el/60:.1f}min  预计剩余 {eta/60:.1f}min",
                  flush=True)
    flush()
    fh.close()

    el = time.time() - t0
    summary = {
        "data": str(args.data), "n": len(files), "unreadable": n_err,
        "elapsed_min": round(el / 60, 2),
        "tier": stats, "decision": dec_stats,
        "thresholds": {"hi": TIER_HI, "lo": TIER_LO},
        "weights": TIER_WEIGHTS,
    }
    Path(str(out_path) + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 52)
    print(f"完成 {len(files)} 张，不可读 {n_err}，耗时 {el/60:.1f} 分钟")
    print(f"档位分布  {stats}")
    print(f"判定分布  {dec_stats}")
    tot = sum(dec_stats.values()) or 1
    print(f"判伪率 {dec_stats['fake']/tot*100:.1f}%  判真率 {dec_stats['real']/tot*100:.1f}%"
          f"  灰区 {dec_stats['uncertain']/tot*100:.1f}%")
    print(f"结果 → {out_path}")


if __name__ == "__main__":
    sys.exit(main())
