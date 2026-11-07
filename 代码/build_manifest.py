# -*- coding: utf-8 -*-
"""统一扫描三份数据集，输出 manifest.csv

字段:
  path        绝对路径
  label       0=real 1=fake
  dataset     数据集名
  method      生成方法 (gan/stylegan, faceswap, simswap, deepfacelab, dit)
  domain      gan / swap / diffusion
  group       视频片段/图像族 ID —— 切分必须以它为单位，否则相邻帧泄漏
  src         cdf / ff / 140k / celeba
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(d: Path):
    out = []
    for root, _dirs, files in os.walk(d):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXT:
                out.append(os.path.join(root, f))
    return out


def add(rows, paths, label, dataset, method, domain, group_fn, src):
    for p in paths:
        rows.append({
            "path": p.replace("\\", "/"),
            "label": label,
            "dataset": dataset,
            "method": method,
            "domain": domain,
            "group": group_fn(p),
            "src": src,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manifest.csv")
    args = ap.parse_args()

    rows = []

    # ---------- 1. 140k-Real-and-Fake-Faces (GAN 域) ----------
    base = paths.HF140K
    if base.exists():
        for split in ("train", "validation", "test"):
            for cls, lab in (("real", 0), ("fake", 1)):
                d = base / split / cls
                if not d.exists():
                    continue
                ps = list_images(d)
                # 每张图独立（GAN 生成图之间无视频关系）
                add(rows, ps, lab, "140k", "gan", "gan",
                    lambda p: Path(p).stem, "140k")

    # ---------- 2. DF40 (换脸 + 扩散) ----------
    df40 = paths.DF40
    if df40.exists():
        # deepfacelab: fake/frames/<clip>/  real/frames/<clip>/
        for cls, lab in (("fake", 1), ("real", 0)):
            d = df40 / "deepfacelab" / cls / "frames"
            if d.exists():
                ps = list_images(d)
                add(rows, ps, lab, "DF40", "deepfacelab", "swap",
                    lambda p: Path(p).parent.name, "celeba")

        # faceswap / simswap: <sub>/frames/<clip>/
        for method in ("faceswap", "simswap"):
            for sub in ("cdf", "ff"):
                d = df40 / method / sub / "frames"
                if not d.exists():
                    continue
                ps = list_images(d)
                add(rows, ps, 1, "DF40", method, "swap",
                    lambda p: Path(p).parent.name, sub)

        # DiT: <sub>/Fake_from_*/<clip>/<frame>
        for sub in ("cdf", "ff"):
            d = df40 / "DiT" / sub
            if not d.exists():
                continue
            ps = list_images(d)
            add(rows, ps, 1, "DF40", "dit", "diffusion",
                lambda p: Path(p).parent.name, sub)

    # ---------- 3. CelebDF_test_aligned ----------
    cdf = paths.CELEBDF
    if cdf.exists():
        for cls, lab in (("real", 0), ("fake", 1)):
            d = cdf / cls
            if not d.exists():
                continue
            ps = list_images(d)
            # clip id 形如 id0_id16_0003
            def g(p):
                stem = Path(p).stem
                m = re.match(r"(id\d+_id\d+_\d+|id\d+_\d+)", stem)
                return m.group(1) if m else stem
            add(rows, ps, lab, "CelebDF", "celebreal", "swap", g, "cdf")

    # ---------- 4. 已训过的 real-vs-fake（用于查重与对照） ----------
    rvf = paths.RVF
    if rvf.exists():
        for cls, lab in (("real", 0), ("fake", 1)):
            d = rvf / cls
            if d.exists():
                ps = list_images(d)
                add(rows, ps, lab, "real-vs-fake", "gan", "gan",
                    lambda p: Path(p).stem, "rvf")

    if not rows:
        sys.exit("未扫描到任何图片")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "dataset",
                                          "method", "domain", "group", "src"])
        w.writeheader()
        w.writerows(rows)

    # ---------- 统计 ----------
    print(f"总计 {len(rows)} 张 -> {args.out}\n")
    agg = defaultdict(lambda: Counter())
    for r in rows:
        agg[f"{r['dataset']}/{r['method']}/{r['src']}"][r["label"]] += 1
    print(f"{'dataset/method/src':38s} {'real':>8s} {'fake':>8s} {'group':>7s}")
    print("-" * 66)
    groups = defaultdict(set)
    for r in rows:
        groups[f"{r['dataset']}/{r['method']}/{r['src']}"].add(r["group"])
    for k in sorted(agg):
        c = agg[k]
        print(f"{k:38s} {c.get(0,0):8d} {c.get(1,0):8d} {len(groups[k]):7d}")

    print("\n按 domain:")
    for dom in ("gan", "swap", "diffusion"):
        sub = [r for r in rows if r["domain"] == dom]
        print(f"  {dom:10s} total={len(sub):7d}  real={sum(1 for r in sub if r['label']==0):6d}"
              f"  fake={sum(1 for r in sub if r['label']==1):6d}")


if __name__ == "__main__":
    main()
