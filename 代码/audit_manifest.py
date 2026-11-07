# -*- coding: utf-8 -*-
"""数据审计：跨数据集重复 / clip 交集 / 尺寸与编码一致性

回答三个决定实验设计的问题：
1. 140k 与 real-vs-fake 是否同一批图（决定能否合并训练）
2. faceswap/simswap/DiT 是否共用同一批源视频 clip（决定切分单位）
3. 各域图像尺寸/格式是否一致（决定是否会被朴素管线特征误导）
"""
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

random.seed(0)


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


df = pd.read_csv("manifest.csv", encoding="utf-8")
print(f"manifest: {len(df)} 行\n")

# ---------- 1. GAN 两数据集是否重复 ----------
print("=" * 68)
print("1. 140k  vs  real-vs-fake  是否重复")
print("=" * 68)
a = df[df.dataset == "140k"].sample(600, random_state=0)
b = df[df.dataset == "real-vs-fake"].sample(600, random_state=0)
ma = {md5(p): p for p in a.path}
mb = {md5(p): p for p in b.path}
inter = set(ma) & set(mb)
print(f"  各抽样 600 张，md5 交集: {len(inter)}")
if inter:
    print(f"  示例: {ma[next(iter(inter))]}")
    print(f"        {mb[next(iter(inter))]}")
else:
    print("  -> 无重复，两个数据集相互独立，可合并作为 GAN 域")

# ---------- 2. clip 交集 ----------
print()
print("=" * 68)
print("2. 各方法之间 clip（片段）交集")
print("=" * 68)
sets = {}
for (m, s), g in df.groupby(["method", "src"]):
    sets[f"{m}/{s}"] = set(g.group.unique())
keys = sorted(sets)
print(f"{'':22s}" + "".join(f"{k.split('/')[0][:7]:>9s}" for k in keys))
for k1 in keys:
    row = f"{k1:22s}"
    for k2 in keys:
        n = len(sets[k1] & sets[k2])
        row += f"{n:9d}"
    print(row)
print()
print("列顺序:", ", ".join(keys))
fs_cdf = sets.get("faceswap/cdf", set())
ss_cdf = sets.get("simswap/cdf", set())
dit_cdf = sets.get("dit/cdf", set())
print(f"\nfaceswap/cdf ∩ simswap/cdf = {len(fs_cdf & ss_cdf)}  "
      f"(占 faceswap {len(fs_cdf)} 的 {len(fs_cdf & ss_cdf)/max(1,len(fs_cdf))*100:.1f}%)")
print(f"faceswap/cdf ∩ dit/cdf     = {len(fs_cdf & dit_cdf)}")
print("-> 交集大 = 同源视频，切分必须按 clip 且跨方法共享 group")

# ---------- 3. clip id 规范化是否能跨方法对齐 ----------
print()
print("=" * 68)
print("3. clip id 是否可跨方法对齐（决定 group key 设计）")
print("=" * 68)


def norm_group(method, src, g):
    """把各方法的 clip id 归一化成同一源视频标识"""
    if method == "dit":
        # DiT 的 group = idX_XXXX，需映射回 cdf 的 idA_idB_XXXX 形式不可行，
        # 但 DiT 目录名带 source 标识，先用原始值
        return f"dit|{src}|{g}"
    if method in ("faceswap", "simswap"):
        return f"{src}|{g}"          # 同源同 clip，跨方法共享
    if method == "deepfacelab":
        return f"dfl|{g}"
    return f"{method}|{g}"


df["group_key"] = [norm_group(m, s, g) for m, s, g in
                   zip(df.method, df.src, df.group)]
n_multi = df.groupby("group_key").method.nunique()
print(f"  归一化后 group 数: {df.group_key.nunique()}")
print(f"  被多个 method 共用的 group: {(n_multi > 1).sum()}")

# ---------- 4. 尺寸/格式一致性 ----------
print()
print("=" * 68)
print("4. 各子集图像尺寸与格式（抽样 200）")
print("=" * 68)
import cv2

cv2.setNumThreads(0)
print(f"{'subset':26s} {'size (mode)':>14s} {'ext':>6s} {'ok':>4s}")
for (ds, m, s), g in df.groupby(["dataset", "method", "src"]):
    smp = g.sample(min(200, len(g)), random_state=0)
    sizes, exts, ok = Counter(), Counter(), 0
    for p in smp.path:
        im = cv2.imread(p)
        exts[Path(p).suffix.lower()] += 1
        if im is not None:
            sizes[f"{im.shape[1]}x{im.shape[0]}"] += 1
            ok += 1
    sz = sizes.most_common(1)[0][0] if sizes else "n/a"
    ex = exts.most_common(1)[0][0] if exts else "n/a"
    print(f"{ds+'/'+m+'/'+s:26s} {sz:>14s} {ex:>6s} {ok:4d}")
