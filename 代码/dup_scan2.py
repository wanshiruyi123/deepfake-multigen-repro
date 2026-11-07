# -*- coding: utf-8 -*-
"""近重复精确检测（分桶法，避免自比较与重复计数）。

先把全量 md5+dHash 落盘缓存，缓存存在则直接复用。
输出:
  - 完全重复组数、跨类完全重复
  - dHash 距离<=6 的真实图片对数量、涉及图片数
  - 这些近重复对里有多少跨越 train/test 边界（真正的泄漏风险）
"""
from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _paths  # noqa: E402

import numpy as np
import pandas as pd
from PIL import Image

ROOT = _paths.RVF
CACHE = Path("dup_hashes.csv")


def dhash(im: Image.Image, size: int = 8) -> int:
    g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def work(p: str):
    try:
        md5 = hashlib.md5(Path(p).read_bytes()).hexdigest()
        with Image.open(p) as im:
            dh = dhash(im)
    except Exception:  # noqa: BLE001
        return None
    cls = "fake" if "/fake/" in p.replace("\\", "/") else "real"
    return (p, cls, md5, dh)


def build_cache() -> pd.DataFrame:
    files = [str(p) for p in ROOT.rglob("*.jpg")]
    print(f"扫描 {len(files)} 文件 ...", flush=True)
    with Pool(10) as pool:
        res = [r for r in pool.imap_unordered(work, files, chunksize=300) if r]
    df = pd.DataFrame(res, columns=["path", "cls", "md5", "dhash"])
    df.to_csv(CACHE, index=False, encoding="utf-8-sig")
    print(f"缓存写入 {CACHE}")
    return df


def band_index(dh: np.ndarray, bands: int = 4) -> dict:
    idx = defaultdict(list)
    bits = 64 // bands
    for i, v in enumerate(dh):
        for b in range(bands):
            key = (b, (int(v) >> (b * bits)) & ((1 << bits) - 1))
            idx[key].append(i)
    return idx


def main() -> None:
    df = pd.read_csv(CACHE) if CACHE.exists() else build_cache()
    print(f"样本 {len(df)}  real={int((df.cls == 'real').sum())} "
          f"fake={int((df.cls == 'fake').sum())}")

    # 完全重复
    g = df.groupby("md5")
    dup = g.filter(lambda x: len(x) > 1)
    print(f"\n[完全重复] {len(dup) // 2 if len(dup) else 0} 张涉及，"
          f"{dup['md5'].nunique() if len(dup) else 0} 组"
          + (f"，跨类组数 {dup.groupby('md5')['cls'].nunique().gt(1).sum()}"
             if len(dup) else ""))

    # 近重复：分桶 + 精确 Hamming
    dh = df["dhash"].to_numpy(np.uint64)
    idx = band_index(dh)
    pairs = set()
    for lst in idx.values():
        if len(lst) < 2 or len(lst) > 400:      # 跳过超大桶（近似均匀纹理）
            continue
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                i, j = lst[a], lst[b]
                if bin(int(dh[i] ^ dh[j])).count("1") <= 6:
                    pairs.add((i, j) if i < j else (j, i))
    print(f"[近似重复] 真实图片对 {len(pairs)} 对，涉及图片 "
          f"{len({i for ij in pairs for i in ij})} 张")

    if pairs:
        from train_own import build_split  # noqa: E402
        tr = {p for p, _ in build_split(ROOT, 30000)[0]}
        paths = df["path"].tolist()
        cross_split = sum(1 for i, j in pairs
                          if (paths[i] in tr) != (paths[j] in tr))
        cross_cls = sum(1 for i, j in pairs
                        if df.cls.iloc[i] != df.cls.iloc[j])
        print(f"           其中跨 train/test 边界 {cross_split} 对，"
              f"跨 real/fake {cross_cls} 对")
        # 抽样展示
        for i, j in list(pairs)[:3]:
            print(f"           {Path(paths[i]).parent.name}/{Path(paths[i]).name}"
                  f"  <->  {Path(paths[j]).parent.name}/{Path(paths[j]).name}")


if __name__ == "__main__":
    main()
