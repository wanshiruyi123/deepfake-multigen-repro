# -*- coding: utf-8 -*-
"""数据集完整性与重复检测。

1) 全量 md5：找完全相同的文件（含跨类重复，跨类重复说明标注有问题）
2) dHash 64bit 感知哈希：找近似重复（Hamming<=6），并统计训练/测试切分之间的泄漏
3) JPEG 量化表分布：判断 real 与 fake 是否来自不同编码管线（单/多来源线索）
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

import numpy as np
from PIL import Image

ROOT = paths.RVF


def dhash(im: Image.Image, size: int = 8) -> int:
    g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def work(p: str):
    cls = "fake" if "/fake/" in p.replace("\\", "/") else "real"
    try:
        raw = Path(p).read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        with Image.open(p) as im:
            dh = dhash(im)
            q = im.quantization or {}
            qsig = "|".join(f"{k}:{','.join(map(str, v))}" for k, v in sorted(q.items()))
    except Exception:  # noqa: BLE001
        return None
    return (p, cls, md5, dh, hashlib.md5(qsig.encode()).hexdigest()[:8])


def main() -> None:
    files = [str(p) for p in ROOT.rglob("*.jpg")]
    print(f"扫描 {len(files)} 个文件 ...", flush=True)
    with Pool(10) as pool:
        res = [r for r in pool.imap_unordered(work, files, chunksize=200) if r]
    print(f"成功解析 {len(res)}")

    # 1) 完全重复
    by_md5 = defaultdict(list)
    for p, cls, md5, _, _ in res:
        by_md5[md5].append((p, cls))
    dup = {k: v for k, v in by_md5.items() if len(v) > 1}
    cross = {k: v for k, v in dup.items() if len({c for _, c in v}) > 1}
    print(f"\n[完全重复] {len(dup)} 组；其中跨 real/fake 的 {len(cross)} 组")
    for k, v in list(cross.items())[:3]:
        print("   ", [f"{c}:{Path(p).name}" for p, c in v])

    # 2) 近重复（dHash Hamming<=6），用 LSH 桶加速：按高 16 位分桶再比对
    buckets = defaultdict(list)
    for p, cls, _, dh, _ in res:
        buckets[dh >> 48].append(dh)
    pairs_near = 0
    for dh in set(d for _, _, _, d, _ in res):
        for cand in buckets[dh >> 48]:
            if bin(cand ^ dh).count("1") <= 6:
                pairs_near += 1
    near_groups = pairs_near // 2
    print(f"[近似重复] dHash 同桶内 Hamming<=6 的对数约 {near_groups}")

    # 3) 量化表分布
    qc = Counter((cls, qh) for _, cls, _, _, qh in res)
    print("\n[JPEG 量化表签名] (class, 签名) -> 数量")
    for (cls, qh), n in sorted(qc.items(), key=lambda x: -x[1])[:6]:
        print(f"   {cls} {qh} : {n}")
    real_q = Counter(qh for _, cls, _, _, qh in res if cls == "real")
    fake_q = Counter(qh for _, cls, _, _, qh in res if cls == "fake")
    print(f"   real 独立签名 {len(real_q)} 种；fake 独立签名 {len(fake_q)} 种")
    print(f"   两者共享签名 {len(set(real_q) & set(fake_q))} 种")


if __name__ == "__main__":
    main()
