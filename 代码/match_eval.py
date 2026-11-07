# -*- coding: utf-8 -*-
"""倾向匹配分析：排除画质混淆后的跨域 AUC

背景：swap / diffusion 测试格里，real 是原始视频帧（锐度约 14），
fake 是 DF40 对齐后的合成人脸（锐度 50~90）。模型可以纯靠"清晰度"
这一与伪造无关的线索拿分，掩盖真实泛化能力。

方法：1:1 最近邻倾向匹配 —— 每个 fake 从 real 池中挑锐度最接近的样本，
限制单张 real 的复用次数，在匹配子集上重算 AUC。匹配后两组锐度分布
近似，"清晰度"捷径被中和。

用法:
    python batch/match_eval.py m1_gan,m2_swap,m3_diff,m4_gan_swap,m5_gan_diff,m6_all
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(0)
ROOT = Path(__file__).resolve().parents[1]
SHARP_CACHE = ROOT / "sharpness_cache.csv"
CELLS = ["gan", "swap_faceswap", "swap_simswap", "swap_dfl", "diffusion", "celebdf"]


def sharpness(path: str) -> float:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return 0.0
    return float(cv2.Laplacian(cv2.resize(im, (256, 256)), cv2.CV_64F).var())


def load_sharpness(paths) -> pd.Series:
    cache = {}
    if SHARP_CACHE.exists():
        c = pd.read_csv(SHARP_CACHE)
        cache = dict(zip(c.path, c.sharp))
    todo = [p for p in paths if p not in cache]
    if todo:
        print(f"  计算锐度: {len(todo)} 张...", flush=True)
        for i, p in enumerate(todo, 1):
            cache[p] = sharpness(p)
            if i % 1000 == 0:
                print(f"    {i}/{len(todo)}", flush=True)
        pd.DataFrame({"path": list(cache), "sharp": list(cache.values())}) \
            .to_csv(SHARP_CACHE, index=False)
    return pd.Series({p: cache[p] for p in paths})


def auc_tie_aware(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC，并列分数取平均秩（sigmoid 饱和会产生大量并列）"""
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def matched_subset(df: pd.DataFrame, max_reuse: int = 3, seed: int = 7):
    """1:1 最近邻锐度匹配，返回 (匹配后的 df, 匹配对数)"""
    rng = np.random.RandomState(seed)
    real = df[df.label == 0].reset_index(drop=True)
    fake = df[df.label == 1].reset_index(drop=True)
    if real.empty or fake.empty:
        return None, 0
    d = np.abs(real.sharp.values[None, :] - fake.sharp.values[:, None])
    nn = d.argmin(axis=1)
    cert = d.min(axis=1)                      # 每个 fake 的匹配距离
    cnt = np.bincount(nn, minlength=len(real))
    # 配额不足时，按匹配距离从小到大保留，保证留下的 fake 匹配质量最好
    keep = np.zeros(len(fake), dtype=bool)
    quota = cnt[nn].copy()
    order = np.argsort(cert, kind="mergesort")
    used = np.zeros(len(real), dtype=np.int64)
    for i in order:
        r = nn[i]
        if used[r] < max_reuse or cnt[r] <= max_reuse:
            keep[i] = True
            used[r] += 1
    pair_real = real.iloc[nn[keep]]
    pair_fake = fake.iloc[keep]
    out = pd.concat([pair_real, pair_fake], ignore_index=True)
    return out, int(keep.sum())


def main():
    models = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        sorted(p.parent.name for p in (ROOT / "models").glob("m*/best.pt"))
    cells = [c for c in CELLS if any((ROOT / f"scores_{m}_{c}.csv").exists()
                                     for m in models)]
    all_paths = set()
    for m in models:
        for c in cells:
            f = ROOT / f"scores_{m}_{c}.csv"
            if f.exists():
                all_paths.update(pd.read_csv(f, usecols=["path"]).path)
    print(f"模型 {models}\n测试格 {cells}\n涉及 {len(all_paths)} 张图")
    sp = load_sharpness(sorted(all_paths))

    print(f"\n{'model':14s} {'cell':16s} {'原始AUC':>9s} {'匹配后':>8s} "
          f"{'差值':>8s} {'锐度(真/假)':>14s} {'对':>6s}")
    print("-" * 78)
    rows = []
    for m in models:
        for c in cells:
            f = ROOT / f"scores_{m}_{c}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d["sharp"] = d.path.map(sp)
            sub, npair = matched_subset(d)
            if sub is None:
                continue
            auc_raw = auc_tie_aware(d.label.values, d.score.values)
            auc_m = auc_tie_aware(sub.label.values, sub.score.values)
            sr = sub.loc[sub.label == 0, "sharp"].median()
            sf = sub.loc[sub.label == 1, "sharp"].median()
            rows.append({"model": m, "cell": c,
                         "auc_raw": round(auc_raw, 4),
                         "auc_matched": round(auc_m, 4),
                         "delta": round(auc_m - auc_raw, 4),
                         "sharp_real": round(float(sr), 1),
                         "sharp_fake": round(float(sf), 1),
                         "pairs": npair})
            print(f"{m:14s} {c:16s} {auc_raw:9.4f} {auc_m:8.4f} "
                  f"{auc_m - auc_raw:+8.4f} {sr:7.1f}/{sf:<6.1f} {npair:6d}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "matched_auc.csv", index=False, encoding="utf-8")
    print("\n匹配后 AUC 矩阵")
    print(out.pivot(index="model", columns="cell", values="auc_matched")
          .round(3).to_string())
    print("\n-> matched_auc.csv")


if __name__ == "__main__":
    main()
