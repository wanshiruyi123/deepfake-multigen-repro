# -*- coding: utf-8 -*-
"""用人工标注的少量样本，反推每个档位的判定阈值

输入:
    --scores  full_result.csv（run_full.py 的输出，含 s_* 列）
    --labels  labels.csv（file,label[,tier]，label: 1=伪 0=真，tier 可留空）

输出:
    thresholds.json  → 给 run_full.py --config 用
    控制台表格：每个档位在 FPR=0.5%/1%/2%/5% 下的阈值与 TPR

用法:
    python 代码/calibrate_tiers.py --scores full_result.csv --labels labels.csv \
           --target-fpr 0.01 --out thresholds.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

TIERS = ("A", "B", "C")


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()}
                for r in csv.DictReader(f)]


def load_pairs(scores_path, labels_path):
    """返回 {tier: (y, fused)}"""
    sc = {r["file"]: r for r in read_csv(scores_path)}
    lab = {r["file"]: r for r in read_csv(labels_path)}
    out = {t: ([], []) for t in TIERS}
    missing = 0
    for name, r in lab.items():
        s = sc.get(name)
        if s is None:
            missing += 1
            continue
        tier = r.get("tier") or s.get("tier") or "C"
        if tier not in out:
            tier = "C"
        out[tier][0].append(int(float(r["label"])))
        out[tier][1].append(float(s["fused"]))
    if missing:
        print(f"[警告] {missing} 条标注在分数表里找不到同名文件，已跳过")
    return out


def thr_for_fpr(y, s, fpr):
    """取使 FPR<=fpr 的最大阈值"""
    real = np.sort(s[y == 0])
    if len(real) == 0:
        return 1.01
    k = int(np.ceil(fpr * len(real)))
    return float(real[-k]) if k > 0 else float(real.max() + 1e-6)


def metrics(y, s, thr):
    p, n = s >= thr, s < thr
    tp = int(((y == 1) & p).sum())
    fp = int(((y == 0) & p).sum())
    fn = int(((y == 1) & n).sum())
    tn = int(((y == 0) & n).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, tpr=round(tpr, 4),
                fpr=round(fpr, 4), prec=round(prec, 4))


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    o = np.argsort(s, kind="mergesort")
    s, y = s[o], y[o]
    r = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="full_result.csv")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--out", default="thresholds.json")
    args = ap.parse_args()

    data = load_pairs(args.scores, args.labels)
    cfg = {"tiers": {}, "target_fpr": args.target_fpr}
    print(f"目标 FPR = {args.target_fpr:.2%}\n")
    print(f"{'档位':6s} {'n':>5s} {'真':>4s} {'伪':>4s} {'AUC':>7s} "
          f"{'阈值':>7s} {'TPR':>7s} {'FPR':>7s} {'精确率':>7s}")
    print("-" * 64)

    for t in TIERS:
        y, s = data[t]
        if len(y) < 30 or len(set(y)) < 2:
            print(f"{t:6s} {len(y):5d}  样本不足（需 ≥30 张且真假都有），跳过")
            continue
        y, s = np.array(y), np.array(s)
        thr = thr_for_fpr(y, s, args.target_fpr)
        m = metrics(y, s, thr)
        a = auc(y, s)
        print(f"{t:6s} {len(y):5d} {int((y==0).sum()):4d} {int((y==1).sum()):4d} "
              f"{a:7.3f} {thr:7.3f} {m['tpr']:7.3f} {m['fpr']:7.3f} {m['prec']:7.3f}")
        cfg["tiers"][t] = {"hi": round(thr, 4), "lo": round(1 - thr, 4),
                           "n": int(len(y)), "auc": round(float(a), 4),
                           "tpr_at_fpr": m["tpr"], "fpr": m["fpr"],
                           "precision": m["prec"]}

    print("-" * 64)
    print("\n各档位在不同 FPR 下的工作点（挑一个当 hi 阈值）：")
    for t in TIERS:
        y, s = data[t]
        if len(y) < 30 or len(set(y)) < 2:
            continue
        y, s = np.array(y), np.array(s)
        row = []
        for fp in (0.005, 0.01, 0.02, 0.05):
            thr = thr_for_fpr(y, s, fp)
            m = metrics(y, s, thr)
            row.append(f"FPR{fp*100:.1f}%→阈值{thr:.2f}/TPR{m['tpr']:.2f}")
        print(f"  {t}: " + "  ".join(row))

    Path(args.out).write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n已写 {args.out}（供 run_full.py --config 使用）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
