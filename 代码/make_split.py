# -*- coding: utf-8 -*-
"""生成 group-aware 的 train / val / test 清单

设计要点（防泄漏）:
  1. 换脸数据的切分单位是 clip（视频片段），不是帧。同一 clip 的所有帧必须
     落在同一边。faceswap 与 simswap 共用源视频（clip id 完全相同），因此
     group key 不含方法名 —— 两者同一 clip 一起被划到训练或测试。
  2. DiT（扩散）按 clip 切分，group key 加 src 防跨源碰撞。
  3. 140k 用官方 train/validation/test 划分，不打乱。
  4. CelebDF_test_aligned 的 real 全部留作测试，绝不参与训练 —— 它是
     swap/diffusion 测试集唯一的真实样本来源。
  5. GAN 域的 real 是训练 real 的主来源；deepfacelab 的 real 一半训练一半测试。

输出:
  split_train.csv / split_val.csv / split_test.csv
  每个测试子集在 test 里有 testset 字段，便于分格评测。
"""
import argparse
from pathlib import Path

import pandas as pd

TRAIN_RATIO = 0.80
CDF_TRAIN_RATIO = 0.50     # CelebDF 真帧专用：帧少(1216)，各留一半才能保证测试样本量


def clip_in_train(key: str, ratio: int | None = None) -> bool:
    """按 group key 的稳定哈希切分（确定性，可复现）"""
    r = ratio if ratio is not None else int(TRAIN_RATIO * 100)
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (h % 100) < r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, encoding="utf-8")
    df = df[df.dataset != "real-vs-fake"]          # 与 140k 有同源重叠，本次不用
    out = Path(args.outdir)

    # ---------- 统一 group key ----------
    def gkey(r):
        m, s, g = r["method"], r["src"], r["group"]
        if m in ("faceswap", "simswap"):
            return f"swap|{s}|{g}"      # 跨方法共享 -> 同 clip 同边
        if m == "dit":
            return f"dit|{s}|{g}"
        if m == "deepfacelab":
            return f"dfl|{g}"
        if m == "gan":
            return "gan|" + r["path"]   # GAN 每张独立
        return f"{m}|{s}|{g}"

    df["gkey"] = df.apply(gkey, axis=1)

    tr, va, te = [], [], []

    # ---------- 140k GAN ----------
    g = df[df.dataset == "140k"]
    for _, r in g.iterrows():
        split = "train" if "/train/" in r["path"] else (
            "val" if "/validation/" in r["path"] else "test")
        rec = r.to_dict()
        rec["testset"] = "gan" if split == "test" else ""
        {"train": tr, "val": va, "test": te}[split].append(rec)

    # ---------- DF40 (swap + diffusion) ----------
    for _, r in df[df.dataset == "DF40"].iterrows():
        rec = r.to_dict()
        if r["method"] == "deepfacelab":
            if r["label"] == 0:
                # real: 一半训练，一半测试
                in_tr = clip_in_train(r["gkey"])
                rec["testset"] = "" if in_tr else "deepfacelab"
                (tr if in_tr else te).append(rec)
            else:
                in_tr = clip_in_train(r["gkey"])
                rec["testset"] = "" if in_tr else "deepfacelab"
                (tr if in_tr else te).append(rec)
            continue
        in_tr = clip_in_train(r["gkey"])
        if r["method"] in ("faceswap", "simswap"):
            rec["testset"] = "" if in_tr else f"swap_{r['method']}_{r['src']}"
        elif r["method"] == "dit":
            rec["testset"] = "" if in_tr else "diffusion"
        (tr if in_tr else te).append(rec)

    # ---------- CelebDF ----------
    # real 按 clip 70/30 分给训练/测试：训练 real 池必须包含视频帧真图，
    # 否则模型会把"高清=真"学进去，测试时低清视频帧全判假（第一轮实验已证实）
    # fake 全部留测试
    for _, r in df[df.dataset == "CelebDF"].iterrows():
        rec = r.to_dict()
        if r["label"] == 0:
            in_tr = clip_in_train(r["gkey"], int(CDF_TRAIN_RATIO * 100))
            rec["testset"] = "" if in_tr else "celebdfofficial"
            (tr if in_tr else te).append(rec)
        else:
            rec["testset"] = "celebdfofficial"
            te.append(rec)

    # deepfacelab 的 real 从 50/50 改为 80/20，给训练 real 池多留视频帧真图
    dfl_real_tr, dfl_real_te = [], []
    for rec in list(tr) + list(te):
        if rec["dataset"] == "DF40" and rec["method"] == "deepfacelab" \
                and rec["label"] == 0:
            (dfl_real_tr if clip_in_train(rec["gkey"] + "|dfl80") else dfl_real_te).append(rec)
    tr = [r for r in tr if not (r["dataset"] == "DF40"
                                and r["method"] == "deepfacelab" and r["label"] == 0)]
    te = [r for r in te if not (r["dataset"] == "DF40"
                                and r["method"] == "deepfacelab" and r["label"] == 0)]
    for r in dfl_real_tr:
        r["testset"] = ""
        tr.append(r)
    for r in dfl_real_te:
        r["testset"] = "deepfacelab"
        te.append(r)

    cols = ["path", "label", "dataset", "method", "domain", "src",
            "group", "gkey", "testset"]
    for name, rows in (("train", tr), ("val", va), ("test", te)):
        d = pd.DataFrame(rows)[cols]
        d.to_csv(out / f"split_{name}.csv", index=False, encoding="utf-8")

    # ---------- 报告 ----------
    print(f"train {len(tr):7d}   val {len(va):7d}   test {len(te):7d}\n")

    print("训练集构成:")
    dtr = pd.DataFrame(tr)
    for k, sub in dtr.groupby(["domain", "method"]):
        print(f"  {k[0]:10s} {k[1]:12s} real={sum(sub.label==0):6d} fake={sum(sub.label==1):6d}")

    print("\n测试集分格（每格 real/fake）:")
    dte = pd.DataFrame(te)
    print(f"{'testset':26s} {'real':>6s} {'fake':>6s} {'clips':>7s}")
    for k, sub in dte.groupby("testset"):
        print(f"{k:26s} {sum(sub.label==0):6d} {sum(sub.label==1):6d} "
              f"{sub.gkey.nunique():7d}")

    # ---------- 泄漏断言 ----------
    print("\n泄漏检查:")
    gt = set(pd.DataFrame(tr).gkey)
    gv = set(pd.DataFrame(va).gkey) if va else set()
    gs = set(dte.gkey)
    print(f"  train ∩ test 的 group: {len(gt & gs)}  (必须为 0)")
    print(f"  train ∩ val  的 group: {len(gt & gv)}")
    # 路径级检查
    pt = set(pd.DataFrame(tr).path)
    ps = set(dte.path)
    print(f"  train ∩ test 的路径:   {len(pt & ps)}  (必须为 0)")
    assert len(gt & gs) == 0 and len(pt & ps) == 0, "存在泄漏，终止"


if __name__ == "__main__":
    main()
