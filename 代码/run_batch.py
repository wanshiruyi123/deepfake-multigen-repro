#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 万张图的一键入口：按 N 个进程分片并行跑，最后合并成一个 CSV 并打印统计。

用法:
  python run_batch.py --input D:/imgs --out results.csv --procs 4
  python run_batch.py --input D:/imgs --out results.csv --procs 4 --resume   # 断点续跑

说明:
  Windows 上多进程比多线程实在（torch CPU 推理是串行热点），
  4 进程 = 4 路独立推理，实测接近线性加速，直到吃满 CPU 核心。
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=3, help="每个进程内的解码线程数")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--face-crop", default="auto", choices=["auto", "off"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    base, ext = os.path.splitext(args.out)
    for f in glob.glob(f"{base}.shard*{ext}"):
        if not args.resume:
            os.remove(f)
    start = time.time()

    procs = []
    for i in range(args.procs):
        cmd = [
            PY, os.path.join(HERE, "batch_detect.py"),
            "--input", args.input,
            "--out", f"{base}.shard{i}{ext}",
            "--shard", f"{i}/{args.procs}",
            "--threshold", str(args.threshold),
            "--workers", str(args.workers),
            "--batch", str(args.batch),
            "--face-crop", args.face_crop,
        ]
        if args.resume:
            cmd.append("--resume")
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        log = open(f"{base}.shard{i}.log", "w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"启动分片 {i}/{args.procs} pid={procs[-1][0].pid}")

    alive = True
    while alive:
        alive = any(p.poll() is None for p, _ in procs)
        if alive:
            time.sleep(5)

    for p, log in procs:
        log.close()
        if p.returncode != 0:
            print(f"[warn] 分片进程退出码 {p.returncode}，见日志 {base}.shard*.log")

    frames = []
    for f in sorted(glob.glob(f"{base}.shard*{ext}")):
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] 读 {f} 失败: {e}")
    if not frames:
        print("没有产出任何结果")
        return 1
    df = pd.concat(frames, ignore_index=True).drop_duplicates("path", keep="last")
    df.to_csv(args.out, index=False, encoding="utf-8")

    ok = df[df["status"] == "ok"]
    fake = int((ok["decision"] == "fake").sum())
    dur = time.time() - start
    print("\n================ 汇总 ================")
    print(f"图片总数      {len(df)}")
    print(f"成功 / 失败   {len(ok)} / {len(df) - len(ok)}")
    print(f"判定伪造      {fake}  ({fake / max(1, len(ok)) * 100:.2f}%)")
    print(f"检出人脸      {int(ok['face'].sum())} 张")
    print(f"总耗时        {dur / 60:.1f} min  ({len(ok) / max(1e-9, dur):.1f} img/s)")
    if len(ok):
        print(f"分数分布      p50={ok['score'].median():.3f}  "
              f"p90={ok['score'].quantile(0.9):.3f}  "
              f"p99={ok['score'].quantile(0.99):.3f}")
    print(f"结果文件      {os.path.abspath(args.out)}")
    print("======================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
