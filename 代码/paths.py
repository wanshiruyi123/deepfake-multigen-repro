# -*- coding: utf-8 -*-
"""数据集路径解析（复现用）。

优先读环境变量，未设置时回落到作者本机默认值。别人复现只需设 4 个环境变量，
不必改任何脚本源码。

环境变量           含义
  DATA_RVF         120k real-vs-fake 根目录（其下 real/ 与 fake/）
  DATA_HF140K      140k-Real-and-Fake-Faces 的 images 目录（其下为 split/class）
  DATA_DF40        DF40 根目录
  DATA_CELEBDF     Celeb-DF 对齐帧根目录（其下 real/ 与 fake/）

Windows PowerShell 设置示例:
  $env:DATA_RVF    = "D:/data/real-vs-fake"
  $env:DATA_HF140K = "D:/data/140k-Real-and-Fake-Faces/images"
  $env:DATA_DF40   = "D:/data/DF40"
  $env:DATA_CELEBDF= "D:/data/CelebDF_test_aligned"

Linux/macOS:
  export DATA_RVF=/data/real-vs-fake
"""
from __future__ import annotations

import os
from pathlib import Path

_SPEC = {
    "RVF":     ("DATA_RVF",     "C:/AIGC_data/real-vs-fake"),
    "HF140K":  ("DATA_HF140K",  "D:/AIGC_data/140k-Real-and-Fake-Faces/images"),
    "DF40":    ("DATA_DF40",    "D:/AIGC_data/DF40"),
    "CELEBDF": ("DATA_CELEBDF", "D:/AIGC_data/CelebDF_test_aligned"),
}


def get(key: str) -> Path:
    env, default = _SPEC[key]
    return Path(os.environ.get(env) or default)


RVF = get("RVF")
HF140K = get("HF140K")
DF40 = get("DF40")
CELEBDF = get("CELEBDF")


def describe() -> str:
    lines = []
    for key, (env, default) in _SPEC.items():
        p = get(key)
        src = "env" if os.environ.get(env) else "default"
        lines.append(f"{env:12s} [{src:7s}] {p}  {'OK' if p.exists() else 'MISSING'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
