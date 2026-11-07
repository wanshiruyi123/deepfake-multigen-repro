#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载三个预训练模型到本地 models/ 目录（支持断点续传，走 hf-mirror）。

huggingface_hub 在镜像上会 503（api/resolve-cache 路径不通），所以直接走
resolve/main 直链，用 HTTP Range 续传。

用法: python fetch_models.py [--mirror https://hf-mirror.com]
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request

REPOS = {
    "wvolf_vit_deepfake": "Wvolf/ViT_Deepfake_Detection",
    "prithiv_v2": "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "umm_ai_image": "umm-maybe/AI-image-detector",
}

CANDIDATES = ["config.json", "preprocessor_config.json",
              "pytorch_model.bin", "model.safetensors"]


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=20) as r:
        return int(r.headers.get("Content-Length", 0))


def fetch(url: str, dst: str) -> bool:
    """带 Range 续传的下载。返回是否成功。"""
    tmp = dst + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab") as f:
            total = have + int(r.headers.get("Content-Length", 0))
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                have += len(chunk)
                if total:
                    pct = have * 100 / total
                    print(f"\r    {os.path.basename(dst)} {pct:5.1f}% "
                          f"({have/1e6:.0f}/{total/1e6:.0f} MB)", end="", flush=True)
        print()
        os.replace(tmp, dst)
        return True
    except Exception as e:                              # noqa: BLE001
        print(f"\n    [warn] {os.path.basename(dst)}: {e}")
        return False


def worker(slug: str, repo: str, mirror: str) -> None:
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", slug)
    out = os.path.normpath(out)
    os.makedirs(out, exist_ok=True)
    print(f"[{slug}] {repo}")
    for name in CANDIDATES:
        url = f"{mirror}/{repo}/resolve/main/{name}"
        dst = os.path.join(out, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1024:
            print(f"    {name} 已存在，跳过")
            continue
        try:
            size = head_size(url)
        except Exception:                               # noqa: BLE001
            continue                                    # 文件不存在
        if size <= 0:
            continue
        if fetch(url, dst):
            print(f"    {name} OK ({size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = ap.parse_args()
    threads = [threading.Thread(target=worker, args=(s, r, args.mirror), daemon=True)
               for s, r in REPOS.items()]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(1)
    print("全部完成 -> models/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
