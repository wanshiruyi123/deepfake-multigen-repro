# -*- coding: utf-8 -*-
"""用 GitHub Git Data API 提交本目录（绕开 git push）。

背景：本机 `github.com:443` 的 git 协议被拦，`git push` 报
`CONNECT tunnel failed, response 502`；而 `api.github.com` 可通。
所以用 blob → tree → commit → 更新 ref 的方式写仓库，效果等同一次普通提交
（不改历史，父提交取远端 main 当前 HEAD）。

前置：gh 已登录（`gh auth login`），仓库已存在。

用法:
    python 代码/push_via_api.py                          # 默认仓库
    python 代码/push_via_api.py --repo owner/name --message "fix: xxx"
    python 代码/push_via_api.py --dry-run                # 只列将提交的文件

注意:
  - 受 .gitignore 约束，只提交 `git ls-files` 出来的文件（权重不会进仓库）
  - 大文件（>50MB）不要走这条路，改用 Release 附件
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"
DEFAULT_REPO = "wanshiruyi123/deepfake-multigen-repro"


def token() -> str:
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except Exception as e:                           # noqa: BLE001
        sys.exit(f"拿不到 gh token，先跑 gh auth login（{e}）")


def call(tk: str, method: str, url: str, payload=None, soft: bool = False):
    headers = {"Authorization": f"Bearer {tk}",
               "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "push-via-api"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url if url.startswith("http") else API + url,
                                data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        if soft:
            return None
        sys.exit(f"HTTP {e.code} {url}\n{e.read()[:500].decode('utf-8', 'replace')}")


def tracked_files() -> list[str]:
    out = subprocess.check_output(
        ["git", "-c", "core.quotePath=false", "ls-files"],
        cwd=ROOT, text=True, encoding="utf-8")
    return [f for f in out.split("\n") if f.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default=None, help="提交信息，默认取 git log -1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = tracked_files()
    total = sum((ROOT / f).stat().st_size for f in files)
    print(f"{len(files)} 个文件, {total / 1024**2:.1f} MB")
    if args.dry_run:
        for f in files:
            print(" ", f)
        return

    tk = token()
    entries = []

    def blob(f: str) -> str:
        content = base64.b64encode((ROOT / f).read_bytes()).decode()
        return call(tk, "POST", f"/repos/{args.repo}/git/blobs",
                    {"content": content, "encoding": "base64"})["sha"]

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(blob, f): f for f in files}
        for i, fut in enumerate(futs, 1):
            f = futs[fut]
            entries.append({"path": f, "mode": "100755" if f.endswith(".sh")
                            else "100644", "type": "blob", "sha": fut.result()})
            if i % 20 == 0:
                print(f"  blob {i}/{len(files)}")

    tree = call(tk, "POST", f"/repos/{args.repo}/git/trees", {"tree": entries})
    print("tree", tree["sha"])

    msg = args.message or subprocess.check_output(
        ["git", "log", "-1", "--pretty=%B"], cwd=ROOT, text=True,
        encoding="utf-8").strip()
    body = {"message": msg, "tree": tree["sha"]}
    ref = call(tk, "GET", f"/repos/{args.repo}/git/ref/heads/{args.branch}",
               soft=True)
    if ref:
        body["parents"] = [ref["object"]["sha"]]
        print("parent", ref["object"]["sha"])
    else:
        print("远端分支为空，将创建首个提交")

    commit = call(tk, "POST", f"/repos/{args.repo}/git/commits", body)
    print("commit", commit["sha"])

    created = call(tk, "POST", f"/repos/{args.repo}/git/refs",
                   {"ref": f"refs/heads/{args.branch}", "sha": commit["sha"]},
                   soft=True)
    if created is None:                              # 分支已存在 → 前移
        call(tk, "PATCH", f"/repos/{args.repo}/git/refs/heads/{args.branch}",
             {"sha": commit["sha"], "force": True})
    print(f"已更新 https://github.com/{args.repo}/commits/{args.branch}")


if __name__ == "__main__":
    main()
