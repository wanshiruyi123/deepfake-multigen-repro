#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成人工标注页面：从批量结果里分层抽样，点几下就能导出 labels.csv。

为什么需要它：公开模型的阈值几乎不可能直接适配你的数据，而校准只需要 200~300 张
人工确认过的样本。这个页面把这件事压缩到 15 分钟。

用法:
  python make_label_page.py --results results.csv --n 300 --out label_page.html

然后浏览器打开 label_page.html，逐张点"真实/伪造"，最后点"导出 labels.csv"。
把导出的文件喂给 calibrate.py。
"""
from __future__ import annotations

import argparse
import html
import os
import random

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="label_page.html")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df[df["status"] == "ok"].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[df["score"].notna()]
    if df.empty:
        raise SystemExit("没有可用记录")

    # 分层：高分 / 灰区 / 低分 各抽 1/3，保证标注样本覆盖整个分数段
    rng = random.Random(args.seed)
    bins = [
        df[df["score"] >= 0.8],
        df[(df["score"] >= 0.4) & (df["score"] < 0.8)],
        df[df["score"] < 0.4],
    ]
    per = max(1, args.n // 3)
    picked = []
    for b in bins:
        if len(b) <= per:
            picked.append(b)
        else:
            picked.append(b.sample(n=per, random_state=rng.randint(0, 10**6)))
    sample = pd.concat(picked).drop_duplicates("path").sort_values("score")
    print(f"抽样 {len(sample)} 张（高 {len(picked[0])} / 灰 {len(picked[1])} / 低 {len(picked[2])}）")

    rows = []
    for i, r in enumerate(sample.itertuples()):
        p = os.path.abspath(r.path).replace("\\", "/")
        rows.append(
            f'<div class="card" data-path="{html.escape(r.path)}">'
            f'<div class="imgbox"><img loading="lazy" src="file:///{html.escape(p)}"></div>'
            f'<div class="meta"><span class="idx">#{i+1}</span>'
            f'<span class="score">score {r.score:.3f}</span>'
            f'<span class="fn">{html.escape(os.path.basename(r.path))}</span></div>'
            f'<div class="btns"><button class="real" onclick="mark(this,0)">真实</button>'
            f'<button class="fake" onclick="mark(this,1)">伪造</button></div></div>'
        )

    doc = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Deepfake 标注</title><style>
:root{--bg:#f7f7f8;--card:#fff;--line:#e3e3e6;--txt:#1a1a1c;--mut:#6b6b73}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif}
header{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header b{font-size:15px}
button{cursor:pointer;border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 14px;font:inherit;color:var(--txt)}
button:hover{border-color:#999}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.imgbox{aspect-ratio:1;background:#eee;display:flex;align-items:center;justify-content:center;overflow:hidden}
.imgbox img{width:100%;height:100%;object-fit:cover}
.meta{padding:6px 8px;display:flex;gap:6px;align-items:center;font-size:12px;color:var(--mut)}
.score{margin-left:auto;font-variant-numeric:tabular-nums}
.fn{max-width:90px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.btns{display:flex;gap:0;border-top:1px solid var(--line)}
.btns button{flex:1;border:0;border-radius:0;padding:10px;font-weight:600}
.real{color:#0a7c3f}.fake{color:#b3261e}
.card[data-l="0"]{outline:2px solid #0a7c3f}.card[data-l="1"]{outline:2px solid #b3261e}
.card[data-l="0"] .real,.card[data-l="1"] .fake{background:#f0f0f2}
</style></head><body>
<header>
  <b>Deepfake 标注</b>
  <span id="prog">已标 0 / __N__</span>
  <button onclick="exportCsv()">导出 labels.csv</button>
  <button onclick="clearAll()">清空</button>
  <span style="color:#6b6b73;font-size:12px">按分数从低到高排列，方便你判断阈值边界</span>
</header>
<div class="grid" id="g">__ROWS__</div>
<script>
const marks={};
function mark(btn,v){
  const card=btn.closest('.card');
  card.dataset.l=v;
  marks[card.dataset.path]=v;
  const n=Object.keys(marks).length;
  document.getElementById('prog').textContent=`已标 ${n} / ${document.querySelectorAll('.card').length}`;
}
function exportCsv(){
  const rows=['path,label'];
  for(const [p,v] of Object.entries(marks)) rows.push(`"${p.replace(/"/g,'""')}",${v}`);
  const blob=new Blob([rows.join('\\n')],{type:'text/csv;charset=utf-8'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='labels.csv'; a.click();
}
function clearAll(){for(const c of document.querySelectorAll('.card'))delete c.dataset.l;for(const k in marks)delete marks[k];document.getElementById('prog').textContent='已标 0 / '+document.querySelectorAll('.card').length}
</script></body></html>"""

    doc = doc.replace("__N__", str(len(sample))).replace("__ROWS__", "\n".join(rows))
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"-> {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
