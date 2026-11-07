# -*- coding: utf-8 -*-
"""汇总跨域矩阵：每模型的支持域、同域/跨域均值、最差格"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "matched_auc.csv")

# 每个模型训练时覆盖的域
SUPPORT = {
    "m1_gan": {"gan"},
    "m2_swap": {"swap_faceswap", "swap_simswap", "swap_dfl"},
    "m3_diff": {"diffusion"},
    "m4_gan_swap": {"gan", "swap_faceswap", "swap_simswap", "swap_dfl"},
    "m5_gan_diff": {"gan", "diffusion"},
    "m6_all": {"gan", "swap_faceswap", "swap_simswap", "swap_dfl", "diffusion"},
}
ORDER = ["m1_gan", "m2_swap", "m3_diff", "m4_gan_swap", "m5_gan_diff", "m6_all"]

rows = []
for m in ORDER:
    d = df[df.model == m]
    sup = SUPPORT[m]
    in_d = d[d.cell.isin(sup)].auc_matched
    out_d = d[~d.cell.isin(sup)].auc_matched
    rows.append({
        "model": m,
        "n_domain": len(sup),
        "in_domain": round(float(in_d.mean()), 4) if len(in_d) else None,
        "cross_domain": round(float(out_d.mean()), 4) if len(out_d) else None,
        "all6_mean": round(float(d.auc_matched.mean()), 4),
        "worst": round(float(d.auc_matched.min()), 4),
        "worst_cell": d.loc[d.auc_matched.idxmin(), "cell"],
    })
s = pd.DataFrame(rows)
print(s.to_string(index=False))
s.to_csv(ROOT / "cross_summary.csv", index=False, encoding="utf-8")

print("\n每个测试格上的最佳模型")
best = df.loc[df.groupby("cell").auc_matched.idxmax()][["cell", "model", "auc_matched"]]
print(best.sort_values("auc_matched", ascending=False).to_string(index=False))
