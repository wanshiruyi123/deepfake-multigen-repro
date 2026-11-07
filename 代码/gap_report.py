# -*- coding: utf-8 -*-
"""量化：同域上限 vs 跨域最好水平的差距；以及逐域增量增益"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "matched_auc.csv")
piv = df.pivot(index="model", columns="cell", values="auc_matched")
SUPPORT = {
    "m1_gan": {"gan"},
    "m2_swap": {"swap_faceswap", "swap_simswap", "swap_dfl"},
    "m3_diff": {"diffusion"},
    "m4_gan_swap": {"gan", "swap_faceswap", "swap_simswap", "swap_dfl"},
    "m5_gan_diff": {"gan", "diffusion"},
    "m6_all": {"gan", "swap_faceswap", "swap_simswap", "swap_dfl", "diffusion"},
}

print("同域最佳 vs 跨域最佳（AUC）")
print(f"{'测试格':16s} {'同域最佳':>10s} {'跨域最佳':>10s} {'差距':>8s}   跨域最佳模型")
print("-" * 70)
rows = []
for cell in piv.columns:
    in_models = [m for m in piv.index if cell in SUPPORT[m]]
    out_models = [m for m in piv.index if cell not in SUPPORT[m]]
    a_in = piv.loc[in_models, cell].max() if in_models else float("nan")
    if out_models:
        a_out = piv.loc[out_models, cell].max()
        who = piv.loc[out_models, cell].idxmax()
    else:
        a_out, who = float("nan"), "-"
    print(f"{cell:16s} {a_in:10.3f} {a_out:10.3f} {a_out - a_in:+8.3f}   {who}")
    rows.append({"cell": cell, "best_in_domain": round(float(a_in), 3),
                 "best_cross": round(float(a_out), 3),
                 "gap": round(float(a_out - a_in), 3), "cross_model": who})
pd.DataFrame(rows).to_csv(ROOT / "generalization_gap.csv", index=False,
                          encoding="utf-8")

print("\n逐域增量：每加一个域，各格变化（正=变好）")
for a, b, label in [("m2_swap", "m4_gan_swap", "换脸 → 加 GAN"),
                    ("m4_gan_swap", "m6_all", "GAN+换脸 → 加扩散")]:
    d = (piv.loc[b] - piv.loc[a]).round(3)
    print(f"\n{label}  均值 {piv.loc[a].mean():.3f} → {piv.loc[b].mean():.3f} "
          f"({piv.loc[b].mean() - piv.loc[a].mean():+.3f})")
    for c, v in d.items():
        print(f"   {c:16s} {v:+.3f}")
