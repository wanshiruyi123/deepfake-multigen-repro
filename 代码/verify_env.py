# -*- coding: utf-8 -*-
"""环境自检 + 数值复现校验。复现第一步就跑这个。

检查项:
  1. Python / 关键库版本
  2. torch 能否看到 GPU
  3. 四个数据集路径是否就位（可用环境变量覆盖）
  4. 8 份权重（含训练起点）是否齐全、大小是否对得上
  5. 用 m6_all / m2_swap / own_resnet50 对 样例/ 五张图打分，
     与 expected.csv 的参考分数比对（容差 0.02）

用法:
  python 代码/verify_env.py
退出码 0 = 环境可用；1 = 有致命问题。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "代码"))
import paths  # noqa: E402

TOL = 0.02                       # 分数容差：同机同版本应 <1e-3，留余量给不同 GPU/库版本
EXPECT_SIZE = {                  # 字节数，允许 ±1%
    "m1_gan": 94340525, "m2_swap": 94340525, "m3_diff": 94340525,
    "m4_gan_swap": 94340525, "m5_gan_diff": 94340525, "m6_all": 94340525,
    "own_resnet50": 94340461, "cnndetection": 282442597,
}
SMOKE_MODELS = ("m6_all", "m2_swap", "own_resnet50")

fails: list[str] = []
warns: list[str] = []


def head(t: str) -> None:
    print(f"\n{'=' * 66}\n{t}\n{'=' * 66}")


def check_libs() -> None:
    head("1. 解释器与依赖")
    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 9):
        fails.append("需要 Python >= 3.9")
    for m in ("numpy", "pandas", "cv2", "torch", "torchvision", "PIL"):
        try:
            mod = __import__(m)
            print(f"  {m:12s} {getattr(mod, '__version__', '?')}")
        except Exception as e:                       # noqa: BLE001
            fails.append(f"缺少 {m}: {e}")
            print(f"  {m:12s} MISSING ({e})")
    try:                                             # 仅 calibrate.py 用到
        import sklearn
        print(f"  {'sklearn':12s} {sklearn.__version__}")
    except Exception:                                # noqa: BLE001
        warns.append("未装 scikit-learn：只有 calibrate.py 需要，主流程可忽略")


def check_gpu() -> None:
    head("2. 训练设备")
    try:
        import torch
    except Exception:                                # noqa: BLE001
        return
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i}  {p.name}  {p.total_memory / 1024**3:.1f} GB  "
                  f"sm_{p.major}{p.minor}")
    else:
        warns.append("没有可用 GPU：推理仍可跑，训练会非常慢")
        print("  CPU only")


def check_data() -> None:
    head("3. 数据集路径（环境变量可覆盖）")
    print(paths.describe())
    for key in ("RVF", "HF140K", "DF40", "CELEBDF"):
        if not paths.get(key).exists():
            warns.append(f"{key} 缺失（{paths.get(key)}）：只做推理可忽略，重训必须补齐")


def check_weights() -> None:
    head("4. 权重完整性")
    for name, size in EXPECT_SIZE.items():
        p = (ROOT / "models" / "cnndetection" / "blur_jpg_prob0.5.pth"
             if name == "cnndetection" else ROOT / "models" / name / "best.pt")
        if not p.exists():
            fails.append(f"权重缺失: {p}")
            print(f"  {name:14s} MISSING  {p}")
            continue
        real = p.stat().st_size
        ok = abs(real - size) <= size * 0.01
        print(f"  {name:14s} {'OK   ' if ok else 'SIZE?'}  {real / 1024**2:.1f} MB")
        if not ok:
            warns.append(f"{name} 大小异常：{real} != {size}")


def check_smoke() -> None:
    head("5. 数值复现校验：样例/ 五张图")
    import cv2
    import numpy as np
    import torch
    from torchvision.models import resnet50

    cv2.setNumThreads(0)
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def imread_cn(p: str):
        buf = np.fromfile(p, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None

    def prep(im):
        x = cv2.cvtColor(cv2.resize(im, (256, 256)), cv2.COLOR_BGR2RGB)
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    sample_dir = ROOT / "样例"
    if not sample_dir.is_dir():
        warns.append(f"未找到 {sample_dir}，跳过数值校验")
        print("  样例目录不存在，跳过")
        return
    exp_path = sample_dir / "expected.csv"
    if not exp_path.exists():
        warns.append("样例/expected.csv 缺失，跳过数值校验")
        print("  参考分数表缺失，跳过")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = {}
    for name in SMOKE_MODELS:
        ck = torch.load(ROOT / "models" / name / "best.pt", map_location="cpu",
                        weights_only=False)
        net = resnet50(num_classes=1)
        net.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        nets[name] = net.eval().to(dev)

    def score(net, im):
        x = prep(im).to(dev)
        with torch.inference_mode():
            return float(torch.sigmoid(net(x)).item())

    ref = {}
    with exp_path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            r = {(k or "").strip().lstrip("\ufeff"): v for k, v in r.items()}
            ref[r["file"].strip()] = r

    rows, bad = [], 0
    for f in sorted(sample_dir.glob("*")):
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        im = imread_cn(str(f))
        if im is None:
            fails.append(f"样例不可读: {f}")
            continue
        # 与 score_files.py 一致：raw 与统一重编码两路取平均
        ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        uni = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else im
        s = {k: (score(n, im) + score(n, uni)) / 2 for k, n in nets.items()}
        pred = "伪造" if s["m6_all"] >= 0.5 else "真实"
        r = ref.get(f.name)
        drift = ""
        if r:
            for k in SMOKE_MODELS:
                want, got = float(r[k]), s[k]
                if abs(want - got) > TOL:
                    drift += f" {k}期望{want:.4f}实测{got:.4f}"
            if drift:
                bad += 1
                fails.append(f"{f.name} 分数漂移:{drift}")
        print(f"  {f.name:18s} m6={s['m6_all']:.4f} m2={s['m2_swap']:.4f} "
              f"own={s['own_resnet50']:.4f} -> {pred}"
              f"{'  [漂移]' + drift if drift else ('  OK' if r else '')}")
        rows.append([f.name, f"{s['m6_all']:.4f}", f"{s['m2_swap']:.4f}",
                     f"{s['own_resnet50']:.4f}", pred])

    out = sample_dir / "smoke_result.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "m6_all", "m2_swap", "own_resnet50", "pred"])
        w.writerows(rows)
    print(f"\n  本次结果 -> {out}   (漂移文件数 {bad}/{len(rows)})")


def main() -> int:
    print(f"报告根目录: {ROOT}")
    check_libs()
    check_gpu()
    check_data()
    check_weights()
    try:
        check_smoke()
    except Exception as e:                           # noqa: BLE001
        fails.append(f"数值校验异常: {e}")
        print(f"  数值校验异常: {e}")

    head("结论")
    for w in warns:
        print(f"[warn] {w}")
    for f in fails:
        print(f"[FAIL] {f}")
    if fails:
        print("\n环境/复现校验未通过，按上面 FAIL 逐条修。")
        return 1
    print("\n通过。下一步: python 代码/score_files.py <图片路径…>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
