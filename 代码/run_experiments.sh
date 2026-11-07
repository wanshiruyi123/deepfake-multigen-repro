#!/usr/bin/env bash
# 跨生成器泛化实验：6 个训练配置，仅训练数据不同
#
# 用法（在报告根目录执行）:
#   bash 代码/run_experiments.sh [per_class] [epochs]
#   PY=python3 bash 代码/run_experiments.sh 4000 2
#
# 前置步骤:
#   python 代码/build_manifest.py --out manifest.csv
#   python 代码/make_split.py --manifest manifest.csv --outdir .
set -u
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
PC="${1:-4000}"
EP="${2:-2}"

run () {
  local name="$1" dom="$2"
  echo "=================================================================="
  echo ">>> $name  domains=$dom  per_class=$PC  epochs=$EP  $(date +%H:%M:%S)"
  echo "=================================================================="
  "$PY" -u 代码/train_multi.py --name "$name" --domains "$dom" \
        --per-class "$PC" --epochs "$EP" --batch 48 --workers 6 --seed 42 \
        > "train_${name}.log" 2>&1
  tail -3 "train_${name}.log"
  echo "<<< $name 结束 $(date +%H:%M:%S)"
}

# 单域基线
run m1_gan        gan
run m2_swap       swap
run m3_diff       diff
# 两两混合
run m4_gan_swap   gan,swap
run m5_gan_diff   gan,diff
# 全混合
run m6_all        gan,swap,diff

echo "ALL DONE $(date +%H:%M:%S)"
