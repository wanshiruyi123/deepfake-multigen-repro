# 权重不随仓库走（900MB），从 Release 下载后解压到本目录

# 下载（v1.0-weights）:
#   https://github.com/wanshiruyi123/deepfake-multigen-repro/releases/tag/v1.0-weights
#
# 解压后结构:
#   models/
#   ├── cnndetection/blur_jpg_prob0.5.pth   (269 MB, 训练起点, 重训必需)
#   ├── m1_gan/best.pt        (90 MB)
#   ├── m2_swap/best.pt
#   ├── m3_diff/best.pt
#   ├── m4_gan_swap/best.pt
#   ├── m5_gan_diff/best.pt
#   ├── m6_all/best.pt
#   └── own_resnet50/best.pt
#
# 校验（SHA256SUMS.txt 在本目录，路径相对 models/）:
#   cd models && sha256sum -c SHA256SUMS.txt            # Linux / macOS / Git Bash
#   cd models; Get-Content SHA256SUMS.txt | ForEach-Object { ... }   # PowerShell 用 Get-FileHash
#
# 放好权重后跑 `python 代码/verify_env.py`，应从"权重缺失"变为"通过"。
