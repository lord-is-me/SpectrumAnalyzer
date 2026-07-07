#!/bin/bash
# 对方法1/2里跑出来的全部12个模型（8个nist_split + 4个legacy）依次做 Grad-CAM 标签聚合
# 用法: bash run_gradcam_aggregate.sh [min_pos]
# 需要先跑过对应的 checkpoints/{exp}/best.pth：
#   nist_split的8个 <- run_nist_split.sh
#   legacy的4个     <- run_all.sh（跑之前记得先 python ../fix_legacy_nist_images.py 修好NIST重绘图的坐标轴）

set -e

MIN_POS="${1:-10}"

EXPS="vgg16_pretrained vgg16_scratch resnet50_pretrained resnet50_scratch resnet101_pretrained resnet101_scratch vit_b_pretrained vit_b_scratch vgg16 resnet50 resnet101 vit_b"

for exp in $EXPS; do
    if [ ! -f "checkpoints/${exp}/best.pth" ]; then
        echo "===== 跳过 ${exp}：找不到 checkpoints/${exp}/best.pth ====="
        continue
    fi
    echo "===== 开始: ${exp} (min_pos=${MIN_POS}) ====="
    python gradcam_aggregate.py --exp "$exp" --min_pos "$MIN_POS"
done

echo "全部跑完，结果分别在 results/{exp}/gradcam_aggregate/test_label_wavenumber_profile.csv"
