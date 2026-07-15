#!/bin/bash
# 对全部16个模型（8个方法1/2 nist_split + 4个legacy + 4个方法3融合模型）依次做
# 2D Grad-CAM 区域重要性聚合（gradcam_region_map.py）
# 用法: bash run_gradcam_region_map.sh [min_pos]
# 前提和 run_gradcam_aggregate.sh 完全一致：
#   nist_split的8个   <- run_nist_split.sh
#   legacy的4个       <- run_all.sh（跑之前记得先 python ../fix_legacy_nist_images.py）
#   方法3融合模型的4个 <- run_fusion.sh

set -e

MIN_POS="${1:-10}"

EXPS="vgg16_pretrained vgg16_scratch resnet50_pretrained resnet50_scratch resnet101_pretrained resnet101_scratch vit_b_pretrained vit_b_scratch vgg16 resnet50 resnet101 vit_b fusion_rnn_3layer fusion_lstm_3layer fusion_gru_3layer fusion_transformer_3layer"

for exp in $EXPS; do
    if [ ! -f "checkpoints/${exp}/best.pth" ]; then
        echo "===== 跳过 ${exp}：找不到 checkpoints/${exp}/best.pth ====="
        continue
    fi
    echo "===== 开始: ${exp} (min_pos=${MIN_POS}) ====="
    python gradcam_region_map.py --exp "$exp" --min_pos "$MIN_POS"
done

echo "全部跑完，结果分别在 results/{exp}/gradcam_region_map/_grid.png"
