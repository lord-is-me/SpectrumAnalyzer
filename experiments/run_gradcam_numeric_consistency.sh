#!/bin/bash
# 对方法3的4个融合模型依次做"图像重要区域 vs 数值重要区域一致性"分析（Phase 4, 5.2节）
# 用法: bash run_gradcam_numeric_consistency.sh [min_pos] [ig_steps]
# 需要先跑过 checkpoints/fusion_*_3layer/best.pth（见 run_fusion.sh）

set -e

MIN_POS="${1:-10}"
IG_STEPS="${2:-20}"

EXPS="fusion_rnn_3layer fusion_lstm_3layer fusion_gru_3layer fusion_transformer_3layer"

for exp in $EXPS; do
    if [ ! -f "checkpoints/${exp}/best.pth" ]; then
        echo "===== 跳过 ${exp}：找不到 checkpoints/${exp}/best.pth ====="
        continue
    fi
    echo "===== 开始: ${exp} (min_pos=${MIN_POS}, ig_steps=${IG_STEPS}) ====="
    python gradcam_numeric_consistency.py --exp "$exp" --min_pos "$MIN_POS" --ig_steps "$IG_STEPS"
done

echo "全部跑完，结果分别在 results/{exp}/gradcam_numeric_consistency/train_label_correlation.csv"
