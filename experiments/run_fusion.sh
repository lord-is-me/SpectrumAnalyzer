#!/bin/bash
# 方法3 baseline：RNN/LSTM/GRU/Transformer 四种架构先都跑3层，建立基线后再决定要不要跑7/9层
# 用法: bash run_fusion.sh
# 先跑过 Phase 0/1 (python ../build_nist_sdbs_manifest.py 和 ../build_dataset.py) 再跑这个

set -e

for m in rnn lstm gru transformer; do
    exp="fusion_${m}_3layer"
    mkdir -p "results/${exp}"
    echo "===== 开始训练: ${exp} ====="
    python train_fusion.py --seq_model "$m" --num_layers 3 --epochs 50 2>&1 | tee "results/${exp}/log.txt"
done

echo "4种架构的3层baseline全部跑完，结果在 results/fusion_{rnn,lstm,gru,transformer}_3layer 下"
echo "如果某个架构效果明显更好/更值得深挖，再单独跑: python train_fusion.py --seq_model <arch> --num_layers 7 (或9)"
