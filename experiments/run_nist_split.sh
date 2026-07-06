#!/bin/bash
# 新数据集(NistSdbsSplit)方法1/2：4个backbone × 预训练/从零训练 = 8次训练，依次跑完
# 用法: bash run_nist_split.sh
# 先跑过 Phase 0/1 (python ../build_nist_sdbs_manifest.py 和 ../build_dataset.py) 再跑这个

set -e

BACKBONES="vgg16 resnet50 resnet101 vit_b"
PRETRAINED_OPTS="1 0"

for bb in $BACKBONES; do
    for pt in $PRETRAINED_OPTS; do
        if [ "$pt" = "1" ]; then tag=pretrained; else tag=scratch; fi
        exp="${bb}_${tag}"
        mkdir -p "results/${exp}"
        echo "===== 开始训练: ${exp} ====="
        python train.py --dataset nist_split --backbone "$bb" --pretrained "$pt" --epochs 50 2>&1 | tee "results/${exp}/log.txt"
    done
done

echo "8组训练全部跑完，结果在 results/{backbone}_pretrained 和 results/{backbone}_scratch 下"
