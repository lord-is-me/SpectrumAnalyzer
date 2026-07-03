#!/bin/bash
# 依次跑 resnet50 / resnet101 / vit_b
# 用法: bash run_remaining.sh

mkdir -p results/resnet50 results/resnet101 results/vit_b

python train.py --backbone resnet50  --epochs 50 2>&1 | tee results/resnet50/log.txt
python train.py --backbone resnet101 --epochs 50 2>&1 | tee results/resnet101/log.txt
python train.py --backbone vit_b     --epochs 50 2>&1 | tee results/vit_b/log.txt
