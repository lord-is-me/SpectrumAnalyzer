#!/bin/bash
# 依次跑四个backbone，日志分别写入对应目录
# 用法: bash run_all.sh

mkdir -p results/vgg16 results/resnet50 results/resnet101 results/vit_b

python train.py --backbone vgg16     --epochs 50 2>&1 | tee results/vgg16/log.txt
python train.py --backbone resnet50  --epochs 50 2>&1 | tee results/resnet50/log.txt
python train.py --backbone resnet101 --epochs 50 2>&1 | tee results/resnet101/log.txt
python train.py --backbone vit_b     --epochs 50 2>&1 | tee results/vit_b/log.txt

python summarize_results.py
