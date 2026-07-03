# 依次跑 resnet50 / resnet101 / vit_b
# 用法: .\run_remaining_windows.ps1

New-Item -ItemType Directory -Force -Path results\resnet50, results\resnet101, results\vit_b | Out-Null

python train.py --backbone resnet50  --epochs 50 2>&1 | Tee-Object -FilePath results\resnet50\log.txt
python train.py --backbone resnet101 --epochs 50 2>&1 | Tee-Object -FilePath results\resnet101\log.txt
python train.py --backbone vit_b     --epochs 50 2>&1 | Tee-Object -FilePath results\vit_b\log.txt
