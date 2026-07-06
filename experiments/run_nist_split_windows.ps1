# 新数据集(NistSdbsSplit)方法1/2：4个backbone × 预训练/从零训练 = 8次训练，依次跑完
# 用法: .\run_nist_split_windows.ps1
# 先跑过 Phase 0/1 (python ..\build_nist_sdbs_manifest.py 和 ..\build_dataset.py) 再跑这个

$Backbones = @("vgg16", "resnet50", "resnet101", "vit_b")
$PretrainedOpts = @(1, 0)

foreach ($bb in $Backbones) {
    foreach ($pt in $PretrainedOpts) {
        if ($pt -eq 1) { $tag = "pretrained" } else { $tag = "scratch" }
        $exp = "${bb}_${tag}"
        New-Item -ItemType Directory -Force -Path "results\$exp" | Out-Null
        Write-Host "===== 开始训练: $exp ====="
        python train.py --dataset nist_split --backbone $bb --pretrained $pt --epochs 50 2>&1 | Tee-Object -FilePath "results\$exp\log.txt"
    }
}

Write-Host "8组训练全部跑完，结果在 results\{backbone}_pretrained 和 results\{backbone}_scratch 下"
