# NIST/SDBS 融合实验 —— 运行说明

配套设计文档见 [nist_fusion_experiment_plan.md](nist_fusion_experiment_plan.md)。本文档只讲怎么跑，从零开始（新服务器/新checkout）到跑出方法1/2的结果。

---

## 0. 前提

- Python 环境已配置好（conda环境、PyTorch等），没配置过先看 [experiments/ENVIRONMENT_SETUP.md](../experiments/ENVIRONMENT_SETUP.md)。
- `data.zip` 已解压到项目根目录下的 `data/`，确保这三个目录存在：
  ```
  data/NIST/{file_id}.jdx        # 原始数值
  data/SDBS/{file_id}.png        # SDBS原图
  data/StandardizedSpectra/      # 已有的标准化图 + all_cleaned.csv
  ```
  `data/` 整个目录是 gitignore 掉的，git pull 不会带过来，必须单独传（scp/rsync `data.zip` 再解压）。

---

## 1. 生成账目表（Phase 0，几秒钟）

```bash
cd ~/SpectrumAnalyzer   # 项目根目录，不是 experiments/
python build_nist_sdbs_manifest.py
```

产出：`data/NistSdbsSplit/manifest.csv`。跑完会打印类似：

```
train       1659
test         617
val          616
excluded    5611
训练集里的重合分子...: 1126
```

数字对不上说明 `data/NIST`、`data/SDBS`、`data/StandardizedSpectra` 没放对地方，先检查目录。

---

## 2. 生成数据集（Phase 1，视机器性能几分钟内）

```bash
python build_dataset.py
# 可选：--bins 改数值向量的分bin数，默认360（10cm-1/bin）
```

产出：

```
data/NistSdbsSplit/
├── train/
│   ├── images/{file_id}.png       # 1659张，NIST原始数据重绘（分段轴，见方案文档3.2.1）
│   ├── images_aug/{file_id}.png   # 1126张，重合分子的SDBS增强图
│   ├── vectors/{file_id}.npy      # 1659个，形状[360,2]（%T + 覆盖掩码）
│   └── labels.csv                 # 2785行（=1659+1126）
├── val/images/ + val/labels.csv   # 616条，直接拷贝自StandardizedSpectra
└── test/images/ + test/labels.csv # 617条，直接拷贝自StandardizedSpectra
```

跑完检查一下 `失败/跳过` 那行是不是0，不是0的话把打印出的失败原因发出来看。

---

## 3. 跑方法1/2：预训练 vs 从零训练消融（Phase 2）

```bash
cd experiments
```

4个backbone × 2种初始化 = 8次训练，每次都要跑：

```bash
python train.py --dataset nist_split --backbone vgg16     --pretrained 1 --epochs 50
python train.py --dataset nist_split --backbone vgg16     --pretrained 0 --epochs 50
python train.py --dataset nist_split --backbone resnet50  --pretrained 1 --epochs 50
python train.py --dataset nist_split --backbone resnet50  --pretrained 0 --epochs 50
python train.py --dataset nist_split --backbone resnet101 --pretrained 1 --epochs 50
python train.py --dataset nist_split --backbone resnet101 --pretrained 0 --epochs 50
python train.py --dataset nist_split --backbone vit_b     --pretrained 1 --epochs 50
python train.py --dataset nist_split --backbone vit_b     --pretrained 0 --epochs 50
```

**建议先跑一次短的验证没问题再跑全部**（比如某个backbone `--epochs 1`），确认能正常输出 `val_f1`/`val_auc` 不报错，再放心跑满50轮。

其他可选参数（`--batch_size`、`--lr`、`--patience`）用法和旧的 legacy 数据集一致，见 [experiments/README.md](../experiments/README.md)。

`--pretrained 0` 从随机初始化训练，正常比 `--pretrained 1` 收敛慢、效果差一截，这是预期中的对照，不是bug。

---

## 4. 看结果

每次训练跑完，产出分两处（参考 [experiments/README.md](../experiments/README.md) 的存放规则）：

```
experiments/checkpoints/{backbone}_pretrained(或_scratch)/best.pth   # 权重，留在服务器
experiments/results/{backbone}_pretrained(或_scratch)/
    ├── history.csv        # 每epoch的loss/val_f1/val_auc
    └── test_result.csv    # 测试集最终指标（含 dataset/pretrained/backbone 等字段）
```

8次跑完想快速看一眼对比，可以：

```bash
cd experiments
python -c "
import pandas as pd
from pathlib import Path
rows = []
for exp in ['vgg16_pretrained','vgg16_scratch','resnet50_pretrained','resnet50_scratch',
            'resnet101_pretrained','resnet101_scratch','vit_b_pretrained','vit_b_scratch']:
    f = Path('results')/exp/'test_result.csv'
    if f.exists():
        rows.append(pd.read_csv(f).iloc[0])
print(pd.DataFrame(rows).to_string(index=False))
"
```

`summarize_results.py` 目前只认旧的4个legacy backbone目录名，新实验命名不一样，暂时没适配（按方案文档，汇总这步等所有实验跑完再统一做，Phase 7）。

---

## 5. 打包结果传回本地

只需要 `results/`，不需要 `checkpoints/`（权重留服务器）：

```bash
tar czf results_nist_split.tar.gz experiments/results
# 本地机器执行：
scp user@server:~/SpectrumAnalyzer/results_nist_split.tar.gz .
```

---

## 常见问题

- **`FileNotFoundError` 找不到 `data/NistSdbsSplit`**：先跑第1、2步，不能跳过直接跑 `train.py --dataset nist_split`。
- **`--dataset` 忘了指定**：默认是 `legacy`（旧的StandardizedSpectra随机划分），不会报错但跑的不是新数据集，结果目录也不会带 `_pretrained`/`_scratch` 后缀，容易和新实验搞混，记得每次都显式写 `--dataset nist_split`。
- **8次训练全跑一遍很花时间**：可以先用 `--epochs 5` 左右快速摸底哪个backbone明显不收敛（参考旧baseline里 vit_b 表现最差），再决定要不要给它更多轮数。
