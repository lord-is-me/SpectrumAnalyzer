# NIST/SDBS 融合实验 —— 运行说明

配套设计文档见 [nist_fusion_experiment_plan.md](nist_fusion_experiment_plan.md)。本文档只讲怎么跑，从零开始（新服务器/新checkout）到跑出方法1/2/3的结果。

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

4个backbone × 2种初始化 = 8次训练。一键跑全部（Linux服务器）：

```bash
bash run_nist_split.sh
```

Windows本地调试用 `run_nist_split_windows.ps1`。两个脚本都是依次跑下面8条命令，日志分别写到各自 `results/{exp}/log.txt`：

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

**建议先跑一次短的验证没问题再跑全部**（比如某个backbone单独跑 `--epochs 1`），确认能正常输出 `val_f1`/`val_auc` 不报错，再用 `run_nist_split.sh` 跑满50轮的全部8组。`run_nist_split.sh` 开头有 `set -e`，中途某个backbone报错会直接停止，不会带着错误继续跑后面几个。

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

## 5. 跑方法3：原始数据+图片融合模型（Phase 3）

图像分支（ResNet50）+ 数值序列分支（RNN/LSTM/GRU/Transformer，深度可选3/7/9层）晚融合，详见 [nist_fusion_experiment_plan.md](nist_fusion_experiment_plan.md) 第4节。

先跑4种架构的3层baseline，建立基线后再决定要不要跑7/9层：

```bash
cd experiments
bash run_fusion.sh
```

等价于依次跑：

```bash
python train_fusion.py --seq_model rnn         --num_layers 3 --epochs 50
python train_fusion.py --seq_model lstm        --num_layers 3 --epochs 50
python train_fusion.py --seq_model gru         --num_layers 3 --epochs 50
python train_fusion.py --seq_model transformer --num_layers 3 --epochs 50
```

看哪个架构效果好、值得深挖，再单独加深度：

```bash
python train_fusion.py --seq_model transformer --num_layers 7
python train_fusion.py --seq_model transformer --num_layers 9
```

**关于val/test的评估口径**：val/test没有数值向量（纯SDBS来源），评估时数值分支统一喂"全掩码=0"的占位输入。训练时默认以 `--modality_dropout 0.3` 的概率随机把数值分支置空，让模型提前适应这种"没有数值数据"的场景，评估结果才不会因为遇到训练时没见过的输入模式而失真——这也意味着**测试集分数本质上反映的是模型只靠图像分支时的表现**，不是"双分支输入"下的表现，方案文档3.4节有解释这为什么是数据本身决定的、不是设计缺陷。

结果存放：`checkpoints/fusion_{seq_model}_{num_layers}layer/` + `results/fusion_{seq_model}_{num_layers}layer/`（结构和方法1/2一致）。

---

## 6. Grad-CAM 按气味标签聚合（Phase 4，5.1节）

对方法1/2/3跑出来的每个模型，把 test 集里每个正例数够多的气味标签的 Grad-CAM 热力图都聚合一遍，换算到统一波数轴上，看每个气味标签平均关注哪个波数区间。

`--exp` 名字自动决定用哪个模型/数据集（也可以用 `--dataset` 强制指定nist_split/legacy）：
- `xxx_pretrained` / `xxx_scratch` → 方法1/2，nist_split（NistSdbsSplit的固定测试集）
- 纯backbone名（`vgg16`/`resnet50`/`resnet101`/`vit_b`，没有后缀）→ 方法1/2的legacy版，StandardizedSpectra，复现 `train.py` 里那个固定 `random_state=42` 的随机测试集
- `fusion_{seq_model}_{num_layers}layer` → 方法3融合模型，只分析图像分支（数值分支喂"全掩码=0"的占位输入，和这些模型在test集上实际被评估时的输入一致，见 `train_fusion.py` 的模态dropout设计），Grad-CAM hook挂在图像分支里ResNet50的最后一个Bottleneck上

**跑legacy模型这条之前，记得先做完 [第0节的两步修复](#0-前提)**：`python ../fix_legacy_nist_images.py` 把StandardizedSpectra里533张NIST重绘图换成分段轴版本 + 重新跑一遍 `bash run_all.sh`，不然legacy的Grad-CAM还是建立在坐标轴对不齐的旧图上。

一键跑全部16个模型（8个方法1/2 nist_split + 4个legacy + 4个方法3融合模型）：

```bash
cd experiments
bash run_gradcam_aggregate.sh          # min_pos默认10
bash run_gradcam_aggregate.sh 8        # 或自己传min_pos
```

哪个模型的 `checkpoints/{exp}/best.pth` 还没跑出来，会自动跳过，不会中断整个批次。单独跑某一个：

```bash
python gradcam_aggregate.py --exp resnet101_pretrained --min_pos 10        # 方法1/2 nist_split
python gradcam_aggregate.py --exp resnet101 --min_pos 10                  # 方法1/2 legacy
python gradcam_aggregate.py --exp fusion_transformer_3layer --min_pos 10  # 方法3融合模型
```

结果存在 `results/{exp}/gradcam_aggregate/{split}_label_wavenumber_profile.csv`（行=360个波数bin，列=达标的气味标签）。

---

## 7. 打包结果传回本地

只需要 `results/`，不需要 `checkpoints/`（权重留服务器）：

```bash
tar czf results_nist_split.tar.gz experiments/results
# 本地机器执行：
scp user@server:~/SpectrumAnalyzer/results_nist_split.tar.gz .
```

---

## 常见问题

- **`FileNotFoundError` 找不到 `data/NistSdbsSplit`**：先跑第1、2步，不能跳过直接跑 `train.py --dataset nist_split` 或 `train_fusion.py`。
- **`--dataset` 忘了指定**（方法1/2）：默认是 `legacy`（旧的StandardizedSpectra随机划分），不会报错但跑的不是新数据集，结果目录也不会带 `_pretrained`/`_scratch` 后缀，容易和新实验搞混，记得每次都显式写 `--dataset nist_split`。
- **8次训练全跑一遍很花时间**：可以先用 `--epochs 5` 左右快速摸底哪个backbone明显不收敛（参考旧baseline里 vit_b 表现最差），再决定要不要给它更多轮数。
- **方法3的7层/9层模型比3层慢很多、还容易过拟合**：训练集只有2785个样本，深层RNN/Transformer很容易过拟合，先看3层baseline的val_f1曲线是否已经明显震荡/不再提升，再决定加深有没有意义。
