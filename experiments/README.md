# 红外光谱多标签气味分类 — Backbone 基线实验

## 任务描述

输入一张红外光谱图（686×322，黑白），预测该分子具有哪些气味描述符（118类二元多标签分类）。

本轮实验目标：用标准视觉 backbone（VGG16 / ResNet50 / ResNet101 / ViT-B）建立性能基线，为后续专用光谱网络的设计提供对比依据。

---

## 目录结构

```
experiments/
├── train.py              # 主训练脚本
├── summarize_results.py  # 汇总所有 backbone 的结果表格
├── run_all.sh            # 一键顺序运行全部 backbone
├── README.md             # 本文件
├── checkpoints/          # 模型权重（体积大），运行后自动生成，留在服务器上，不需要传回本地
│   ├── vgg16/best.pth
│   ├── resnet50/best.pth
│   ├── resnet101/best.pth
│   └── vit_b/best.pth
└── results/              # 轻量结果，运行后自动生成，跑完直接打包传回本地
    ├── vgg16/
    │   ├── history.csv     # 每 epoch 的 loss / val_f1 / val_auc
    │   ├── test_result.csv # 测试集最终指标
    │   └── log.txt         # 训练日志
    ├── resnet50/
    ├── resnet101/
    ├── vit_b/
    └── comparison.csv    # 四个 backbone 的汇总对比表
```

模型权重（`checkpoints/`）和结果（`results/`）分开存放：训练在云端服务器上跑，权重文件没必要传回本地，只需要把整个 `results/` 目录打包传回来就有全部指标和日志；`checkpoints/` 留在服务器上供 Grad-CAM 等后续分析直接读取。

---

## 数据说明

| 项目 | 内容 |
|---|---|
| 光谱图 | `data/StandardizedSpectra/images/{file_id}.png` |
| 标签文件 | `data/StandardizedSpectra/all_cleaned.csv` |
| 有效样本数 | 2892（spectrum_type == 1 的行） |
| 标签数 | 118 个气味描述符（二元多标签） |
| 划分比例 | 训练 60% / 验证 20% / 测试 20%，随机种子固定为 42 |

---

## 环境依赖

```bash
pip install torch torchvision scikit-learn pandas pillow
```

---

## 运行步骤

### Step 1：确认路径

打开 `train.py`，修改第 19 行的 `DATA_ROOT` 为服务器上的实际路径：

```python
DATA_ROOT = Path("/your/server/path/to/StandardizedSpectra")
```

### Step 2：运行单个 backbone（调试用）

```bash
python train.py --backbone resnet50 --epochs 50
```

可选参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--backbone` | resnet50 | vgg16 / resnet50 / resnet101 / vit_b |
| `--dataset` | legacy | legacy=StandardizedSpectra随机60/20/20划分；nist_split=NistSdbsSplit固定划分（见docs/nist_fusion_experiment_plan.md） |
| `--pretrained` | 1 | 1=加载ImageNet预训练权重，0=随机初始化从零训练（预训练vs从零训练消融，只在 `--dataset nist_split` 下有意义） |
| `--epochs` | 50 | 最大训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr` | 1e-4 | 初始学习率 |
| `--patience` | 10 | 早停耐心值（验证 F1 不提升的 epoch 数） |

`--dataset nist_split` 下，权重和结果分别存到 `checkpoints/{backbone}_pretrained(或_scratch)/` 和 `results/{backbone}_pretrained(或_scratch)/`，和旧的 `--dataset legacy`（存到 `checkpoints/{backbone}/`）互不冲突。

```bash
# 新数据集：预训练 vs 从零训练消融
python train.py --dataset nist_split --backbone resnet101 --pretrained 1
python train.py --dataset nist_split --backbone resnet101 --pretrained 0
```

### Step 3：运行全部 backbone

```bash
bash run_all.sh
```

四个 backbone 依次运行，日志分别写入各自目录。

### Step 4：汇总结果

```bash
python summarize_results.py
```

输出示例：

```
===== Backbone 对比结果 =====
Backbone    Test Macro-F1  Test Macro-AUC  Best Val-F1  Epochs
vgg16            0.xxxx         0.xxxx       0.xxxx        xx
resnet50         0.xxxx         0.xxxx       0.xxxx        xx
resnet101        0.xxxx         0.xxxx       0.xxxx        xx
vit_b            0.xxxx         0.xxxx       0.xxxx        xx
```

---

## 实验说明

**图片预处理**：灰度图复制为三通道 RGB，再用 ImageNet 均值方差归一化（backbone 是在 RGB 图像上预训练的）。

**损失函数**：逐标签加权 BCE（`pos_weight = N_neg / N_pos`），补偿各标签正负样本数量差异。

**指标**：Macro F1（主指标）和 Macro AUC（辅助指标），均只统计训练集中至少有 1 个正例的标签。

**早停**：验证集 Macro F1 连续 10 个 epoch 不提升则停止，保存最优权重用于测试集评估。

---

## 后续计划

基线实验完成后，根据对比结果设计针对光谱特点的专用网络（峰值 Token Transformer），预期在现有 baseline 上进一步提升性能。
