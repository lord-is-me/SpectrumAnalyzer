# 总体网络结构方案

## 系统总览

```
输入层
 ├── [路径A] 红外光谱数值向量 (360维, 400-4000 cm-1)
 └── [路径B] 分子结构指纹 (2048维 Morgan FP, 可选)
        │                │
        ▼                ▼
  SpectrumEncoder   StructureEncoder
  (1D-CNN / Transformer)  (MLP)
        │                │
        └──── Fusion ────┘
                 │
           MultiLabelHead
                 │
         120维 Sigmoid 输出
```

---

## 模块一：Spectrum Encoder（核心创新）

### 版本1：细粒度分箱 + 1D-CNN（推荐起点）

```
输入: [B, 360]  (每个 bin 的最小%T)
  │
  ├── Embedding: Linear(360, 256) + LayerNorm
  │
  ├── 1D-Conv Block × 3:
  │   ├── Conv1d(256, 256, kernel=7, padding=3)
  │   ├── BatchNorm1d(256)
  │   └── GELU
  │
  ├── Global Average Pool: [B, 256]
  │
  └── 输出: spectrum_emb [B, 256]
```

实现简单，参数量小（< 500K），适合 3000 样本起步。

---

### 版本2：Peak Token Transformer（论文核心方案）

```
输入: 变长峰序列，每个峰 = (wavenumber, depth, fwhm)

步骤1: 峰检测
  %T曲线 → scipy.find_peaks(prominence, width) → Top-K 峰
  K = 30（不足补零，有 padding_mask）

步骤2: 峰嵌入
  每个峰 [3] → Linear(3, d_model=128) + 位置编码(wavenumber/4000归一化)
  输出: [B, K, 128]

步骤3: CLS Token 拼接
  [CLS, peak_1, peak_2, ..., peak_K] → [B, K+1, 128]

步骤4: Transformer Encoder
  Layers = 4
  Heads = 4
  d_ff = 512
  Dropout = 0.3
  Attention 带 padding_mask（屏蔽零填充的峰）

步骤5: 输出
  取 CLS Token → spectrum_emb [B, 128]
```

**Attention 可视化**：每个注意力头的权重矩阵可以展示"哪个峰对哪个气味标签贡献最大"，是论文里重要的定性分析材料。

---

## 模块二：Structure Encoder（可选，用于融合实验）

```
输入: Morgan FP [B, 2048]  (radius=2, nbits=2048)

  Linear(2048, 512) → GELU → Dropout(0.2)
  Linear(512, 256) → GELU → Dropout(0.2)
  Linear(256, 128)

输出: struct_emb [B, 128]
```

---

## 模块三：Fusion

```
光谱 only:    spectrum_emb → MultiLabelHead
结构 only:    struct_emb → MultiLabelHead
光谱 + 结构:  Concat([spectrum_emb, struct_emb]) → Linear(256, 256) → MultiLabelHead
```

融合方式备选：
- **Late fusion（当前）**：各自编码完再拼接
- **Cross-Attention**：光谱特征作为 Query，结构特征作为 Key/Value（或反过来），让两路信息交互

---

## 模块四：MultiLabelHead

```
输入: emb [B, D]
  │
  ├── Linear(D, 256) → GELU → Dropout(0.3)
  ├── Linear(256, 120)
  └── Sigmoid → [B, 120]  (每个标签独立概率)
```

---

## 损失函数

```python
# 逐标签加权 Focal Loss
def focal_loss(pred, target, gamma=2.0):
    pos_weight = (N_neg / N_pos).to(device)  # shape [120]
    bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
    pt = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()
```

---

## 完整模型参数量估计

| 组件 | 参数量 |
|---|---|
| Peak Token Transformer | ~280K |
| Structure Encoder（可选） | ~1.1M |
| MultiLabelHead | ~35K |
| **总计（光谱only）** | **~315K** |
| **总计（光谱+结构）** | **~1.4M** |

3000样本下建议总参数 < 2M，否则过拟合风险高。

---

## 训练配置

```yaml
optimizer: AdamW
lr: 3e-4
weight_decay: 1e-4
scheduler: CosineAnnealingLR (T_max=100)
epochs: 200 (early stopping, patience=20)
batch_size: 64
grad_clip: 1.0

augmentation (光谱向量):
  - 加性高斯噪声 σ=0.01
  - 随机基线漂移 (加一个小的线性斜率)
  - 随机水平翻转 x 轴（物理上对应 x 轴颠倒，谨慎使用）
```

---

## 各方案对比汇总

| 方案 | 架构 | 输入 | 参数量 | 解释性 | 适合样本量 |
|---|---|---|---|---|---|
| B1 基线 | RF / MLP | Morgan FP | 小 | 低 | 任意 |
| B3 基线 | 2D-CNN | 谱图图片 | ~2M | 低 | 需 >5K |
| B4 | MLP | 360维分箱向量 | ~200K | 中 | 1K+ |
| **主模型** | **Peak Transformer** | **峰序列** | **~315K** | **高** | **1K+** |
| 融合版 | Peak Trans + MLP | 峰序列 + FP | ~1.4M | 高 | 2K+ |

---

## 可视化与可解释性

1. **Attention Heatmap**：针对某个气味标签，聚合各样本里对应峰的注意力权重，画出"该气味最依赖的波数段"

2. **Peak importance map**：横轴为波数 (400-4000 cm⁻¹)，纵轴为注意力权重均值，对比不同气味类别的敏感区域差异

3. **t-SNE/UMAP**：将 spectrum_emb 降维可视化，验证同类气味的谱图在嵌入空间里是否聚集
