# Grad-CAM 可视化使用指南

用于查看模型预测某个气味标签时，光谱图上哪一部分（哪些吸收峰）贡献最大，是训练完成后的可解释性分析工具。

---

## 前置条件

必须先训练完对应模型，确认 `checkpoints/{exp}/best.pth` 已经生成。Grad-CAM 依赖这个权重文件，不存在会直接报 `FileNotFoundError`。

## 1. 安装依赖

`matplotlib` 已加入 `requirements.txt`：

```bash
cd ~/SpectrumAnalyzer/experiments
conda activate deep
pip install -r requirements.txt
```

## 2. 运行

`--exp` 的解析规则和 `gradcam_aggregate.py`/`gradcam_region_map.py` 完全一致：纯backbone名（vgg16/resnet50/resnet101/vit_b）=legacy，带 `_pretrained`/`_scratch` 后缀=nist_split，`fusion_{rnn,lstm,gru,transformer}_{layers}layer`=方法3融合模型（只分析图像分支）。

```bash
# 从test集随机抽5张图，每张展示预测概率最高的3个标签的热力图
python grad_cam.py --exp resnet101_pretrained

# 调整抽样数量和展示的标签数
python grad_cam.py --exp resnet101_pretrained --num_samples 8 --topk_labels 4

# 指定具体 file_id 和标签（不再随机抽样）——比如挑一个有代表性的分子具体看
python grad_cam.py --exp resnet101_pretrained --file_ids 101,205 --labels floral,woody

# legacy / 方法3融合模型
python grad_cam.py --exp resnet101
python grad_cam.py --exp fusion_transformer_3layer
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--exp` | 必填 | 实验名，对应 `checkpoints/{exp}/best.pth` |
| `--dataset` | auto | 不填就按 `--exp` 名字自动判断，只对方法1/2有意义 |
| `--split` | test | test / val |
| `--num_samples` | 5 | 随机抽取的样本数 |
| `--file_ids` | 无 | 逗号分隔，指定具体 file_id 时忽略 `--num_samples` |
| `--labels` | 无 | 逗号分隔，指定要看的标签名；不填则自动取预测概率最高的几个 |
| `--topk_labels` | 3 | 未指定 `--labels` 时，每张图展示概率最高的前 K 个标签 |
| `--seed` | 0 | 随机抽样种子 |

样本默认从 **test** 集里抽（legacy 和 `train.py` 用完全相同的固定随机划分，nist_split 读 `NistSdbsSplit/test/labels.csv`），避免可视化到训练/验证集造成误导。

**想看指定分子（比如乙醇）**：数据里没有单独的化合物名字段，只有 SMILES，先按 SMILES 查出 `file_id` 再传给 `--file_ids`：

```python
import pandas as pd
df = pd.read_csv("data/StandardizedSpectra/all_cleaned.csv")   # legacy
# 或 df = pd.read_csv("data/NistSdbsSplit/test/labels.csv")     # nist_split
df["file_id"] = df.index + 2   # 仅legacy需要这一步，nist_split的labels.csv本身就有file_id列
print(df[df["SMILES"] == "CCO"][["file_id", "SMILES"]])   # 乙醇
```

## 3. 输出结果

```
results/{exp}/gradcam/{file_id}.png
```

每张图是"原图 + 各标签热力图叠加"的横向对比图，标题里带预测概率 `pred=` 和真实标签 `true=`（1 表示该分子确实带有此气味）。热力图会插值放大回原生686×322比例再叠加在未拉伸变形的原图上，不是被CNN输入分辨率（224×224正方形）压扁过的版本。

## 4. 把结果传回本地查看

`results/` 和 `checkpoints/` 是分开的：`checkpoints/{exp}/best.pth` 是模型权重，体积大，留在服务器上不用管；`results/{exp}/` 下面只有 history.csv、test_result.csv、log.txt、gradcam 图这些轻量文件，直接打包整个 `results/` 传回本地即可，不会带上权重文件。

```bash
# 本地机器上执行（从服务器拉图片，不涉及权重文件）
scp -r lefan@<服务器地址>:~/SpectrumAnalyzer/experiments/results/resnet101_pretrained/gradcam ./gradcam_resnet101
```

---

## 实现说明

- **vgg16 / resnet50 / resnet101**：标准 Grad-CAM，hook 最后一个卷积特征层（ResNet 用整个 `layer4[-1]` 模块的输出，即残差相加 + ReLU 之后，而不是内部的 `conv3`，这样才是实际送进 avgpool 的特征图）。
- **vit_b**：ViT 没有卷积特征图，用最后一个 Transformer Block 的输出、去掉 class token 后 reshape 成 14×14 网格做近似 Grad-CAM。这是简化处理，可解释性不如卷积网络上的 Grad-CAM 严谨，但能大致看出关注区域。
- 多标签分类下，Grad-CAM 的目标分数取该标签对应输出 logit（sigmoid 之前），而不是像单标签分类那样取 argmax 类别。
