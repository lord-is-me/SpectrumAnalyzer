# Grad-CAM 可视化使用指南

用于查看模型预测某个气味标签时，光谱图上哪一部分（哪些吸收峰）贡献最大，是训练完成后的可解释性分析工具。

---

## 前置条件

必须先训练完对应 backbone，确认 `results/{backbone}/best.pth` 已经生成。Grad-CAM 依赖这个权重文件，不存在会直接报 `FileNotFoundError`。

## 1. 安装依赖

`matplotlib` 已加入 `requirements.txt`：

```bash
cd ~/SpectrumAnalyzer/experiments
conda activate deep
pip install -r requirements.txt
```

## 2. 运行

```bash
# 从测试集随机抽 5 张图，每张展示预测概率最高的 3 个标签的热力图
python grad_cam.py --backbone resnet50

# 调整抽样数量和展示的标签数
python grad_cam.py --backbone resnet101 --num_samples 8 --topk_labels 4

# 指定具体 file_id 和标签（不再随机抽样）
python grad_cam.py --backbone vit_b --file_ids 101,205 --labels floral,woody
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--backbone` | 必填 | vgg16 / resnet50 / resnet101 / vit_b |
| `--num_samples` | 5 | 从测试集随机抽取的样本数 |
| `--file_ids` | 无 | 逗号分隔，指定具体 file_id 时忽略 `--num_samples` |
| `--labels` | 无 | 逗号分隔，指定要看的标签名；不填则自动取预测概率最高的几个 |
| `--topk_labels` | 3 | 未指定 `--labels` 时，每张图展示概率最高的前 K 个标签 |
| `--seed` | 0 | 随机抽样种子 |

样本只从**测试集**里抽（和 `train.py` 用完全相同的划分方式，`random_state=42`），避免可视化到训练/验证集造成误导。

## 3. 输出结果

```
results/{backbone}/gradcam/{file_id}.png
```

每张图是"原图 + 各标签热力图叠加"的横向对比图，标题里带预测概率 `pred=` 和真实标签 `true=`（1 表示该分子确实带有此气味）。

## 4. 把结果传回本地查看

```bash
# 本地机器上执行（从服务器拉图片）
scp -r lefan@<服务器地址>:~/SpectrumAnalyzer/experiments/results/resnet50/gradcam ./gradcam_resnet50
```

---

## 实现说明

- **vgg16 / resnet50 / resnet101**：标准 Grad-CAM，hook 最后一个卷积特征层（ResNet 用整个 `layer4[-1]` 模块的输出，即残差相加 + ReLU 之后，而不是内部的 `conv3`，这样才是实际送进 avgpool 的特征图）。
- **vit_b**：ViT 没有卷积特征图，用最后一个 Transformer Block 的输出、去掉 class token 后 reshape 成 14×14 网格做近似 Grad-CAM。这是简化处理，可解释性不如卷积网络上的 Grad-CAM 严谨，但能大致看出关注区域。
- 多标签分类下，Grad-CAM 的目标分数取该标签对应输出 logit（sigmoid 之前），而不是像单标签分类那样取 argmax 类别。
