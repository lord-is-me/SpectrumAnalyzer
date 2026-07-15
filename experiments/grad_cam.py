"""
Grad-CAM 可视化：查看模型预测某个气味标签时，光谱图上哪一部分（哪些吸收峰）贡献最大。
单样本可视化，和 gradcam_aggregate.py（1D波数聚合）/ gradcam_region_map.py（2D区域聚合）是
互补关系——那两个是"多样本按标签平均"，这个是"看单个分子的具体几张热力图"，适合挑一个有代表
性的分子（比如乙醇）具体看模型判断每个气味标签时关注哪里。

--exp 的解析规则和 gradcam_aggregate.py 完全一致：
    纯backbone名(vgg16/resnet50/resnet101/vit_b) = legacy
    带_pretrained/_scratch后缀                   = nist_split
    fusion_{rnn,lstm,gru,transformer}_{layers}layer = 方法3融合模型（只分析图像分支）

用法:
    # 从test集随机抽5张图，每张展示预测概率最高的3个标签的热力图
    python grad_cam.py --exp resnet101_pretrained

    # 指定具体的 file_id 和标签
    python grad_cam.py --exp resnet101_pretrained --file_ids 101,205 --labels floral,woody

    # legacy / 方法3融合模型
    python grad_cam.py --exp resnet101
    python grad_cam.py --exp fusion_transformer_3layer

结果保存在 results/{exp}/gradcam/
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from standardize_spectra import TARGET_SIZE  # noqa: E402  (686, 322) = (W, H)，标准化光谱图的原生尺寸

from train import (  # noqa: E402
    build_model, get_transforms, LABEL_COLS, NUM_LABELS,
    IMG_DIR, NIST_SPLIT_ROOT, RESULT_ROOT, CHECKPOINT_ROOT,
)
from fusion_model import FusionModel  # noqa: E402
from gradcam_common import (  # noqa: E402
    LEGACY_BACKBONES, backbone_name_from_exp, parse_fusion_exp, load_legacy_rows, ImageOnlyWrapper,
)


# ──────────────────────────────────────────────
# Grad-CAM 核心实现
# ──────────────────────────────────────────────
class GradCAM:
    """对给定 target_layer 的输出做 Grad-CAM。reshape_transform 用于把非
    卷积特征（比如 ViT 的 token 序列）reshape 成 (B, C, H, W) 形式。"""

    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.reshape_transform = reshape_transform
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, label_idx):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, label_idx].sum()
        score.backward(retain_graph=True)

        activations, gradients = self.activations, self.gradients
        if self.reshape_transform is not None:
            activations = self.reshape_transform(activations)
            gradients = self.reshape_transform(gradients)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        prob = torch.sigmoid(logits)[:, label_idx].item()
        return cam, prob


def find_last_conv(module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def disable_inplace_relu(model):
    # register_full_backward_hook 的输出是一个 view，如果紧跟着 inplace ReLU
    # 会在这个 view 上直接改内存，触发 RuntimeError，所以推理时统一关掉 inplace
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


def vit_reshape_transform(tensor):
    # tensor: (B, num_patches+1, hidden_dim) -> 去掉 class token 再还原成网格
    tokens = tensor[:, 1:, :]
    n = tokens.shape[1]
    h = w = int(n ** 0.5)
    tokens = tokens.reshape(tensor.shape[0], h, w, tensor.shape[2])
    return tokens.permute(0, 3, 1, 2)


def get_target_layer(model, backbone):
    if backbone == "vgg16":
        return find_last_conv(model.features), None
    if backbone in ("resnet50", "resnet101"):
        # 用整个最后一个 Bottleneck 的输出（残差相加 + ReLU 之后），
        # 而不是内部的 conv3，这样才是实际送进 avgpool 的特征图
        return model.layer4[-1], None
    if backbone == "vit_b":
        # ViT 没有卷积特征图，用最后一个 Encoder Block 的输出近似做 Grad-CAM
        return model.encoder.layers[-1], vit_reshape_transform
    raise ValueError(f"Unknown backbone: {backbone}")


def overlay_heatmap(orig_img, cam, alpha=0.45):
    """cam 是224x224方形（CNN输入分辨率算出来的），直接按cam.shape取原图会把686x322的光谱图
    再压扁一次；这里把cam插值放大回原生 TARGET_SIZE 比例，叠加在未拉伸变形的原图上，保证
    展示图和训练图长宽比例一致（同一处理见 gradcam_region_map.py）。"""
    cam_native = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(TARGET_SIZE, Image.BILINEAR)
    ).astype(np.float32) / 255.0
    orig = np.array(orig_img.resize(TARGET_SIZE)).astype(np.float32) / 255.0
    if orig.ndim == 2:
        orig = np.stack([orig] * 3, axis=-1)
    heatmap = cm.jet(cam_native)[..., :3]
    overlay = np.clip((1 - alpha) * orig + alpha * heatmap, 0, 1)
    return overlay


# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                        help="纯backbone名(legacy)/带_pretrained/_scratch后缀(nist_split)/"
                             "fusion_{rnn,lstm,gru,transformer}_{layers}layer(方法3)，"
                             "对应 checkpoints/{exp}/best.pth")
    parser.add_argument("--dataset", type=str, default="auto", choices=["auto", "legacy", "nist_split"],
                        help="不填就按--exp名字自动判断，只对方法1/2有意义")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--num_samples", type=int, default=5, help="随机抽取的样本数")
    parser.add_argument("--file_ids", type=str, default=None, help="逗号分隔，指定 file_id 时忽略 num_samples")
    parser.add_argument("--labels", type=str, default=None, help="逗号分隔，指定要看的标签名；不填则自动取预测概率最高的几个")
    parser.add_argument("--topk_labels", type=int, default=3, help="未指定 --labels 时，每张图展示概率最高的前 K 个标签")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CHECKPOINT_ROOT / args.exp / "best.pth"

    fusion_info = parse_fusion_exp(args.exp)
    if fusion_info is not None:
        seq_arch, num_layers = fusion_info
        print(f"模型类型: 方法3融合模型 (seq_model={seq_arch}, num_layers={num_layers})，只分析图像分支")
        fusion_model = FusionModel(NUM_LABELS, seq_arch=seq_arch, seq_layers=num_layers, image_pretrained=False)
        fusion_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model = ImageOnlyWrapper(fusion_model).to(device).eval()
        disable_inplace_relu(model)
        if seq_arch in ("rnn", "lstm", "gru"):
            # cuDNN的RNN/LSTM/GRU反向传播要求模块处于training模式，见 gradcam_aggregate.py 同名注释
            model.fusion_model.seq_encoder.encoder.train()
        img_size = 224
        target_layer = model.fusion_model.image_encoder.features[-2][-1]
        reshape_transform = None
        is_legacy = False
    else:
        backbone = backbone_name_from_exp(args.exp)
        is_legacy = args.dataset == "legacy" or (args.dataset == "auto" and args.exp in LEGACY_BACKBONES)
        model, img_size = build_model(backbone, NUM_LABELS, pretrained=False)  # 权重马上被checkpoint整个覆盖
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model = model.to(device).eval()
        disable_inplace_relu(model)
        target_layer, reshape_transform = get_target_layer(model, backbone)

    cam_engine = GradCAM(model, target_layer, reshape_transform)

    if is_legacy:
        print("数据集来源: legacy（StandardizedSpectra，随机60/20/20划分，random_state=42）")
        df = load_legacy_rows(args.split)

        def resolve_img_path(row):
            return IMG_DIR / f"{int(row['file_id'])}.png"
    else:
        print(f"数据集来源: nist_split（NistSdbsSplit/{args.split}）")
        split_dir = NIST_SPLIT_ROOT / args.split
        df = pd.read_csv(split_dir / "labels.csv")

        def resolve_img_path(row):
            return split_dir / row["image_path"]

    if args.file_ids:
        file_ids = [int(x) for x in args.file_ids.split(",")]
        rows = df[df["file_id"].isin(file_ids)]
    else:
        random.seed(args.seed)
        sampled = random.sample(list(df.index), min(args.num_samples, len(df)))
        rows = df.loc[sampled]

    label_names = args.labels.split(",") if args.labels else None

    transform = get_transforms(img_size, is_train=False)
    out_dir = RESULT_ROOT / args.exp / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in rows.iterrows():
        file_id = int(row["file_id"])
        img_path = resolve_img_path(row)
        orig_img = Image.open(img_path).convert("RGB")
        x = transform(orig_img).unsqueeze(0).to(device)

        if label_names is not None:
            target_labels = label_names
        else:
            with torch.no_grad():
                probs = torch.sigmoid(model(x))[0].cpu().numpy()
            top_idx = np.argsort(probs)[::-1][:args.topk_labels]
            target_labels = [LABEL_COLS[i] for i in top_idx]

        fig, axes = plt.subplots(1, len(target_labels) + 1, figsize=(4 * (len(target_labels) + 1), 4 * TARGET_SIZE[1] / TARGET_SIZE[0]))
        axes[0].imshow(orig_img.resize(TARGET_SIZE), interpolation="nearest")
        axes[0].set_title(f"file_id={file_id}\noriginal")
        axes[0].axis("off")

        for ax, label_name in zip(axes[1:], target_labels):
            label_idx = LABEL_COLS.index(label_name)
            cam, prob = cam_engine(x, label_idx)
            overlay = overlay_heatmap(orig_img, cam)
            true_label = int(row[label_name])
            ax.imshow(overlay, interpolation="nearest")
            ax.set_title(f"{label_name}\npred={prob:.2f} true={true_label}")
            ax.axis("off")

        fig.tight_layout()
        save_path = out_dir / f"{file_id}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
