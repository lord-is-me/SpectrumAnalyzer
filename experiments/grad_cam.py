"""
Grad-CAM 可视化：查看模型预测某个气味标签时，光谱图上哪一部分（哪些吸收峰）贡献最大。
用法:
    # 从测试集随机抽 5 张图，每张展示预测概率最高的 3 个标签的热力图
    python grad_cam.py --backbone resnet50

    # 指定具体的 file_id 和标签
    python grad_cam.py --backbone resnet50 --file_ids 101,205 --labels floral,woody

结果保存在 results/{backbone}/gradcam/
"""
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from train import (
    build_model, get_transforms, LABEL_COLS, NUM_LABELS,
    CSV_PATH, IMG_DIR, RESULT_ROOT, CHECKPOINT_ROOT,
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


# ──────────────────────────────────────────────
# 数据准备（与 train.py 保持一致的划分，只从测试集里取样）
# ──────────────────────────────────────────────
def load_test_split():
    df_all = pd.read_csv(CSV_PATH)
    df_spec = df_all[df_all["spectrum_type"] == 1].copy()
    df_spec["file_id"] = df_spec.index + 2
    df_spec = df_spec.reset_index(drop=True)

    all_idx = list(range(len(df_spec)))
    tmp_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
    return df_spec, test_idx


def overlay_heatmap(orig_img, cam, alpha=0.45):
    orig = np.array(orig_img.resize((cam.shape[1], cam.shape[0]))).astype(np.float32) / 255.0
    if orig.ndim == 2:
        orig = np.stack([orig] * 3, axis=-1)
    heatmap = cm.jet(cam)[..., :3]
    overlay = np.clip((1 - alpha) * orig + alpha * heatmap, 0, 1)
    return overlay


# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["vgg16", "resnet50", "resnet101", "vit_b"])
    parser.add_argument("--num_samples", type=int, default=5, help="从测试集随机抽取的样本数")
    parser.add_argument("--file_ids", type=str, default=None, help="逗号分隔，指定 file_id 时忽略 num_samples")
    parser.add_argument("--labels", type=str, default=None, help="逗号分隔，指定要看的标签名；不填则自动取预测概率最高的几个")
    parser.add_argument("--topk_labels", type=int, default=3, help="未指定 --labels 时，每张图展示概率最高的前 K 个标签")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = build_model(args.backbone, NUM_LABELS)
    ckpt_path = CHECKPOINT_ROOT / args.backbone / "best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device).eval()
    disable_inplace_relu(model)

    target_layer, reshape_transform = get_target_layer(model, args.backbone)
    cam_engine = GradCAM(model, target_layer, reshape_transform)

    df_spec, test_idx = load_test_split()

    if args.file_ids:
        file_ids = [int(x) for x in args.file_ids.split(",")]
        rows = df_spec[df_spec["file_id"].isin(file_ids)]
    else:
        random.seed(args.seed)
        sampled = random.sample(test_idx, min(args.num_samples, len(test_idx)))
        rows = df_spec.iloc[sampled]

    label_names = args.labels.split(",") if args.labels else None

    transform = get_transforms(img_size, is_train=False)
    out_dir = RESULT_ROOT / args.backbone / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in rows.iterrows():
        file_id = int(row["file_id"])
        img_path = IMG_DIR / f"{file_id}.png"
        orig_img = Image.open(img_path).convert("RGB")
        x = transform(orig_img).unsqueeze(0).to(device)

        if label_names is not None:
            target_labels = label_names
        else:
            with torch.no_grad():
                probs = torch.sigmoid(model(x))[0].cpu().numpy()
            top_idx = np.argsort(probs)[::-1][:args.topk_labels]
            target_labels = [LABEL_COLS[i] for i in top_idx]

        fig, axes = plt.subplots(1, len(target_labels) + 1, figsize=(4 * (len(target_labels) + 1), 4))
        axes[0].imshow(orig_img.resize((img_size, img_size)))
        axes[0].set_title(f"file_id={file_id}\n原图")
        axes[0].axis("off")

        for ax, label_name in zip(axes[1:], target_labels):
            label_idx = LABEL_COLS.index(label_name)
            cam, prob = cam_engine(x, label_idx)
            overlay = overlay_heatmap(orig_img, cam)
            true_label = int(row[label_name])
            ax.imshow(overlay)
            ax.set_title(f"{label_name}\npred={prob:.2f} true={true_label}")
            ax.axis("off")

        fig.tight_layout()
        save_path = out_dir / f"{file_id}.png"
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
