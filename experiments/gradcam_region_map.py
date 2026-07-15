"""
按气味标签聚合 Grad-CAM 的二维区域重要性——不像 gradcam_aggregate.py 那样把热力图坍缩成
1D波数曲线，而是保留图像原始的二维结构，看模型判断某个气味标签时，光谱图上哪一片**区域**
（不仅是哪个波数，还包括是曲线的峰顶/峰谷/基线附近）平均被关注。

对每个标签，把 test 集里该标签全部正例样本的 Grad-CAM 热力图逐像素取平均，叠加在这些样本的
平均光谱图背景上。和 gradcam_aggregate.py 的1D波数曲线是互补关系：那边告诉你"哪个波数重要"，
这边告诉你"图像上哪个区域重要"（同一个波数在图像上对应一整条竖线，2D图能看出模型是主要看峰顶
还是峰的其他部分）。

CNN 实际吃的是把686x322硬拉伸成224x224正方形的输入，Grad-CAM 算出来的热力图也是224x224的
正方形。这里把热力图插值放大回标准化光谱图的原生686x322比例（TARGET_SIZE），叠加在未经方形
裁剪/拉伸的原图上再保存——保证输出图和训练图长宽比例一致，不是被拉伸压扁过的版本。

支持方法1/2的4个backbone（vgg16/resnet50/resnet101/vit_b，纯名=legacy，带_pretrained/
_scratch后缀=nist_split）和方法3的4个融合模型（fusion_{rnn,lstm,gru,transformer}_{layers}
layer，只分析图像分支，数值分支喂占位输入，和 gradcam_aggregate.py 的规则完全一致，直接
复用它的 ImageOnlyWrapper/parse_fusion_exp/backbone_name_from_exp/load_legacy_rows）。

用法:
    python gradcam_region_map.py --exp resnet101_pretrained --min_pos 10   # nist_split
    python gradcam_region_map.py --exp resnet101 --min_pos 10             # legacy
    python gradcam_region_map.py --exp fusion_transformer_3layer --min_pos 10  # 方法3融合模型

结果保存在 results/{exp}/gradcam_region_map/
    {label}.png    每个达标标签一张：该标签正例样本的平均Grad-CAM热力图，叠加在平均光谱图背景上
    _grid.png      所有达标标签拼成一张总览网格图
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import (  # noqa: E402
    build_model, get_transforms, LABEL_COLS, NUM_LABELS,
    NIST_SPLIT_ROOT, CHECKPOINT_ROOT, RESULT_ROOT, IMG_DIR,
)
from grad_cam import GradCAM, get_target_layer, disable_inplace_relu  # noqa: E402
from fusion_model import FusionModel  # noqa: E402
from gradcam_common import (  # noqa: E402
    backbone_name_from_exp, load_legacy_rows, LEGACY_BACKBONES,
    parse_fusion_exp, ImageOnlyWrapper,
)
from standardize_spectra import TARGET_SIZE  # noqa: E402  (686, 322) = (W, H)，标准化光谱图的原生尺寸

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def overlay_heatmap(bg_arr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """bg_arr: [H,W,3] 0-1 背景图；cam: [H,W] 0-1 热力图。"""
    heatmap = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * bg_arr + alpha * heatmap, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                        help="纯backbone名(legacy)或带_pretrained/_scratch后缀(nist_split)，对应 checkpoints/{exp}/best.pth")
    parser.add_argument("--dataset", type=str, default="auto", choices=["auto", "legacy", "nist_split"])
    parser.add_argument("--min_pos", type=int, default=10,
                        help="标签在该split正例数至少要有这么多才纳入分析")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--max_samples_per_label", type=int, default=60,
                        help="每个标签最多平均多少张正例样本，0=不限制")
    parser.add_argument("--grid_cols", type=int, default=6)
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
        image_root = IMG_DIR

        def resolve_img_path(row):
            return image_root / f"{int(row['file_id'])}.png"
    else:
        print(f"数据集来源: nist_split（NistSdbsSplit/{args.split}）")
        split_dir = NIST_SPLIT_ROOT / args.split
        df = pd.read_csv(split_dir / "labels.csv")

        def resolve_img_path(row):
            return split_dir / row["image_path"]

    transform = get_transforms(img_size, is_train=False)

    label_counts = df[LABEL_COLS].sum(axis=0)
    valid_labels = [l for l in LABEL_COLS if label_counts[l] >= args.min_pos]
    print(f"共 {len(LABEL_COLS)} 个标签，{len(valid_labels)} 个在 {args.split} 集正例数 >= {args.min_pos}，只分析这些")

    out_dir = RESULT_ROOT / args.exp / "gradcam_region_map"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_entries = []
    for label in valid_labels:
        pos_rows = df[df[label] == 1]
        if args.max_samples_per_label > 0 and len(pos_rows) > args.max_samples_per_label:
            pos_rows = pos_rows.sample(n=args.max_samples_per_label, random_state=42)
        label_idx = LABEL_COLS.index(label)

        cam_sum = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]))   # (H, W)
        img_sum = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3))
        n = 0
        for _, row in pos_rows.iterrows():
            img_path = resolve_img_path(row)
            orig_img = Image.open(img_path).convert("RGB")
            x = transform(orig_img).unsqueeze(0).to(device)
            cam, _prob = cam_engine(x, label_idx)  # [img_size, img_size]（方形，CNN输入分辨率），单样本已归一化到0-1

            # CNN 输入是把686x322硬拉伸成224x224正方形算出来的cam，直接当背景图叠加会把光谱图
            # 压扁变形；这里把cam插值放大回原生686x322比例再叠加，背景图也用未经方形裁剪的原图，
            # 保证输出图和训练图长宽比例、清晰度一致，不是被拉伸压缩过的版本
            cam_native = np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(TARGET_SIZE, Image.BILINEAR)
            ).astype(np.float32) / 255.0

            cam_sum += cam_native
            img_sum += np.array(orig_img.resize(TARGET_SIZE)).astype(np.float32) / 255.0
            n += 1

        cam_avg = cam_sum / n
        # 多样本平均后重新拉伸到0-1，纯为了显示对比度——每个标签的原始信号强弱不影响这一步，
        # 数值层面的"哪些标签整体激活更强"不该看这张图，要看 gradcam_aggregate.py 的1D曲线
        cam_avg = (cam_avg - cam_avg.min()) / (cam_avg.max() - cam_avg.min() + 1e-8)
        img_avg = img_sum / n
        overlay = overlay_heatmap(img_avg, cam_avg)

        # figsize按TARGET_SIZE的真实长宽比来，不用凑整数，避免matplotlib再引入一次拉伸变形
        fig, ax = plt.subplots(figsize=(TARGET_SIZE[0] / 100, TARGET_SIZE[1] / 100 + 0.4))
        ax.imshow(overlay, interpolation="nearest")
        ax.set_title(f"{label}  (n={n})", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{label}.png", dpi=200)
        plt.close(fig)

        grid_entries.append((label, n, overlay))
        print(f"  {label:15s}  n={n:3d}  完成")

    cols = args.grid_cols
    rows = (len(grid_entries) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 1.6 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < len(grid_entries):
            label, n, overlay = grid_entries[i]
            ax.imshow(overlay, interpolation="nearest")
            ax.set_title(f"{label} (n={n})", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "_grid.png", dpi=150)
    plt.close(fig)

    print(f"\n已保存到 {out_dir}/")


if __name__ == "__main__":
    main()
