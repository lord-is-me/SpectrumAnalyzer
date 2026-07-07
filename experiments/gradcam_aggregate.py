"""
Phase 4 (5.1)：按气味标签聚合 Grad-CAM 热力图，找每个气味标签最敏感的波数区间。

对每个标签，把 test 集里所有该标签为正例的样本单独算一次 Grad-CAM，把热力图沿波数轴
（不是原始像素列，用 standardize_spectra.pixel_to_wavenumber 换算，NIST重绘图和SDBS图
现在共用同一套换算公式）重采样到统一的360-bin网格再取平均，得到"这个气味标签平均看哪
里"的波数敏感曲线。

支持两种数据集来源（根据 --exp 名字自动判断，也可以用 --dataset 强制指定）：
  - nist_split：exp 名带 _pretrained/_scratch 后缀，读 data/NistSdbsSplit/{split}/labels.csv
  - legacy    ：exp 就是纯 backbone 名（vgg16/resnet50/resnet101/vit_b），复现 train.py
                里那个固定 random_state=42 的 60/20/20 随机划分，从 StandardizedSpectra 取图

用法:
    python gradcam_aggregate.py --exp resnet101_pretrained --min_pos 10   # nist_split
    python gradcam_aggregate.py --exp resnet101 --min_pos 10             # legacy

只分析 test 集里正例数 >= --min_pos 的标签，正例太少统计噪声太大，没意义。

结果保存在 results/{exp}/gradcam_aggregate/{split}_label_wavenumber_profile.csv
（行=360个波数bin，列=每个达标的气味标签）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from standardize_spectra import pixel_to_wavenumber, TARGET_SIZE  # noqa: E402

from train import (  # noqa: E402
    build_model, get_transforms, LABEL_COLS, NUM_LABELS,
    NIST_SPLIT_ROOT, CHECKPOINT_ROOT, RESULT_ROOT, CSV_PATH, IMG_DIR,
)
from grad_cam import GradCAM, get_target_layer, disable_inplace_relu  # noqa: E402

N_BINS = 360
WN_LOW, WN_HIGH = 400.0, 4000.0
LEGACY_BACKBONES = ("vgg16", "resnet50", "resnet101", "vit_b")


def backbone_name_from_exp(exp_name: str) -> str:
    """"resnet101_pretrained" -> "resnet101"，"vgg16" -> "vgg16"（legacy实验没有后缀）。"""
    for bb in ("resnet101", "resnet50", "vgg16", "vit_b"):
        if exp_name.startswith(bb):
            return bb
    raise ValueError(f"无法从实验名解析backbone: {exp_name}")


def load_legacy_rows(split: str) -> pd.DataFrame:
    """复现 train.py::prepare_legacy_datasets 里那个固定 random_state=42 的
    60/20/20 划分，取出 val 或 test 对应的那部分行（file_id + 118个标签列）。"""
    df_all = pd.read_csv(CSV_PATH)
    df_spec = df_all[df_all["spectrum_type"] == 1].copy()
    df_spec["file_id"] = df_spec.index + 2
    df_spec = df_spec.reset_index(drop=True)

    all_idx = list(range(len(df_spec)))
    tmp_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(tmp_idx, test_size=0.25, random_state=42)
    idx = val_idx if split == "val" else test_idx
    return df_spec.iloc[idx].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="resnet101_pretrained",
                        help="要分析的实验名，对应 checkpoints/{exp}/best.pth；"
                             "纯backbone名(vgg16/resnet50/resnet101/vit_b)=legacy，带_pretrained/_scratch后缀=nist_split")
    parser.add_argument("--dataset", type=str, default="auto", choices=["auto", "legacy", "nist_split"],
                         help="不填就按--exp名字自动判断")
    parser.add_argument("--min_pos", type=int, default=10,
                        help="标签在该split正例数至少要有这么多才纳入分析")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    args = parser.parse_args()

    backbone = backbone_name_from_exp(args.exp)
    is_legacy = args.dataset == "legacy" or (args.dataset == "auto" and args.exp in LEGACY_BACKBONES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, img_size = build_model(backbone, NUM_LABELS, pretrained=False)  # 权重马上被checkpoint整个覆盖
    ckpt_path = CHECKPOINT_ROOT / args.exp / "best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device).eval()
    disable_inplace_relu(model)

    target_layer, reshape_transform = get_target_layer(model, backbone)
    cam_engine = GradCAM(model, target_layer, reshape_transform)

    if is_legacy:
        print(f"数据集来源: legacy（StandardizedSpectra，随机60/20/20划分，random_state=42）")
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

    bin_edges = np.linspace(WN_LOW, WN_HIGH, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    label_counts = df[LABEL_COLS].sum(axis=0)
    valid_labels = [l for l in LABEL_COLS if label_counts[l] >= args.min_pos]
    print(f"共 {len(LABEL_COLS)} 个标签，{len(valid_labels)} 个在 {args.split} 集正例数 >= {args.min_pos}，只分析这些")

    profiles = {}
    for label in valid_labels:
        pos_rows = df[df[label] == 1]
        label_idx = LABEL_COLS.index(label)
        bin_vals = []
        for _, row in pos_rows.iterrows():
            img_path = resolve_img_path(row)
            orig_img = Image.open(img_path).convert("RGB")
            x = transform(orig_img).unsqueeze(0).to(device)
            cam, _prob = cam_engine(x, label_idx)  # cam: [img_size, img_size]，0-1

            col_importance = cam.max(axis=0)  # 每一列取最大激活 -> [img_size]
            px_686 = np.linspace(0, TARGET_SIZE[0] - 1, num=len(col_importance))
            wn = pixel_to_wavenumber(px_686, TARGET_SIZE[0])  # 随 px_686 递增而递减
            # np.interp 要求 x 递增，wn是递减的，两边一起倒序
            resampled = np.interp(bin_centers, wn[::-1], col_importance[::-1])
            bin_vals.append(resampled)

        profiles[label] = np.mean(bin_vals, axis=0)
        print(f"  {label:15s}  n_pos={len(pos_rows):3d}  完成")

    out = pd.DataFrame(profiles, index=bin_centers.astype(int))
    out.index.name = "wavenumber_cm-1"
    out_dir = RESULT_ROOT / args.exp / "gradcam_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_label_wavenumber_profile.csv"
    out.to_csv(out_path)
    print(f"\n已保存到 {out_path}")


if __name__ == "__main__":
    main()
