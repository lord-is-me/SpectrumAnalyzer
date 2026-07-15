"""
Grad-CAM 系列脚本共用的 --exp 解析 / 数据集加载辅助函数。被 grad_cam.py（单样本可视化）、
gradcam_aggregate.py（1D波数聚合）、gradcam_region_map.py（2D区域聚合）、
gradcam_numeric_consistency.py（图像/数值跨模态一致性）共用。

单独拆出这个文件是因为 grad_cam.py 定义了被其余三个脚本导入的 GradCAM 核心类，如果这些
--exp 解析函数留在 gradcam_aggregate.py 里，grad_cam.py 想复用它们就会反过来 import
gradcam_aggregate.py，形成循环 import。
"""
import re

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from train import CSV_PATH
from fusion_model import FusionModel

LEGACY_BACKBONES = ("vgg16", "resnet50", "resnet101", "vit_b")
FUSION_RE = re.compile(r"^fusion_(rnn|lstm|gru|transformer)_(\d+)layer$")


def backbone_name_from_exp(exp_name: str) -> str:
    """"resnet101_pretrained" -> "resnet101"，"vgg16" -> "vgg16"（legacy实验没有后缀）。"""
    for bb in ("resnet101", "resnet50", "vgg16", "vit_b"):
        if exp_name.startswith(bb):
            return bb
    raise ValueError(f"无法从实验名解析backbone: {exp_name}")


def parse_fusion_exp(exp_name: str):
    """"fusion_transformer_3layer" -> ("transformer", 3)，不是融合实验名就返回 None。"""
    m = FUSION_RE.match(exp_name)
    return (m.group(1), int(m.group(2))) if m else None


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


class ImageOnlyWrapper(nn.Module):
    """Grad-CAM 只分析融合模型的图像分支：包成单参数(image)的forward，数值分支喂
    "全掩码=0"的占位输入——和val/test实际评估时的输入完全一致（它们没有真实数值向量）。"""
    def __init__(self, fusion_model: FusionModel, n_bins: int = 360):
        super().__init__()
        self.fusion_model = fusion_model
        self.n_bins = n_bins

    def forward(self, image):
        b = image.shape[0]
        values = torch.full((b, self.n_bins), 100.0, device=image.device)
        mask = torch.zeros((b, self.n_bins), device=image.device)
        return self.fusion_model(image, values, mask)
