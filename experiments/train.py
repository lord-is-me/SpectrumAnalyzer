"""
红外光谱多标签气味分类 - Backbone 对比实验
用法:
    # 旧数据集（StandardizedSpectra，随机60/20/20划分）
    python train.py --backbone resnet50
    python train.py --backbone resnet101
    python train.py --backbone vgg16
    python train.py --backbone vit_b

    # 新数据集（NistSdbsSplit，train=NIST覆盖分子固定划分，val/test=纯SDBS固定划分）
    # 方法1/2：预训练 vs 从零训练消融
    python train.py --dataset nist_split --backbone resnet101 --pretrained 1
    python train.py --dataset nist_split --backbone resnet101 --pretrained 0

结果保存在 results/{实验名}/，权重保存在 checkpoints/{实验名}/
"""
import argparse
import os
import json
import time
from pathlib import Path

from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# ──────────────────────────────────────────────
# 路径配置（根据服务器路径修改这里）
# ──────────────────────────────────────────────
DATA_ROOT   = Path("../data/StandardizedSpectra")   # all_cleaned.csv 所在目录（旧数据集，随机划分）
IMG_DIR     = DATA_ROOT / "images"                  # {file_id}.png 所在目录
CSV_PATH    = DATA_ROOT / "all_cleaned.csv"
NIST_SPLIT_ROOT = Path("../data/NistSdbsSplit")     # 新数据集（NIST覆盖分子=train，纯SDBS=val/test，固定划分）
RESULT_ROOT = Path("results")        # 轻量结果（history/test_result/log/gradcam），跑完直接打包传回本地
CHECKPOINT_ROOT = Path("checkpoints")  # 模型权重（体积大），留在云端服务器，不参与传输

LABEL_COLS = [
    'alcoholic','aldehydic','alliaceous','almond','amber','ambre','animal',
    'anisic','apple','apricot','aromatic','balsamic','banana','berry','bland',
    'brandy','burnt','buttery','camphor','caramel','cedar','celery','cheesy',
    'cherry','chocolate','citrus','clean','cocoa','coconut','coffee','cognac',
    'cooked','cortex','creamy','cucumber','dairy','dry','earthy','ethereal',
    'fatty','fermented','fishy','floral','fresh','fruity','garlic','gassy',
    'geranium','grape','grapefruit','grassy','green','hay','hazelnut',
    'herbaceous','herbal','honey','hyacinth','jasmin','lactonic','lavender',
    'leafy','leathery','lemon','lily','marine','meaty','medicinal','melon',
    'metallic','mild','milky','mint','muguet','mushroom','musk','musty',
    'natural','nut','odorless','oily','onion','orange','orris','ozone',
    'patchouli','peach','pear','phenolic','pine','pineapple','plum','popcorn',
    'powdery','pungent','ripe','roasted','rose','rum','sandalwood','savory',
    'soapy','sour','spicy','strawberry','sulfurous','sweet','tea','tobacco',
    'tropical','vanilla','vegetable','vetiver','violet','warm','waxy',
    'winey','woody'
]
NUM_LABELS = len(LABEL_COLS)   # 118


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class SpectrumDataset(Dataset):
    def __init__(self, df, indices, transform):
        self.df = df.iloc[indices].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = int(row["file_id"])
        img_path = IMG_DIR / f"{file_id}.png"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        labels = torch.tensor(row[LABEL_COLS].values.astype(float), dtype=torch.float32)
        return img, labels


class NistSplitImageDataset(Dataset):
    """读 data/NistSdbsSplit/{split}/labels.csv，image_path 列是相对 split_dir 的相对路径
    （train 里有 images/{file_id}.png 和 images_aug/{file_id}.png 两种，val/test 只有 images/）。"""
    def __init__(self, df, split_dir: Path, transform):
        self.df = df.reset_index(drop=True)
        self.split_dir = split_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.split_dir / row["image_path"]).convert("RGB")
        img = self.transform(img)
        labels = torch.tensor(row[LABEL_COLS].values.astype(float), dtype=torch.float32)
        return img, labels


def get_transforms(img_size, is_train):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.0),   # 光谱图不适合水平翻转（会改变波数方向）
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# ──────────────────────────────────────────────
# Backbone 构建
# ──────────────────────────────────────────────
def build_model(backbone: str, num_labels: int, pretrained: bool = True):
    """pretrained=False 时不加载 ImageNet 权重，从随机初始化开始训练
    （方法1/2消融：预训练 vs 从零训练，见 docs/nist_fusion_experiment_plan.md 第4节）。"""
    if backbone == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(4096, num_labels)
        img_size = 224

    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
        img_size = 224

    elif backbone == "resnet101":
        weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet101(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
        img_size = 224

    elif backbone == "vit_b":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_labels)
        img_size = 224

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model, img_size


# ──────────────────────────────────────────────
# 损失：逐标签加权 BCE（处理正负样本不平衡）
# ──────────────────────────────────────────────
def compute_pos_weight(df, indices):
    labels = df.iloc[indices][LABEL_COLS].values.astype(float)
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


# ──────────────────────────────────────────────
# 评估指标
# ──────────────────────────────────────────────
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.numpy())
    probs   = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    preds   = (probs >= threshold).astype(int)

    # 只对至少有1个正例的标签计算指标，避免 AUC 报错
    valid = targets.sum(axis=0) > 0
    macro_f1  = f1_score(targets[:, valid], preds[:, valid], average="macro", zero_division=0)
    try:
        macro_auc = roc_auc_score(targets[:, valid], probs[:, valid], average="macro")
    except Exception:
        macro_auc = float("nan")
    return macro_f1, macro_auc


# ──────────────────────────────────────────────
# 主训练逻辑
# ──────────────────────────────────────────────
def prepare_legacy_datasets(img_size):
    """旧数据集：StandardizedSpectra，60/20/20随机划分（SDBS/NIST来源混在一起，random_state=42）。"""
    df_all  = pd.read_csv(CSV_PATH)
    df_spec = df_all[df_all["spectrum_type"] == 1].copy()
    df_spec["file_id"] = df_spec.index + 2   # 过滤前保存原始行号，用于定位图片文件
    df_spec = df_spec.reset_index(drop=True)
    print(f"有光谱的样本数: {len(df_spec)}")

    all_idx  = list(range(len(df_spec)))
    tmp_idx, test_idx = train_test_split(all_idx, test_size=0.2,  random_state=42)
    train_idx, val_idx = train_test_split(tmp_idx, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    train_ds = SpectrumDataset(df_spec, train_idx, get_transforms(img_size, True))
    val_ds   = SpectrumDataset(df_spec, val_idx,   get_transforms(img_size, False))
    test_ds  = SpectrumDataset(df_spec, test_idx,  get_transforms(img_size, False))
    pos_weight = compute_pos_weight(df_spec, train_idx)
    return train_ds, val_ds, test_ds, pos_weight


def prepare_nist_split_datasets(img_size):
    """新数据集：NistSdbsSplit，train=NIST覆盖分子（固定，含SDBS增强图），
    val/test=纯SDBS分子（固定划分），见 docs/nist_fusion_experiment_plan.md。"""
    splits = {}
    for split in ("train", "val", "test"):
        split_dir = NIST_SPLIT_ROOT / split
        df = pd.read_csv(split_dir / "labels.csv")
        splits[split] = df
        print(f"{split} 样本数: {len(df)}")

    train_ds = NistSplitImageDataset(splits["train"], NIST_SPLIT_ROOT / "train", get_transforms(img_size, True))
    val_ds   = NistSplitImageDataset(splits["val"],   NIST_SPLIT_ROOT / "val",   get_transforms(img_size, False))
    test_ds  = NistSplitImageDataset(splits["test"],  NIST_SPLIT_ROOT / "test",  get_transforms(img_size, False))
    pos_weight = compute_pos_weight(splits["train"], list(range(len(splits["train"]))))
    return train_ds, val_ds, test_ds, pos_weight


def train(args):
    # ── 数据准备 ──
    pretrained = bool(args.pretrained)
    model, img_size = build_model(args.backbone, NUM_LABELS, pretrained=pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Dataset: {args.dataset}  Pretrained: {pretrained}")
    model = model.to(device)

    if args.dataset == "nist_split":
        train_ds, val_ds, test_ds, pos_weight = prepare_nist_split_datasets(img_size)
        exp_name = f"{args.backbone}_{'pretrained' if pretrained else 'scratch'}"
    else:
        train_ds, val_ds, test_ds, pos_weight = prepare_legacy_datasets(img_size)
        exp_name = args.backbone

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    pos_weight = pos_weight.to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 输出目录 ──
    # out_dir: 轻量结果（history/test_result），跑完打包传回本地
    # ckpt_dir: 模型权重（体积大），留在服务器，不参与传输
    out_dir = RESULT_ROOT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINT_ROOT / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 训练循环 ──
    best_val_f1 = -1
    history = []
    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False,
                    ncols=90, unit="batch")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        val_f1, val_auc = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        elapsed  = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  val_f1={val_f1:.4f}  val_auc={val_auc:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "loss": avg_loss, "val_f1": val_f1, "val_auc": val_auc})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── 测试集评估 ──
    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))
    test_f1, test_auc = evaluate(model, test_loader, device)
    print(f"\n[{exp_name}] Test  Macro-F1={test_f1:.4f}  Macro-AUC={test_auc:.4f}")

    # ── 保存结果 ──
    result = {
        "backbone":   args.backbone,
        "exp_name":   exp_name,
        "dataset":    args.dataset,
        "pretrained": pretrained,
        "test_f1":    test_f1,
        "test_auc":   test_auc,
        "best_val_f1": best_val_f1,
        "epochs_ran": len(history),
    }
    pd.DataFrame([result]).to_csv(out_dir / "test_result.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    print(f"结果已保存到 {out_dir}/")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",   type=str, default="resnet50",
                        choices=["vgg16", "resnet50", "resnet101", "vit_b"])
    parser.add_argument("--dataset",    type=str, default="legacy",
                        choices=["legacy", "nist_split"],
                        help="legacy=StandardizedSpectra随机划分；nist_split=NistSdbsSplit固定划分")
    parser.add_argument("--pretrained", type=int, default=1, choices=[0, 1],
                        help="1=加载ImageNet预训练权重，0=随机初始化从零训练（方法1/2消融）")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int, default=10)
    args = parser.parse_args()
    train(args)
