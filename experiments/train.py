"""
红外光谱多标签气味分类 - Backbone 对比实验
用法:
    python train.py --backbone resnet50
    python train.py --backbone resnet101
    python train.py --backbone vgg16
    python train.py --backbone vit_b

结果保存在 results/{backbone}/
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
DATA_ROOT   = Path("../data/StandardizedSpectra")   # all_cleaned.csv 所在目录
IMG_DIR     = DATA_ROOT / "images"                  # {file_id}.png 所在目录
CSV_PATH    = DATA_ROOT / "all_cleaned.csv"
RESULT_ROOT = Path("results")

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
def build_model(backbone: str, num_labels: int):
    if backbone == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(4096, num_labels)
        img_size = 224

    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
        img_size = 224

    elif backbone == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
        img_size = 224

    elif backbone == "vit_b":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
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
def train(args):
    # ── 数据准备 ──
    df_all  = pd.read_csv(CSV_PATH)
    df_spec = df_all[df_all["spectrum_type"] == 1].copy()
    df_spec["file_id"] = df_spec.index + 2   # 过滤前保存原始行号，用于定位图片文件
    df_spec = df_spec.reset_index(drop=True)
    print(f"有光谱的样本数: {len(df_spec)}")

    all_idx  = list(range(len(df_spec)))
    tmp_idx, test_idx = train_test_split(all_idx, test_size=0.2,  random_state=42)
    train_idx, val_idx = train_test_split(tmp_idx, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    model, img_size = build_model(args.backbone, NUM_LABELS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)

    train_ds = SpectrumDataset(df_spec, train_idx, get_transforms(img_size, True))
    val_ds   = SpectrumDataset(df_spec, val_idx,   get_transforms(img_size, False))
    test_ds  = SpectrumDataset(df_spec, test_idx,  get_transforms(img_size, False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    pos_weight = compute_pos_weight(df_spec, train_idx).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 输出目录 ──
    out_dir = RESULT_ROOT / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)

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
            torch.save(model.state_dict(), out_dir / "best.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── 测试集评估 ──
    model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device))
    test_f1, test_auc = evaluate(model, test_loader, device)
    print(f"\n[{args.backbone}] Test  Macro-F1={test_f1:.4f}  Macro-AUC={test_auc:.4f}")

    # ── 保存结果 ──
    result = {
        "backbone":  args.backbone,
        "test_f1":   test_f1,
        "test_auc":  test_auc,
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
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int, default=10)
    args = parser.parse_args()
    train(args)
