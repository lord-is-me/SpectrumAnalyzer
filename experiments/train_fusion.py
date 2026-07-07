"""
方法3：原始数据 + 图片融合模型（RNN/LSTM/GRU/Transformer 序列分支 + CNN图像分支）
只能在 NistSdbsSplit 上跑（train里才有数值向量，见 build_dataset.py）。

用法:
    python train_fusion.py --seq_model transformer --num_layers 3
    python train_fusion.py --seq_model lstm        --num_layers 7

val/test 没有数值向量（只有纯SDBS图像），评估时数值分支统一喂"全掩码=0"的占位输入
（代表"没有数值数据"）。为了不让模型在评估时遇到训练时从没见过的输入模式，训练时按
--modality_dropout 概率随机把数值分支整个置空，让模型学会缺numeric输入时只靠图像分支
也能给出合理预测（见 docs/nist_fusion_experiment_plan.md 方法3小节）。

结果保存在 results/fusion_{seq_model}_{num_layers}layer/，权重保存在同名 checkpoints/ 下。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from train import (
    LABEL_COLS, NUM_LABELS, NIST_SPLIT_ROOT, RESULT_ROOT, CHECKPOINT_ROOT,
    get_transforms, compute_pos_weight,
)
from fusion_model import FusionModel

N_BINS = 360


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class FusionDataset(Dataset):
    def __init__(self, df, split_dir: Path, transform, has_vector: bool, modality_dropout: float = 0.0):
        self.df = df.reset_index(drop=True)
        self.split_dir = split_dir
        self.transform = transform
        self.has_vector = has_vector
        self.modality_dropout = modality_dropout

    def __len__(self):
        return len(self.df)

    def _null_vector(self):
        values = np.full(N_BINS, 100.0, dtype=np.float32)
        mask = np.zeros(N_BINS, dtype=np.float32)
        return values, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.split_dir / row["image_path"]).convert("RGB")
        img = self.transform(img)
        labels = torch.tensor(row[LABEL_COLS].values.astype(float), dtype=torch.float32)

        if self.has_vector:
            drop = self.modality_dropout > 0 and np.random.rand() < self.modality_dropout
            if drop:
                values, mask = self._null_vector()
            else:
                vec = np.load(self.split_dir / row["vector_path"])  # [N, 2]
                values = vec[:, 0].astype(np.float32)
                mask = vec[:, 1].astype(np.float32)
        else:
            values, mask = self._null_vector()

        return img, torch.from_numpy(values), torch.from_numpy(mask), labels


# ──────────────────────────────────────────────
# 评估
# ──────────────────────────────────────────────
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, values, mask, labels in loader:
            imgs, values, mask = imgs.to(device), values.to(device), mask.to(device)
            logits = model(imgs, values, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.numpy())
    probs = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    preds = (probs >= threshold).astype(int)

    valid = targets.sum(axis=0) > 0
    macro_f1 = f1_score(targets[:, valid], preds[:, valid], average="macro", zero_division=0)
    try:
        macro_auc = roc_auc_score(targets[:, valid], probs[:, valid], average="macro")
    except Exception:
        macro_auc = float("nan")
    return macro_f1, macro_auc


# ──────────────────────────────────────────────
# 主训练逻辑
# ──────────────────────────────────────────────
def train(args):
    exp_name = f"fusion_{args.seq_model}_{args.num_layers}layer"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = pd.read_csv(NIST_SPLIT_ROOT / split / "labels.csv")
        print(f"{split} 样本数: {len(splits[split])}")

    train_ds = FusionDataset(splits["train"], NIST_SPLIT_ROOT / "train", get_transforms(224, True),
                              has_vector=True, modality_dropout=args.modality_dropout)
    val_ds = FusionDataset(splits["val"], NIST_SPLIT_ROOT / "val", get_transforms(224, False),
                            has_vector=False)
    test_ds = FusionDataset(splits["test"], NIST_SPLIT_ROOT / "test", get_transforms(224, False),
                             has_vector=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = FusionModel(NUM_LABELS, seq_arch=args.seq_model, seq_layers=args.num_layers,
                         image_pretrained=bool(args.image_pretrained)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}  exp: {exp_name}  参数量: {n_params/1e6:.2f}M  modality_dropout={args.modality_dropout}")

    pos_weight = compute_pos_weight(splits["train"], list(range(len(splits["train"])))).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = RESULT_ROOT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINT_ROOT / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1
    history = []
    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False, ncols=90, unit="batch")
        for imgs, values, mask, labels in pbar:
            imgs, values, mask, labels = imgs.to(device), values.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, values, mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        val_f1, val_auc = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0
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

    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))
    test_f1, test_auc = evaluate(model, test_loader, device)
    print(f"\n[{exp_name}] Test（val/test无数值向量，数值分支吃占位输入，等价于图像分支为主）"
          f"  Macro-F1={test_f1:.4f}  Macro-AUC={test_auc:.4f}")

    result = {
        "exp_name": exp_name,
        "seq_model": args.seq_model,
        "num_layers": args.num_layers,
        "image_pretrained": bool(args.image_pretrained),
        "modality_dropout": args.modality_dropout,
        "params_m": round(n_params / 1e6, 3),
        "test_f1": test_f1,
        "test_auc": test_auc,
        "best_val_f1": best_val_f1,
        "epochs_ran": len(history),
    }
    pd.DataFrame([result]).to_csv(out_dir / "test_result.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    print(f"结果已保存到 {out_dir}/")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_model", type=str, default="transformer",
                        choices=["rnn", "lstm", "gru", "transformer"])
    parser.add_argument("--num_layers", type=int, default=3, choices=[3, 7, 9])
    parser.add_argument("--image_pretrained", type=int, default=1, choices=[0, 1])
    parser.add_argument("--modality_dropout", type=float, default=0.3,
                         help="训练时随机把数值向量整个置空的概率，让模型学会没有数值输入时只靠图像分支预测")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    train(args)
