"""
方法3的模型定义：图像分支（CNN）+ 数值序列分支（RNN/LSTM/GRU/Transformer）+ 晚融合。
结构对应 docs/network_architecture.md 的 Fusion 设计。
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """ResNet50 去掉分类头，全局平均池化后接一个投影层。"""
    def __init__(self, pretrained: bool = True, out_dim: int = 256):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # -> [B, 2048, 1, 1]
        self.proj = nn.Linear(2048, out_dim)

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.proj(feat)


class SequenceEncoder(nn.Module):
    """输入 [B, N, 2]（第0列%T值，第1列覆盖掩码：1=真实测量，0=填充/无数据）。
    四种架构共用同一套 masked mean pooling，保证深度/架构之间对比公平。"""
    def __init__(self, arch: str = "transformer", num_layers: int = 3,
                 d_model: int = 128, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.arch = arch
        self.input_proj = nn.Linear(2, d_model)

        rnn_dropout = dropout if num_layers > 1 else 0.0
        if arch == "rnn":
            self.encoder = nn.RNN(d_model, d_model, num_layers=num_layers, batch_first=True,
                                   dropout=rnn_dropout, nonlinearity="tanh")
        elif arch == "lstm":
            self.encoder = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True,
                                    dropout=rnn_dropout)
        elif arch == "gru":
            self.encoder = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True,
                                   dropout=rnn_dropout)
        elif arch == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown seq arch: {arch}")

        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(torch.stack([values, mask], dim=-1))  # [B, N, d_model]

        if self.arch == "transformer":
            key_padding_mask = (mask == 0)  # True = 屏蔽该位置
            # 极端情况：整条序列全被屏蔽（val/test模态dropout场景），Transformer的
            # src_key_padding_mask 不允许某一行全True（会导致NaN），退化为不加mask。
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked] = False
            out = self.encoder(h, src_key_padding_mask=key_padding_mask)
        else:
            out, _ = self.encoder(h)

        m = mask.unsqueeze(-1)  # [B, N, 1]
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = (out * m).sum(dim=1) / denom
        all_masked = (mask.sum(dim=1) == 0)
        if all_masked.any():
            pooled_fallback = out.mean(dim=1)
            pooled = torch.where(all_masked.unsqueeze(-1), pooled_fallback, pooled)
        return self.out_proj(pooled)


class FusionModel(nn.Module):
    def __init__(self, num_labels: int, seq_arch: str = "transformer", seq_layers: int = 3,
                 image_pretrained: bool = True, emb_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=image_pretrained, out_dim=emb_dim)
        self.seq_encoder = SequenceEncoder(arch=seq_arch, num_layers=seq_layers,
                                            out_dim=emb_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, num_labels),
        )

    def forward(self, image, vec_values, vec_mask):
        img_emb = self.image_encoder(image)
        seq_emb = self.seq_encoder(vec_values, vec_mask)
        fused = torch.cat([img_emb, seq_emb], dim=1)
        return self.head(fused)
