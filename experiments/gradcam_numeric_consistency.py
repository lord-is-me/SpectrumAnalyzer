"""
Phase 4 (5.2)：图像重要区域 vs 数值重要区域是否一致。

只对方法3的融合模型有意义（exp 形如 fusion_{seq_model}_{num_layers}layer），而且只能用
train 集——val/test 里的数值向量是"全掩码=0"的占位输入（见 train_fusion.py 顶部注释），
根本没有真实数值数据可分析；train 集里每个 NIST 来源样本才同时有真实图像+真实数值向量。

对每个正例样本，同时算两条重要性曲线（都在同一条 400-4000 cm⁻¹ / 360-bin 网格上）：
  - 图像侧：Grad-CAM（这次数值分支喂的是该样本真实的向量，不是占位输入，反映融合模型
            训练时实际会遇到的真实前向传播），沿波数轴重采样到360-bin，逻辑和
            gradcam_aggregate.py 完全一致（resample_cam_to_wavenumber）。
  - 数值侧：Transformer 用 self-attention 权重（跨层跨query取平均，近似"这个位置平均被
            关注多少"）；RNN/LSTM/GRU 用积分梯度（Integrated Gradients，手写实现，baseline
            取全序列填充值100.0——和build_dataset.py里"无覆盖bin"的填充值同源，物理含义是
            "一条没有任何吸收信息的平谱"）。

按标签聚合时只累加 mask==1（真实测量覆盖）的bin，被掩码标记为缺失的bin不计入平均，也不
参与相关系数计算——避免"模型正确地避开了假填充区"被误读成一个化学发现（这一点在
docs/nist_fusion_experiment_plan.md 5.2节里强调过）。

用法:
    python gradcam_numeric_consistency.py --exp fusion_transformer_3layer --min_pos 10
    python gradcam_numeric_consistency.py --exp fusion_lstm_3layer --min_pos 10 --ig_steps 20

结果保存在 results/{exp}/gradcam_numeric_consistency/
    train_label_profiles.csv          行=360个波数bin，列=每个达标标签的 image/numeric/coverage 三列
    train_label_correlation.csv       行=每个达标标签，列=n_pos/n_valid_bins/pearson_r/pearson_p/spearman_r/spearman_p
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import (  # noqa: E402
    get_transforms, LABEL_COLS, NUM_LABELS, NIST_SPLIT_ROOT, CHECKPOINT_ROOT, RESULT_ROOT,
)
from grad_cam import GradCAM, disable_inplace_relu  # noqa: E402
from fusion_model import FusionModel  # noqa: E402
from gradcam_aggregate import (  # noqa: E402
    N_BINS, WN_LOW, WN_HIGH, parse_fusion_exp, resample_cam_to_wavenumber,
)

BASELINE_VALUE = 100.0  # 和 build_dataset.py::build_vector 里"无覆盖bin"的填充值同源


class FusionGradCAMWrapper(nn.Module):
    """Grad-CAM 要求 model(x) 只吃图像一个参数；这里把当前样本真实的数值向量/掩码存成
    可更新的属性，数值分支用真实输入（不是5.1节 ImageOnlyWrapper 那种占位输入），这样
    Grad-CAM 反映的是融合模型训练/验证时实际会遇到的真实前向传播。"""
    def __init__(self, fusion_model: FusionModel):
        super().__init__()
        self.fusion_model = fusion_model
        self._values = None
        self._mask = None

    def set_vector(self, values: torch.Tensor, mask: torch.Tensor):
        self._values = values
        self._mask = mask

    def forward(self, image):
        return self.fusion_model(image, self._values, self._mask)


def _replay_transformer_layer(layer: nn.TransformerEncoderLayer, x: torch.Tensor,
                               key_padding_mask: torch.Tensor):
    """手动重放一层 TransformerEncoderLayer 的计算，显式用 need_weights=True 调用
    self_attn 拿到真实注意力权重。不能靠给 self_attn.forward 打补丁再调用
    layer(...)/encoder(...)：eval()+no_grad() 下 TransformerEncoderLayer 会整层走融合的
    fast path（"BetterTransformer"，torch._transformer_encoder_layer_fwd），根本不会调用
    子模块的 forward，补丁挂了也不会触发（实测：这样跑出来的注意力矩阵全是0）。这里绕开
    layer.forward()/TransformerEncoder.forward()，直接调子模块，保证真的算了attention。"""
    attn = layer.self_attn
    if layer.norm_first:
        normed = layer.norm1(x)
        attn_out, w = attn(normed, normed, normed, key_padding_mask=key_padding_mask,
                            need_weights=True, average_attn_weights=True)
        x = x + layer.dropout1(attn_out)
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
        x = x + layer.dropout2(ff)
    else:
        attn_out, w = attn(x, x, x, key_padding_mask=key_padding_mask,
                            need_weights=True, average_attn_weights=True)
        x = layer.norm1(x + layer.dropout1(attn_out))
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm2(x + layer.dropout2(ff))
    return x, w


def transformer_attention_importance(seq_encoder: nn.Module, values: torch.Tensor,
                                      mask: torch.Tensor) -> np.ndarray:
    """重放 SequenceEncoder(transformer分支) 的前向计算，逐层拿真实注意力权重，
    跨层、跨query维取平均，得到每个位置(=key，和360个波数bin一一对应)平均被关注的强度。"""
    with torch.no_grad():
        x = seq_encoder.input_proj(torch.stack([values, mask], dim=-1))
        key_padding_mask = (mask == 0)
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False

        layer_weights = []
        for layer in seq_encoder.encoder.layers:
            x, w = _replay_transformer_layer(layer, x, key_padding_mask)
            layer_weights.append(w)

        stacked = torch.stack(layer_weights, dim=0)  # [L, 1, N, N]
        avg = stacked.mean(dim=0).squeeze(0)  # [N, N]
        return avg.mean(dim=0).cpu().numpy()  # 对query维平均 -> [N]


def integrated_gradients_numeric(fusion_model: FusionModel, img_emb: torch.Tensor,
                                  values: torch.Tensor, mask: torch.Tensor, label_idx: int,
                                  steps: int) -> np.ndarray:
    """手写积分梯度：只沿数值向量(values)从baseline(全序列填充值100.0)插值到真实输入，
    mask 全程固定。图像分支的嵌入 img_emb 在外面用真实图像只算一次再传进来——它跟
    interp 无关，插值这些步骤里重复跑一遍ResNet50纯属浪费。返回长度360的有符号归因，
    外面再取绝对值当"重要性"。"""
    baseline = torch.full_like(values, BASELINE_VALUE)
    diff = values - baseline
    total_grad = torch.zeros_like(values)
    for i in range(1, steps + 1):
        alpha = i / steps
        interp = (baseline + alpha * diff).detach().requires_grad_(True)
        seq_emb = fusion_model.seq_encoder(interp, mask)
        fused = torch.cat([img_emb, seq_emb], dim=1)
        logits = fusion_model.head(fused)
        score = logits[:, label_idx].sum()
        grad, = torch.autograd.grad(score, interp)
        total_grad += grad
    avg_grad = total_grad / steps
    attributions = (diff * avg_grad).detach().squeeze(0).cpu().numpy()
    return attributions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                        help="fusion_{rnn,lstm,gru,transformer}_{layers}layer，对应 checkpoints/{exp}/best.pth")
    parser.add_argument("--min_pos", type=int, default=10,
                        help="标签在train集正例数至少要有这么多才纳入分析")
    parser.add_argument("--min_bin_coverage", type=int, default=5,
                        help="一个波数bin至少要有这么多正例样本在该bin有真实测量覆盖(mask==1)，才纳入该标签的曲线/相关系数")
    parser.add_argument("--ig_steps", type=int, default=20,
                        help="RNN/LSTM/GRU 积分梯度的插值步数，越大越准但越慢")
    parser.add_argument("--max_samples_per_label", type=int, default=0,
                        help="每个标签最多分析多少个正例样本，0=不限制（正例多的标签在RNN/LSTM/GRU上跑IG很慢，可用这个封顶）")
    args = parser.parse_args()

    fusion_info = parse_fusion_exp(args.exp)
    if fusion_info is None:
        raise ValueError(f"--exp 必须形如 fusion_{{rnn,lstm,gru,transformer}}_{{layers}}layer，收到: {args.exp}")
    seq_arch, num_layers = fusion_info

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CHECKPOINT_ROOT / args.exp / "best.pth"

    print(f"模型类型: 方法3融合模型 (seq_model={seq_arch}, num_layers={num_layers})")
    fusion_model = FusionModel(NUM_LABELS, seq_arch=seq_arch, seq_layers=num_layers, image_pretrained=False)
    fusion_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    fusion_model = fusion_model.to(device).eval()

    wrapper = FusionGradCAMWrapper(fusion_model).to(device).eval()
    disable_inplace_relu(wrapper)

    if seq_arch in ("rnn", "lstm", "gru"):
        # cuDNN的RNN/LSTM/GRU反向传播要求模块处于training模式，eval模式下backward会直接报错。
        # 这里只把序列分支这一个模块设回train()；同时把它内部的层间dropout清零，因为这次数值
        # 分支喂的是真实向量（不是5.1节那种无关的占位输入），dropout噪声会直接污染我们要分析
        # 的数值侧重要性——train()只是为了满足cuDNN的backward前提，行为上仍要等价于eval。
        fusion_model.seq_encoder.encoder.train()
        fusion_model.seq_encoder.encoder.dropout = 0.0

    # ImageEncoder.features = resnet50.children()[:-1]，只去掉了fc，avgpool还在最后一位，
    # 所以 layer4 是倒数第二个（features[-1]=avgpool，features[-2]=layer4），取它最后一个Bottleneck
    target_layer = wrapper.fusion_model.image_encoder.features[-2][-1]
    cam_engine = GradCAM(wrapper, target_layer)

    print("数据集来源: NistSdbsSplit/train（唯一同时有真实图像+真实数值向量的split）")
    split_dir = NIST_SPLIT_ROOT / "train"
    df = pd.read_csv(split_dir / "labels.csv")
    img_size = 224
    transform = get_transforms(img_size, is_train=False)

    bin_edges = np.linspace(WN_LOW, WN_HIGH, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    label_counts = df[LABEL_COLS].sum(axis=0)
    valid_labels = [l for l in LABEL_COLS if label_counts[l] >= args.min_pos]
    print(f"共 {len(LABEL_COLS)} 个标签，{len(valid_labels)} 个在 train 集正例数 >= {args.min_pos}，只分析这些")

    profile_cols = {}
    corr_rows = []
    for label in valid_labels:
        pos_rows = df[df[label] == 1]
        if args.max_samples_per_label > 0 and len(pos_rows) > args.max_samples_per_label:
            pos_rows = pos_rows.sample(n=args.max_samples_per_label, random_state=42)
        label_idx = LABEL_COLS.index(label)

        img_sum = np.zeros(N_BINS)
        num_sum = np.zeros(N_BINS)
        coverage = np.zeros(N_BINS)

        for _, row in pos_rows.iterrows():
            img_path = split_dir / row["image_path"]
            orig_img = Image.open(img_path).convert("RGB")
            x = transform(orig_img).unsqueeze(0).to(device)

            vec = np.load(split_dir / row["vector_path"])  # [N_BINS, 2]
            values = torch.from_numpy(vec[:, 0]).unsqueeze(0).to(device)
            mask = torch.from_numpy(vec[:, 1]).unsqueeze(0).to(device)
            mask_np = mask.squeeze(0).cpu().numpy()
            if mask_np.sum() == 0:
                continue  # 这条样本在400-4000范围内一个真实测量点都没有，跳过

            wrapper.set_vector(values, mask)
            cam, _prob = cam_engine(x, label_idx)
            img_profile = resample_cam_to_wavenumber(cam, bin_centers)

            if seq_arch == "transformer":
                num_profile = transformer_attention_importance(fusion_model.seq_encoder, values, mask)
            else:
                with torch.no_grad():
                    img_emb = fusion_model.image_encoder(x)
                num_profile = np.abs(integrated_gradients_numeric(
                    fusion_model, img_emb, values, mask, label_idx, steps=args.ig_steps))

            img_sum += img_profile * mask_np
            num_sum += num_profile * mask_np
            coverage += mask_np

        valid_bins = coverage >= args.min_bin_coverage
        n_valid_bins = int(valid_bins.sum())
        img_profile_avg = np.divide(img_sum, coverage, out=np.full(N_BINS, np.nan), where=coverage > 0)
        num_profile_avg = np.divide(num_sum, coverage, out=np.full(N_BINS, np.nan), where=coverage > 0)

        profile_cols[f"{label}__image"] = img_profile_avg
        profile_cols[f"{label}__numeric"] = num_profile_avg
        profile_cols[f"{label}__coverage"] = coverage

        if n_valid_bins >= 5:
            a, b = img_profile_avg[valid_bins], num_profile_avg[valid_bins]
            pear_r, pear_p = pearsonr(a, b)
            spear_r, spear_p = spearmanr(a, b)
        else:
            pear_r = pear_p = spear_r = spear_p = float("nan")

        corr_rows.append({
            "label": label, "n_pos": len(pos_rows), "n_valid_bins": n_valid_bins,
            "pearson_r": pear_r, "pearson_p": pear_p,
            "spearman_r": spear_r, "spearman_p": spear_p,
        })
        print(f"  {label:15s}  n_pos={len(pos_rows):3d}  n_valid_bins={n_valid_bins:3d}  "
              f"pearson_r={pear_r:.3f}  spearman_r={spear_r:.3f}")

    out_dir = RESULT_ROOT / args.exp / "gradcam_numeric_consistency"
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles_out = pd.DataFrame(profile_cols, index=bin_centers.astype(int))
    profiles_out.index.name = "wavenumber_cm-1"
    profiles_out.to_csv(out_dir / "train_label_profiles.csv")

    corr_out = pd.DataFrame(corr_rows).sort_values("pearson_r", ascending=False)
    corr_out.to_csv(out_dir / "train_label_correlation.csv", index=False)

    print(f"\n已保存到 {out_dir}/")


if __name__ == "__main__":
    main()
