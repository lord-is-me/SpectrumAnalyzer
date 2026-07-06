"""Phase 1：根据 manifest.csv 生成 NIST/SDBS 融合实验的正式数据集。

- train : 全部用 data/NIST/{file_id}.jdx 原始数值重新画图（images/{file_id}.png），
          同时生成同源的数值向量（vectors/{file_id}.npy，形状[N,2] = %T + 覆盖掩码）。
          其中 is_overlap 的分子额外拷贝一份现成的 SDBS 图作为第二视角增强样本
          （images_aug/{file_id}.png），标签不变，数值向量仍只有 NIST 那一份。
- val/test : 直接从 data/StandardizedSpectra/images/{file_id}.png 拷贝，不做任何处理。

用法: python build_dataset.py [--bins 360]
输出: data/NistSdbsSplit/{train,val,test}/
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from standardize_spectra import parse_jdx, convert_x_to_wavenumber, compute_pct_t_and_ylim, process_nist
from build_nist_sdbs_manifest import load_labels

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
SPLIT_DIR = DATA / 'NistSdbsSplit'
STD_IMG_DIR = DATA / 'StandardizedSpectra' / 'images'

WN_LOW, WN_HIGH = 400.0, 4000.0


def build_vector(jdx_path: Path, n_bins: int):
    """返回形状 [n_bins, 2] 的数组：第0列 %T（同源气相拉伸过），第1列覆盖掩码(1=真实测量, 0=填充)。"""
    meta, x, y = parse_jdx(jdx_path)
    x = convert_x_to_wavenumber(meta, x)
    pct_t, ylim = compute_pct_t_and_ylim(meta, y)
    if x is None or pct_t is None:
        raise ValueError(f'{jdx_path.name}: 单位不合法，不应该进入 train（manifest 应已过滤）')

    # 和 process_nist 画图时的 ylim 缩放严格同源：把可视范围 [ylim[0], 100] 重新映射到 [0, 100]，
    # 这样数值向量的数值大小才对应曲线在图像里的实际视觉高度。
    stretched = (pct_t - ylim[0]) / (ylim[1] - ylim[0]) * 100.0

    edges = np.linspace(WN_LOW, WN_HIGH, n_bins + 1)
    bin_width = (WN_HIGH - WN_LOW) / n_bins

    in_range = (x >= WN_LOW) & (x <= WN_HIGH)
    xr, sr = x[in_range], stretched[in_range]

    values = np.full(n_bins, np.inf)
    if len(xr) > 0:
        bin_idx = np.clip(((xr - WN_LOW) / bin_width).astype(int), 0, n_bins - 1)
        np.minimum.at(values, bin_idx, sr)

    mask = np.isfinite(values).astype(np.float32)
    values = np.where(np.isfinite(values), values, 100.0).astype(np.float32)
    return np.stack([values, mask], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=360, help='400-4000cm-1 切成多少个bin（默认360，即10cm-1/bin）')
    args = parser.parse_args()

    manifest = pd.read_csv(SPLIT_DIR / 'manifest.csv')
    labels_df, label_cols = load_labels()
    labels_df = labels_df.set_index('file_id')

    for split in ('train', 'val', 'test'):
        (SPLIT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / 'train' / 'images_aug').mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / 'train' / 'vectors').mkdir(parents=True, exist_ok=True)

    train_rows, val_rows, test_rows = [], [], []
    failed = []

    for i, row in manifest.iterrows():
        file_id = int(row['file_id'])
        split = row['split']
        if split == 'excluded':
            continue

        label_vals = labels_df.loc[file_id, label_cols] if file_id in labels_df.index else None
        if label_vals is None:
            failed.append((file_id, 'no label row'))
            continue

        if split == 'train':
            jdx_path = DATA / 'NIST' / f'{file_id}.jdx'
            try:
                im = process_nist(jdx_path)
                vec = build_vector(jdx_path, args.bins)
            except Exception as e:
                failed.append((file_id, str(e)))
                continue
            if im is None or vec is None:
                failed.append((file_id, 'process_nist/build_vector returned None'))
                continue

            im.save(SPLIT_DIR / 'train' / 'images' / f'{file_id}.png')
            np.save(SPLIT_DIR / 'train' / 'vectors' / f'{file_id}.npy', vec)
            train_rows.append({
                'sample_id': f'{file_id}_nist', 'file_id': file_id, 'SMILES': row['SMILES'],
                'source': 'nist_original',
                'image_path': f'images/{file_id}.png',
                'vector_path': f'vectors/{file_id}.npy',
                **label_vals.to_dict(),
            })

            if row['is_overlap']:
                src = STD_IMG_DIR / f'{file_id}.png'
                if src.exists():
                    shutil.copyfile(src, SPLIT_DIR / 'train' / 'images_aug' / f'{file_id}.png')
                    train_rows.append({
                        'sample_id': f'{file_id}_aug', 'file_id': file_id, 'SMILES': row['SMILES'],
                        'source': 'sdbs_paired_aug',
                        'image_path': f'images_aug/{file_id}.png',
                        'vector_path': f'vectors/{file_id}.npy',   # 数值向量仍只有NIST这一份
                        **label_vals.to_dict(),
                    })
                else:
                    failed.append((file_id, 'is_overlap but StandardizedSpectra image missing'))

        else:  # val / test
            src = STD_IMG_DIR / f'{file_id}.png'
            if not src.exists():
                failed.append((file_id, f'{split}: StandardizedSpectra image missing'))
                continue
            shutil.copyfile(src, SPLIT_DIR / split / 'images' / f'{file_id}.png')
            record = {
                'sample_id': f'{file_id}_sdbs', 'file_id': file_id, 'SMILES': row['SMILES'],
                'source': 'sdbs',
                'image_path': f'images/{file_id}.png',
                **label_vals.to_dict(),
            }
            (val_rows if split == 'val' else test_rows).append(record)

        if (i + 1) % 500 == 0:
            print(f'progress: {i + 1}/{len(manifest)}')

    pd.DataFrame(train_rows).to_csv(SPLIT_DIR / 'train' / 'labels.csv', index=False)
    pd.DataFrame(val_rows).to_csv(SPLIT_DIR / 'val' / 'labels.csv', index=False)
    pd.DataFrame(test_rows).to_csv(SPLIT_DIR / 'test' / 'labels.csv', index=False)

    print('\n===== 汇总 =====')
    print(f'train 样本数（含增强）: {len(train_rows)}  '
          f"(nist_original={sum(1 for r in train_rows if r['source']=='nist_original')}, "
          f"sdbs_paired_aug={sum(1 for r in train_rows if r['source']=='sdbs_paired_aug')})")
    print(f'val 样本数: {len(val_rows)}')
    print(f'test 样本数: {len(test_rows)}')
    print(f'失败/跳过: {len(failed)}')
    if failed:
        print('前20条失败原因:', failed[:20])
    print(f'\n已保存到 {SPLIT_DIR}/')


if __name__ == '__main__':
    main()
