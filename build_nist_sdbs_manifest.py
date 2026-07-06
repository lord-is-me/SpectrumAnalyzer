"""Phase 0：生成 NIST/SDBS 融合实验的总账表，不碰图片/数值，先把账算清楚。

对 data/StandardizedSpectra/smiles_spectrum_nist.csv 的每一行（=每个 file_id），判定：
  - train  : NIST==1 且 jdx 存在、XUNITS/YUNITS 可解析（process_nist 不会返回 None）
  - val/test: 不满足 train 条件，但有 SDBS 图（spectrum_type in {1,2}），
              在这个候选池里按固定随机种子做 val/test 二切分
  - excluded: 两者都不满足，没有任何可用图

train 集合里同时有 SDBS 图的分子（is_overlap=True）会在 Phase 1 额外拿到一张 SDBS 版本的
增强图（详见 docs/nist_fusion_experiment_plan.md 2.2 / 3.2 节）。

用法: python build_nist_sdbs_manifest.py
输出: data/NistSdbsSplit/manifest.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from standardize_spectra import parse_jdx, jdx_axis_valid

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
OUT_DIR = DATA / 'NistSdbsSplit'
SEED = 42
VAL_FRACTION = 0.5   # 验证/测试候选池里，验证集占比（其余为测试集）


def load_labels():
    """all_cleaned.csv 与 smiles_spectrum_nist.csv 按行号对齐（同一套 file_id = 行号+2），
    这里只取 120 个气味标签列，用于 val/test 候选池的分层切分。"""
    df = pd.read_csv(DATA / 'StandardizedSpectra' / 'all_cleaned.csv')
    label_cols = [c for c in df.columns if c not in ('SMILES', 'spectrum_type')]
    label_df = df[label_cols].copy()
    label_df['file_id'] = label_df.index + 2
    return label_df, label_cols


def stratified_val_test_split(candidate_ids, labels_df, label_cols):
    """优先做多标签分层切分（需要 iterstrat），缺依赖时退化为普通随机切分。"""
    sub = labels_df[labels_df['file_id'].isin(candidate_ids)].reset_index(drop=True)
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        y = sub[label_cols].values
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=1 - VAL_FRACTION, random_state=SEED)
        val_idx, test_idx = next(splitter.split(sub[['file_id']].values, y))
        val_ids = set(sub.loc[val_idx, 'file_id'])
        test_ids = set(sub.loc[test_idx, 'file_id'])
        print(f'[val/test 切分] 使用多标签分层 (iterstrat)，val={len(val_ids)} test={len(test_ids)}')
    except ImportError:
        rng = np.random.RandomState(SEED)
        shuffled = sub['file_id'].sample(frac=1.0, random_state=rng).tolist()
        n_val = round(len(shuffled) * VAL_FRACTION)
        val_ids = set(shuffled[:n_val])
        test_ids = set(shuffled[n_val:])
        print(f'[val/test 切分] 未安装 iterstrat，退化为普通随机切分（seed={SEED}），'
              f'val={len(val_ids)} test={len(test_ids)}')
    return val_ids, test_ids


def main():
    df = pd.read_csv(DATA / 'StandardizedSpectra' / 'smiles_spectrum_nist.csv')
    labels_df, label_cols = load_labels()

    records = []
    for i, row in df.iterrows():
        file_id = i + 2
        spectrum_type = int(row['spectrum_type'])
        nist_flag = int(row['NIST'])
        has_sdbs_image = spectrum_type in (1, 2)

        jdx_path = DATA / 'NIST' / f'{file_id}.jdx'
        jdx_exists = nist_flag == 1 and jdx_path.exists()
        jdx_yunits = jdx_xunits = ''
        jdx_valid = False
        if jdx_exists:
            try:
                meta, _, _ = parse_jdx(jdx_path)
                jdx_yunits = meta.get('YUNITS', '')
                jdx_xunits = meta.get('XUNITS', '')
                jdx_valid = jdx_axis_valid(meta)
            except Exception as e:
                jdx_yunits = f'PARSE_ERROR: {e}'

        records.append({
            'file_id': file_id,
            'SMILES': row['SMILES'],
            'spectrum_type': spectrum_type,
            'NIST': nist_flag,
            'has_sdbs_image': has_sdbs_image,
            'jdx_exists': jdx_exists,
            'jdx_yunits': jdx_yunits,
            'jdx_xunits': jdx_xunits,
            'jdx_valid': jdx_valid,
        })

        if (i + 1) % 2000 == 0:
            print(f'progress: {i + 1}/{len(df)}')

    manifest = pd.DataFrame(records)
    manifest['is_train_candidate'] = manifest['jdx_valid']
    manifest['is_overlap'] = manifest['is_train_candidate'] & manifest['has_sdbs_image']

    valtest_candidates = manifest.loc[
        (~manifest['is_train_candidate']) & manifest['has_sdbs_image'], 'file_id'
    ].tolist()
    val_ids, test_ids = stratified_val_test_split(valtest_candidates, labels_df, label_cols)

    def assign_split(r):
        if r['is_train_candidate']:
            return 'train'
        if r['file_id'] in val_ids:
            return 'val'
        if r['file_id'] in test_ids:
            return 'test'
        return 'excluded'

    manifest['split'] = manifest.apply(assign_split, axis=1)
    manifest = manifest.drop(columns=['is_train_candidate'])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT_DIR / 'manifest.csv', index=False)

    print('\n===== 汇总 =====')
    print(manifest['split'].value_counts().to_string())
    print(f"\n训练集里的重合分子（有效NIST数据 + 也有SDBS图，Phase 1 会额外加一张SDBS增强图）: "
          f"{manifest['is_overlap'].sum()}")
    print(f"完全没有可用图的行数（train/val/test都进不去）: "
          f"{(manifest['split'] == 'excluded').sum()}")
    print(f'\n已保存到 {OUT_DIR / "manifest.csv"}')


if __name__ == '__main__':
    main()
