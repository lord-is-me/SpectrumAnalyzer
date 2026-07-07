"""修复 data/StandardizedSpectra 里NIST重绘图仍在用旧版(纯线性)X轴的问题。

背景：process_nist 这一轮改成了和SDBS一致的分段线性X轴（见 standardize_spectra.py 的
wavenumber_to_pixel，折点2000cm-1），但当时只重新生成了新数据集(data/NistSdbsSplit)里的
1659张训练图。StandardizedSpectra 里那些NIST重绘图（spectrum_type==0 且 NIST==1，共544张，
其中533张在NistSdbsSplit里有对应的新版本）还是用旧的纯线性轴生成的，和另外2359张SDBS来源图
坐标系不一致——这正是最初发现"SDBS和NIST有偏移"这个bug时，NistSdbsSplit已经修了、
StandardizedSpectra却漏掉没修的部分。

直接从已经用新公式重新生成过的 data/NistSdbsSplit/train/images/{file_id}.png 拷贝覆盖过去，
不用重新解析jdx画一遍。

用法: python fix_legacy_nist_images.py
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / 'data'


def main():
    df = pd.read_csv(DATA / 'StandardizedSpectra' / 'labels.csv')
    # image 字段失败时是 NaN，不是空字符串，用 notna() 才能正确排除掉那11个从未成功产出图片的行
    nist_rows = df[(df['spectrum_type'] == 0) & (df['NIST'] == 1) & (df['image'].notna())]

    src_dir = DATA / 'NistSdbsSplit' / 'train' / 'images'
    dst_dir = DATA / 'StandardizedSpectra' / 'images'

    replaced, missing = [], []
    for fid in nist_rows['file_id']:
        src = src_dir / f'{fid}.png'
        if src.exists():
            shutil.copyfile(src, dst_dir / f'{fid}.png')
            replaced.append(int(fid))
        else:
            missing.append(int(fid))

    print(f'共 {len(nist_rows)} 张NIST重绘图（StandardizedSpectra里spectrum_type==0且NIST==1）')
    print(f'替换成功: {len(replaced)} 张')
    if missing:
        print(f'找不到对应新版本、维持旧版不动: {len(missing)} 张（多半是jdx本身有解析问题，和这次的轴修复无关）')
        print('file_id:', missing)


if __name__ == '__main__':
    main()
