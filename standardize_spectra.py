"""把 data/SDBS (类型1/2) 和 data/NIST (纯NIST) 三种来源的光谱图统一成同一种标准样式：
黑色坐标轴框 + 黑色曲线，无刻度、无文字、无版权信息，统一尺寸 686x322，
横轴固定 4000->400 cm-1，纵轴固定 0-100 %T。

来源优先级（按 csv 每一行）：
  spectrum_type == 2          -> data/SDBS/{id}.png （已是干净样式，裁剪+去刻度）
  spectrum_type == 1          -> data/SDBS/{id}.png （裁剪掉信息框/峰值表/结构图）
  spectrum_type == 0, NIST==1 -> data/NIST/{id}.jdx （解析原始数值重绘）
  spectrum_type == 0, NIST==0 -> 跳过，没有图可用

文件名与 csv 行号的对应关系（已用样例验证）：把表头算第1行，
csv 第 N 行 <-> 文件名 {N}.png / {N}.jdx，等价于 pandas 0-based 的 iloc[N-2]。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
OUT_DIR = DATA / 'StandardizedSpectra'
OUT_IMG_DIR = OUT_DIR / 'images'

TARGET_SIZE = (686, 322)          # 最终统一尺寸，基于类型1原生框尺寸
TYPE1_CANVAS_SIZE = (715, 553)    # 类型1版式的画布尺寸，用于识别个别标签错误的图
BOX1 = (29, 96, 715, 418)         # 类型1坐标轴框（左,上,右,下），裁剪后正好是 TARGET_SIZE

# 类型2坐标轴框内侧的刻度线位置，按box尺寸的比例表示（不同子版式的框大小不完全一样，
# 已验证按比例算出来的位置在两种版式上都对得上），裁剪后用小补丁去掉，避免整条带误删曲线
_REF_BOX2 = (80, 22, 777, 389)
_REF_LEFT_TICK_ROWS = [30, 31, 59, 95, 132, 168, 205, 242, 278, 315, 351]
_REF_BOTTOM_TICK_COLS = [107, 134, 160, 187, 214, 241, 267, 294, 321, 348, 374,
                          401, 428, 455, 482, 508, 535, 562, 589, 615, 642, 669,
                          696, 722, 749]
LEFT_TICK_FRACS = [(r - _REF_BOX2[1]) / (_REF_BOX2[3] - _REF_BOX2[1]) for r in _REF_LEFT_TICK_ROWS]
BOTTOM_TICK_FRACS = [(c - _REF_BOX2[0]) / (_REF_BOX2[2] - _REF_BOX2[0]) for c in _REF_BOTTOM_TICK_COLS]


def detect_box(im: Image.Image, row_thresh=0.5, col_thresh=0.5):
    """检测坐标轴框的左上右下边界。类型2有至少两种子版式，框的具体像素位置不固定，
    必须每张图自己检测，不能写死坐标（67.png 这种版式 B 用版式 A 的坐标裁就会整个偏掉）。"""
    arr = np.array(im.convert('L'))
    dark = arr < 200
    row_frac = dark.mean(axis=1)
    col_frac = dark.mean(axis=0)
    hor = [i for i, f in enumerate(row_frac) if f > row_thresh]
    ver = [i for i, f in enumerate(col_frac) if f > col_thresh]
    return ver[0], hor[0], ver[-1] + 1, hor[-1] + 1   # left, top, right(exclusive), bottom(exclusive)


def process_type1(path: Path) -> Image.Image:
    im = Image.open(path).convert('L').convert('RGB')
    return im.crop(BOX1)


def process_type2(path: Path) -> Image.Image:
    im = Image.open(path).convert('L').convert('RGB')
    im.info.pop('transparency', None)
    box = detect_box(im)
    left, top, right, bottom = box
    crop = im.crop(box)
    draw = ImageDraw.Draw(crop)
    h, w = bottom - top, right - left
    for frac in LEFT_TICK_FRACS:
        r = round(frac * h)
        draw.rectangle([1, r - 1, 9, r + 1], fill='white')
    for frac in BOTTOM_TICK_FRACS:
        c = round(frac * w)
        draw.rectangle([c - 1, h - 9, c + 1, h - 2], fill='white')
    return crop.resize(TARGET_SIZE, Image.LANCZOS)


def parse_jdx(path: Path):
    text = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    meta = {}
    data_start = None
    for i, line in enumerate(text):
        if line.startswith('##XYDATA'):
            data_start = i + 1
            break
        if line.startswith('##') and '=' in line:
            k, v = line[2:].split('=', 1)
            meta[k.strip()] = v.strip()
    if data_start is None:
        raise ValueError('no ##XYDATA section')

    xfactor = float(meta.get('XFACTOR', 1.0))
    yfactor = float(meta.get('YFACTOR', 1.0))
    firstx = float(meta['FIRSTX'])
    if 'DELTAX' in meta:
        deltax = float(meta['DELTAX'])
    else:
        lastx = float(meta['LASTX'])
        npoints = int(meta['NPOINTS'])
        deltax = (lastx - firstx) / (npoints - 1)

    xs, ys = [], []
    for line in text[data_start:]:
        line = line.strip()
        if not line or line.startswith('##') or not re.match(r'^-?\d', line):
            continue
        toks = line.split()
        x0 = float(toks[0]) * xfactor
        for j, t in enumerate(toks[1:]):
            xs.append(x0 + j * deltax)
            ys.append(float(t) * yfactor)
    return meta, np.array(xs), np.array(ys)


def process_nist(path: Path) -> Image.Image | None:
    meta, x, y = parse_jdx(path)

    xunits = meta.get('XUNITS', '').upper()
    if xunits == 'MICROMETERS':
        x = 10000.0 / x
    elif xunits != '1/CM':
        return None

    yunits = meta.get('YUNITS', '').upper()
    if yunits == 'ABSORBANCE':
        # 这部分几乎全是气相(GC-IR)数据。只有当整条曲线都没跌破50%时才算"被压扁"，
        # 此时按这张谱图自己的范围自动缩放（类似NIST WebBook的展示方式）；
        # 如果曲线本身已经有深吸收（跌破50%），说明对比度已经足够，维持固定0-100%刻度。
        pct_t = 100.0 * 10 ** (-y)
        y_min = pct_t.min()
        if y_min > 50:
            margin = (100 - y_min) * 0.05
            ylim = (y_min - margin, 100)
        else:
            ylim = (0, 100)
    elif yunits == 'TRANSMITTANCE':
        # 液相/固相/溶液法数据，本来就是直接量到的%T，对比度天然足够，维持固定0-100%刻度。
        pct_t = y * 100.0 if float(meta.get('MAXY', 1)) <= 1.5 else y
        ylim = (0, 100)
    else:
        return None   # Reflectance / absorption index / dispersion index 等，物理量不同，跳过

    # 坐标轴框如果紧贴画布边缘（add_axes([0,0,1,1])），右/下边框线会被画布裁切掉一半线宽
    # 导致基本看不见，所以渲染时四周多留 PAD 像素余量，存图前再裁掉这圈余量。
    dpi = 100
    pad = 4
    render_w, render_h = TARGET_SIZE[0] + 2 * pad, TARGET_SIZE[1] + 2 * pad
    fig = plt.figure(figsize=(render_w / dpi, render_h / dpi), dpi=dpi)
    ax = fig.add_axes([pad / render_w, pad / render_h, TARGET_SIZE[0] / render_w, TARGET_SIZE[1] / render_h])
    ax.plot(x, pct_t, color='black', linewidth=0.7)
    ax.set_xlim(4000, 400)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    im = Image.fromarray(buf).convert('RGB')
    plt.close(fig)
    # 裁剪时两侧各多留1px，确保右/下边框线完整保留，再resize回精确的目标尺寸
    crop = im.crop((pad - 1, pad - 1, pad + TARGET_SIZE[0] + 1, pad + TARGET_SIZE[1] + 1))
    return crop.resize(TARGET_SIZE, Image.LANCZOS)


def main():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT_DIR / 'smiles_spectrum_nist.csv')

    records = []
    failed = []
    for i, row in df.iterrows():
        file_id = i + 2
        image_name = ''
        try:
            if row['spectrum_type'] == 2:
                src = DATA / 'SDBS' / f'{file_id}.png'
                im = process_type2(src) if src.exists() else None
            elif row['spectrum_type'] == 1:
                src = DATA / 'SDBS' / f'{file_id}.png'
                if src.exists():
                    # 极少数(4/1767)标了spectrum_type=1，但图实际是类型2版式（800x441等），
                    # 不是类型1的715x553裁剪信息框版式，按实际画布尺寸分流，而不是死信csv标签
                    im = process_type1(src) if Image.open(src).size == TYPE1_CANVAS_SIZE else process_type2(src)
                else:
                    im = None
            elif row['spectrum_type'] == 0 and row['NIST'] == 1:
                src = DATA / 'NIST' / f'{file_id}.jdx'
                im = process_nist(src) if src.exists() else None
            else:
                im = None
        except Exception as e:
            im = None
            failed.append((file_id, str(e)))

        if im is not None:
            image_name = f'{file_id}.png'
            im.save(OUT_IMG_DIR / image_name)

        records.append({
            'file_id': file_id,
            'SMILES': row['SMILES'],
            'spectrum_type': row['spectrum_type'],
            'NIST': row['NIST'],
            'image': image_name,
        })

        if (i + 1) % 500 == 0:
            print(f'progress: {i + 1}/{len(df)}')

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_DIR / 'labels.csv', index=False)

    n_total = len(out_df)
    n_with_image = (out_df['image'] != '').sum()
    print(f'total rows: {n_total}, with image: {n_with_image}, failed: {len(failed)}')
    if failed:
        print('failed file ids (first 30):', failed[:30])


if __name__ == '__main__':
    main()
