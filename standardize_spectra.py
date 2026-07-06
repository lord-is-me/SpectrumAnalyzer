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


VALID_XUNITS = {'1/CM', 'CM-1'}   # 两种写法等价，'CM-1'是少数文件的写法差异
VALID_YUNITS = {'ABSORBANCE', 'TRANSMITTANCE'}


def jdx_axis_valid(meta: dict) -> bool:
    """XUNITS/YUNITS 是否是我们能处理的物理量，供 process_nist 和账目脚本共用判断逻辑。"""
    xunits = meta.get('XUNITS', '').upper()
    yunits = meta.get('YUNITS', '').upper()
    if xunits not in VALID_XUNITS and xunits != 'MICROMETERS':
        return False
    return yunits in VALID_YUNITS


def convert_x_to_wavenumber(meta: dict, x: np.ndarray) -> np.ndarray | None:
    """按 XUNITS 把 x 轴统一转成 cm-1。单位不认识返回 None（调用方应已用 jdx_axis_valid 提前过滤）。"""
    xunits = meta.get('XUNITS', '').upper()
    if xunits == 'MICROMETERS':
        return 10000.0 / x
    if xunits in VALID_XUNITS:
        return x
    return None


def compute_pct_t_and_ylim(meta: dict, y: np.ndarray):
    """把 y 轴统一转成 %T，并算出这条曲线该用的显示范围 ylim（含气相拉伸逻辑）。
    返回 (pct_t, ylim)，单位/物理量不认识时返回 (None, None)。

    这个函数是画图（process_nist）和生成数值向量（build_dataset.py）共用的唯一实现，
    保证两边的"气相拉伸"变换严格一致——图像和数值向量的 y 轴变换必须同源，
    否则"图像重要区域 vs 数值重要区域对比"这个分析就没有意义了。
    """
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
        return None, None   # Reflectance / absorption index / dispersion index 等，物理量不同，跳过
    return pct_t, ylim


# SDBS 原图的X轴不是纯线性的：实测 type1/type2 多个样本，2000 cm-1 处有个折点，
# 折点以下（指纹区，2000-400）的像素密度是折点以上（2000-4000）的整整2倍
# （SDBS的historical惯例，指纹区峰密集，故意画得更宽）。NIST重绘图原来用纯线性轴，
# 和SDBS图在同一个"4000->400, 686x322"画布里其实是两套不同的像素<->波数换算关系。
# 这里让NIST重绘图也采用同一套分段规则，两种来源的图从此共享一套换算公式；
# 这是无损操作——NIST是原始数值点，改的只是画在哪个像素，不是重采样丢信息。
AXIS_BREAK_WN = 2000.0


def wavenumber_to_pixel(wn, width=TARGET_SIZE[0]):
    """波数 -> 像素列（分段线性，4000cm-1在0，400cm-1在width，折点在AXIS_BREAK_WN）。"""
    scale_high = ((4000.0 - AXIS_BREAK_WN) + 2.0 * (AXIS_BREAK_WN - 400.0)) / width  # cm-1/px，折点以上
    scale_low = scale_high / 2.0                                                     # cm-1/px，折点以下，密度翻倍
    break_px = (4000.0 - AXIS_BREAK_WN) / scale_high
    wn = np.asarray(wn, dtype=float)
    return np.where(
        wn >= AXIS_BREAK_WN,
        (4000.0 - wn) / scale_high,
        break_px + (AXIS_BREAK_WN - wn) / scale_low,
    )


def pixel_to_wavenumber(px, width=TARGET_SIZE[0]):
    """wavenumber_to_pixel 的反函数，供 Grad-CAM 等后续分析换算像素位置对应的波数。"""
    scale_high = ((4000.0 - AXIS_BREAK_WN) + 2.0 * (AXIS_BREAK_WN - 400.0)) / width
    scale_low = scale_high / 2.0
    break_px = (4000.0 - AXIS_BREAK_WN) / scale_high
    px = np.asarray(px, dtype=float)
    return np.where(
        px <= break_px,
        4000.0 - px * scale_high,
        AXIS_BREAK_WN - (px - break_px) * scale_low,
    )


def process_nist(path: Path) -> Image.Image | None:
    meta, x, y = parse_jdx(path)

    x = convert_x_to_wavenumber(meta, x)
    if x is None:
        return None

    pct_t, ylim = compute_pct_t_and_ylim(meta, y)
    if pct_t is None:
        return None

    px = wavenumber_to_pixel(x)

    # 坐标轴框如果紧贴画布边缘（add_axes([0,0,1,1])），右/下边框线会被画布裁切掉一半线宽
    # 导致基本看不见，所以渲染时四周多留 PAD 像素余量，存图前再裁掉这圈余量。
    dpi = 100
    pad = 4
    render_w, render_h = TARGET_SIZE[0] + 2 * pad, TARGET_SIZE[1] + 2 * pad
    fig = plt.figure(figsize=(render_w / dpi, render_h / dpi), dpi=dpi)
    ax = fig.add_axes([pad / render_w, pad / render_h, TARGET_SIZE[0] / render_w, TARGET_SIZE[1] / render_h])
    ax.plot(px, pct_t, color='black', linewidth=0.7)
    ax.set_xlim(0, TARGET_SIZE[0])
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
