"""
Phase 5：新旧数据集划分方法对比图。读 results/phase5_split_comparison.csv
（旧随机划分 vs 新NIST/SDBS划分，同backbone同ImageNet预训练初始化），画F1和AUC两张
并排的分组柱状图。

用法: python plot_phase5_comparison.py
输出: results/phase5_split_comparison.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

RESULT_ROOT = Path(__file__).parent / "results"

COLOR_OLD = "#2a78d6"   # categorical slot 1 (blue) -- 旧随机划分
COLOR_NEW = "#1baf7a"   # categorical slot 2 (aqua) -- 新NIST/SDBS划分
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"


def draw_panel(ax, labels, old_vals, new_vals, title, ylabel, fmt="{:.3f}"):
    x = np.arange(len(labels))
    width = 0.32  # bar thickness, leaves air in the slot

    bars_old = ax.bar(x - width / 2 - 0.02, old_vals, width, color=COLOR_OLD,
                       label="旧随机划分 (legacy)", zorder=3)
    bars_new = ax.bar(x + width / 2 + 0.02, new_vals, width, color=COLOR_NEW,
                       label="新NIST/SDBS划分", zorder=3)

    for bars in (bars_old, bars_new):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.006, fmt.format(h),
                     ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)

    ax.set_title(title, fontsize=13, color=INK_PRIMARY, pad=12, loc="left")
    ax.set_ylabel(ylabel, fontsize=10, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=INK_PRIMARY)
    ax.set_ylim(0, max(max(old_vals), max(new_vals)) * 1.22)

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0, colors=INK_MUTED)
    ax.set_facecolor(SURFACE)


def main():
    df = pd.read_csv(RESULT_ROOT / "phase5_split_comparison.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.patch.set_facecolor(PAGE)

    draw_panel(axes[0], df["backbone"], df["legacy_test_f1"], df["nist_split_test_f1"],
               "Test Macro-F1：旧划分 vs 新划分", "Macro-F1")
    draw_panel(axes[1], df["backbone"], df["legacy_test_auc"], df["nist_split_test_auc"],
               "Test Macro-AUC：旧划分 vs 新划分", "Macro-AUC")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.995))

    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out_path = RESULT_ROOT / "phase5_split_comparison.png"
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"已保存到 {out_path}")


if __name__ == "__main__":
    main()
