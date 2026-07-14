"""
plot_retention_lines.py
========================
基于 plot_retention_different_k.py 中的数据，
模仿 plot_investigation.py 的论文级折线风格，
在一张图中用两个子图分别绘制 NQA 和 SQuAD 的 Accuracy vs Time Step 折线，
共用一个图例。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

# ============================================================
#  全局样式 — 论文级排版（与 plot_investigation.py 一致）
# ============================================================
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "font.size":          18,
    "axes.labelsize":     18,
    "axes.titlesize":     18,
    "xtick.labelsize":    18,
    "ytick.labelsize":    18,
    "legend.fontsize":    14,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.6,
})

os.makedirs("plot_figures", exist_ok=True)

# ============================================================
#  DATA
# ============================================================
borderline_nqa = 0.146
borderline_squad = 0.074

vanilla_nqa = [0.732, 0.698, 0.608, 0.548, 0.546, 0.378, 0.286, 0.21, 0.22, 0.218, 0.18, 0.156, 0.156, 0.156, 0.128, 0.086, 0.064, 0.1, 0.086, 0.094]
vanilla_squad = [0.776, 0.788, 0.754, 0.75, 0.674, 0.604, 0.606, 0.548, 0.53, 0.54, 0.474, 0.462, 0.368, 0.394, 0.364, 0.226, 0.178, 0.194, 0.15, 0.19]

top_all_nqa = [0.732, 0.714, 0.704, 0.652, 0.61, 0.54, 0.482, 0.412, 0.374, 0.36, 0.332, 0.322, 0.314, 0.278, 0.282, 0.246, 0.244, 0.236, 0.232, 0.226]
top_all_squad = [0.776, 0.792, 0.776, 0.748, 0.702, 0.684, 0.654, 0.632, 0.598, 0.588, 0.572, 0.542, 0.502, 0.462, 0.456, 0.428, 0.416, 0.4, 0.376, 0.344]

top_1_nqa = [0.732, 0.69, 0.708, 0.72, 0.716, 0.72, 0.706, 0.676, 0.688, 0.672, 0.672, 0.662, 0.662, 0.666, 0.68, 0.682, 0.678, 0.67, 0.66, 0.664]
top_1_squad = [0.776, 0.774, 0.776, 0.77, 0.772, 0.766, 0.768, 0.766, 0.762, 0.764, 0.758, 0.756, 0.75, 0.734, 0.728, 0.73, 0.722, 0.716, 0.69, 0.672]

top_2_nqa = [0.732, 0.714, 0.704, 0.702, 0.718, 0.706, 0.706, 0.684, 0.68, 0.634, 0.662, 0.66, 0.658, 0.678, 0.676, 0.69, 0.678, 0.674, 0.688, 0.672]
top_2_squad = [0.776, 0.792, 0.8, 0.78, 0.78, 0.774, 0.78, 0.784, 0.78, 0.788, 0.778, 0.788, 0.778, 0.78, 0.76, 0.76, 0.758, 0.752, 0.728, 0.718]

top_3_nqa = [0.732, 0.712, 0.706, 0.67, 0.698, 0.682, 0.662, 0.642, 0.646, 0.614, 0.622, 0.624, 0.608, 0.616, 0.618, 0.642, 0.63, 0.612, 0.636, 0.614]
top_3_squad = [0.776, 0.794, 0.778, 0.78, 0.762, 0.764, 0.77, 0.774, 0.756, 0.762, 0.768, 0.766, 0.762, 0.752, 0.736, 0.728, 0.728, 0.744, 0.718, 0.722]

top_4_nqa = [0.732, 0.714, 0.702, 0.656, 0.65, 0.646, 0.638, 0.608, 0.632, 0.588, 0.608, 0.608, 0.564, 0.596, 0.6, 0.594, 0.608, 0.594, 0.612, 0.608]
top_4_squad = [0.776, 0.792, 0.774, 0.748, 0.75, 0.732, 0.746, 0.746, 0.716, 0.744, 0.736, 0.75, 0.736, 0.73, 0.74, 0.73, 0.722, 0.724, 0.704, 0.71]

top_5_nqa = [0.732, 0.718, 0.704, 0.656, 0.608, 0.596, 0.598, 0.57, 0.574, 0.548, 0.562, 0.566, 0.546, 0.534, 0.556, 0.56, 0.546, 0.546, 0.56, 0.56]
top_5_squad = [0.776, 0.794, 0.778, 0.752, 0.704, 0.7, 0.708, 0.69, 0.698, 0.706, 0.702, 0.722, 0.716, 0.716, 0.716, 0.702, 0.702, 0.706, 0.682, 0.692]

top_6_nqa = [0.732, 0.718, 0.7, 0.648, 0.606, 0.54, 0.546, 0.49, 0.524, 0.504, 0.528, 0.498, 0.482, 0.492, 0.486, 0.512, 0.472, 0.508, 0.518, 0.498]
top_6_squad = [0.776, 0.792, 0.778, 0.75, 0.702, 0.686, 0.676, 0.674, 0.68, 0.688, 0.686, 0.69, 0.686, 0.688, 0.706, 0.684, 0.678, 0.686, 0.648, 0.642]

# ============================================================
#  公共设置
# ============================================================
num_steps = 20
steps = np.arange(1, num_steps + 1)

# 折线配置: (label, data_nqa, data_squad, color, marker)
lines_config = [
    ("Vanilla",   vanilla_nqa,  vanilla_squad,  "#B0B0B0",  "x"),
    ("Top-all",   top_all_nqa,  top_all_squad,  "#4A4A4A",  "d"),
    ("Top-1",     top_1_nqa,    top_1_squad,    "#2563EB",  "o"),
    ("Top-2",     top_2_nqa,    top_2_squad,    "#E45756",  "s"),
    ("Top-3",     top_3_nqa,    top_3_squad,    "#59A14F",  "^"),
    ("Top-4",     top_4_nqa,    top_4_squad,    "#F28E2B",  "v"),
    ("Top-5",     top_5_nqa,    top_5_squad,    "#B07AA1",  "D"),
    ("Top-6",     top_6_nqa,    top_6_squad,    "#76B7B2",  "P"),
]


def plot_dataset(ax, lines, borderline, title):
    """在给定 ax 上绘制一个数据集的所有折线。"""
    for label, data, color, marker in lines:
        ax.plot(
            steps, data,
            marker=marker, linewidth=2, markersize=5,
            color=color,
            markeredgecolor=color, markeredgewidth=0.8,
            label=label, zorder=3,
        )

    # Borderline 虚横线（不加入 legend，直接在图中标注数值）
    ax.axhline(
        borderline, color="#D62728", linewidth=1.5,
        linestyle="--", zorder=2,
    )
    # 在虚线左端（靠近 step=0）标注 "Borderline = xx.x%"
    ax.text(
        1.0, borderline,
        f"Borderline = {borderline * 100:.1f}%",
        color="#D62728", fontsize=12,
        va="bottom", ha="left",
        zorder=4,
    )

    ax.set_title(title)
    ax.set_xlabel("Time Step ($n$)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.5, num_steps + 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(2, num_steps + 1, 2))
    # y 轴用百分比显示
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, axis="both", linestyle="--")


# ============================================================
#  创建包含两个子图的 figure（每个子图近似正方形）
# ============================================================
# figsize 适度收窄，避免 tight_layout 把多余宽度塞进子图间距
fig, (ax_nqa, ax_squad) = plt.subplots(
    1, 2, figsize=(10, 5.5), sharey=True,
)

# 让每个子图的数据区呈正方形
for ax in (ax_nqa, ax_squad):
    ax.set_box_aspect(1)

# --- NQA 子图 ---
nqa_lines = [(label, d_nqa, color, marker)
             for label, d_nqa, _, color, marker in lines_config]
plot_dataset(ax_nqa, nqa_lines, borderline_nqa, "NaturalQA")

# --- SQuAD 子图 ---
squad_lines = [(label, d_squad, color, marker)
               for label, _, d_squad, color, marker in lines_config]
plot_dataset(ax_squad, squad_lines, borderline_squad, "SQuAD")

# 右子图共享 y 轴后隐藏 y label 避免重复
ax_squad.set_ylabel("")

# --- 共用图例（取自左子图的 handles） ---
handles, labels = ax_nqa.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.92),
    ncol=4,
    framealpha=0.9,
    edgecolor="#CCCCCC",
    columnspacing=1.2,
    handletextpad=0.5,
)

fig.tight_layout(rect=[0, 0.02, 1, 0.9])  # 顶部留出图例空间
fig.subplots_adjust(wspace=0.08)          # 压紧两个子图之间的水平间距
fig.savefig("plot_figures/retention_different_k.pdf")
print("[✓] Saved: plot_figures/retention_different_k.pdf")

plt.show()
