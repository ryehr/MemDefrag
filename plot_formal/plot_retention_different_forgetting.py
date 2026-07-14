"""
plot_retention_different_forgetting.py
=======================================
模仿 plot_retention_different_k.py 的论文级折线风格，
以 2×2 网格绘制 4 张子图，对比 Random 与 Informativeness-based 遗忘策略，
并标注 Forgetting 从第 26 步开始。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

# ============================================================
#  全局样式 — 论文级排版（与 plot_retention_different_k.py 一致）
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

top_2_random_nqa = [0.732, 0.71, 0.686, 0.702, 0.72, 0.706, 0.71, 0.684, 0.678, 0.67, 0.676, 0.672, 0.672, 0.662, 0.684, 0.668, 0.668, 0.67, 0.672, 0.67, 0.672, 0.648, 0.66, 0.646, 0.656, 0.652, 0.624, 0.582, 0.578, 0.534, 0.526, 0.484, 0.458, 0.442, 0.386, 0.404, 0.354, 0.356, 0.356, 0.34, 0.296, 0.282, 0.29, 0.262, 0.276, 0.248, 0.242, 0.22, 0.216, 0.208]
top_2_perplxity_nqa = [0.732, 0.712, 0.692, 0.704, 0.714, 0.71, 0.702, 0.684, 0.67, 0.668, 0.676, 0.662, 0.652, 0.678, 0.67, 0.686, 0.672, 0.658, 0.686, 0.674, 0.664, 0.636, 0.636, 0.666, 0.636, 0.65, 0.658, 0.642, 0.628, 0.622, 0.602, 0.588, 0.582, 0.562, 0.552, 0.524, 0.542, 0.5, 0.512, 0.5, 0.504, 0.51, 0.484, 0.456, 0.44, 0.442, 0.424, 0.428, 0.396, 0.378]

top_2_random_squad = [0.776, 0.794, 0.79, 0.78, 0.774, 0.782, 0.778, 0.784, 0.768, 0.774, 0.774, 0.782, 0.78, 0.78, 0.76, 0.758, 0.748, 0.736, 0.734, 0.73, 0.712, 0.714, 0.696, 0.742, 0.7, 0.672, 0.658, 0.602, 0.57, 0.574, 0.514, 0.508, 0.476, 0.444, 0.4, 0.408, 0.384, 0.334, 0.334, 0.288, 0.316, 0.272, 0.238, 0.216, 0.214, 0.208, 0.198, 0.198, 0.212, 0.162]
top_2_perplxity_squad = [0.776, 0.792, 0.788, 0.782, 0.772, 0.776, 0.766, 0.782, 0.772, 0.782, 0.79, 0.786, 0.774, 0.786, 0.76, 0.766, 0.744, 0.758, 0.734, 0.74, 0.724, 0.696, 0.702, 0.704, 0.678, 0.688, 0.662, 0.68, 0.646, 0.608, 0.574, 0.59, 0.536, 0.52, 0.518, 0.49, 0.484, 0.466, 0.45, 0.4, 0.396, 0.362, 0.326, 0.336, 0.318, 0.302, 0.33, 0.284, 0.29, 0.272]

top_3_random_nqa = [0.732, 0.71, 0.706, 0.67, 0.696, 0.692, 0.674, 0.66, 0.636, 0.606, 0.644, 0.632, 0.596, 0.614, 0.636, 0.63, 0.618, 0.624, 0.644, 0.626, 0.632, 0.622, 0.636, 0.618, 0.628, 0.62, 0.596, 0.576, 0.548, 0.504, 0.476, 0.45, 0.418, 0.394, 0.366, 0.358, 0.314, 0.296, 0.276, 0.248, 0.254, 0.246, 0.238, 0.224, 0.226, 0.208, 0.212, 0.176, 0.184, 0.186]
top_3_perplxity_nqa = [0.732, 0.712, 0.7, 0.668, 0.692, 0.676, 0.672, 0.66, 0.65, 0.628, 0.626, 0.622, 0.628, 0.636, 0.628, 0.634, 0.632, 0.614, 0.65, 0.62, 0.632, 0.612, 0.618, 0.604, 0.608, 0.622, 0.618, 0.606, 0.576, 0.592, 0.56, 0.524, 0.542, 0.524, 0.506, 0.496, 0.486, 0.476, 0.484, 0.446, 0.448, 0.438, 0.448, 0.418, 0.408, 0.394, 0.384, 0.376, 0.36, 0.362]

top_3_random_squad = [0.776, 0.792, 0.778, 0.772, 0.778, 0.77, 0.772, 0.77, 0.772, 0.77, 0.774, 0.778, 0.768, 0.772, 0.744, 0.75, 0.728, 0.734, 0.708, 0.716, 0.724, 0.718, 0.684, 0.72, 0.712, 0.674, 0.642, 0.6, 0.594, 0.568, 0.52, 0.484, 0.468, 0.45, 0.39, 0.38, 0.382, 0.332, 0.308, 0.278, 0.302, 0.252, 0.236, 0.224, 0.22, 0.204, 0.2, 0.202, 0.182, 0.168]
top_3_perplxity_squad = [0.776, 0.796, 0.774, 0.774, 0.77, 0.78, 0.774, 0.758, 0.758, 0.768, 0.752, 0.756, 0.746, 0.764, 0.744, 0.744, 0.742, 0.746, 0.696, 0.738, 0.718, 0.69, 0.704, 0.718, 0.694, 0.68, 0.656, 0.668, 0.626, 0.602, 0.556, 0.56, 0.534, 0.524, 0.514, 0.472, 0.464, 0.444, 0.446, 0.416, 0.398, 0.362, 0.326, 0.332, 0.308, 0.3, 0.286, 0.268, 0.274, 0.25]

# ============================================================
#  公共设置
# ============================================================
num_steps = 50
steps = np.arange(1, num_steps + 1)
forgetting_start = 26  # Forgetting 从第 26 步开始

# 颜色与标记
COLOR_RANDOM = "#2563EB"       # 蓝色 — Random
COLOR_PERPLEXITY = "#E45756"   # 红色 — Informativeness-based
MARKER_RANDOM = "o"
MARKER_PERPLEXITY = "s"

# 4 张子图的配置: (title, data_random, data_perplexity, borderline)
# 2×2 布局按行优先顺序排布：
#   [0,0] NaturalQA Top-2   [0,1] NaturalQA Top-3
#   [1,0] SQuAD Top-2       [1,1] SQuAD Top-3
subplots_config = [
    ("NaturalQA (Top-2)", top_2_random_nqa,   top_2_perplxity_nqa,   borderline_nqa),
    ("NaturalQA (Top-3)", top_3_random_nqa,   top_3_perplxity_nqa,   borderline_nqa),
    ("SQuAD (Top-2)",     top_2_random_squad,  top_2_perplxity_squad, borderline_squad),
    ("SQuAD (Top-3)",     top_3_random_squad,  top_3_perplxity_squad, borderline_squad),
]


def plot_dataset(ax, data_random, data_perplexity, borderline, title):
    """在给定 ax 上绘制一个子图：2 条折线 + borderline + forgetting 标注。"""

    # Random 折线
    ax.plot(
        steps, data_random,
        marker=MARKER_RANDOM, linewidth=2, markersize=5,
        color=COLOR_RANDOM,
        markeredgecolor=COLOR_RANDOM, markeredgewidth=0.8,
        label="Random", zorder=3, markevery=2,
    )

    # Informativeness-based 折线
    ax.plot(
        steps, data_perplexity,
        marker=MARKER_PERPLEXITY, linewidth=2, markersize=5,
        color=COLOR_PERPLEXITY,
        markeredgecolor=COLOR_PERPLEXITY, markeredgewidth=0.8,
        label="Informativeness-based", zorder=3, markevery=2,
    )

    # Borderline 虚横线
    ax.axhline(
        borderline, color="#D62728", linewidth=1.5,
        linestyle="--", zorder=2,
    )
    # 在虚线左端标注 "Borderline = xx.x%"
    ax.text(
        1.0, borderline,
        f"Borderline = {borderline * 100:.1f}%",
        color="#D62728", fontsize=12,
        va="bottom", ha="left",
        zorder=4,
    )

    # 标注 Forgetting 起始（第 26 步竖虚线 + 文字）
    ax.axvline(
        forgetting_start, color="#888888", linewidth=1.0,
        linestyle=":", zorder=2,
    )
    ax.text(
        forgetting_start + 0.8, 0.96,
        "Forgetting\nstarts",
        fontsize=12, color="#555555",
        ha="left", va="top",
        zorder=5,
    )

    ax.set_title(title, pad=6)
    ax.set_xlim(0.5, num_steps + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(5, num_steps + 1, 10))
    # y 轴用百分比显示
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, axis="both", linestyle="--")


# ============================================================
#  创建 2×2 网格的 figure
# ============================================================
fig, axes = plt.subplots(
    2, 2, figsize=(10, 10), sharey=True, sharex=True,
)
axes_flat = axes.flatten()

for idx, (title, d_random, d_perplexity, bl) in enumerate(subplots_config):
    ax = axes_flat[idx]
    plot_dataset(ax, d_random, d_perplexity, bl, title)

    row, col = divmod(idx, 2)

    # 只有左列显示 y label
    if col == 0:
        ax.set_ylabel("Accuracy")
    else:
        ax.set_ylabel("")

    # 只有最后一行显示 x label
    if row == 1:
        ax.set_xlabel("Time Step ($n$)")
    else:
        ax.set_xlabel("")

# --- 共用图例（放在顶部） ---
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.92),
    ncol=2,
    framealpha=0.9,
    edgecolor="#CCCCCC",
    columnspacing=1.5,
    handletextpad=0.5,
)

fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.subplots_adjust(wspace=0.08, hspace=0.25)
fig.savefig("plot_figures/retention_different_forgetting.pdf")
fig.savefig("plot_figures/retention_different_forgetting.png")
print("[✓] Saved: plot_figures/retention_different_forgetting.pdf")
print("[✓] Saved: plot_figures/retention_different_forgetting.png")

plt.show()
