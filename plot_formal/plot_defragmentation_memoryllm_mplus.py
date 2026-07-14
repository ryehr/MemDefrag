"""
plot_defragmentation_memoryllm_mplus.py
=======================================
对比 MemoryLLM / M+ (baseline) 与其 Defragmentation 版本
在 NaturalQA 和 SQuAD 上的 Accuracy vs Time Step 折线 (50 steps)。
模仿 plot_retention_comparison.py 的论文级排版风格。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

# ============================================================
#  全局样式 — 论文级排版（与 plot_retention_comparison.py 一致）
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
#  DATA — 50 steps
# ============================================================
# --- Baseline 方法 ---
memoryllm_nqa = [0.302, 0.286, 0.26, 0.232, 0.274, 0.23, 0.23, 0.244, 0.226, 0.234, 0.202, 0.202, 0.248, 0.222, 0.2, 0.178, 0.212, 0.202, 0.196, 0.182, 0.198, 0.228, 0.2, 0.182, 0.212, 0.2, 0.234, 0.214, 0.184, 0.184, 0.19, 0.182, 0.176, 0.158, 0.16, 0.17, 0.156, 0.158, 0.15, 0.18, 0.148, 0.144, 0.146, 0.154, 0.146, 0.152, 0.164, 0.15, 0.158, 0.174]
memoryllm_squad = [0.42, 0.37, 0.328, 0.358, 0.314, 0.318, 0.302, 0.298, 0.304, 0.29, 0.298, 0.294, 0.29, 0.298, 0.284, 0.234, 0.234, 0.248, 0.26, 0.242, 0.236, 0.202, 0.214, 0.224, 0.242, 0.218, 0.208, 0.2, 0.22, 0.202, 0.216, 0.192, 0.198, 0.204, 0.184, 0.19, 0.202, 0.186, 0.164, 0.19, 0.178, 0.19, 0.146, 0.168, 0.138, 0.166, 0.178, 0.16, 0.194, 0.154]

mplus_nqa = [0.394, 0.362, 0.32, 0.284, 0.302, 0.308, 0.306, 0.276, 0.276, 0.262, 0.274, 0.258, 0.272, 0.24, 0.218, 0.2, 0.22, 0.198, 0.204, 0.196, 0.226, 0.202, 0.196, 0.206, 0.214, 0.222, 0.222, 0.234, 0.228, 0.21, 0.176, 0.192, 0.206, 0.226, 0.21, 0.194, 0.184, 0.17, 0.174, 0.172, 0.192, 0.194, 0.19, 0.19, 0.208, 0.192, 0.204, 0.198, 0.202, 0.176]
mplus_squad = [0.488, 0.5, 0.436, 0.448, 0.466, 0.412, 0.458, 0.438, 0.446, 0.44, 0.41, 0.446, 0.374, 0.396, 0.362, 0.356, 0.38, 0.348, 0.36, 0.364, 0.38, 0.336, 0.346, 0.356, 0.392, 0.336, 0.326, 0.36, 0.376, 0.332, 0.336, 0.346, 0.35, 0.36, 0.318, 0.31, 0.298, 0.31, 0.306, 0.296, 0.284, 0.288, 0.274, 0.268, 0.262, 0.248, 0.242, 0.234, 0.236, 0.202]

# --- Defragmentation 版本 ---
memoryllm_nqa_defrag = [0.302, 0.288, 0.272, 0.270, 0.264, 0.260, 0.260, 0.258, 0.246, 0.252, 0.248, 0.242, 0.238, 0.238, 0.236, 0.234, 0.240, 0.236, 0.228, 0.230, 0.220, 0.222, 0.216, 0.218, 0.212, 0.204, 0.208, 0.214, 0.196, 0.188, 0.194, 0.192, 0.184, 0.178, 0.176, 0.172, 0.176, 0.178, 0.176, 0.164, 0.166, 0.168, 0.172, 0.158, 0.160, 0.162, 0.164, 0.168, 0.172, 0.180]
memoryllm_squad_defrag = [0.42, 0.378, 0.356, 0.354, 0.334, 0.326, 0.312, 0.298, 0.306, 0.302, 0.310, 0.304, 0.300, 0.308, 0.286, 0.284, 0.274, 0.278, 0.26, 0.248, 0.246, 0.230, 0.254, 0.244, 0.244, 0.232, 0.228, 0.230, 0.228, 0.214, 0.220, 0.208, 0.212, 0.214, 0.194, 0.198, 0.204, 0.206, 0.184, 0.196, 0.172, 0.19, 0.206, 0.208, 0.178, 0.190, 0.198, 0.188, 0.196, 0.180]

mplus_nqa_defrag = [0.394, 0.378, 0.354, 0.328, 0.332, 0.338, 0.330, 0.314, 0.302, 0.290, 0.298, 0.276, 0.274, 0.262, 0.258, 0.238, 0.246, 0.224, 0.228, 0.216, 0.230, 0.230, 0.242, 0.240, 0.228, 0.238, 0.246, 0.244, 0.234, 0.23, 0.220, 0.222, 0.238, 0.240, 0.232, 0.218, 0.214, 0.206, 0.200, 0.198, 0.208, 0.206, 0.200, 0.210, 0.210, 0.214, 0.204, 0.212, 0.210, 0.192]
mplus_squad_defrag = [0.488, 0.492, 0.446, 0.452, 0.460, 0.452, 0.468, 0.444, 0.460, 0.452, 0.462, 0.440, 0.420, 0.422, 0.408, 0.418, 0.426, 0.428, 0.398, 0.396, 0.388, 0.354, 0.380, 0.376, 0.390, 0.374, 0.378, 0.380, 0.382, 0.376, 0.368, 0.382, 0.368, 0.374, 0.346, 0.344, 0.328, 0.324, 0.336, 0.330, 0.310, 0.306, 0.298, 0.304, 0.300, 0.280, 0.268, 0.260, 0.254, 0.238]

# ============================================================
#  公共设置
# ============================================================
num_steps = 50
steps = np.arange(1, num_steps + 1)

# 折线配置: (label, data_nqa, data_squad, color, marker, linestyle, linewidth)
# Baseline 用虚线，Defragmentation 用实线；同一模型同色系
lines_config = [
    # --- MemoryLLM ---
    ("MemoryLLM",                  memoryllm_nqa,        memoryllm_squad,        "#B0B0B0",  "x",  "--", 2.0),
    ("MemoryLLM w/ Defragmentation", memoryllm_nqa_defrag, memoryllm_squad_defrag, "#2563EB",  "o",  "-",  2.0),
    # --- M+ ---
    ("M+",                          mplus_nqa,            mplus_squad,            "#4A4A4A",  "d",  "--", 2.0),
    ("M+ w/ Defragmentation",        mplus_nqa_defrag,     mplus_squad_defrag,     "#E45756",  "s",  "-",  2.0),
]


def plot_dataset(ax, lines, title):
    """在给定 ax 上绘制一个数据集的所有折线。"""
    for label, data, color, marker, ls, lw in lines:
        ax.plot(
            steps, data,
            marker=marker, linewidth=lw, markersize=5,
            color=color,
            markeredgecolor=color, markeredgewidth=0.8,
            markevery=2,          # 每隔 2 步标一个 marker，避免 50 步太密
            linestyle=ls,
            label=label, zorder=3,
        )

    ax.set_title(title)
    ax.set_xlabel("Time Step ($n$)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.5, num_steps + 0.5)
    ax.set_ylim(0.14, 0.5)
    ax.set_xticks(np.arange(5, num_steps + 1, 5))
    # y 轴用百分比显示
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, axis="both", linestyle="--")


# ============================================================
#  创建包含两个子图的 figure（每个子图近似正方形）
# ============================================================
fig, (ax_nqa, ax_squad) = plt.subplots(
    1, 2, figsize=(10, 5.5), sharey=True,
)

# 让每个子图的数据区呈正方形
for ax in (ax_nqa, ax_squad):
    ax.set_box_aspect(1)

# --- NQA 子图 ---
nqa_lines = [(label, d_nqa, color, marker, ls, lw)
             for label, d_nqa, _, color, marker, ls, lw in lines_config]
plot_dataset(ax_nqa, nqa_lines, "NaturalQA")

# --- SQuAD 子图 ---
squad_lines = [(label, d_squad, color, marker, ls, lw)
               for label, _, d_squad, color, marker, ls, lw in lines_config]
plot_dataset(ax_squad, squad_lines, "SQuAD")

# 右子图共享 y 轴后隐藏 y label 避免重复
ax_squad.set_ylabel("")

# --- 共用图例（取自左子图的 handles） ---
handles, labels = ax_nqa.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.88),
    ncol=2,                 # 4 条折线分 2 行 × 2 列
    framealpha=0.9,
    edgecolor="#CCCCCC",
    columnspacing=1.2,
    handletextpad=0.5,
)

fig.tight_layout(rect=[0, 0.02, 1, 0.86])  # 顶部留出两行图例的空间
fig.subplots_adjust(wspace=0.08)           # 压紧两个子图之间的水平间距
fig.savefig("plot_figures/defragmentation_memoryllm_mplus.pdf")
print("[✓] Saved: plot_figures/defragmentation_memoryllm_mplus.pdf")

plt.show()
