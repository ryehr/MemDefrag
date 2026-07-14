"""
make_cost_table.py
==================
合并 cost_scaling 三次测量（GPU0: full/top1/top2/vanilla；GPU1: top3；GPU2: top4），
输出合并后的 Markdown + LaTeX 表和论文风格折线图。
full_context / vanilla_latent 取主运行（GPU0）的数据（三次运行同 seed 同数据）。
"""

import json
import numpy as np

MAIN = "./adaptive_logs/cost_scaling.json"
EXTRA = [("./adaptive_logs/cost_scaling_k3.json", "top3"),
         ("./adaptive_logs/cost_scaling_k4.json", "top4")]
OUT_PDF = "./adaptive_logs/cost_scaling_all.pdf"

LABELS = {
    "full_context": "Full-context",
    "top1": "MemDefrag (Top-1)",
    "top2": "MemDefrag (Top-2)",
    "top3": "MemDefrag (Top-3)",
    "top4": "MemDefrag (Top-4)",
    "vanilla_latent": "Vanilla latent memory",
}
ORDER = ["full_context", "top1", "top2", "top3", "top4", "vanilla_latent"]


def main():
    with open(MAIN) as f:
        main_data = json.load(f)
    ns = main_data["ns"]
    times = {c: {int(n): v for n, v in d.items()}
             for c, d in main_data["times"].items()}
    for path, key in EXTRA:
        with open(path) as f:
            d = json.load(f)
        times[key] = {int(n): v for n, v in d["times"][key].items()}

    ctx = {int(n): v for n, v in main_data["ctx_tokens"].items()}
    mem = {int(n): v for n, v in main_data["mem_tokens"].items()}

    configs = [c for c in ORDER if c in times]
    print("### Per-query inference latency (s) vs injected fragments "
          "(Llama-3.1-8B, NQA, 10 groups x 3 repeats, H200)\n")
    print("| n | ctx tokens | mem tokens | " + " | ".join(LABELS[c] for c in configs) + " |")
    print("|" + "---|" * (3 + len(configs)))
    for n in ns:
        row = f"| {n} | {int(ctx[n])} | {int(mem[n])} "
        for c in configs:
            v = np.array(times[c][n])
            row += f"| {v.mean():.3f} ± {v.std():.3f} "
        print(row + "|")

    fc50 = np.mean(times["full_context"][max(ns)])
    print("\nSpeedup at n=50 (full-context / MemDefrag):")
    for c in configs:
        if c.startswith("top"):
            print(f"  {LABELS[c]}: {fc50 / np.mean(times[c][max(ns)]):.2f}x")

    print("\n%% LaTeX")
    print("n & " + " & ".join(LABELS[c] for c in configs) + r" \\")
    for n in ns:
        cells = [f"{np.mean(times[c][n]):.2f}$\\pm${np.std(times[c][n]):.2f}"
                 for c in configs]
        print(f"{n} & " + " & ".join(cells) + r" \\")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    style = {
        "full_context": ("#4A4A4A", "d", "-"),
        "top1": ("#2563EB", "o", "-"),
        "top2": ("#E45756", "s", "-"),
        "top3": ("#59A14F", "^", "-"),
        "top4": ("#F28E2B", "v", "-"),
        "vanilla_latent": ("#B0B0B0", "x", "--"),
    }
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for c in configs:
        color, marker, ls = style[c]
        means = [np.mean(times[c][n]) for n in ns]
        stds = [np.std(times[c][n]) for n in ns]
        ax.errorbar(ns, means, yerr=stds, label=LABELS[c], color=color,
                    marker=marker, linestyle=ls, linewidth=2, markersize=6, capsize=3)
    ax.set_xlabel("Injected knowledge fragments ($n$)")
    ax.set_ylabel("Per-query inference time (s)")
    ax.set_xticks(ns)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=11)
    fig.savefig(OUT_PDF)
    print(f"\nsaved: {OUT_PDF}")


if __name__ == "__main__":
    main()
