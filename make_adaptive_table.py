"""
make_adaptive_table.py
======================
汇总 adaptive-K 实验结果：
  - static Top-1..4 曲线从 plot_formal/plot_retention_comparison.py (8B backbone)
    与 plot_formal/plot_other_models.py (Qwen/Mistral/Gemma) 中解析（500 组、50 步原始数据）；
  - adaptive-K 结果从 adaptive_logs/adaptive_<tag>.log 中解析
    （eval steps [1,10,20,30,40,50] 对应的 accuracy 列表与 mean-K 统计）。
输出：Markdown 表 + LaTeX 表（adaptive vs. best/worst static K, n ∈ {10,30,50}）。
"""

import ast
import re
import os

STEPS_REPORT = [10, 30, 50]
EVAL_STEPS = [1, 10, 20, 30, 40, 50]

CONFIGS = [
    # (tag, model display, dataset display, static array name pattern)
    ("base8b_nqa",   "8B backbone",              "NaturalQA", "top_{k}_nqa"),
    ("base8b_squad", "8B backbone",              "SQuAD",     "top_{k}_squad"),
    ("qwen_nqa",      "Qwen2.5-7B-Instruct",      "NaturalQA", "qwen_nqa_top{k}"),
    ("qwen_squad",    "Qwen2.5-7B-Instruct",      "SQuAD",     "qwen_squad_top{k}"),
    ("mistral_nqa",   "Mistral-7B-Instruct-v0.3", "NaturalQA", "mistral_nqa_top{k}"),
    ("mistral_squad", "Mistral-7B-Instruct-v0.3", "SQuAD",     "mistral_squad_top{k}"),
    ("gemma_nqa",     "Gemma-2-9b-it",            "NaturalQA", "gemma_nqa_top{k}"),
    ("gemma_squad",   "Gemma-2-9b-it",            "SQuAD",     "gemma_squad_top{k}"),
]


def parse_arrays(path):
    """解析 python 源文件中 `name = [ ... ]` 形式的数值数组。"""
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    arrays = {}
    for m in re.finditer(r"^(\w+)\s*=\s*(?:\w+\s*=\s*)?(\[[^\]]*\])", src, re.M):
        try:
            arrays[m.group(1)] = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            pass
    return arrays


def parse_adaptive_out(path, n_groups=100):
    """
    从 stdout 文件解析恰好完成 n_groups 组时的累计 accuracy 与 mean-K。
    每组结束时的打印顺序：accuracy 列表 → lengths 列表 → Adaptive-K mean per step → Step i: Time taken。
    因此在遇到 "Step {n_groups}: Time taken" 时，缓存的最近 accuracy/mean-K 即为 n_groups 组均值。
    """
    acc, mean_k_per_step = None, None
    last_acc, last_k = None, None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("['"):
                try:
                    vals = [float(v) for v in ast.literal_eval(line)]
                except (ValueError, SyntaxError):
                    continue
                if len(vals) == len(EVAL_STEPS) and max(vals) <= 1.01:
                    last_acc = vals
            elif line.startswith("Adaptive-K mean per step:"):
                d = ast.literal_eval(line.split(":", 1)[1].strip())
                last_k = {int(k): float(v) for k, v in d.items()}
            elif line.startswith(f"Step {n_groups}: Time taken"):
                acc, mean_k_per_step = last_acc, last_k
                break
    return acc, mean_k_per_step, None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    static = {}
    static.update(parse_arrays(os.path.join(here, "plot_formal", "plot_retention_comparison.py")))
    static.update(parse_arrays(os.path.join(here, "plot_formal", "plot_other_models.py")))

    md_rows, tex_rows = [], []
    for tag, model, dataset, pat in CONFIGS:
        out_path = os.path.join(here, "adaptive_logs", f"adaptive_{tag}.out")
        acc = mean_k = overall = None
        if os.path.exists(out_path):
            acc, mean_k, overall = parse_adaptive_out(out_path, n_groups=100)

        static_curves = {k: static[pat.format(k=k)] for k in range(1, 5)}

        md_cells, tex_cells = [], []
        for n in STEPS_REPORT:
            per_k = {k: static_curves[k][n - 1] for k in range(1, 5)}
            best_k = max(per_k, key=per_k.get)
            worst_k = min(per_k, key=per_k.get)
            if acc is not None:
                a = acc[EVAL_STEPS.index(n)]
                mk = mean_k.get(n) if mean_k else None
                a_str = f"{a*100:.1f}"
                mk_str = f"{mk:.2f}" if mk is not None else "-"
            else:
                a_str, mk_str = "??", "-"
            md_cells.append(
                f"{a_str} (K̄={mk_str}) | {per_k[best_k]*100:.1f} (K={best_k}) | {per_k[worst_k]*100:.1f} (K={worst_k})"
            )
            tex_cells.append(
                f"{a_str} & {per_k[best_k]*100:.1f}\\,(K{{=}}{best_k}) & {per_k[worst_k]*100:.1f}\\,(K{{=}}{worst_k})"
            )

        md_rows.append(f"| {model} | {dataset} | " + " | ".join(md_cells) + " |")
        tex_rows.append(f"{model} & {dataset} & " + " & ".join(tex_cells) + r" \\")
        if overall:
            md_rows.append(f"| _{tag}: {overall}_ |" + " |" * (1 + 3 * len(STEPS_REPORT)))

    print("### Adaptive-K vs. static K (accuracy %, 50-step knowledge retention)")
    print("(adaptive: 100 groups; static reference: 500 groups from the paper's runs)\n")
    header = "| Model | Dataset | " + " | ".join(
        f"n={n} Adaptive | n={n} Best static | n={n} Worst static" for n in STEPS_REPORT
    ) + " |"
    sep = "|" + "---|" * (2 + 3 * len(STEPS_REPORT))
    print(header)
    print(sep)
    for r in md_rows:
        print(r)

    print("\n\n%% ===== LaTeX =====")
    print(r"\begin{tabular}{llccc ccc ccc}")
    print(r"\toprule")
    print(r" & & \multicolumn{3}{c}{$n=10$} & \multicolumn{3}{c}{$n=30$} & \multicolumn{3}{c}{$n=50$} \\")
    print(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}")
    print(r"Model & Dataset & Adap. & Best & Worst & Adap. & Best & Worst & Adap. & Best & Worst \\")
    print(r"\midrule")
    for r in tex_rows:
        print(r)
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
