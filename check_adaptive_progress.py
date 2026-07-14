"""
check_adaptive_progress.py
==========================
运行中查看 adaptive-K 实验的即时结果快照。
从 adaptive_logs/adaptive_<tag>.out 中取：
  - 已完成组数与平均每组耗时（估算 ETA）；
  - 最新一行累计 accuracy（eval steps [1,10,20,30,40,50] 上、已完成组的平均）；
  - 最新的 Adaptive-K mean per step。
注意：Python stdout 块缓冲，.out 文件约有几组的延迟；组数少时数值噪声大。
"""

import ast
import os
import re

EVAL_STEPS = [1, 10, 20, 30, 40, 50]
TAGS = ["llama_nqa", "llama_squad", "qwen_nqa", "qwen_squad",
        "mistral_nqa", "mistral_squad", "gemma_nqa", "gemma_squad"]
TOTAL_GROUPS = 100


def snapshot(path):
    if not os.path.exists(path):
        return None
    groups, last_acc, last_k, times = 0, None, None, []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Step ") and "Time taken" in line:
                groups += 1
                m = re.search(r"Time taken = ([\d.]+)", line)
                if m:
                    times.append(float(m.group(1)))
            elif line.startswith("['"):
                try:
                    vals = [float(v) for v in ast.literal_eval(line)]
                except (ValueError, SyntaxError):
                    continue
                if len(vals) == len(EVAL_STEPS) and max(vals) <= 1.01:
                    last_acc = vals
            elif line.startswith("Adaptive-K mean per step:"):
                last_k = line.split(":", 1)[1].strip()
    return groups, last_acc, last_k, times


def main():
    print(f"{'config':<15}{'groups':>7}  {'ETA':>6}  acc@n=" + ",".join(map(str, EVAL_STEPS)))
    print("-" * 100)
    for tag in TAGS:
        snap = snapshot(f"./adaptive_logs/adaptive_{tag}.out")
        if snap is None:
            print(f"{tag:<15}{'—':>7}  (not started)")
            continue
        groups, acc, k, times = snap
        if groups == 0 or acc is None:
            print(f"{tag:<15}{groups:>7}  (warming up)")
            continue
        avg_t = sum(times[-50:]) / len(times[-50:])
        eta_min = (TOTAL_GROUPS - groups) * avg_t / 60
        acc_str = " ".join(f"{a*100:5.1f}" for a in acc)
        print(f"{tag:<15}{groups:>7}  {eta_min:5.0f}m  [{acc_str}]  ({avg_t:.1f}s/group)")
        if k:
            print(f"{'':<15}{'':>7}  {'':>6}  mean-K per step: {k}")
    print("\n(accuracy 为已完成组上的累计平均，随组数增加收敛；<100 组时波动较大)")


if __name__ == "__main__":
    main()
