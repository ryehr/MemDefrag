"""
cost_scaling.py
===============
Per-query 推理延迟 vs 注入上下文规模（rebuttal "How the cost scales"）。

对比（同一硬件、同一 8B 骨干模型、greedy、固定 32 生成 token）：
  - full-context：全部 n 个 512-token 片段以原文放入 prompt（512n + |q| 的
    prefill + 解码），每次查询都要重新编码全部上下文；
  - MemDefrag Top-K：latent memory 已构建（形成成本离线摊销），每次查询 =
    SDPA tracing（tracer 层探针）+ 切片 defrag + Top-K 截断记忆上的生成；
  - vanilla latent（参考线）：全量 latent memory 前缀上的生成。

协议：每组按知识保持流程增量注入 50 个片段（target + unrelated，
perplexity 比例遗忘，N_max=12,800），在 n ∈ {10,20,30,40,50} 处测量。
每个测点先预热一遍，再计时 repeats 次（torch.cuda.synchronize + perf_counter）。
输出：汇总表（mean±std, s）、JSON、论文风格折线图 PDF。
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utilities import (
    get_max_memory,
    get_input_device,
    model_knowledge,
    concatenate_knowledge_latent,
    rebuild_indices_from_segment_lengths,
    maybe_forget_before_concat,
    generate_with_hidden_prefix,
)
from sdpa_tracing import probe_densities_sdpa, defrag_from_densities


def generate_text_timed(model_obj, input_ids, max_new_tokens):
    """纯文本 prompt 的 prefill + 固定步数解码（与 generate_with_hidden_prefix
    相同的手工解码循环，不因 EOS 提前停止），用于 full-context 计时。"""
    with torch.no_grad():
        out = model_obj(input_ids=input_ids, use_cache=True, return_dict=True)
        logits, past = out.logits, out.past_key_values
        for _ in range(max_new_tokens):
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = model_obj(input_ids=next_id, past_key_values=past,
                            use_cache=True, return_dict=True)
            logits, past = out.logits, out.past_key_values


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="nqa")
    parser.add_argument("--ns", type=str, default="10,20,30,40,50")
    parser.add_argument("--keep_nums", type=str, default="1,2")
    parser.add_argument("--reorder_base_layer", type=int, default=13)
    parser.add_argument("--query_mode", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--max_knowledge_tokens", type=int, default=12800)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_prefix", type=str, default="./adaptive_logs/cost_scaling")
    args = parser.parse_args()
    print(args)

    ns = [int(x) for x in args.ns.split(",")]
    keep_nums = [int(x) for x in args.keep_nums.split(",")]
    n_max_frag = max(ns)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory=get_max_memory()).eval()
    input_device = get_input_device(model)

    path = f"./data_new/{args.dataset}/"
    with open(path + "val_data.json", encoding="utf-8-sig") as f:
        test_data = json.load(f)
    with open(path + "unrelated_contexts_from_train.json", encoding="utf-8-sig") as f:
        unrelated_data = json.load(f)
    random.shuffle(test_data)
    random.shuffle(unrelated_data)
    test_data = [it for it in test_data
                 if len(tokenizer(it["context"])["input_ids"]) <= 512][: args.groups]

    configs = ["full_context"] + [f"top{k}" for k in keep_nums] + ["vanilla_latent"]
    times = {c: {n: [] for n in ns} for c in configs}
    ctx_tokens = {n: [] for n in ns}   # full-context 的实际 prompt token 数
    mem_tokens = {n: [] for n in ns}   # latent memory 的实际长度

    for gi, item in enumerate(test_data):
        t0 = time.time()
        question = item["question"] + " Answer: "
        texts = [item["context"] + "\n\n"]
        # 与 retention 协议一致：target 先注入，unrelated 逐步追加（所有组共用同一批）
        unrel = unrelated_data[: n_max_frag - 1]

        knowledge_latent = model_knowledge(
            model_obj=model, tokenizer=tokenizer,
            knowledge=texts[0], add_special_tokens=True)

        frag_idx = 1
        for n in range(2, n_max_frag + 1):
            ids = tokenizer(unrel[frag_idx - 1], return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(input_device)[:, :512]
            truncated = tokenizer.decode(ids[0]) + "\n\n"
            texts.append(truncated)
            temp = model_knowledge(model_obj=model, tokenizer=tokenizer,
                                   knowledge=truncated, add_special_tokens=False)
            knowledge_latent = maybe_forget_before_concat(
                knowledge_latent=knowledge_latent, new_knowledge_latent=temp,
                forget_strategy="perplexity",
                max_knowledge_tokens=args.max_knowledge_tokens)
            knowledge_latent = concatenate_knowledge_latent(knowledge_latent, temp)
            frag_idx += 1

            if n not in ns:
                continue

            indices = rebuild_indices_from_segment_lengths(
                knowledge_latent["segment_lengths"])
            prompt_ids = tokenizer("".join(texts) + question,
                                   return_tensors="pt")["input_ids"].to(input_device)
            ctx_tokens[n].append(prompt_ids.shape[1])
            mem_tokens[n].append(knowledge_latent["input_ids"].shape[1])

            def run_topk(k):
                dens = probe_densities_sdpa(
                    model_obj=model, tokenizer=tokenizer,
                    knowledge_latent=knowledge_latent, question=question,
                    layer_idx=args.reorder_base_layer, indices=indices,
                    query_mode=args.query_mode)
                reordered = defrag_from_densities(knowledge_latent, indices, dens, k)
                generate_with_hidden_prefix(
                    model_obj=model, tokenizer=tokenizer,
                    knowledge_latent=reordered, question=question,
                    max_new_tokens=args.max_new_tokens, ignore_eos=True)

            def run_vanilla():
                generate_with_hidden_prefix(
                    model_obj=model, tokenizer=tokenizer,
                    knowledge_latent=knowledge_latent, question=question,
                    max_new_tokens=args.max_new_tokens, ignore_eos=True)

            # 预热（每个测点、每条流水线各一次，含所有 K 的形状）
            generate_text_timed(model, prompt_ids, args.max_new_tokens)
            for k in keep_nums:
                run_topk(k)
            run_vanilla()

            for _ in range(args.repeats):
                times["full_context"][n].append(
                    timed(lambda: generate_text_timed(model, prompt_ids,
                                                      args.max_new_tokens)))
                for k in keep_nums:
                    times[f"top{k}"][n].append(timed(lambda: run_topk(k)))
                times["vanilla_latent"][n].append(timed(run_vanilla))

        del knowledge_latent
        torch.cuda.empty_cache()
        print(f"[group {gi+1}/{len(test_data)}] {time.time()-t0:.1f}s")

    # ===== 汇总 =====
    print(f"\n===== Per-query inference latency (s), {args.model_name}, "
          f"{args.groups} groups x {args.repeats} repeats =====")
    header = "| n | ctx tokens | mem tokens | " + " | ".join(
        {"full_context": "Full-context",
         "vanilla_latent": "Vanilla latent"}.get(c, f"MemDefrag Top-{c[3:]}")
        for c in configs) + " |"
    print(header)
    print("|" + "---|" * (3 + len(configs)))
    summary = {}
    for n in ns:
        row = [f"| {n} | {int(np.mean(ctx_tokens[n]))} | {int(np.mean(mem_tokens[n]))} "]
        for c in configs:
            v = np.array(times[c][n])
            summary[f"{c}@{n}"] = (float(v.mean()), float(v.std()))
            row.append(f"| {v.mean():.3f} ± {v.std():.3f} ")
        print("".join(row) + "|")

    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump({"args": vars(args), "ns": ns,
                   "ctx_tokens": {n: float(np.mean(v)) for n, v in ctx_tokens.items()},
                   "mem_tokens": {n: float(np.mean(v)) for n, v in mem_tokens.items()},
                   "times": {c: {n: times[c][n] for n in ns} for c in configs}},
                  f, indent=1)
    print(f"saved: {args.out_prefix}.json")

    # ===== 论文风格折线图 =====
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    style = {
        "full_context": ("Full-context", "#4A4A4A", "d", "-"),
        "vanilla_latent": ("Vanilla latent memory", "#B0B0B0", "x", "--"),
    }
    palette = ["#2563EB", "#E45756", "#59A14F", "#F28E2B"]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for i, c in enumerate(configs):
        label, color, marker, ls = style.get(
            c, (f"MemDefrag (Top-{c[3:]})", palette[(i - 1) % len(palette)], "os^v"[(i - 1) % 4], "-"))
        means = [np.mean(times[c][n]) for n in ns]
        stds = [np.std(times[c][n]) for n in ns]
        ax.errorbar(ns, means, yerr=stds, label=label, color=color,
                    marker=marker, linestyle=ls, linewidth=2, markersize=6, capsize=3)
    ax.set_xlabel("Injected knowledge fragments ($n$)")
    ax.set_ylabel("Per-query inference time (s)")
    ax.set_xticks(ns)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    fig.savefig(f"{args.out_prefix}.pdf")
    print(f"saved: {args.out_prefix}.pdf")


if __name__ == "__main__":
    main()
