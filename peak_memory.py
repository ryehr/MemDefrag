"""
peak_memory.py
==============
Peak GPU memory 对比（论文 rebuttal Memory consumption 用）。

设置：Llama-3.1-8B-Instruct（单模型，SDPA），NaturalQA，n=50 片段，
perplexity 比例遗忘使 N_n = N_max = 12,800。

两条路径（每个 sample 分别 reset 峰值统计后测量）：
  - vanilla   : 以全量 12,800 记忆为前缀生成 32 token
  - MemDefrag : SDPA tracing（probe_density_optimized，仅算 prompt 行注意力）
                → 按密度重排 + Top-2 截断（纯张量切片）→ 以 Top-2 记忆生成 32 token
峰值含模型权重与常驻的 latent memory（真实部署口径）。
"""

import argparse
import json
import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utilities import (
    get_max_memory,
    generate_with_hidden_prefix,
)
from latency_breakdown import build_memory, probe_density_optimized

GB = 1024 ** 3


def defrag_from_densities(knowledge_latent, indices, densities, keep_num):
    """给定 SDPA 探针算出的密度，做重排 + Top-K 截断（与 reorder() 的张量重建逻辑一致）。"""
    prefix_len = knowledge_latent["attention_mask"].shape[1]

    reduced_ends = []
    for k in range(1, len(indices)):
        idx = int(indices[k])
        reduced_ends.append(max(0, min(idx - 1, prefix_len - 1)))
    cleaned_ends, prev = [], 0
    for e in reduced_ends:
        if e > prev:
            cleaned_ends.append(e)
            prev = e
    if len(cleaned_ends) == 0 or cleaned_ends[-1] < (prefix_len - 1):
        cleaned_ends.append(prefix_len - 1)
    block_ranges, start = [], 0
    for end in cleaned_ends:
        if end > start:
            block_ranges.append((start, end))
            start = end
    assert len(block_ranges) == len(densities)

    sorted_ids = sorted(range(len(block_ranges)), key=lambda i: densities[i])
    kept = sorted_ids[-keep_num:] if keep_num > 0 else sorted_ids
    slices = [(s + 1, e + 1) for s, e in block_ranges]  # +1 跳过 bos 偏移

    ids_parts = [knowledge_latent["input_ids"][:, :1]]
    am_parts = [knowledge_latent["attention_mask"][:, :1]]
    for b in kept:
        s, e = slices[b]
        ids_parts.append(knowledge_latent["input_ids"][:, s:e])
        am_parts.append(knowledge_latent["attention_mask"][:, s:e])
    hidden = []
    for layer_h in knowledge_latent["hidden_states"]:
        parts = [layer_h[:, :1, :]]
        for b in kept:
            s, e = slices[b]
            parts.append(layer_h[:, s:e, :])
        hidden.append(torch.cat(parts, dim=1))
    new_ids = torch.cat(ids_parts, dim=1)
    return {
        "input_ids": new_ids,
        "attention_mask": torch.cat(am_parts, dim=1),
        "hidden_states": tuple(hidden),
        "segment_lengths": [new_ids.shape[1]],
        "perplexity_scores": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--reorder_base_layer", type=int, default=13)
    parser.add_argument("--query_mode", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--keep_num", type=int, default=2)
    parser.add_argument("--n_fragments", type=int, default=50)
    parser.add_argument("--max_knowledge_tokens", type=int, default=12800)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory=get_max_memory()).eval()
    weights_gb = torch.cuda.memory_allocated() / GB
    print(f"model weights allocated: {weights_gb:.2f} GB")

    with open("./data_new/nqa/val_data.json", encoding="utf-8-sig") as f:
        test_data = json.load(f)
    with open("./data_new/nqa/unrelated_contexts_from_train.json", encoding="utf-8-sig") as f:
        unrelated_data = json.load(f)
    random.shuffle(test_data)
    random.shuffle(unrelated_data)
    test_data = [it for it in test_data
                 if len(tokenizer(it["context"])["input_ids"]) <= 512][: args.n_samples]
    unrelated_texts = unrelated_data[: args.n_fragments - 1]

    peaks = {"baseline": [], "vanilla": [], "memdefrag": []}

    for gi, item in enumerate(test_data):
        question = item["question"] + " Answer: "
        context = item["context"] + "\n\n"
        knowledge_latent, indices = build_memory(
            model, tokenizer, context, unrelated_texts, args.max_knowledge_tokens)
        n_tok = knowledge_latent["input_ids"].shape[1]

        if gi == 0:  # warmup（cuBLAS workspace 等一次性分配计入后续两条路径之前）
            probe_density_optimized(model, tokenizer, knowledge_latent, question,
                                    args.reorder_base_layer, indices, args.query_mode)
            generate_with_hidden_prefix(model, tokenizer, knowledge_latent, question,
                                        max_new_tokens=2, ignore_eos=True)

        # ---- 常驻基线：权重 + 存储的 latent memory ----
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated() / GB
        peaks["baseline"].append(baseline)

        # ---- vanilla：全量记忆生成 ----
        torch.cuda.reset_peak_memory_stats()
        generate_with_hidden_prefix(model, tokenizer, knowledge_latent, question,
                                    max_new_tokens=args.max_new_tokens, ignore_eos=True)
        torch.cuda.synchronize()
        peak_v = torch.cuda.max_memory_allocated() / GB
        peaks["vanilla"].append(peak_v)

        # ---- MemDefrag：SDPA tracing -> defrag(Top-K) -> 生成 ----
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        densities = probe_density_optimized(
            model, tokenizer, knowledge_latent, question,
            args.reorder_base_layer, indices, args.query_mode)
        reordered = defrag_from_densities(knowledge_latent, indices, densities, args.keep_num)
        generate_with_hidden_prefix(model, tokenizer, reordered, question,
                                    max_new_tokens=args.max_new_tokens, ignore_eos=True)
        torch.cuda.synchronize()
        peak_m = torch.cuda.max_memory_allocated() / GB
        peaks["memdefrag"].append(peak_m)

        print(f"[sample {gi+1}] N={n_tok}, topk_len={reordered['input_ids'].shape[1]}, "
              f"baseline={baseline:.2f} GB, vanilla peak={peak_v:.2f} GB, "
              f"MemDefrag peak={peak_m:.2f} GB")

        del knowledge_latent, reordered
        torch.cuda.empty_cache()

    def fmt(key):
        v = peaks[key]
        return f"{np.mean(v):.2f} ± {np.std(v):.2f}"

    latent_gb = peaks["baseline"][0] - weights_gb
    print(f"\n===== Peak GPU memory (GB, allocated), Llama-3.1-8B, NaturalQA, "
          f"n={args.n_fragments}, N=12800, {args.n_samples} samples =====")
    print(f"| Path | Peak (GB) |\n|---|---|")
    print(f"| resident baseline (weights {weights_gb:.2f} + latent memory {latent_gb:.2f}) "
          f"| {fmt('baseline')} |")
    print(f"| Vanilla (generation w/ full memory) | {fmt('vanilla')} |")
    print(f"| MemDefrag Top-{args.keep_num} (tracing + defrag + generation) | {fmt('memdefrag')} |")
    dv = np.mean(peaks["vanilla"]) - np.mean(peaks["baseline"])
    dm = np.mean(peaks["memdefrag"]) - np.mean(peaks["baseline"])
    print(f"\nTransient overhead over resident baseline: "
          f"vanilla +{dv:.2f} GB, MemDefrag +{dm:.2f} GB")


if __name__ == "__main__":
    main()
