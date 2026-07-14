"""
latency_breakdown.py
====================
Per-prompt latency 分解（论文 rebuttal Runtime breakdown 用）。

设置：Llama-3.1-8B-Instruct，NaturalQA，n=50 个记忆片段，
perplexity 比例遗忘使 N_n = N_max = 12,800（与 50 步知识保持实验的末态一致）。

测量组件（每组记忆重复 repeats 次，报告 mean±std, ms）：
  1. trace_forward    — 前向到 tracer 层（layer 13）并捕获注意力（含 question 编码与 hook 开销）
  2. reorder_filter   — 密度计算 + 排序 + 重排/Top-K 截断的张量重建
  3. gen_topk         — 以 Top-K 记忆为前缀生成 32 token（禁用 EOS 提前停止）
  4. MemDefrag total  = 1 + 2 + 3
对照：
  5. gen_full         — vanilla：以全量 12,800 记忆为前缀生成 32 token
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
    model_knowledge,
    concatenate_knowledge_latent,
    rebuild_indices_from_segment_lengths,
    maybe_forget_before_concat,
    generate_with_hidden_prefix,
    reorder,
    get_input_device,
    _register_hidden_prefix_hooks,
)
from transformers.models.llama.modeling_llama import repeat_kv


class _StopProbe(Exception):
    pass


def _rope(t, cos, sin):
    """对 [1, H, L, D] 张量应用 RoPE；cos/sin: [1, L, D]。"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    half = t.shape[-1] // 2
    rotated = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
    return t * cos + rotated * sin


def probe_density_optimized(model_obj, tokenizer, knowledge_latent, question,
                            layer_idx, indices, query_mode="last"):
    """
    优化版 tracer 探针：在 SDPA 模型上前向到 tracer 层，仅对 prompt 行手工计算
    注意力（softmax 逐行独立，结果与 eager 全矩阵后取行完全一致），
    不物化 S×S 注意力矩阵。返回 (densities, prefix_len)。
    """
    input_device = get_input_device(model_obj)
    question_ids = tokenizer(question, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(input_device)
    prefix_len = knowledge_latent["attention_mask"].shape[1]
    q_embeds = model_obj.model.embed_tokens(question_ids)
    dummy = torch.zeros((q_embeds.shape[0], prefix_len, q_embeds.shape[2]),
                        dtype=q_embeds.dtype, device=q_embeds.device)
    full_embeds = torch.cat([dummy, q_embeds], dim=1)
    attn_mask = torch.ones((1, full_embeds.shape[1]), dtype=torch.long, device=input_device)

    cfg = model_obj.config
    num_heads = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // num_heads

    handles = _register_hidden_prefix_hooks(
        model_obj, knowledge_latent["hidden_states"], prefix_len)
    result = {}
    target_layer = model_obj.model.layers[layer_idx]

    def capture(module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        S = h.shape[1]
        p = S - prefix_len
        x = module.input_layernorm(h)
        attn = module.self_attn

        if query_mode == "last":
            xq = x[:, -1:, :]
        else:
            xq = x[:, -p:, :]
        rows = xq.shape[1]
        q = attn.q_proj(xq).view(1, rows, num_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(1, S, num_kv, head_dim).transpose(1, 2)

        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None:
            cos, sin = pos_emb
        else:
            pos_ids = torch.arange(S, device=h.device).unsqueeze(0)
            cos, sin = model_obj.model.rotary_emb(x, pos_ids)
        q = _rope(q, cos[:, -rows:], sin[:, -rows:])
        k = _rope(k, cos, sin)
        k = repeat_kv(k, num_heads // num_kv)

        scores = (q @ k.transpose(-1, -2)) / (head_dim ** 0.5)  # [1, H, rows, S]
        if query_mode == "mean":
            # prompt 行之间的 causal mask（prefix 列都在所有 prompt 位置之前，无需 mask）
            causal = torch.full((rows, rows), float("-inf"), device=h.device)
            causal = torch.triu(causal, diagonal=1)
            scores[:, :, :, prefix_len:] = scores[:, :, :, prefix_len:] + causal
        probs = torch.softmax(scores.float(), dim=-1)
        if query_mode == "last":
            a = probs[0, :, -1, :prefix_len].mean(dim=0)          # [prefix_len]
        else:
            a = probs[0, :, :, :prefix_len].mean(dim=0).mean(dim=0)
        result["a"] = a
        raise _StopProbe()

    cap_handle = target_layer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            try:
                model_obj(inputs_embeds=full_embeds, attention_mask=attn_mask,
                          use_cache=False, return_dict=True)
            except _StopProbe:
                pass
    finally:
        cap_handle.remove()
        for hd in handles:
            hd.remove()

    # 与 reorder 相同的分块与密度计算逻辑
    attn_wo_bos = result["a"][1:]
    reduced_ends = []
    for kk in range(1, len(indices)):
        idx = int(indices[kk])
        end_exclusive = max(0, min(idx - 1, prefix_len - 1))
        reduced_ends.append(end_exclusive)
    cleaned_ends, prev = [], 0
    for e in reduced_ends:
        if e > prev:
            cleaned_ends.append(e)
            prev = e
    if len(cleaned_ends) == 0 or cleaned_ends[-1] < (prefix_len - 1):
        cleaned_ends.append(prefix_len - 1)
    densities, start = [], 0
    for end in cleaned_ends:
        if end > start:
            densities.append((attn_wo_bos[start:end].sum() / (end - start)).item())
            start = end
    return densities


def build_memory(model_gen, tokenizer, target_context, unrelated_texts, max_knowledge_tokens):
    """按 QA_benchmark 的记忆演化流程构建 n=50 片段、N=N_max 的 latent memory。"""
    input_device = get_input_device(model_gen)
    knowledge_latent = model_knowledge(
        model_obj=model_gen, tokenizer=tokenizer,
        knowledge=target_context, add_special_tokens=True,
    )
    for unrelated_text in unrelated_texts:
        ids = tokenizer(unrelated_text, return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(input_device)[:, :512]
        truncated = tokenizer.decode(ids[0]) + "\n\n"
        temp = model_knowledge(model_obj=model_gen, tokenizer=tokenizer,
                               knowledge=truncated, add_special_tokens=False)
        knowledge_latent = maybe_forget_before_concat(
            knowledge_latent=knowledge_latent, new_knowledge_latent=temp,
            forget_strategy="perplexity", max_knowledge_tokens=max_knowledge_tokens,
        )
        knowledge_latent = concatenate_knowledge_latent(knowledge_latent, temp)
    indices = rebuild_indices_from_segment_lengths(knowledge_latent["segment_lengths"])
    return knowledge_latent, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--reorder_base_layer", type=int, default=13)
    parser.add_argument("--query_mode", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--keep_num", type=int, default=2)
    parser.add_argument("--n_fragments", type=int, default=50)
    parser.add_argument("--max_knowledge_tokens", type=int, default=12800)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--n_groups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_raw", action="store_true",
                        help="打印每次测量的原始 trace_forward/gen 时间，用于诊断方差来源")
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dtype = torch.bfloat16
    max_memory = get_max_memory()
    model_reorder = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto",
        max_memory=max_memory, attn_implementation="eager").eval()
    model_gen = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto",
        max_memory=max_memory).eval()

    with open("./data_new/nqa/val_data.json", encoding="utf-8-sig") as f:
        test_data = json.load(f)
    with open("./data_new/nqa/unrelated_contexts_from_train.json", encoding="utf-8-sig") as f:
        unrelated_data = json.load(f)
    random.shuffle(test_data)
    random.shuffle(unrelated_data)
    test_data = [it for it in test_data
                 if len(tokenizer(it["context"])["input_ids"]) <= 512][: args.n_groups]
    unrelated_texts = unrelated_data[: args.n_fragments - 1]

    stats = {k: [] for k in ["trace_forward", "trace_forward_opt",
                             "reorder_filter", "gen_topk", "gen_full"]}
    len_full, len_topk = [], []

    for gi, item in enumerate(test_data):
        question = item["question"] + " Answer: "
        context = item["context"] + "\n\n"
        knowledge_latent, indices = build_memory(
            model_gen, tokenizer, context, unrelated_texts, args.max_knowledge_tokens)
        n_seg = len(knowledge_latent["segment_lengths"])
        n_tok = knowledge_latent["input_ids"].shape[1]
        print(f"[group {gi+1}] memory built: {n_seg} fragments, {n_tok} tokens")

        # 每组预热一遍完整流水线：排除 CUDA kernel 编译与显存分配器的首次开销
        # （传 timings={} 使 warmup 也走"不 empty_cache"的稳态路径，预热分配器池）
        r = reorder(model_reorder, tokenizer, knowledge_latent, question,
                    args.reorder_base_layer, indices, keep_num=args.keep_num,
                    query_mode=args.query_mode, timings={})
        probe_density_optimized(model_gen, tokenizer, knowledge_latent, question,
                                args.reorder_base_layer, indices, args.query_mode)
        generate_with_hidden_prefix(model_gen, tokenizer, r, question,
                                    max_new_tokens=args.max_new_tokens, ignore_eos=True)
        generate_with_hidden_prefix(model_gen, tokenizer, knowledge_latent, question,
                                    max_new_tokens=args.max_new_tokens, ignore_eos=True)

        for rep in range(args.repeats):
            t = {}
            reordered = reorder(model_reorder, tokenizer, knowledge_latent, question,
                                args.reorder_base_layer, indices, keep_num=args.keep_num,
                                query_mode=args.query_mode, timings=t)
            stats["trace_forward"].append(t["trace_forward"])
            stats["reorder_filter"].append(t["reorder_filter"])

            torch.cuda.synchronize(); t0 = time.perf_counter()
            dens_opt = probe_density_optimized(
                model_gen, tokenizer, knowledge_latent, question,
                args.reorder_base_layer, indices, args.query_mode)
            torch.cuda.synchronize()
            stats["trace_forward_opt"].append(time.perf_counter() - t0)

            if rep == 0:  # 验证优化探针与 eager 路径的密度一致性
                dens_eager = t["densities"]
                order_match = (np.argsort(dens_eager).tolist()
                               == np.argsort(dens_opt).tolist())
                topk_match = (set(np.argsort(dens_eager)[-args.keep_num:])
                              == set(np.argsort(dens_opt)[-args.keep_num:]))
                max_rel = max(abs(a - b) / max(abs(b), 1e-12)
                              for a, b in zip(dens_opt, dens_eager))
                print(f"  [validate] full-ranking match={order_match}, "
                      f"Top-{args.keep_num} match={topk_match}, max rel diff={max_rel:.2e}")

            torch.cuda.synchronize(); t0 = time.perf_counter()
            generate_with_hidden_prefix(model_gen, tokenizer, reordered, question,
                                        max_new_tokens=args.max_new_tokens, ignore_eos=True)
            torch.cuda.synchronize(); stats["gen_topk"].append(time.perf_counter() - t0)

            torch.cuda.synchronize(); t0 = time.perf_counter()
            generate_with_hidden_prefix(model_gen, tokenizer, knowledge_latent, question,
                                        max_new_tokens=args.max_new_tokens, ignore_eos=True)
            torch.cuda.synchronize(); stats["gen_full"].append(time.perf_counter() - t0)

            len_full.append(n_tok)
            len_topk.append(reordered["input_ids"].shape[1])

        if args.print_raw:
            k = args.repeats
            print(f"  raw trace_forward (ms): "
                  f"{[f'{v*1e3:.0f}' for v in stats['trace_forward'][-k:]]}")
            print(f"  raw gen_topk (ms):      "
                  f"{[f'{v*1e3:.0f}' for v in stats['gen_topk'][-k:]]}")
            print(f"  raw gen_full (ms):      "
                  f"{[f'{v*1e3:.0f}' for v in stats['gen_full'][-k:]]}")

        del knowledge_latent
        torch.cuda.empty_cache()

    ms = {k: (np.mean(v) * 1e3, np.std(v) * 1e3, np.median(v) * 1e3)
          for k, v in stats.items()}

    def total_of(trace_key):
        samples = [a + b + c for a, b, c in
                   zip(stats[trace_key], stats["reorder_filter"], stats["gen_topk"])]
        return np.mean(samples) * 1e3, np.std(samples) * 1e3, np.median(samples) * 1e3

    total_eager = total_of("trace_forward")
    total_opt = total_of("trace_forward_opt")
    trace_pct = ms["trace_forward"][0] / total_eager[0] * 100
    trace_pct_opt = ms["trace_forward_opt"][0] / total_opt[0] * 100
    speedup = ms["gen_full"][0] / ms["gen_topk"][0]

    print("\n===== Per-prompt latency breakdown "
          f"(Llama-3.1-8B, NaturalQA, n={args.n_fragments}, N={int(np.mean(len_full))}, "
          f"Top-{args.keep_num} => {int(np.mean(len_topk))} tokens, "
          f"{args.max_new_tokens} generated tokens) =====")
    rows = [
        (f"Tracing forward to layer {args.reorder_base_layer} (as implemented, eager)",
         ms["trace_forward"]),
        (f"Tracing forward to layer {args.reorder_base_layer} (optimized probe, SDPA)",
         ms["trace_forward_opt"]),
        ("Reorder + Top-K filter", ms["reorder_filter"]),
        (f"Generation w/ Top-{args.keep_num} memory", ms["gen_topk"]),
        ("MemDefrag total (as implemented)", total_eager),
        ("MemDefrag total (optimized probe)", total_opt),
        ("Vanilla generation w/ full memory", ms["gen_full"]),
    ]
    print(f"\n| Component | Latency mean ± std (ms) | median (ms) |\n|---|---|---|")
    for name, (m, s, med) in rows:
        print(f"| {name} | {m:.1f} ± {s:.1f} | {med:.1f} |")
    print(f"\nTracing share of MemDefrag total: {trace_pct:.1f}% (as implemented) / "
          f"{trace_pct_opt:.1f}% (optimized)")
    print(f"Reorder+filter: {ms['reorder_filter'][0]:.1f} ms")
    print(f"Generation speedup (full / Top-{args.keep_num}): {speedup:.2f}x")
    print(f"MemDefrag total vs vanilla full-memory generation: "
          f"{ms['gen_full'][0] / total_eager[0]:.2f}x (as implemented), "
          f"{ms['gen_full'][0] / total_opt[0]:.2f}x (optimized)")

    print("\n%% LaTeX")
    for name, (m, s, _) in rows:
        print(f"{name} & {m:.1f} $\\pm$ {s:.1f} \\\\")


if __name__ == "__main__":
    main()
