"""
big_model_layer_sweep.py
========================
在更大模型上做 NaturalQA 全层 tracer 扫描（不限 [L/3, L/2] 带）。

协议（镜像 QA_benchmark_investigation_rank.py，规模缩减）：
  - 每组：target 知识 + 19 条固定 unrelated 知识（各 ≤512 token）独立编码；
  - target 轮换放置在 --positions 指定的位置（默认 0,4,9,14,19，共 5 个）；
  - 每个 (组, 位置) 一次全深度 eager 前向，hook 内把每层注意力当场归约为
    prompt 行向量（last-token 与 all-token 两种模式一次前向同时得到），
    不缓存任何 S×S 矩阵 —— 大模型显存可控；
  - 统计每层每模式下 target 片段的密度排名：mean rank、Top-1..5 准确率。

用法：
  python big_model_layer_sweep.py --model_name Qwen/Qwen2.5-32B-Instruct
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sdpa_tracing import repeat_kv

from utilities import (
    get_max_memory,
    get_input_device,
    model_knowledge,
    place_target_at_position,
    _register_hidden_prefix_hooks,
)

MODES = ["last", "mean"]


def _rope(t, cos, sin):
    """对 [1, H, L, D] 张量应用 RoPE；cos/sin: [1, L, D]。"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    half = t.shape[-1] // 2
    rotated = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
    return t * cos + rotated * sin


def all_layer_densities_both_modes(model_obj, tokenizer, knowledge_latent, question,
                                   tracing="eager"):
    """
    一次前向得到每层 last / mean 两种模式的 [prefix_len] 行向量并计算分段密度。
    tracing="eager"：HF eager 注意力（各架构语义精确，含 softcapping 等），
                     hook 内立即归约、丢弃 S×S 大矩阵；
    tracing="sdpa" ：模型全程 SDPA，不物化任何 S×S 矩阵；在每层 pre-hook 中
                     用该层输入手工计算 prompt 行的注意力（RoPE+GQA+行 softmax，
                     数学上与 eager 全矩阵后取行一致；适用标准 RoPE+GQA decoder 架构）。
    返回 {mode: [num_layers][num_segments] densities}。
    """
    prefix_len = knowledge_latent["attention_mask"].shape[1]
    num_layers = len(model_obj.model.layers)
    segment_lengths = knowledge_latent["segment_lengths"]
    input_device = get_input_device(model_obj)

    question_ids = tokenizer(question, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(input_device)
    q_embeds = model_obj.model.embed_tokens(question_ids)
    dummy = torch.zeros((q_embeds.shape[0], prefix_len, q_embeds.shape[2]),
                        dtype=q_embeds.dtype, device=q_embeds.device)
    full_embeds = torch.cat([dummy, q_embeds], dim=1)
    attn_mask = torch.ones((1, full_embeds.shape[1]), dtype=torch.long,
                           device=input_device)

    handles = _register_hidden_prefix_hooks(
        model_obj, knowledge_latent["hidden_states"], prefix_len)

    reduced = {m: {} for m in MODES}
    orig_forwards, cap_handles = {}, []

    if tracing == "eager":
        for li in range(num_layers):
            attn_module = model_obj.model.layers[li].self_attn
            orig_forwards[li] = attn_module.forward

            def make_wrapped(orig_fwd):
                def wrapped(*a, **kw):
                    kw["output_attentions"] = True
                    return orig_fwd(*a, **kw)
                return wrapped
            attn_module.forward = make_wrapped(orig_forwards[li])

        def make_hook(li):
            def hook(module, inputs, outputs):
                if isinstance(outputs, tuple) and len(outputs) >= 2 and outputs[1] is not None:
                    aw = outputs[1]  # [1, H, S, S]
                    reduced["last"][li] = aw[0, :, -1, :prefix_len].mean(dim=0).float().detach()
                    reduced["mean"][li] = (aw[0, :, prefix_len:, :prefix_len]
                                           .mean(dim=0).mean(dim=0).float().detach())
            return hook

        cap_handles = [
            model_obj.model.layers[li].self_attn.register_forward_hook(make_hook(li))
            for li in range(num_layers)]
    else:  # sdpa：每层 pre-hook 手工计算 prompt 行注意力
        cfg = model_obj.config
        num_heads = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // num_heads

        def make_pre_hook(li):
            def hook(module, h_args, h_kwargs):
                h = h_args[0] if h_args else h_kwargs["hidden_states"]
                S = h.shape[1]
                p = S - prefix_len
                x = module.input_layernorm(h)
                attn = module.self_attn
                q = attn.q_proj(x[:, -p:]).view(1, p, num_heads, head_dim).transpose(1, 2)
                k = attn.k_proj(x).view(1, S, num_kv, head_dim).transpose(1, 2)
                pos_emb = h_kwargs.get("position_embeddings", None)
                if pos_emb is not None:
                    cos, sin = pos_emb
                else:
                    pos_ids = torch.arange(S, device=h.device).unsqueeze(0)
                    cos, sin = model_obj.model.rotary_emb(x, pos_ids)
                q = _rope(q, cos[:, -p:], sin[:, -p:])
                k = _rope(k, cos, sin)
                k = repeat_kv(k, num_heads // num_kv)
                scaling = getattr(attn, "scaling", None) or head_dim ** -0.5
                scores = (q @ k.transpose(-1, -2)) * scaling  # [1, H, p, S]
                # Gemma-2：attention logit softcapping（先 capping 再加 mask，与 HF eager 一致）
                softcap = getattr(model_obj.config, "attn_logit_softcapping", None)
                if softcap:
                    scores = torch.tanh(scores / softcap) * softcap
                # 滑窗层（如 Gemma-2 交替层，window=4096）：query 只能看见近 window 的 key
                sw = getattr(attn, "sliding_window", None)
                if sw and S > sw:
                    kpos = torch.arange(S, device=h.device)
                    qpos = torch.arange(S - p, S, device=h.device)
                    too_far = (qpos.unsqueeze(1) - kpos.unsqueeze(0)) >= sw
                    scores = scores.masked_fill(too_far.view(1, 1, p, S), float("-inf"))
                causal = torch.triu(
                    torch.full((p, p), float("-inf"), device=h.device), diagonal=1)
                scores[:, :, :, prefix_len:] = scores[:, :, :, prefix_len:] + causal
                probs = torch.softmax(scores.float(), dim=-1)
                reduced["last"][li] = probs[0, :, -1, :prefix_len].mean(dim=0).detach()
                reduced["mean"][li] = (probs[0, :, :, :prefix_len]
                                       .mean(dim=0).mean(dim=0).detach())
            return hook

        cap_handles = [
            model_obj.model.layers[li].register_forward_pre_hook(
                make_pre_hook(li), with_kwargs=True)
            for li in range(num_layers)]

    try:
        with torch.no_grad():
            model_obj(inputs_embeds=full_embeds, attention_mask=attn_mask,
                      output_attentions=False, use_cache=False, return_dict=True)
    finally:
        for ch in cap_handles:
            ch.remove()
        for li in orig_forwards:
            model_obj.model.layers[li].self_attn.forward = orig_forwards[li]
        for h in handles:
            h.remove()

    seg_ranges, cum = [], 0
    for s_idx, seg_len in enumerate(segment_lengths):
        seg_ranges.append((1, seg_len) if s_idx == 0 else (cum, cum + seg_len))
        cum += seg_len

    out = {}
    for m in MODES:
        per_layer = []
        for li in range(num_layers):
            vec = reduced[m].get(li)
            if vec is None:
                per_layer.append([0.0] * len(segment_lengths))
                continue
            vec = vec.to("cpu")
            per_layer.append([
                (vec[s:e].sum() / (e - s)).item() if e > s else 0.0
                for s, e in seg_ranges])
        out[m] = per_layer
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="nqa")
    parser.add_argument("--group_num", type=int, default=100)
    parser.add_argument("--unrelated_num", type=int, default=19)
    parser.add_argument("--positions", type=str, default="0,4,9,14,19")
    parser.add_argument("--tracing", type=str, default="sdpa", choices=["sdpa", "eager"],
                        help="sdpa（默认，推荐）=不物化 S×S，每层手工算 prompt 行注意力，"
                             "显存开销小、速度快；eager=HF eager 注意力+hook 归约（参考实现）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    positions = [int(p) for p in args.positions.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto",
                       max_memory=get_max_memory())
    if args.tracing == "eager":
        load_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs).eval()
    num_layers = len(model.model.layers)
    print(f"model loaded: {num_layers} layers, tracing={args.tracing}")
    input_device = get_input_device(model)

    path = f"./data_new/{args.dataset}/"
    with open(path + "val_data.json", encoding="utf-8-sig") as f:
        test_data = json.load(f)
    with open(path + "unrelated_contexts_from_train.json", encoding="utf-8-sig") as f:
        unrelated_data = json.load(f)
    random.shuffle(test_data)
    random.shuffle(unrelated_data)
    test_data = [it for it in test_data
                 if len(tokenizer(it["context"])["input_ids"]) <= 512][: args.group_num]
    unrelated_texts = unrelated_data[: args.unrelated_num]

    # 19 条 unrelated 对所有组相同：只编码一次
    unrelated_latents = []
    for text in unrelated_texts:
        ids = tokenizer(text, return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(input_device)[:, :512]
        truncated = tokenizer.decode(ids[0]) + "\n\n"
        unrelated_latents.append(model_knowledge(
            model_obj=model, tokenizer=tokenizer, knowledge=truncated,
            add_special_tokens=False))
    print(f"{len(unrelated_latents)} unrelated latents formed")

    # ranks[mode][layer] -> list of target ranks (1-based)
    ranks = {m: [[] for _ in range(num_layers)] for m in MODES}

    for gi, item in enumerate(test_data):
        t0 = time.time()
        question = item["question"] + " Answer: "
        context = item["context"] + "\n\n"
        target_latent = model_knowledge(
            model_obj=model, tokenizer=tokenizer, knowledge=context,
            add_special_tokens=True)
        latents_list = [target_latent] + unrelated_latents

        for pos in positions:
            combined, target_seg = place_target_at_position(
                knowledge_latents_list=latents_list, target_idx=0, position=pos)
            dens = all_layer_densities_both_modes(model, tokenizer, combined, question,
                                                  tracing=args.tracing)
            for m in MODES:
                for li in range(num_layers):
                    d = dens[m][li]
                    order = np.argsort(d)[::-1]  # 降序
                    rank = int(np.where(order == target_seg)[0][0]) + 1
                    ranks[m][li].append(rank)
            del combined
            torch.cuda.empty_cache()

        del target_latent
        torch.cuda.empty_cache()
        print(f"[group {gi+1}/{len(test_data)}] {time.time()-t0:.1f}s")

        if (gi + 1) % 10 == 0 or gi == len(test_data) - 1:
            for m in MODES:
                mean_ranks = [np.mean(r) for r in ranks[m]]
                best = int(np.argmin(mean_ranks))
                top1 = np.mean([1 for r in ranks[m][best] if r == 1]) * 100 \
                    if ranks[m][best] else 0
                print(f"  [{m}] best layer so far: {best} "
                      f"(rank {mean_ranks[best]:.2f}, top1 {top1:.1f}%)")

    # ===== 汇总 =====
    band_lo, band_hi = num_layers // 3, (num_layers + 1) // 2
    results = {}
    for m in MODES:
        table = []
        for li in range(num_layers):
            r = np.array(ranks[m][li])
            row = dict(layer=li, mean_rank=float(r.mean()),
                       **{f"top{k}": float((r <= k).mean() * 100) for k in (1, 2, 3, 4, 5)})
            table.append(row)
        table.sort(key=lambda x: x["mean_rank"])
        results[m] = table
        print(f"\n===== {args.model_name} ({num_layers} layers), {m}-token attention, "
              f"n={len(ranks[m][0])} (groups x positions), "
              f"band [L/3, L/2] = [{band_lo}, {band_hi}] =====")
        print("layer  rank   top1   top2   top3   top4   top5   in-band")
        for row in table[:10]:
            in_band = band_lo <= row["layer"] <= band_hi
            print(f"{row['layer']:>5d}  {row['mean_rank']:5.2f}  "
                  + "  ".join(f"{row[f'top{k}']:5.1f}" for k in (1, 2, 3, 4, 5))
                  + f"   {'yes' if in_band else 'NO'}")

    out_json = args.out_json or f"./adaptive_logs/bigmodel_sweep_{args.model_name.split('/')[-1]}.json"
    with open(out_json, "w") as f:
        json.dump({"model": args.model_name, "num_layers": num_layers,
                   "groups": len(test_data), "positions": positions,
                   "results": results}, f, indent=1)
    print(f"\nsaved: {out_json}")


if __name__ == "__main__":
    main()
