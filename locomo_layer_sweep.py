"""
locomo_layer_sweep.py
=====================
在 LoCoMo（conv-30, single-hop）上做 tracer 层扫描：
对每道题一次 eager 前向捕获全部层的注意力，计算各层 hit@K
（evidence 所在 session 是否进入该层密度 Top-K），
以确定 LoCoMo 域上的最优 tracer 层（论文的 [L/3, L/2] 带内轻量扫描）。
需 4 卡（全层注意力捕获约 330GB，device_map=auto 分摊）。
"""

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utilities import (
    get_max_memory,
    get_input_device,
    _register_hidden_prefix_hooks,
)
from LoCoMo_benchmark import load_locomo, build_memory, parse_evidence_sessions


def all_layer_densities_reduced(model_obj, tokenizer, knowledge_latent, question,
                                query_mode="last"):
    """
    一次 eager 前向，捕获每层注意力后在 hook 内立即归约为 [prefix_len] 行向量
    （避免缓存 num_layers 个 S×S 大矩阵导致 OOM），返回 [num_layers][num_segments] 密度。
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

    orig_forwards, reduced = {}, {}
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
                if query_mode == "last":
                    vec = aw[0, :, -1, :prefix_len].mean(dim=0)
                else:
                    vec = aw[0, :, prefix_len:, :prefix_len].mean(dim=0).mean(dim=0)
                reduced[li] = vec.float().detach()
        return hook

    cap_handles = [model_obj.model.layers[li].self_attn.register_forward_hook(make_hook(li))
                   for li in range(num_layers)]
    try:
        with torch.no_grad():
            model_obj(inputs_embeds=full_embeds, attention_mask=attn_mask,
                      output_attentions=False, use_cache=False, return_dict=True)
    finally:
        for ch in cap_handles:
            ch.remove()
        for li in range(num_layers):
            model_obj.model.layers[li].self_attn.forward = orig_forwards[li]
        for h in handles:
            h.remove()

    seg_ranges, cum = [], 0
    for s_idx, seg_len in enumerate(segment_lengths):
        seg_ranges.append((1, seg_len) if s_idx == 0 else (cum, cum + seg_len))
        cum += seg_len

    all_seg = []
    for li in range(num_layers):
        vec = reduced.get(li)
        if vec is None:
            all_seg.append([0.0] * len(segment_lengths))
            continue
        all_seg.append([
            (vec[s:e].sum() / (e - s)).item() if e > s else 0.0
            for s, e in seg_ranges])
    return all_seg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data_path", type=str, default="./data_new/locomo/locomo10.json")
    parser.add_argument("--category", type=str, default="4")
    parser.add_argument("--max_conv_tokens", type=int, default=16000)
    parser.add_argument("--conv_ids", type=str, default="conv-30")
    parser.add_argument("--query_mode", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--max_knowledge_tokens", type=int, default=12800)
    parser.add_argument("--topk_list", type=str, default="1,2,3")
    args = parser.parse_args()
    print(args)

    topks = [int(k) for k in args.topk_list.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory=get_max_memory(), attn_implementation="eager").eval()
    num_layers = len(model.model.layers)

    conv_ids = set(args.conv_ids.split(",")) if args.conv_ids else None
    convs = load_locomo(args.data_path, tokenizer, args.category,
                        args.max_conv_tokens, conv_ids)
    print(f"conversations: {[(c[0], c[3], len(c[2])) for c in convs]}")

    # hits[layer][k] 累计
    hits = np.zeros((num_layers, len(topks)))
    n_q = 0

    for sample_id, frags, qa, _ in convs:
        latent, indices = build_memory(model, tokenizer, frags,
                                       "perplexity", args.max_knowledge_tokens)
        session_ks = [k for k, _ in frags]
        print(f"[{sample_id}] memory {latent['input_ids'].shape[1]} tokens, "
              f"{len(frags)} segments")

        for qi, q in enumerate(qa):
            ev = parse_evidence_sessions(q.get("evidence", []))
            if not ev:
                continue
            ev_seg_ids = {session_ks.index(s) for s in ev if s in session_ks}
            if not ev_seg_ids:
                continue
            all_seg = all_layer_densities_reduced(
                model_obj=model, tokenizer=tokenizer, knowledge_latent=latent,
                question=q["question"] + " Answer: ", query_mode=args.query_mode)
            n_q += 1
            for li in range(num_layers):
                order = np.argsort(all_seg[li])  # 升序
                for kj, k in enumerate(topks):
                    topset = set(order[-k:])
                    if ev_seg_ids & topset:
                        hits[li][kj] += 1
            if (qi + 1) % 10 == 0:
                print(f"  {qi+1}/{len(qa)} questions done")
        del latent
        torch.cuda.empty_cache()

    print(f"\n===== Layer sweep ({args.query_mode}-token attention, n={n_q}) =====")
    print("layer  " + "  ".join(f"hit@{k:<2d}" for k in topks))
    rates = hits / max(n_q, 1) * 100
    for li in range(num_layers):
        print(f"{li:>5d}  " + "  ".join(f"{rates[li][kj]:6.1f}" for kj in range(len(topks))))
    best = np.argsort(-rates[:, min(1, len(topks) - 1)])[:8]
    print(f"\nTop-8 layers by hit@{topks[min(1, len(topks)-1)]}: "
          f"{[(int(l), round(float(rates[l][min(1, len(topks)-1)]), 1)) for l in best]}")


if __name__ == "__main__":
    main()
