"""
sdpa_tracing.py
===============
SDPA 版 tracing + defrag（大模型单卡场景：不加载 eager 双模型副本）。

probe_densities_sdpa：模型全程 SDPA，前向到 tracer 层的 pre-hook 中用该层输入
手工计算 prompt 行注意力（RoPE + GQA + softcapping + 滑窗掩码，行 softmax 与
eager 全矩阵后取行数学等价），得到各 fragment 密度后立即终止前向。
defrag_from_densities：按密度升序重排 + Top-K 截断（与 utilities.reorder 的
张量重建逻辑一致），纯切片、无前向。
层排序正确性已在四个 7-9B 骨干模型上与论文 Table 1/5/6/7 的 eager 调查结果对照验证。
"""

import torch

from utilities import get_input_device, _register_hidden_prefix_hooks


def repeat_kv(hidden_states, n_rep):
    """GQA 的 KV 头展开：[B, KV, S, D] -> [B, KV*n_rep, S, D]。"""
    if n_rep == 1:
        return hidden_states
    b, kv, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, kv, n_rep, s, d)
    return hidden_states.reshape(b, kv * n_rep, s, d)


class _StopProbe(Exception):
    pass


def _rope(t, cos, sin):
    """对 [1, H, L, D] 张量应用 RoPE；cos/sin: [1, L, D]。"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    half = t.shape[-1] // 2
    rotated = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
    return t * cos + rotated * sin


def _block_ranges(indices, prefix_len):
    """与 utilities.reorder 相同的分块逻辑（去 bos 偏移后的 [start, end) 区间）。"""
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
    ranges, start = [], 0
    for end in cleaned_ends:
        if end > start:
            ranges.append((start, end))
            start = end
    return ranges


def probe_densities_sdpa(model_obj, tokenizer, knowledge_latent, question,
                         layer_idx, indices, query_mode="last"):
    """返回各 fragment 的注意力密度（tracer 层 layer_idx，last/mean 模式）。"""
    prefix_len = knowledge_latent["attention_mask"].shape[1]
    input_device = get_input_device(model_obj)

    question_ids = tokenizer(question, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(input_device)
    q_embeds = model_obj.model.embed_tokens(question_ids)
    dummy = torch.zeros((q_embeds.shape[0], prefix_len, q_embeds.shape[2]),
                        dtype=q_embeds.dtype, device=q_embeds.device)
    full_embeds = torch.cat([dummy, q_embeds], dim=1)
    attn_mask = torch.ones((1, full_embeds.shape[1]), dtype=torch.long,
                           device=input_device)

    cfg = model_obj.config
    num_heads = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // num_heads
    softcap = getattr(cfg, "attn_logit_softcapping", None)

    handles = _register_hidden_prefix_hooks(
        model_obj, knowledge_latent["hidden_states"], prefix_len)
    result = {}
    target_layer = model_obj.model.layers[layer_idx]

    def capture(module, h_args, h_kwargs):
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
        if softcap:
            scores = torch.tanh(scores / softcap) * softcap
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
        if query_mode == "last":
            result["a"] = probs[0, :, -1, :prefix_len].mean(dim=0)
        else:
            result["a"] = probs[0, :, :, :prefix_len].mean(dim=0).mean(dim=0)
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

    attn_wo_bos = result["a"][1:]
    densities = []
    for s, e in _block_ranges(indices, prefix_len):
        densities.append((attn_wo_bos[s:e].sum() / (e - s)).item())
    return densities


def defrag_from_densities(knowledge_latent, indices, densities, keep_num):
    """按密度重排 + Top-K 截断（keep_num<0 表示只重排不截断）。"""
    prefix_len = knowledge_latent["attention_mask"].shape[1]
    block_ranges = _block_ranges(indices, prefix_len)
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
