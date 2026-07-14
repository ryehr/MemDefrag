import random
import time
import torch
import numpy as np
import string
from collections import Counter
import re
# ============================================================
#  基础工具
# ============================================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def get_input_device(model_obj):
    """
    对于 device_map='auto' 的模型，输入应放到 embedding 所在设备。
    """
    return model_obj.model.embed_tokens.weight.device


def col_one_ratio_np(mat):
    a = np.asarray(mat, dtype=np.int8)
    if a.size == 0:
        return []
    return (a.mean(axis=0)).tolist()


def get_max_memory():
    """
    为每张 GPU 预留一部分空间，避免 device_map 把显存吃得太满。
    """
    if not torch.cuda.is_available():
        return None

    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
        usable_gib = max(1, total_gib - 6)
        if i == 0:
            usable_gib = max(1, usable_gib - 4)
        max_memory[i] = f"{usable_gib}GiB"

    max_memory["cpu"] = "256GiB"
    return max_memory


# ============================================================
#  Token-level Perplexity
# ============================================================

def compute_token_perplexity_scores(model_obj, tokenizer, input_ids, attention_mask):
    """
    为 input_ids 的每一个位置计算一个 token-level perplexity 分数。
    在前面人为 prepend 一个 bos，使第一个 token 也有分数。
    返回 shape: [1, L]
    """
    input_device = get_input_device(model_obj)
    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)

    if tokenizer.bos_token_id is not None:
        prefix_id = tokenizer.bos_token_id
    elif tokenizer.eos_token_id is not None:
        prefix_id = tokenizer.eos_token_id
    else:
        prefix_id = input_ids[0, 0].item()

    prefix_ids = torch.tensor([[prefix_id]], dtype=input_ids.dtype, device=input_device)
    prefix_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=input_device)

    scorer_input_ids = torch.cat([prefix_ids, input_ids], dim=1)
    scorer_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    with torch.no_grad():
        outputs = model_obj(
            input_ids=scorer_input_ids,
            attention_mask=scorer_attention_mask,
            use_cache=False,
            return_dict=True
        )

    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids

    token_nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none"
    ).view_as(labels)

    token_ppl = torch.exp(torch.clamp(token_nll, max=30.0)).detach()
    return token_ppl


# ============================================================
#  Knowledge Latent 构造 / 拼接 / 索引
# ============================================================

def model_knowledge(model_obj, tokenizer, knowledge, add_special_tokens=True):
    """
    提取 knowledge 的层级前缀 hidden states（不保留最后一层输出）。
    同时计算每个 token 的 perplexity 分数。
    """
    inputs = tokenizer(
        knowledge,
        return_tensors="pt",
        add_special_tokens=add_special_tokens
    )
    input_device = get_input_device(model_obj)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_obj(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )

    kept_hidden_states = tuple(h.detach() for h in outputs.hidden_states[:-1])

    token_perplexity_scores = compute_token_perplexity_scores(
        model_obj=model_obj,
        tokenizer=tokenizer,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    knowledge_latent = {
        "input_ids": inputs["input_ids"].detach(),
        "attention_mask": inputs["attention_mask"].detach(),
        "hidden_states": kept_hidden_states,
        "perplexity_scores": token_perplexity_scores.detach(),
        "segment_lengths": [inputs["input_ids"].shape[1]],
    }
    return knowledge_latent


def concatenate_knowledge_latent(knowledge_latent, new_knowledge_latent):
    hidden_states = []
    for layer_id in range(len(knowledge_latent["hidden_states"])):
        hidden_states.append(
            torch.cat(
                [knowledge_latent["hidden_states"][layer_id],
                 new_knowledge_latent["hidden_states"][layer_id]],
                dim=1
            )
        )

    merged = {
        "input_ids": torch.cat(
            [knowledge_latent["input_ids"], new_knowledge_latent["input_ids"]], dim=1
        ),
        "attention_mask": torch.cat(
            [knowledge_latent["attention_mask"], new_knowledge_latent["attention_mask"]], dim=1
        ),
        "hidden_states": tuple(hidden_states),
        "segment_lengths": knowledge_latent["segment_lengths"] + new_knowledge_latent["segment_lengths"],
    }

    if (knowledge_latent.get("perplexity_scores", None) is not None
            and new_knowledge_latent.get("perplexity_scores", None) is not None):
        merged["perplexity_scores"] = torch.cat(
            [knowledge_latent["perplexity_scores"],
             new_knowledge_latent["perplexity_scores"]], dim=1
        )
    else:
        merged["perplexity_scores"] = None

    return merged


def rebuild_indices_from_segment_lengths(segment_lengths):
    """
    根据 segment_lengths 重建 indices。
    indices[0] = 1，后续每个元素表示截至当前 segment 结束时的总长度（包含 bos）。
    """
    indices = [1]
    cumulative = 0
    for seg_len in segment_lengths:
        cumulative += int(seg_len)
        indices.append(cumulative)
    return indices


# ============================================================
#  Forget 机制（剪枝）
# ============================================================

def _allocate_deletions_proportionally(segment_lengths, overflow):
    """
    根据每个 knowledge 的长度，按近似等比例分配删除数量。
    第一个 knowledge 的第一个位置（bos）不可删。
    """
    if overflow <= 0:
        return [0 for _ in segment_lengths]

    total_len = sum(segment_lengths)
    if total_len <= 0:
        return [0 for _ in segment_lengths]

    capacities = []
    for i, seg_len in enumerate(segment_lengths):
        if i == 0:
            capacities.append(max(0, seg_len - 1))
        else:
            capacities.append(max(0, seg_len))

    max_deletable = sum(capacities)
    target_delete = min(overflow, max_deletable)

    raw = [target_delete * seg_len / total_len for seg_len in segment_lengths]
    deletions = [min(int(np.floor(x)), cap) for x, cap in zip(raw, capacities)]

    assigned = sum(deletions)
    remain = target_delete - assigned

    while remain > 0:
        candidates = []
        for i in range(len(segment_lengths)):
            if deletions[i] < capacities[i]:
                frac = raw[i] - np.floor(raw[i])
                remaining_capacity = capacities[i] - deletions[i]
                candidates.append((frac, remaining_capacity, i))

        if len(candidates) == 0:
            break

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        progressed = False
        for _, _, i in candidates:
            if remain <= 0:
                break
            if deletions[i] < capacities[i]:
                deletions[i] += 1
                remain -= 1
                progressed = True

        if not progressed:
            break

    return deletions


def _build_keep_mask_by_strategy(knowledge_latent, deletions_per_segment, strategy):
    """
    根据 strategy（random / perplexity）构建全局 keep_mask。
    """
    total_len = knowledge_latent["input_ids"].shape[1]
    keep_mask = torch.ones(total_len, dtype=torch.bool)

    segment_lengths = knowledge_latent["segment_lengths"]
    perplexity_scores = knowledge_latent.get("perplexity_scores", None)

    start = 0
    for seg_id, (seg_len, del_num) in enumerate(zip(segment_lengths, deletions_per_segment)):
        end = start + seg_len

        if del_num <= 0:
            start = end
            continue

        if seg_id == 0:
            candidate_positions = list(range(start + 1, end))
        else:
            candidate_positions = list(range(start, end))

        if len(candidate_positions) == 0:
            start = end
            continue

        del_num = min(del_num, len(candidate_positions))
        if del_num <= 0:
            start = end
            continue

        if strategy == "random":
            delete_positions = random.sample(candidate_positions, del_num)
        elif strategy == "perplexity":
            if perplexity_scores is None:
                raise ValueError(
                    "forget_strategy='perplexity' 但 knowledge_latent 中没有 perplexity_scores"
                )
            seg_scores = perplexity_scores[0, candidate_positions]
            _, local_idx = torch.topk(seg_scores, k=del_num, largest=False)
            delete_positions = [candidate_positions[idx.item()] for idx in local_idx]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        keep_mask[delete_positions] = False
        start = end

    if total_len > 0:
        keep_mask[0] = True

    return keep_mask


def prune_knowledge_latent(knowledge_latent, deletions_per_segment, strategy):
    """
    按 deletions_per_segment 对当前 knowledge_latent 做删除。
    """
    if sum(deletions_per_segment) <= 0:
        return knowledge_latent

    keep_mask = _build_keep_mask_by_strategy(
        knowledge_latent=knowledge_latent,
        deletions_per_segment=deletions_per_segment,
        strategy=strategy
    )

    new_hidden_states = []
    for h in knowledge_latent["hidden_states"]:
        new_hidden_states.append(h[:, keep_mask, :])

    new_segment_lengths = []
    start = 0
    for seg_len in knowledge_latent["segment_lengths"]:
        end = start + seg_len
        new_segment_lengths.append(int(keep_mask[start:end].sum().item()))
        start = end

    pruned = {
        "input_ids": knowledge_latent["input_ids"][:, keep_mask],
        "attention_mask": knowledge_latent["attention_mask"][:, keep_mask],
        "hidden_states": tuple(new_hidden_states),
        "segment_lengths": new_segment_lengths,
    }

    if "perplexity_scores" in knowledge_latent and knowledge_latent["perplexity_scores"] is not None:
        pruned["perplexity_scores"] = knowledge_latent["perplexity_scores"][:, keep_mask]
    else:
        pruned["perplexity_scores"] = None

    return pruned


def maybe_forget_before_concat(knowledge_latent, new_knowledge_latent, forget_strategy, max_knowledge_tokens):
    """
    在拼接新 knowledge 前，根据 forget_strategy 决定是否对旧 knowledge 做删除。
    """
    if forget_strategy == "none":
        return knowledge_latent

    current_len = knowledge_latent["input_ids"].shape[1]
    new_len = new_knowledge_latent["input_ids"].shape[1]
    overflow = current_len + new_len - max_knowledge_tokens

    if overflow <= 0:
        return knowledge_latent

    deletions_per_segment = _allocate_deletions_proportionally(
        segment_lengths=knowledge_latent["segment_lengths"],
        overflow=overflow
    )

    knowledge_latent = prune_knowledge_latent(
        knowledge_latent=knowledge_latent,
        deletions_per_segment=deletions_per_segment,
        strategy=forget_strategy
    )
    return knowledge_latent


# ============================================================
#  Hidden Prefix Hooks & 生成
# ============================================================

def _register_hidden_prefix_hooks(model_obj, prefix_hidden_states, prefix_len):
    """
    对每一层注册 pre-hook：在该层 forward 前，将前 prefix_len 个位置的
    hidden states 替换成保存好的该层输入前缀。
    """
    handles = []
    decoder_layers = model_obj.model.layers
    num_layers = len(decoder_layers)

    if len(prefix_hidden_states) != num_layers:
        raise ValueError(
            f"len(prefix_hidden_states)={len(prefix_hidden_states)} != num_layers={num_layers}. "
            f'应该保存"embedding + 前 num_layers-1 层输出"，总数正好等于 num_layers。'
        )

    for layer_idx, layer in enumerate(decoder_layers):
        layer_prefix = prefix_hidden_states[layer_idx]

        def make_hook(saved_prefix):
            def hook(module, inputs):
                hidden_states = inputs[0]
                _, total_len, _ = hidden_states.shape

                if saved_prefix.shape[1] != prefix_len:
                    raise ValueError(
                        f"saved_prefix len {saved_prefix.shape[1]} != prefix_len {prefix_len}"
                    )
                if total_len < prefix_len:
                    raise ValueError(
                        f"total_len {total_len} < prefix_len {prefix_len}"
                    )

                new_hidden_states = hidden_states.clone()
                new_hidden_states[:, :prefix_len, :] = saved_prefix.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
                return (new_hidden_states,) + inputs[1:]

            return hook

        handles.append(layer.register_forward_pre_hook(make_hook(layer_prefix)))

    return handles


def prefill_question_with_hidden_prefix(model_obj, knowledge_latent, question_ids):
    """
    对 question 做一次 prefill：knowledge 通过 hidden prefix 注入，
    打开 use_cache=True，返回 logits 和 past_key_values。
    """
    prefix_hidden_states = knowledge_latent["hidden_states"]
    prefix_len = knowledge_latent["attention_mask"].shape[1]

    input_device = get_input_device(model_obj)
    question_ids = question_ids.to(input_device)
    question_attention_mask = torch.ones_like(question_ids, device=input_device)

    q_embeds = model_obj.model.embed_tokens(question_ids)

    prefix_embeds_dummy = torch.zeros(
        (q_embeds.shape[0], prefix_len, q_embeds.shape[2]),
        dtype=q_embeds.dtype,
        device=q_embeds.device
    )

    full_inputs_embeds = torch.cat([prefix_embeds_dummy, q_embeds], dim=1)
    full_attention_mask = torch.cat([
        torch.ones(
            (q_embeds.shape[0], prefix_len),
            dtype=question_attention_mask.dtype,
            device=input_device
        ),
        question_attention_mask
    ], dim=1)

    handles = _register_hidden_prefix_hooks(model_obj, prefix_hidden_states, prefix_len)

    try:
        with torch.no_grad():
            outputs = model_obj(
                inputs_embeds=full_inputs_embeds,
                attention_mask=full_attention_mask,
                use_cache=True,
                return_dict=True
            )
    finally:
        for h in handles:
            h.remove()

    return outputs


def generate_with_hidden_prefix(model_obj, tokenizer, knowledge_latent, question,
                                max_new_tokens=32, do_sample=False, ignore_eos=False):
    """
    生成逻辑：
    1. knowledge 以 hidden states 形式保存
    2. 对 question 做一次 hidden-prefix prefill，拿到 past_key_values
    3. 后续生成 answer 时，只输入上一个 token，复用 KV cache
    """
    input_device = get_input_device(model_obj)

    question_ids = tokenizer(
        question,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"].to(input_device)

    outputs = prefill_question_with_hidden_prefix(
        model_obj=model_obj,
        knowledge_latent=knowledge_latent,
        question_ids=question_ids
    )

    logits = outputs.logits
    past_key_values = outputs.past_key_values
    generated_answer_ids = []

    for _ in range(max_new_tokens):
        next_token_logits = logits[:, -1, :]

        if do_sample:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if (not ignore_eos and tokenizer.eos_token_id is not None
                and next_token_id.item() == tokenizer.eos_token_id):
            break

        generated_answer_ids.append(next_token_id)

        with torch.no_grad():
            outputs = model_obj(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )

        logits = outputs.logits
        past_key_values = outputs.past_key_values

    if len(generated_answer_ids) > 0:
        answer_ids = torch.cat(generated_answer_ids, dim=1)
    else:
        answer_ids = torch.empty(
            (question_ids.shape[0], 0),
            dtype=question_ids.dtype,
            device=question_ids.device
        )

    full_ids = torch.cat([question_ids, answer_ids], dim=1)
    full_text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    answer_text = tokenizer.decode(answer_ids[0], skip_special_tokens=True)

    return {
        "full_text": full_text,
        "answer_text": answer_text
    }


# ============================================================
#  Reorder / Shuffle / Rotate
# ============================================================
def reorder(model_obj, tokenizer, knowledge_latent, question,
            reorder_base_layer, indices, keep_num=-1,
            query_mode="last",  # ← 新增
            adaptive_k=False, adaptive_tau=2.0, adaptive_k_max=4,
            k_record=None,  # 传入 list 时记录每次自适应选出的 K
            timings=None,  # 传入 dict 时记录各阶段耗时（秒）：trace_forward / reorder_filter
            ):
    class _StopForward(Exception):
        pass

    if timings is not None:
        torch.cuda.synchronize()
        _t_start = time.perf_counter()

    prefix_len = knowledge_latent["attention_mask"].shape[1]
    num_layers = len(model_obj.model.layers)

    if reorder_base_layer < 0 or reorder_base_layer >= num_layers:
        raise ValueError(
            f"reorder_base_layer={reorder_base_layer} 超出范围，应在 [0, {num_layers-1}] 内"
        )

    if prefix_len <= 1 or len(indices) <= 1:
        return knowledge_latent

    input_device = get_input_device(model_obj)

    question_ids = tokenizer(
        question, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)

    prefix_hidden_states = knowledge_latent["hidden_states"]
    q_embeds = model_obj.model.embed_tokens(question_ids)

    prefix_embeds_dummy = torch.zeros(
        (q_embeds.shape[0], prefix_len, q_embeds.shape[2]),
        dtype=q_embeds.dtype, device=q_embeds.device
    )

    full_inputs_embeds = torch.cat([prefix_embeds_dummy, q_embeds], dim=1)
    full_attention_mask = torch.cat([
        torch.ones((q_embeds.shape[0], prefix_len), dtype=torch.long, device=input_device),
        torch.ones_like(question_ids, device=input_device)
    ], dim=1)

    handles = _register_hidden_prefix_hooks(model_obj, prefix_hidden_states, prefix_len)

    captured = {"attn": None}
    target_attn_module = model_obj.model.layers[reorder_base_layer].self_attn
    orig_attn_forward = target_attn_module.forward

    def wrapped_attn_forward(*args, **kwargs):
        kwargs["output_attentions"] = True
        return orig_attn_forward(*args, **kwargs)

    def capture_attn_and_stop(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            attn_weights = outputs[1]
            if attn_weights is not None:
                captured["attn"] = attn_weights.detach()
                raise _StopForward()

    target_attn_module.forward = wrapped_attn_forward
    stop_handle = target_attn_module.register_forward_hook(capture_attn_and_stop)

    try:
        with torch.no_grad():
            try:
                model_obj(
                    inputs_embeds=full_inputs_embeds,
                    attention_mask=full_attention_mask,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True
                )
            except _StopForward:
                pass
    finally:
        stop_handle.remove()
        target_attn_module.forward = orig_attn_forward
        for h in handles:
            h.remove()

    if captured["attn"] is None:
        raise RuntimeError(
            "没有成功捕获目标层 self_attn 的 attention。"
            "建议先 print(type(model_obj.model.layers[reorder_base_layer].self_attn)) "
            "确认具体 attention 模块类型。"
        )

    if timings is not None:
        torch.cuda.synchronize()
        _t_trace = time.perf_counter()
        timings["trace_forward"] = _t_trace - _t_start

    layer_attn = captured["attn"]  # [1, num_heads, seq_len, seq_len]

    # ========== 改动部分 ==========
    if query_mode == "last":
        # 原逻辑：只看 question 最后一个 token，avg over heads
        query_attn = layer_attn[0, :, -1, :prefix_len].mean(dim=0)  # [prefix_len]
    elif query_mode == "mean":
        # 所有 question token 位置对 prefix 的注意力
        q_start = prefix_len
        q_end = layer_attn.shape[2]
        query_attn = layer_attn[0, :, q_start:q_end, :prefix_len].mean(dim=0).mean(dim=0)  # [prefix_len]
    else:
        raise ValueError(f"Unknown query_mode: {query_mode!r}. Expected 'last' or 'mean'.")

    attn_wo_bos = query_attn[1:]  # ← 原来是 last_token_attn[1:]，现在统一用 query_attn
    # ========== 改动结束 ==========

    reduced_ends = []
    for k in range(1, len(indices)):
        idx = int(indices[k])
        end_exclusive = idx - 1
        end_exclusive = max(0, min(end_exclusive, prefix_len - 1))
        reduced_ends.append(end_exclusive)

    cleaned_ends = []
    prev = 0
    for e in reduced_ends:
        if e > prev:
            cleaned_ends.append(e)
            prev = e

    if len(cleaned_ends) == 0 or cleaned_ends[-1] < (prefix_len - 1):
        cleaned_ends.append(prefix_len - 1)

    block_ranges_reduced = []
    start = 0
    for end in cleaned_ends:
        if end > start:
            block_ranges_reduced.append((start, end))
            start = end

    if len(block_ranges_reduced) == 0:
        return knowledge_latent

    densities = []
    for start, end in block_ranges_reduced:
        block_scores = attn_wo_bos[start:end]
        block_len = end - start
        density = (block_scores.sum() / block_len).item()
        densities.append(density)

    if timings is not None:
        timings["densities"] = [float(d) for d in densities]

    sorted_block_ids = sorted(
        range(len(block_ranges_reduced)),
        key=lambda i: densities[i]
    )

    total_blocks = len(sorted_block_ids)

    if adaptive_k:
        # 自适应 K：设降序密度为 rho_(1) >= rho_(2) >= ...，从 K=1 开始，
        # 若 rho_(K)/rho_(K+1) > tau（边际足够大，截断安全）则停止，
        # 否则密度分布平坦、追踪信号含糊，K += 1，上限 adaptive_k_max。
        desc_densities = sorted(densities, reverse=True)
        chosen_k = 1
        k_cap = max(1, min(adaptive_k_max, total_blocks))
        while chosen_k < k_cap:
            cur = desc_densities[chosen_k - 1]   # rho_(K)
            nxt = desc_densities[chosen_k]       # rho_(K+1)
            if nxt <= 0 or cur / nxt > adaptive_tau:
                break
            chosen_k += 1
        if k_record is not None:
            k_record.append(chosen_k)
        keep_num = chosen_k

    if keep_num < 0:
        kept_sorted_block_ids = sorted_block_ids
    else:
        keep_num = min(keep_num, total_blocks)
        kept_sorted_block_ids = sorted_block_ids[-keep_num:] if keep_num > 0 else []

    original_block_slices = []
    for start, end in block_ranges_reduced:
        original_block_slices.append((start + 1, end + 1))

    new_input_ids_parts = [knowledge_latent["input_ids"][:, :1]]
    new_attention_mask_parts = [knowledge_latent["attention_mask"][:, :1]]

    for block_id in kept_sorted_block_ids:
        s, e = original_block_slices[block_id]
        new_input_ids_parts.append(knowledge_latent["input_ids"][:, s:e])
        new_attention_mask_parts.append(knowledge_latent["attention_mask"][:, s:e])

    reordered_hidden_states = []
    for layer_h in knowledge_latent["hidden_states"]:
        parts = [layer_h[:, :1, :]]
        for block_id in kept_sorted_block_ids:
            s, e = original_block_slices[block_id]
            parts.append(layer_h[:, s:e, :])
        reordered_hidden_states.append(torch.cat(parts, dim=1))

    reordered_input_ids = torch.cat(new_input_ids_parts, dim=1)
    reordered_attention_mask = torch.cat(new_attention_mask_parts, dim=1)

    reordered_knowledge_latent = {
        "input_ids": reordered_input_ids,
        "attention_mask": reordered_attention_mask,
        "hidden_states": tuple(reordered_hidden_states),
        "segment_lengths": [reordered_input_ids.shape[1]]
    }

    if "perplexity_scores" in knowledge_latent and knowledge_latent["perplexity_scores"] is not None:
        ppl_parts = [knowledge_latent["perplexity_scores"][:, :1]]
        for block_id in kept_sorted_block_ids:
            s, e = original_block_slices[block_id]
            ppl_parts.append(knowledge_latent["perplexity_scores"][:, s:e])
        reordered_knowledge_latent["perplexity_scores"] = torch.cat(ppl_parts, dim=1)
    else:
        reordered_knowledge_latent["perplexity_scores"] = None

    if timings is not None:
        torch.cuda.synchronize()
        timings["reorder_filter"] = time.perf_counter() - _t_trace

    del layer_attn, query_attn, attn_wo_bos  # ← 清理变量名同步更新
    if timings is None:
        # 计时模式下保留分配器缓存以测稳态延迟；
        # 生产路径维持原行为（长跑防 OOM）。
        torch.cuda.empty_cache()

    return reordered_knowledge_latent


def shuffle_knowledge(knowledge_latent, indices):
    """
    在保持第一个 bos token 不变的情况下，随机打乱 knowledge blocks。
    返回 (shuffled_knowledge_latent, shuffled_indices)。
    """
    prefix_len = knowledge_latent["attention_mask"].shape[1]

    if prefix_len <= 1 or len(indices) <= 1:
        return knowledge_latent, indices

    reduced_ends = []
    for k in range(1, len(indices)):
        idx = int(indices[k])
        end_exclusive = idx - 1
        end_exclusive = max(0, min(end_exclusive, prefix_len - 1))
        reduced_ends.append(end_exclusive)

    cleaned_ends = []
    prev = 0
    for e in reduced_ends:
        if e > prev:
            cleaned_ends.append(e)
            prev = e

    if len(cleaned_ends) == 0 or cleaned_ends[-1] < (prefix_len - 1):
        cleaned_ends.append(prefix_len - 1)

    block_ranges_reduced = []
    start = 0
    for end in cleaned_ends:
        if end > start:
            block_ranges_reduced.append((start, end))
            start = end

    if len(block_ranges_reduced) == 0:
        return knowledge_latent, indices

    original_block_slices = [
        (start + 1, end + 1) for start, end in block_ranges_reduced
    ]

    shuffled_block_ids = list(range(len(original_block_slices)))
    random.shuffle(shuffled_block_ids)

    new_input_ids_parts = [knowledge_latent["input_ids"][:, :1]]
    new_attention_mask_parts = [knowledge_latent["attention_mask"][:, :1]]

    for block_id in shuffled_block_ids:
        s, e = original_block_slices[block_id]
        new_input_ids_parts.append(knowledge_latent["input_ids"][:, s:e])
        new_attention_mask_parts.append(knowledge_latent["attention_mask"][:, s:e])

    shuffled_hidden_states = []
    for layer_h in knowledge_latent["hidden_states"]:
        parts = [layer_h[:, :1, :]]
        for block_id in shuffled_block_ids:
            s, e = original_block_slices[block_id]
            parts.append(layer_h[:, s:e, :])
        shuffled_hidden_states.append(torch.cat(parts, dim=1))

    shuffled_input_ids = torch.cat(new_input_ids_parts, dim=1)
    shuffled_attention_mask = torch.cat(new_attention_mask_parts, dim=1)

    shuffled_knowledge_latent = {
        "input_ids": shuffled_input_ids,
        "attention_mask": shuffled_attention_mask,
        "hidden_states": tuple(shuffled_hidden_states),
        "segment_lengths": [shuffled_input_ids.shape[1]]
    }

    if "perplexity_scores" in knowledge_latent and knowledge_latent["perplexity_scores"] is not None:
        ppl_parts = [knowledge_latent["perplexity_scores"][:, :1]]
        for block_id in shuffled_block_ids:
            s, e = original_block_slices[block_id]
            ppl_parts.append(knowledge_latent["perplexity_scores"][:, s:e])
        shuffled_knowledge_latent["perplexity_scores"] = torch.cat(ppl_parts, dim=1)
    else:
        shuffled_knowledge_latent["perplexity_scores"] = None

    shuffled_indices = [1]
    cumulative_non_bos_len = 0
    for block_id in shuffled_block_ids:
        s, e = original_block_slices[block_id]
        block_len = e - s
        cumulative_non_bos_len += block_len
        shuffled_indices.append(cumulative_non_bos_len + 1)

    return shuffled_knowledge_latent, shuffled_indices

def rotate(model_obj, tokenizer, knowledge_latent, question,
           rotate_base_layer, indices, window_size, rotate_cutoff=False,
           query_mode="last",  # ← 新增
           ):
    class _StopForward(Exception):
        pass

    prefix_len = knowledge_latent["attention_mask"].shape[1]
    num_layers = len(model_obj.model.layers)

    if rotate_base_layer < 0 or rotate_base_layer >= num_layers:
        raise ValueError(
            f"rotate_base_layer={rotate_base_layer} 超出范围，应在 [0, {num_layers-1}] 内"
        )

    if prefix_len <= 1 or len(indices) <= 1:
        return knowledge_latent

    input_device = get_input_device(model_obj)

    question_ids = tokenizer(
        question, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)

    prefix_hidden_states = knowledge_latent["hidden_states"]
    q_embeds = model_obj.model.embed_tokens(question_ids)

    prefix_embeds_dummy = torch.zeros(
        (q_embeds.shape[0], prefix_len, q_embeds.shape[2]),
        dtype=q_embeds.dtype, device=q_embeds.device
    )

    full_inputs_embeds = torch.cat([prefix_embeds_dummy, q_embeds], dim=1)
    full_attention_mask = torch.cat([
        torch.ones((q_embeds.shape[0], prefix_len), dtype=torch.long, device=input_device),
        torch.ones_like(question_ids, device=input_device)
    ], dim=1)

    handles = _register_hidden_prefix_hooks(model_obj, prefix_hidden_states, prefix_len)

    captured = {"attn": None}
    target_attn_module = model_obj.model.layers[rotate_base_layer].self_attn
    orig_attn_forward = target_attn_module.forward

    def wrapped_attn_forward(*args, **kwargs):
        kwargs["output_attentions"] = True
        return orig_attn_forward(*args, **kwargs)

    def capture_attn_and_stop(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            attn_weights = outputs[1]
            if attn_weights is not None:
                captured["attn"] = attn_weights.detach()
                raise _StopForward()

    target_attn_module.forward = wrapped_attn_forward
    stop_handle = target_attn_module.register_forward_hook(capture_attn_and_stop)

    try:
        with torch.no_grad():
            try:
                model_obj(
                    inputs_embeds=full_inputs_embeds,
                    attention_mask=full_attention_mask,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True
                )
            except _StopForward:
                pass
    finally:
        stop_handle.remove()
        target_attn_module.forward = orig_attn_forward
        for h in handles:
            h.remove()

    if captured["attn"] is None:
        raise RuntimeError(
            "没有成功捕获目标层 self_attn 的 attention。"
            "建议先 print(type(model_obj.model.layers[rotate_base_layer].self_attn)) "
            "确认具体 attention 模块类型。"
        )

    layer_attn = captured["attn"]  # [1, num_heads, seq_len, seq_len]

    # ========== 改动部分 ==========
    if query_mode == "last":
        query_attn = layer_attn[0, :, -1, :prefix_len].mean(dim=0)
    elif query_mode == "mean":
        q_start = prefix_len
        q_end = layer_attn.shape[2]
        query_attn = layer_attn[0, :, q_start:q_end, :prefix_len].mean(dim=0).mean(dim=0)
    else:
        raise ValueError(f"Unknown query_mode: {query_mode!r}. Expected 'last' or 'mean'.")

    attn_wo_bos = query_attn[1:]
    # ========== 改动结束 ==========

    reduced_ends = []
    for k in range(1, len(indices)):
        idx = int(indices[k])
        end_exclusive = idx - 1
        end_exclusive = max(0, min(end_exclusive, prefix_len - 1))
        reduced_ends.append(end_exclusive)

    cleaned_ends = []
    prev = 0
    for e in reduced_ends:
        if e > prev:
            cleaned_ends.append(e)
            prev = e

    if len(cleaned_ends) == 0 or cleaned_ends[-1] < (prefix_len - 1):
        cleaned_ends.append(prefix_len - 1)

    block_ranges_reduced = []
    start = 0
    for end in cleaned_ends:
        if end > start:
            block_ranges_reduced.append((start, end))
            start = end

    if len(block_ranges_reduced) == 0:
        return knowledge_latent

    num_blocks = len(block_ranges_reduced)

    if window_size <= 0:
        raise ValueError(f"window_size 应为正整数，当前为 {window_size}")

    window_size = min(window_size, num_blocks)

    densities = []
    for start, end in block_ranges_reduced:
        block_scores = attn_wo_bos[start:end]
        block_len = end - start
        density = (block_scores.sum() / block_len).item()
        densities.append(density)

    doubled = densities + densities
    curr_sum = sum(doubled[:window_size])
    best_sum = curr_sum
    best_start = 0

    for s in range(1, num_blocks):
        curr_sum = curr_sum - doubled[s - 1] + doubled[s + window_size - 1]
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_start = s

    best_window_block_ids = [
        (best_start + offset) % num_blocks for offset in range(window_size)
    ]

    original_block_slices = [
        (start + 1, end + 1) for start, end in block_ranges_reduced
    ]

    if rotate_cutoff:
        new_input_ids_parts = [knowledge_latent["input_ids"][:, :1]]
        new_attention_mask_parts = [knowledge_latent["attention_mask"][:, :1]]

        for block_id in best_window_block_ids:
            s, e = original_block_slices[block_id]
            new_input_ids_parts.append(knowledge_latent["input_ids"][:, s:e])
            new_attention_mask_parts.append(knowledge_latent["attention_mask"][:, s:e])

        rotated_hidden_states = []
        for layer_h in knowledge_latent["hidden_states"]:
            parts = [layer_h[:, :1, :]]
            for block_id in best_window_block_ids:
                s, e = original_block_slices[block_id]
                parts.append(layer_h[:, s:e, :])
            rotated_hidden_states.append(torch.cat(parts, dim=1))

    else:
        target_start = num_blocks - window_size
        shift = (best_start - target_start) % num_blocks

        block_order = list(range(num_blocks))
        rotated_block_ids = block_order[shift:] + block_order[:shift]

        new_input_ids_parts = [knowledge_latent["input_ids"][:, :1]]
        new_attention_mask_parts = [knowledge_latent["attention_mask"][:, :1]]

        for block_id in rotated_block_ids:
            s, e = original_block_slices[block_id]
            new_input_ids_parts.append(knowledge_latent["input_ids"][:, s:e])
            new_attention_mask_parts.append(knowledge_latent["attention_mask"][:, s:e])

        rotated_hidden_states = []
        for layer_h in knowledge_latent["hidden_states"]:
            parts = [layer_h[:, :1, :]]
            for block_id in rotated_block_ids:
                s, e = original_block_slices[block_id]
                parts.append(layer_h[:, s:e, :])
            rotated_hidden_states.append(torch.cat(parts, dim=1))

    rotated_input_ids = torch.cat(new_input_ids_parts, dim=1)
    rotated_attention_mask = torch.cat(new_attention_mask_parts, dim=1)

    rotated_knowledge_latent = {
        "input_ids": rotated_input_ids,
        "attention_mask": rotated_attention_mask,
        "hidden_states": tuple(rotated_hidden_states),
        "segment_lengths": [rotated_input_ids.shape[1]]
    }

    if "perplexity_scores" in knowledge_latent and knowledge_latent["perplexity_scores"] is not None:
        if rotate_cutoff:
            ppl_parts = [knowledge_latent["perplexity_scores"][:, :1]]
            for block_id in best_window_block_ids:
                s, e = original_block_slices[block_id]
                ppl_parts.append(knowledge_latent["perplexity_scores"][:, s:e])
        else:
            ppl_parts = [knowledge_latent["perplexity_scores"][:, :1]]
            for block_id in rotated_block_ids:
                s, e = original_block_slices[block_id]
                ppl_parts.append(knowledge_latent["perplexity_scores"][:, s:e])

        rotated_knowledge_latent["perplexity_scores"] = torch.cat(ppl_parts, dim=1)
    else:
        rotated_knowledge_latent["perplexity_scores"] = None

    del layer_attn, query_attn, attn_wo_bos  # ← 变量名同步
    torch.cuda.empty_cache()

    return rotated_knowledge_latent


# ============================================================
#  Attention Density（所有层）
# ============================================================

def compute_attention_density_all_layers(
    model_obj, tokenizer, knowledge_latent, question, first_context_len
):
    """
    对所有 decoder layer 一次 forward 捕获 attention weights，
    计算 question 最后一个 token 对第一个 context block 的注意力密度。

    Args:
        model_obj:           Hugging Face 模型对象
        tokenizer:           对应的 tokenizer
        knowledge_latent:    knowledge latent dict（含 hidden_states, attention_mask 等）
        question:            question 文本
        first_context_len:   第一个 context 在 knowledge_latent 中占据的 token 数
                             （包含 bos，即 indices[1]）

    Returns:
        densities: list[float]，长度 = num_layers，
                   densities[l] = 第 l 层上 question 最后 token 对
                   第一个 context block（去掉 bos 后的 token 范围 [1, first_context_len)）
                   的平均注意力值。
    """
    prefix_hidden_states = knowledge_latent["hidden_states"]
    prefix_len = knowledge_latent["attention_mask"].shape[1]
    num_layers = len(model_obj.model.layers)

    input_device = get_input_device(model_obj)

    question_ids = tokenizer(
        question, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)

    q_embeds = model_obj.model.embed_tokens(question_ids)

    prefix_embeds_dummy = torch.zeros(
        (q_embeds.shape[0], prefix_len, q_embeds.shape[2]),
        dtype=q_embeds.dtype, device=q_embeds.device
    )

    full_inputs_embeds = torch.cat([prefix_embeds_dummy, q_embeds], dim=1)
    full_attention_mask = torch.cat([
        torch.ones((q_embeds.shape[0], prefix_len), dtype=torch.long, device=input_device),
        torch.ones_like(question_ids, device=input_device)
    ], dim=1)

    # 注册 hidden prefix hooks
    handles = _register_hidden_prefix_hooks(model_obj, prefix_hidden_states, prefix_len)

    # 对每一层的 self_attn 包装，使其输出 attention weights
    orig_forwards = {}
    for layer_idx in range(num_layers):
        attn_module = model_obj.model.layers[layer_idx].self_attn
        orig_forwards[layer_idx] = attn_module.forward

        def make_wrapped(orig_fwd):
            def wrapped(*args, **kwargs):
                kwargs["output_attentions"] = True
                return orig_fwd(*args, **kwargs)
            return wrapped

        attn_module.forward = make_wrapped(orig_forwards[layer_idx])

    # 收集每一层的 attention
    captured_attns = {}

    def make_capture_hook(layer_idx):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                attn_weights = outputs[1]
                if attn_weights is not None:
                    captured_attns[layer_idx] = attn_weights.detach()
        return hook

    capture_handles = []
    for layer_idx in range(num_layers):
        attn_module = model_obj.model.layers[layer_idx].self_attn
        capture_handles.append(
            attn_module.register_forward_hook(make_capture_hook(layer_idx))
        )

    try:
        with torch.no_grad():
            model_obj(
                inputs_embeds=full_inputs_embeds,
                attention_mask=full_attention_mask,
                output_attentions=False,
                use_cache=False,
                return_dict=True
            )
    finally:
        # 清理所有 hooks 和包装
        for ch in capture_handles:
            ch.remove()
        for layer_idx in range(num_layers):
            model_obj.model.layers[layer_idx].self_attn.forward = orig_forwards[layer_idx]
        for h in handles:
            h.remove()

    # 计算每层 density
    # first_context_len 对应 indices[1]，即第一个 context 的 token 范围是 [0, first_context_len)
    # 去掉 bos（位置 0）后，block 范围是 [1, first_context_len)
    block_start = 1
    block_end = first_context_len
    block_len = block_end - block_start

    densities = []
    for layer_idx in range(num_layers):
        if layer_idx not in captured_attns:
            densities.append(0.0)
            continue

        layer_attn = captured_attns[layer_idx]  # [1, num_heads, seq_len, seq_len]
        # question 最后一个 token 对 prefix 中各位置的 attention
        last_token_attn = layer_attn[0, :, -1, :prefix_len]  # [num_heads, prefix_len]
        last_token_attn = last_token_attn.mean(dim=0)  # [prefix_len]

        if block_len > 0:
            block_scores = last_token_attn[block_start:block_end]
            density = (block_scores.sum() / block_len).item()
        else:
            density = 0.0

        densities.append(density)

    # 清理
    del captured_attns
    torch.cuda.empty_cache()

    return densities


def place_target_at_position(knowledge_latents_list, target_idx, position):
    """
    将 target_idx 对应的 knowledge_latent 放到指定 position (0-based)，
    其他 knowledge 保持相对顺序，然后 **lazy 拼接** 成一个 knowledge_latent。

    与旧版不同，这里使用 torch.cat 的 view-list 方式，只在最终拼接时才创建新 tensor。
    对于 hidden_states，直接用 torch.cat 一次性拼接（避免反复创建中间结果）。

    约定：knowledge_latents_list[target_idx] 含 bos（由 add_special_tokens=True 生成），
    其他 latent 不含 bos（add_special_tokens=False）。

    拼接结构：[bos_from_target] + [segment_0] + [segment_1] + ... + [segment_{n-1}]

    Args:
        knowledge_latents_list: list of knowledge_latent dicts
        target_idx:             被测试知识在 list 中的原始索引 (通常为 0)
        position:               目标位置 (0-based)，被测知识要放到第 position 个位置

    Returns:
        combined_knowledge_latent: 拼接后的 knowledge_latent dict
        target_segment_idx:        被测知识在拼接结果中的 segment 索引 (0-based)
    """
    n = len(knowledge_latents_list)
    target_latent = knowledge_latents_list[target_idx]

    # 构建非 target 的索引列表（保持原始顺序）
    others = [i for i in range(n) if i != target_idx]

    # 构建最终排列顺序（不含 target 的 bos）
    # target 的 bos 单独处理，target 的 rest（去掉 bos）放到 position 位置
    # ordered_pieces[k] 存的是 (latent, start_col, end_col)
    # 其中 target 去掉 bos 后为 [:, 1:, :]

    # 收集各 piece 的 hidden_states slices 和 metadata
    # 最终一次性 torch.cat
    target_segment_idx = position

    # 构建 pieces 顺序：others[:position] + [target_rest] + others[position:]
    ordered_indices = others[:position] + [target_idx] + others[position:]

    # 构建 segment_lengths
    segment_lengths = []
    for k, idx in enumerate(ordered_indices):
        seg_len = knowledge_latents_list[idx]["input_ids"].shape[1]
        if idx == target_idx:
            seg_len -= 1  # target 去掉 bos
        segment_lengths.append(seg_len)

    # bos 合并进第一个 segment
    segment_lengths[0] += 1  # +1 for bos

    # 一次性拼接 hidden_states: [bos] + ordered pieces
    num_layers = len(target_latent["hidden_states"])
    combined_hidden = []
    for l in range(num_layers):
        parts = [target_latent["hidden_states"][l][:, :1, :]]  # bos
        for idx in ordered_indices:
            lat = knowledge_latents_list[idx]
            if idx == target_idx:
                parts.append(lat["hidden_states"][l][:, 1:, :])  # skip bos
            else:
                parts.append(lat["hidden_states"][l])
        combined_hidden.append(torch.cat(parts, dim=1))

    # 拼接 input_ids 和 attention_mask
    id_parts = [target_latent["input_ids"][:, :1]]  # bos
    mask_parts = [target_latent["attention_mask"][:, :1]]
    for idx in ordered_indices:
        lat = knowledge_latents_list[idx]
        if idx == target_idx:
            id_parts.append(lat["input_ids"][:, 1:])
            mask_parts.append(lat["attention_mask"][:, 1:])
        else:
            id_parts.append(lat["input_ids"])
            mask_parts.append(lat["attention_mask"])

    result = {
        "input_ids": torch.cat(id_parts, dim=1),
        "attention_mask": torch.cat(mask_parts, dim=1),
        "hidden_states": tuple(combined_hidden),
        "segment_lengths": segment_lengths,
        "perplexity_scores": None,
    }

    return result, target_segment_idx


def compute_attention_density_all_segments(
    model_obj, tokenizer, knowledge_latent, question, target_segment_idx,
    query_mode="last",  # ← 新增: "last" = 只看最后一个token, "mean" = 所有question位置平均
):
    """
    对所有 decoder layer 一次 forward 捕获 attention weights，
    计算 question tokens 对 **每个 segment** 的注意力密度，
    并返回目标 segment 在各层的密度值及排名。

    Args:
        model_obj:           Hugging Face 模型对象
        tokenizer:           对应的 tokenizer
        knowledge_latent:    knowledge latent dict（含 hidden_states, attention_mask, segment_lengths 等）
        question:            question 文本
        target_segment_idx:  要观察的目标 segment 的索引 (0-based)
        query_mode:          "last" = 只用 question 最后一个 token 的注意力;
                             "mean" = 对所有 question token 位置的注意力取平均。

    Returns:
        target_densities:    list[float]，长度 = num_layers，
                             target_densities[l] = 第 l 层上目标 segment 的注意力密度。
        target_ranks:        list[int]，长度 = num_layers，
                             target_ranks[l] = 第 l 层上目标 segment 的密度排名 (1-based, 1=最高)。
        all_seg_densities:   list[list[float]]，shape = [num_layers, num_segments]，
                             all_seg_densities[l][s] = 第 l 层上第 s 个 segment 的注意力密度。
    """
    prefix_hidden_states = knowledge_latent["hidden_states"]
    prefix_len = knowledge_latent["attention_mask"].shape[1]
    num_layers = len(model_obj.model.layers)
    segment_lengths = knowledge_latent["segment_lengths"]
    num_segments = len(segment_lengths)

    input_device = get_input_device(model_obj)

    question_ids = tokenizer(
        question, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)

    q_embeds = model_obj.model.embed_tokens(question_ids)

    prefix_embeds_dummy = torch.zeros(
        (q_embeds.shape[0], prefix_len, q_embeds.shape[2]),
        dtype=q_embeds.dtype, device=q_embeds.device
    )

    full_inputs_embeds = torch.cat([prefix_embeds_dummy, q_embeds], dim=1)
    full_attention_mask = torch.cat([
        torch.ones((q_embeds.shape[0], prefix_len), dtype=torch.long, device=input_device),
        torch.ones_like(question_ids, device=input_device)
    ], dim=1)

    # 注册 hidden prefix hooks
    handles = _register_hidden_prefix_hooks(model_obj, prefix_hidden_states, prefix_len)

    # 对每一层的 self_attn 包装，使其输出 attention weights
    orig_forwards = {}
    for layer_idx in range(num_layers):
        attn_module = model_obj.model.layers[layer_idx].self_attn
        orig_forwards[layer_idx] = attn_module.forward

        def make_wrapped(orig_fwd):
            def wrapped(*args, **kwargs):
                kwargs["output_attentions"] = True
                return orig_fwd(*args, **kwargs)
            return wrapped

        attn_module.forward = make_wrapped(orig_forwards[layer_idx])

    # 收集每一层的 attention weights
    captured_attns = {}

    def make_capture_hook(layer_idx):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                attn_weights = outputs[1]
                if attn_weights is not None:
                    captured_attns[layer_idx] = attn_weights.detach()
        return hook

    capture_handles = []
    for layer_idx in range(num_layers):
        attn_module = model_obj.model.layers[layer_idx].self_attn
        capture_handles.append(
            attn_module.register_forward_hook(make_capture_hook(layer_idx))
        )

    try:
        with torch.no_grad():
            model_obj(
                inputs_embeds=full_inputs_embeds,
                attention_mask=full_attention_mask,
                output_attentions=False,
                use_cache=False,
                return_dict=True
            )
    finally:
        for ch in capture_handles:
            ch.remove()
        for layer_idx in range(num_layers):
            model_obj.model.layers[layer_idx].self_attn.forward = orig_forwards[layer_idx]
        for h in handles:
            h.remove()

    # 预计算 segment 范围（去掉 bos）
    seg_ranges = []
    cum = 0
    for s_idx, seg_len in enumerate(segment_lengths):
        if s_idx == 0:
            seg_ranges.append((1, seg_len))
        else:
            seg_ranges.append((cum, cum + seg_len))
        cum += seg_len

    # ========== 改动部分 ==========
    # 计算每层每个 segment 的 density
    all_seg_densities = []
    target_densities = []
    target_ranks = []

    for layer_idx in range(num_layers):
        if layer_idx not in captured_attns:
            all_seg_densities.append([0.0] * num_segments)
            target_densities.append(0.0)
            target_ranks.append(num_segments)
            continue

        layer_attn = captured_attns[layer_idx]  # [1, num_heads, seq_len, seq_len]

        # ---- 根据 query_mode 选择 question 侧的聚合方式 ----
        if query_mode == "last":
            # 原逻辑：只看 question 最后一个 token，avg over heads
            query_attn = layer_attn[0, :, -1, :prefix_len].mean(dim=0)  # [prefix_len]
        elif query_mode == "mean":
            # 所有 question token 位置对 prefix 的注意力
            # question tokens 在 full sequence 中从 prefix_len 到末尾
            q_start = prefix_len
            q_end = layer_attn.shape[2]
            # [num_heads, q_len, prefix_len] -> avg heads -> [q_len, prefix_len] -> avg q_pos -> [prefix_len]
            query_attn = layer_attn[0, :, q_start:q_end, :prefix_len].mean(dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown query_mode: {query_mode!r}. Expected 'last' or 'mean'.")

        layer_seg_densities = []
        for s_idx in range(num_segments):
            s_start, s_end = seg_ranges[s_idx]
            s_len = s_end - s_start
            if s_len > 0:
                density = (query_attn[s_start:s_end].sum() / s_len).item()
            else:
                density = 0.0
            layer_seg_densities.append(density)

        all_seg_densities.append(layer_seg_densities)

        target_d = layer_seg_densities[target_segment_idx]
        target_densities.append(target_d)

        rank = 1
        for d in layer_seg_densities:
            if d > target_d:
                rank += 1
        target_ranks.append(rank)
    # ========== 改动结束 ==========

    del captured_attns
    torch.cuda.empty_cache()

    return target_densities, target_ranks, all_seg_densities

