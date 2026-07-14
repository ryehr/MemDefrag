import argparse
import torch
import json
import random
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

from utilities import (
    get_max_memory,
)


# ============================================================
#  数据加载（与 QA_benchmark_investigation_rank.py 完全一致）
# ============================================================

def load_qa_data(dataset_name: str, group_num: int, unrelated_num: int):
    global tokenizer, args

    if dataset_name == "nqa":
        path = "./data_new/nqa/"
    elif dataset_name == "squad":
        path = "./data_new/squad/"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    test_path = path + "val_data.json"
    unrelated_path = path + "unrelated_contexts_from_train.json"

    with open(test_path, 'r', encoding="utf-8-sig") as file:
        test_data = json.load(file)
    with open(unrelated_path, 'r', encoding="utf-8-sig") as file:
        unrelated_data = json.load(file)

    random.shuffle(test_data)
    random.shuffle(unrelated_data)

    filtered_test_data = []
    filtered_unrelated_data = []

    for item in test_data:
        if len(filtered_test_data) >= group_num:
            break
        context = item["context"]
        if len(tokenizer(context)["input_ids"]) <= 512:
            filtered_test_data.append(item)

    for item in unrelated_data:
        if len(filtered_unrelated_data) >= unrelated_num:
            break
        filtered_unrelated_data.append(item)

    return filtered_test_data, filtered_unrelated_data


# ============================================================
#  纯文本：把 target 放到指定 position，构造完整 input_ids 与 segment_lengths
# ============================================================

def place_target_at_position_text(
    target_ids,            # [1, T_target_with_bos]  (含 bos)
    unrelated_ids_list,    # list of [1, T_i]        (不含 bos)
    target_idx,            # 通常为 0
    position,              # 0-based
):
    """
    与 utilities.place_target_at_position 对齐，只是处理 token ids 而非 hidden_states。

    构建：
        [bos] + ordered_pieces_in_token_ids

    其中 ordered = others[:position] + [target_rest] + others[position:]，
    target_rest = target_ids 去掉 bos。

    segment_lengths:
        - 与 latent 版本一致：每段长度按 token 数计；
        - target 段长度 = len(target) - 1（去掉 bos）；
        - 但 bos 会被并入 segment_lengths[0]（最左段 +1）。

    Returns:
        full_ids: [1, total_len]
        segment_lengths: list[int]
        target_segment_idx: int  (= position)
    """
    n = 1 + len(unrelated_ids_list)
    others_token_ids = [unrelated_ids_list[i] for i in range(len(unrelated_ids_list))]
    # ordered_indices 不直接需要，因为我们已经按物理顺序排好
    # target 去掉 bos
    target_rest = target_ids[:, 1:]
    target_bos = target_ids[:, :1]

    pieces_before = others_token_ids[:position]
    pieces_after = others_token_ids[position:]

    # ordered token pieces (不含 bos)
    ordered_pieces = pieces_before + [target_rest] + pieces_after

    # 构建 full_ids: [bos] + ordered_pieces
    full_ids = torch.cat([target_bos] + ordered_pieces, dim=1)

    # 构建 segment_lengths
    segment_lengths = [p.shape[1] for p in ordered_pieces]
    # bos 并入第一个 segment
    segment_lengths[0] += 1

    target_segment_idx = position
    return full_ids, segment_lengths, target_segment_idx


# ============================================================
#  文本版：question 对每个 segment 的 attention density + target rank
#  （hook 式逐层处理，避免所有层 attention 同时驻留显存）
# ============================================================

def compute_attention_density_all_segments_text(
    model, tokenizer,
    full_prefix_ids,      # [1, prefix_len]   prefix 的 token ids
    segment_lengths,      # list[int]，与 prefix 对齐（segment_lengths[0] 含 bos）
    question,             # 文本
    target_segment_idx,   # 0-based
    query_mode="last",
):
    """
    将 prefix_ids + question_ids 整体送入模型，逐层用 forward hook 抓 attention，
    在每层抓到后立刻：
        1) 提取 question 侧对 prefix 的注意力（按 query_mode）；
        2) 对每个 segment 求平均得到 density；
        3) 计算 target rank；
        4) 释放该层 attention 矩阵。

    Returns:
        target_densities: list[float]
        target_ranks:     list[int]
        all_seg_densities: list[list[float]]
    """
    input_device = next(model.parameters()).device

    prefix_len = full_prefix_ids.shape[1]
    question_ids = tokenizer(
        question, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)

    input_ids = torch.cat([full_prefix_ids.to(input_device), question_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, device=input_device)
    seq_len = input_ids.shape[1]

    # 预计算每个 segment 在 prefix 中的 token 范围（与 latent 版本对齐：第一段跳过 bos）
    num_segments = len(segment_lengths)
    seg_ranges = []
    cum = 0
    for s_idx, seg_len in enumerate(segment_lengths):
        if s_idx == 0:
            seg_ranges.append((1, seg_len))   # 跳过 bos
        else:
            seg_ranges.append((cum, cum + seg_len))
        cum += seg_len

    num_layers = len(model.model.layers)

    all_seg_densities = [[0.0] * num_segments for _ in range(num_layers)]
    target_densities = [0.0] * num_layers
    target_ranks = [num_segments] * num_layers

    # 包装每层 self_attn 让其返回 attention 权重
    orig_forwards = {}
    for layer_idx in range(num_layers):
        attn_module = model.model.layers[layer_idx].self_attn
        orig_forwards[layer_idx] = attn_module.forward

        def make_wrapped(orig_fwd):
            def wrapped(*a, **kw):
                kw["output_attentions"] = True
                return orig_fwd(*a, **kw)
            return wrapped

        attn_module.forward = make_wrapped(orig_forwards[layer_idx])

    # forward hook：拿到 attention → 立刻算 density + rank → 丢弃
    def make_capture_hook(layer_idx):
        def hook(module, inputs, outputs):
            if not (isinstance(outputs, tuple) and len(outputs) >= 2):
                return
            attn_weights = outputs[1]
            if attn_weights is None:
                return
            # attn_weights: [1, num_heads, seq_len, seq_len]
            if query_mode == "last":
                # 只看 question 最后一个 token 对 prefix 的注意力
                query_attn = attn_weights[0, :, -1, :prefix_len].mean(dim=0)  # [prefix_len]
            elif query_mode == "mean":
                q_start = prefix_len
                q_end = attn_weights.shape[2]
                query_attn = attn_weights[0, :, q_start:q_end, :prefix_len].mean(dim=0).mean(dim=0)
            else:
                raise ValueError(f"Unknown query_mode: {query_mode!r}")

            layer_seg_densities = []
            for s_idx in range(num_segments):
                s_start, s_end = seg_ranges[s_idx]
                s_len = s_end - s_start
                if s_len > 0:
                    d = (query_attn[s_start:s_end].sum() / s_len).item()
                else:
                    d = 0.0
                layer_seg_densities.append(d)

            all_seg_densities[layer_idx] = layer_seg_densities
            target_d = layer_seg_densities[target_segment_idx]
            target_densities[layer_idx] = target_d

            rank = 1
            for d in layer_seg_densities:
                if d > target_d:
                    rank += 1
            target_ranks[layer_idx] = rank
            # 不持有 attn_weights / query_attn 的引用
        return hook

    capture_handles = []
    for layer_idx in range(num_layers):
        attn_module = model.model.layers[layer_idx].self_attn
        capture_handles.append(
            attn_module.register_forward_hook(make_capture_hook(layer_idx))
        )

    try:
        with torch.inference_mode():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,  # 不让 HF 在 outputs 中保留所有层 attentions
                use_cache=False,
                return_dict=True,
            )
    finally:
        for h in capture_handles:
            h.remove()
        for layer_idx in range(num_layers):
            model.model.layers[layer_idx].self_attn.forward = orig_forwards[layer_idx]

    del input_ids, attention_mask, question_ids
    torch.cuda.empty_cache()

    return target_densities, target_ranks, all_seg_densities


# ============================================================
#  QA Benchmark 主循环（文本版）
# ============================================================

def run_qa_benchmark(filtered_test_data, filtered_unrelated_data, model):
    """
    与 QA_benchmark_investigation_rank.py 完全一致的流程，只是：
    - 把所有"knowledge_latent"换成纯 token-ids 的 prefix；
    - 用 model 直接 forward，hook 抓 attention。
    """
    global tokenizer, args
    num_positions = 1 + len(filtered_unrelated_data)  # 20 (1 target + 19 unrelated)
    density_all_groups = []  # [group] -> [num_layers][num_positions] 目标 density
    rank_all_groups = []     # [group] -> [num_layers][num_positions] 目标 rank

    input_device = next(model.parameters()).device
    num_layers = len(model.model.layers)

    for i, item in enumerate(filtered_test_data):
        begin_time = time.time()
        this_density = [[] for _ in range(num_layers)]
        this_rank = [[] for _ in range(num_layers)]

        question = item["question"] + " Answer: "
        context = item["context"] + "\n\n"
        print(f"\n[Group {i+1}]")
        print(f"Question: {question}")

        # ---- 步骤 1: 预编码所有"知识" (纯文本 → token ids) ----
        # 被测知识 (index 0)，带 bos
        target_ids = tokenizer(
            context, return_tensors="pt", add_special_tokens=True
        )["input_ids"].to(input_device)

        # unrelated 知识 (index 1~)，不带 bos，每条截断到 ≤512 token 再回写 "\n\n"
        unrelated_ids_list = []
        for unrelated_text in filtered_unrelated_data:
            unrelated_text_ids = tokenizer(
                unrelated_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(input_device)[:, :512]
            truncated_text = tokenizer.decode(unrelated_text_ids[0]) + "\n\n"
            tids = tokenizer(
                truncated_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(input_device)
            unrelated_ids_list.append(tids)

        # ---- 步骤 2: 遍历 20 个位置 ----
        for pos in range(num_positions):
            full_prefix_ids, segment_lengths, target_seg_idx = place_target_at_position_text(
                target_ids=target_ids,
                unrelated_ids_list=unrelated_ids_list,
                target_idx=0,
                position=pos,
            )

            target_densities, target_ranks, _ = compute_attention_density_all_segments_text(
                model=model,
                tokenizer=tokenizer,
                full_prefix_ids=full_prefix_ids,
                segment_lengths=segment_lengths,
                question=question,
                target_segment_idx=target_seg_idx,
                query_mode=args.query_mode,
            )

            for l in range(num_layers):
                this_density[l].append(target_densities[l])
                this_rank[l].append(target_ranks[l])
            del target_densities, target_ranks, full_prefix_ids

            print(f"  Position {pos}: done")
            torch.cuda.empty_cache()

        density_all_groups.append(this_density)
        rank_all_groups.append(this_rank)

        # 截至当前的 rank 分布（按 layer, pos order）
        num_groups_so_far = len(rank_all_groups)
        rank_dist_tmp = []
        avg_rank_tmp = []
        for l in range(num_layers):
            rank_counter = {}
            rank_sum = 0
            rank_count = 0
            for g in range(num_groups_so_far):
                for p in range(num_positions):
                    r = rank_all_groups[g][l][p]
                    rank_counter[r] = rank_counter.get(r, 0) + 1
                    rank_sum += r
                    rank_count += 1
            total = sum(rank_counter.values())
            layer_dist = {f"rank{r}": round(c / total * 100, 1) for r, c in sorted(rank_counter.items())}
            rank_dist_tmp.append(layer_dist)
            avg_rank_tmp.append(round(rank_sum / rank_count, 2) if rank_count > 0 else 0.0)
        print(f"  rank_distribution (by layer, pos order) = {repr(rank_dist_tmp)}")
        print(f"  avg_rank_per_layer (pos order) = {repr(avg_rank_tmp)}")

        end_time = time.time()
        print(f"Group {i+1}: Time taken = {end_time - begin_time:.2f} seconds")

        # 清理本 group 的独立 ids
        del target_ids, unrelated_ids_list
        torch.cuda.empty_cache()

    # ---- 反转: pos=0(最远) -> step20, pos=19(最近) -> step1 ----
    for g in range(len(density_all_groups)):
        for l in range(num_layers):
            density_all_groups[g][l] = density_all_groups[g][l][::-1]
            rank_all_groups[g][l] = rank_all_groups[g][l][::-1]

    # ---- 汇总: 对所有 group 求平均 density 矩阵 (L x positions) ----
    num_groups = len(density_all_groups)

    avg_density_matrix = np.zeros((num_layers, num_positions), dtype=np.float64)
    for g in range(num_groups):
        for l in range(num_layers):
            for s in range(min(len(density_all_groups[g][l]), num_positions)):
                avg_density_matrix[l, s] += density_all_groups[g][l][s]
    if num_groups > 0:
        avg_density_matrix /= num_groups

    # ---- 汇总排名分布（按 layer） ----
    rank_distribution = []
    avg_rank_per_layer = []
    for l in range(num_layers):
        rank_counter = {}
        rank_sum = 0
        rank_count = 0
        for g in range(num_groups):
            for s in range(num_positions):
                r = rank_all_groups[g][l][s]
                rank_counter[r] = rank_counter.get(r, 0) + 1
                rank_sum += r
                rank_count += 1
        total = sum(rank_counter.values())
        rank_dist = {}
        for rank, count in sorted(rank_counter.items()):
            rank_dist[f"rank{rank}"] = f"{count / total * 100:.1f}%"
        rank_distribution.append(rank_dist)
        avg_rank_per_layer.append(round(rank_sum / rank_count, 2) if rank_count > 0 else 0.0)

    return avg_density_matrix, rank_distribution, avg_rank_per_layer


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-level QA Benchmark: attention density & rank")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HF model name or local path")
    # meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, google/gemma-2-9b-it
    parser.add_argument("--device", type=str,
                        default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dataset", type=str, default="nqa",
                        help="Dataset to use for QA benchmark", choices=["nqa", "squad"])
    parser.add_argument("--group_num", type=int, default=3)
    parser.add_argument("--unrelated_num", type=int, default=19)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--query_mode", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--device_map_mode", type=str, default="auto",
                        choices=["auto", "balanced", "balanced_low_0", "sequential"],
                        help="HF accelerate device_map mode")

    args = parser.parse_args()
    print(args)

    logging.basicConfig(
        filename="./investigation_new_models_text.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("Rank_Text")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    max_memory = get_max_memory()

    # eager attention 实现：确保 hook 能拿到真实 attention 权重
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device_map_mode,
        "max_memory": max_memory,
        "attn_implementation": "eager",
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.eval()

    filtered_test_data, filtered_unrelated_data = load_qa_data(
        dataset_name=args.dataset,
        group_num=args.group_num,
        unrelated_num=args.unrelated_num,
    )

    avg_density_matrix, rank_distribution, avg_rank_per_layer = run_qa_benchmark(
        filtered_test_data,
        filtered_unrelated_data,
        model=model,
    )

    print(
        f"\nModel: {args.model_name}, Dataset: {args.dataset}, Group num: {args.group_num}, "
        f"Unrelated num: {args.unrelated_num}"
    )
    print(args)
    print("\n[Final results: step1=nearest to query, step20=farthest from query]")

    num_layers, num_positions = avg_density_matrix.shape

    density_list = avg_density_matrix.tolist()
    logger.info(f"# === Model: {args.model_name}, Dataset: {args.dataset}, Groups: {args.group_num}, Unrelated: {args.unrelated_num} ===")
    logger.info(f"# Shape: ({num_layers}, {num_positions})")
    logger.info(f"density_matrix = {repr(density_list)}")

    rank_dist_python = []
    for l in range(num_layers):
        layer_dict = {}
        for k, v in rank_distribution[l].items():
            layer_dict[k] = float(v.replace("%", ""))
        rank_dist_python.append(layer_dict)

    logger.info(f"rank_distribution = {repr(rank_dist_python)}")
    print(f"\nrank_distribution = {repr(rank_dist_python)}")

    logger.info(f"avg_rank_per_layer = {repr(avg_rank_per_layer)}")
    print(f"\navg_rank_per_layer = {repr(avg_rank_per_layer)}")
