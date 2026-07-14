import argparse
import torch
import json
import random
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from llmlingua import PromptCompressor

from utilities import (
    get_input_device,
    get_max_memory,
    model_knowledge,
    concatenate_knowledge_latent,
    rebuild_indices_from_segment_lengths,
    maybe_forget_before_concat,
    reorder,
    shuffle_knowledge,
    rotate,
    compute_attention_density_all_layers,
    compute_attention_density_all_segments,
    place_target_at_position,
)


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




def run_qa_benchmark(filtered_test_data, filtered_unrelated_data, model_gen):
    """
    只计算注意力密度和排名（不做推理）：
    1. 对每个 group，先把被测知识（第一个 context）和 19 个 unrelated 知识
       各自独立编码成 knowledge_latent。
    2. 遍历位置 0~19（共 20 个位置），将被测知识放到该位置，其他 19 个知识按顺序填充。
    3. 在每个位置上计算注意力密度和排名。
    """
    global tokenizer, args
    num_positions = 1 + len(filtered_unrelated_data)  # 20 (1 target + 19 unrelated)
    density_all_groups = []  # [group] -> [num_layers][num_positions] 目标 density
    rank_all_groups = []     # [group] -> [num_layers][num_positions] 目标 rank

    input_device = get_input_device(model_gen)
    num_layers = len(model_gen.model.layers)

    for i, item in enumerate(filtered_test_data):
        begin_time = time.time()
        this_density = [[] for _ in range(num_layers)]
        this_rank = [[] for _ in range(num_layers)]

        question = item["question"] + " Answer: "
        context = item["context"] + "\n\n"
        print(f"\n[Group {i+1}]")
        print(f"Question: {question}")

        # ---- 步骤 1: 预编码所有知识 ----
        # 被测知识 (index 0)，带 bos
        target_latent = model_knowledge(
            model_obj=model_gen,
            tokenizer=tokenizer,
            knowledge=context,
            add_special_tokens=True
        )

        # unrelated 知识 (index 1~19)，不带 bos
        unrelated_latents = []
        for j, unrelated_text in enumerate(filtered_unrelated_data):
            unrelated_text_ids = tokenizer(
                unrelated_text,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"].to(input_device)[:, :512]

            truncated_text = tokenizer.decode(unrelated_text_ids[0]) + "\n\n"

            temp_latent = model_knowledge(
                model_obj=model_gen,
                tokenizer=tokenizer,
                knowledge=truncated_text,
                add_special_tokens=False
            )
            unrelated_latents.append(temp_latent)

        # knowledge_latents_list[0] = target (带 bos)
        # knowledge_latents_list[1..19] = unrelated (不带 bos)
        knowledge_latents_list = [target_latent] + unrelated_latents

        # ---- 步骤 2: 遍历 20 个位置 ----
        for pos in range(num_positions):
            # 将被测知识 (index 0) 放到 position=pos
            combined_latent, target_seg_idx = place_target_at_position(
                knowledge_latents_list=knowledge_latents_list,
                target_idx=0,
                position=pos
            )

            # --- 注意力密度和排名 ---
            target_densities, target_ranks, _ = compute_attention_density_all_segments(
                model_obj=model_gen,
                tokenizer=tokenizer,
                knowledge_latent=combined_latent,
                question=question,
                target_segment_idx=target_seg_idx,
                query_mode=args.query_mode
            )
            for l in range(num_layers):
                this_density[l].append(target_densities[l])
                this_rank[l].append(target_ranks[l])
            del target_densities, target_ranks

            print(f"  Position {pos}: done")

            del combined_latent
            torch.cuda.empty_cache()

        density_all_groups.append(this_density)
        rank_all_groups.append(this_rank)

        # 截至当前的 rank 分布 (按 layer 记录, pos order)
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

        # 清理本 group 的独立 latent
        del target_latent, unrelated_latents, knowledge_latents_list
        torch.cuda.empty_cache()

    # ---- 反转: pos=0(最远) -> step20, pos=19(最近) -> step1 ----
    # 将所有按 pos 收集的数据反转，使 index 0 = step1(最近), index 19 = step20(最远)
    for g in range(len(density_all_groups)):
        for l in range(num_layers):
            density_all_groups[g][l] = density_all_groups[g][l][::-1]
            rank_all_groups[g][l] = rank_all_groups[g][l][::-1]

    # ---- 汇总: 对所有 group 求平均 density 矩阵 (L x steps) ----
    num_groups = len(density_all_groups)

    avg_density_matrix = np.zeros((num_layers, num_positions), dtype=np.float64)
    for g in range(num_groups):
        for l in range(num_layers):
            for s in range(min(len(density_all_groups[g][l]), num_positions)):
                avg_density_matrix[l, s] += density_all_groups[g][l][s]
    if num_groups > 0:
        avg_density_matrix /= num_groups

    # ---- 汇总排名分布 (按 layer) ----
    # rank_distribution[l][rank] = 第 l 层上，在所有 group × 所有 step 中目标知识获得该排名的比例
    # avg_rank_per_layer[l] = 第 l 层上所有 group × 所有 step 的平均 rank
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

        # 转换为比例
        total = sum(rank_counter.values())
        rank_dist = {}
        for rank, count in sorted(rank_counter.items()):
            rank_dist[f"rank{rank}"] = f"{count / total * 100:.1f}%"
        rank_distribution.append(rank_dist)
        avg_rank_per_layer.append(round(rank_sum / rank_count, 2) if rank_count > 0 else 0.0)

    return avg_density_matrix, rank_distribution, avg_rank_per_layer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-model, batch=1, no-vLLM KV test")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it",
                        help="HF model name or local path") 
    # e.g. Qwen/Qwen2.5-7B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, google/gemma-2-9b-it
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
        filename="./investigation_new_models.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("Rank")

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


    load_kwargs_reorder = {
        "torch_dtype": dtype,
        "device_map": args.device_map_mode,
        "max_memory": max_memory,
        "attn_implementation": "eager",
    }
    model_gen = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **load_kwargs_reorder
    )
    model_gen.eval()



    filtered_test_data, filtered_unrelated_data = load_qa_data(
        dataset_name=args.dataset,
        group_num=args.group_num,
        unrelated_num=args.unrelated_num
    )

    avg_density_matrix, rank_distribution, avg_rank_per_layer = run_qa_benchmark(
        filtered_test_data,
        filtered_unrelated_data,
        model_gen=model_gen
    )

    print(
        f"\nModel: {args.model_name}, Dataset: {args.dataset}, Group num: {args.group_num}, "
        f"Unrelated num: {args.unrelated_num}"
    )
    print(args)
    print("\n[Final results: step1=nearest to query, step20=farthest from query]")

    # ---- 以 Python 可直接复制的格式输出 ----
    num_layers, num_positions = avg_density_matrix.shape

    # attention density 矩阵: list[list[float]], shape = [num_layers, num_positions]
    density_list = avg_density_matrix.tolist()
    logger.info(f"# === Model: {args.model_name}, Dataset: {args.dataset}, Groups: {args.group_num}, Unrelated: {args.unrelated_num} ===")
    logger.info(f"# Shape: ({num_layers}, {num_positions})")
    logger.info(f"density_matrix = {repr(density_list)}")

    # 排名分布: list[dict], 每个 layer 一个 dict
    # 把百分比字符串转成浮点数方便后续处理
    rank_dist_python = []
    for l in range(num_layers):
        layer_dict = {}
        for k, v in rank_distribution[l].items():
            layer_dict[k] = float(v.replace("%", ""))
        rank_dist_python.append(layer_dict)

    logger.info(f"rank_distribution = {repr(rank_dist_python)}")
    print(f"\nrank_distribution = {repr(rank_dist_python)}")

    # 每层平均 rank: list[float], 长度 = num_layers
    logger.info(f"avg_rank_per_layer = {repr(avg_rank_per_layer)}")
    print(f"\navg_rank_per_layer = {repr(avg_rank_per_layer)}")
