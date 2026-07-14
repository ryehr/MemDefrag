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
    col_one_ratio_np,
    model_knowledge,
    concatenate_knowledge_latent,
    rebuild_indices_from_segment_lengths,
    maybe_forget_before_concat,
    generate_with_hidden_prefix,
    reorder,
    shuffle_knowledge,
    rotate,
    compute_attention_density_all_layers,
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
    global tokenizer, args
    result_collection = []
    length_collection = []
    # density_all_groups[g] 是第 g 个 group 的 L×steps 矩阵（list of lists）
    density_all_groups = []
    input_device = get_input_device(model_gen)

    num_layers = len(model_gen.model.layers)

    for i, item in enumerate(filtered_test_data):
        begin_time = time.time()
        indices = [1]
        this_length = []
        this_result = []
        # 本 group 的 density 矩阵: num_layers 行, 每行随 step 增长
        this_density = [[] for _ in range(num_layers)]

        question = item["question"] + " Answer: "
        context = item["context"] + "\n\n"

        gold_answer = item["answer"]
        print(f"\n[Group {i+1}]")

        knowledge_latent = model_knowledge(
            model_obj=model_gen,
            tokenizer=tokenizer,
            knowledge=context,
            add_special_tokens=True
        )

        # 记录第一个 context 的 token 长度（含 bos），用于后续 density 计算
        first_context_len = knowledge_latent["input_ids"].shape[1]

        candidate_answer = generate_with_hidden_prefix(
            model_obj=model_gen,
            tokenizer=tokenizer,
            knowledge_latent=knowledge_latent,
            question=question,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample
        )

        print(f"Question: {question}")
        print(f"Gold answer: {gold_answer}")
        print(f"Candidate answer: {candidate_answer['answer_text']}")

        if gold_answer.replace("</s>", "").strip().lower() in candidate_answer['answer_text'].lower():
            this_result.append(1)
        else:
            this_result.append(0)

        this_length.append(knowledge_latent["input_ids"].shape[1])
        indices.append(knowledge_latent["input_ids"].shape[1])

        # ---- Step 0 (仅第一个 context): 计算所有层的 attention density ----
        step0_densities = compute_attention_density_all_layers(
            model_obj=model_gen,
            tokenizer=tokenizer,
            knowledge_latent=knowledge_latent,
            question=question,
            first_context_len=first_context_len
        )
        for l in range(num_layers):
            this_density[l].append(step0_densities[l])

        for j, unrelated_text in enumerate(filtered_unrelated_data):
            unrelated_text_ids = tokenizer(
                unrelated_text,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"].to(input_device)[:, :512]

            truncated_text = tokenizer.decode(unrelated_text_ids[0]) + "\n\n"

            temp_knowledge_latent = model_knowledge(
                model_obj=model_gen,
                tokenizer=tokenizer,
                knowledge=truncated_text,
                add_special_tokens=False
            )

            knowledge_latent = maybe_forget_before_concat(
                knowledge_latent=knowledge_latent,
                new_knowledge_latent=temp_knowledge_latent,
                forget_strategy=args.forget_strategy,
                max_knowledge_tokens=args.max_knowledge_tokens
            )

            # forget 后，旧 indices 已经过期，立刻按当前 segment_lengths 重建
            indices = rebuild_indices_from_segment_lengths(
                knowledge_latent["segment_lengths"]
            )

            knowledge_latent = concatenate_knowledge_latent(
                knowledge_latent=knowledge_latent,
                new_knowledge_latent=temp_knowledge_latent
            )

            # 拼接新 knowledge 后，再次重建 indices，保证与当前 latent 完全一致
            indices = rebuild_indices_from_segment_lengths(
                knowledge_latent["segment_lengths"]
            )

            this_length.append(knowledge_latent["input_ids"].shape[1])

            reordered_knowledge_latent = knowledge_latent

            candidate_answer = generate_with_hidden_prefix(
                model_obj=model_gen,
                tokenizer=tokenizer,
                knowledge_latent=reordered_knowledge_latent,
                question=question,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample
            )

            del reordered_knowledge_latent
            torch.cuda.empty_cache()

            if gold_answer.replace("</s>", "").strip().lower() in candidate_answer['answer_text'].lower():
                this_result.append(1)
            else:
                this_result.append(0)

            # ---- Step j+1: 计算所有层的 attention density ----
            # 注意: forget 可能缩短了第一个 context 的实际长度，
            # 但这里仍以原始 first_context_len 为准（如超出 prefix_len 则自动截断到 prefix_len）
            actual_first_ctx_len = min(first_context_len, knowledge_latent["input_ids"].shape[1])
            step_densities = compute_attention_density_all_layers(
                model_obj=model_gen,
                tokenizer=tokenizer,
                knowledge_latent=knowledge_latent,
                question=question,
                first_context_len=actual_first_ctx_len
            )
            for l in range(num_layers):
                this_density[l].append(step_densities[l])

        density_all_groups.append(this_density)
        result_collection.append(this_result)
        length_collection.append(this_length)
        accuracy = col_one_ratio_np(result_collection)
        avg_length = np.mean(length_collection, axis=0).tolist()
        print([f"{acc:.4f}" for acc in accuracy])
        print([f"{length:.1f}" for length in avg_length])
        end_time = time.time()
        print(f"Step {i+1}: Time taken = {end_time - begin_time:.2f} seconds")

    # ---- 汇总: 对所有 group 求平均 density 矩阵 (L x steps) ----
    num_groups = len(density_all_groups)
    num_steps = 1 + len(filtered_unrelated_data)  # step 0 + unrelated_num steps

    avg_density_matrix = np.zeros((num_layers, num_steps), dtype=np.float64)
    for g in range(num_groups):
        for l in range(num_layers):
            for s in range(min(len(density_all_groups[g][l]), num_steps)):
                avg_density_matrix[l, s] += density_all_groups[g][l][s]

    if num_groups > 0:
        avg_density_matrix /= num_groups

    return accuracy, avg_length, avg_density_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-model, batch=1, no-vLLM KV test")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HF model name or local path")
    parser.add_argument("--device", type=str,
                        default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dataset", type=str, default="nqa",
                        help="Dataset to use for QA benchmark", choices=["nqa", "squad"])
    parser.add_argument("--group_num", type=int, default=500)
    parser.add_argument("--unrelated_num", type=int, default=19)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device_map_mode", type=str, default="auto",
                        choices=["auto", "balanced", "balanced_low_0", "sequential"],
                        help="HF accelerate device_map mode")

    args = parser.parse_args()
    print(args)



    logging.basicConfig(
        filename="./investigation.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("Accuracy")

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

    accuracy, avg_length, avg_density_matrix = run_qa_benchmark(
        filtered_test_data,
        filtered_unrelated_data,
        model_gen=model_gen
    )

    logger.info("\n")
    logger.info(
        f"Model: {args.model_name}, Dataset: {args.dataset}, Group num: {args.group_num}, "
        f"Unrelated num: {args.unrelated_num}"
    )
    logger.info(args)
    print("\n[Final accuracy with increasing unrelated contexts]")
    logger.info(avg_length)
    logger.info(accuracy)

    # 记录 attention density 矩阵 (L x steps)
    logger.info("=== Attention Density Matrix (L x steps) ===")
    logger.info(f"Shape: {avg_density_matrix.shape}")
    num_layers, num_steps = avg_density_matrix.shape
    # 表头：step 编号
    header = "Layer\\Step | " + " | ".join([f"step{s}" for s in range(num_steps)])
    logger.info(header)
    for l in range(num_layers):
        row_str = f"Layer {l:2d}  | " + " | ".join(
            [f"{avg_density_matrix[l, s]:.6f}" for s in range(num_steps)]
        )
        logger.info(row_str)
    # 同时以 list 格式记录，方便程序化读取
    logger.info("Density matrix (nested list):")
    logger.info(avg_density_matrix.tolist())
