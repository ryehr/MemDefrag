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
    col_one_ratio_np,
)


# ============================================================
#  数据加载（与 QA_benchmark_investigation_acc.py 保持一致）
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
#  纯文本生成（普通 prompt 拼接 + generate）
# ============================================================

def generate_with_context(model, tokenizer, context_text, question,
                          max_new_tokens, do_sample):
    """
    将 context + question 拼成普通 prompt，直接走 model.generate。
    """
    prompt = context_text + question
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer_ids = output_ids[:, input_ids.shape[1]:]
    answer_text = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    return answer_text


# ============================================================
#  文本版本：question 最后 token 对“第一个 context block”
#  在所有 decoder layer 上的 attention density
# ============================================================

def compute_attention_density_all_layers_text(
    model, tokenizer, context_text, question, first_context_len
):
    """
    将 context_text + question 整体送入模型 forward，逐层用 forward hook 抓
    attention 权重，**只保留 question 最后 token 对应的那一行**，立刻算出
    density 标量后释放该层 attention 矩阵，避免显存中同时驻留所有层的
    [1, H, N, N] 张量。

    返回:
        densities: list[float]，长度 = num_layers
    """
    input_device = next(model.parameters()).device

    # 整体编码 prompt（context + question），add_special_tokens=True 保留 bos
    prompt = context_text + question
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(input_device)
    attention_mask = enc["attention_mask"].to(input_device)

    seq_len = input_ids.shape[1]

    # 防御性截断：避免 first_context_len 超过实际长度
    first_context_len = min(first_context_len, seq_len)
    block_start = 1
    block_end = first_context_len
    block_len = max(0, block_end - block_start)

    num_layers = len(model.model.layers)
    densities = [0.0] * num_layers

    # 包装每层 self_attn.forward 让其输出 attentions（eager 实现下生效）
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

    # forward hook：拿到 attention 后立刻提取最后一行 → 算 density → 丢弃
    def make_capture_hook(layer_idx):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                attn_weights = outputs[1]
                if attn_weights is not None:
                    # 只取 question 最后一个 token 对前缀的注意力
                    # attn_weights: [1, num_heads, seq_len, seq_len]
                    last_row = attn_weights[0, :, -1, :seq_len]  # [H, N]
                    last_row = last_row.mean(dim=0)              # [N]
                    if block_len > 0:
                        density = (last_row[block_start:block_end].sum()
                                   / block_len).item()
                    else:
                        density = 0.0
                    densities[layer_idx] = density
                    # 不返回任何东西，原 outputs 仍按正常 forward 流程传递，
                    # 但本 hook 不再持有 attn_weights 的引用
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
                output_attentions=False,  # 不让 HF 把所有层 attentions 累到 outputs
                use_cache=False,
                return_dict=True,
            )
    finally:
        for h in capture_handles:
            h.remove()
        for layer_idx in range(num_layers):
            model.model.layers[layer_idx].self_attn.forward = orig_forwards[layer_idx]

    del input_ids, attention_mask
    torch.cuda.empty_cache()
    return densities


# ============================================================
#  QA Benchmark 主循环（文本版）
# ============================================================

def run_qa_benchmark(filtered_test_data, filtered_unrelated_data, model):
    global tokenizer, args
    result_collection = []
    length_collection = []
    # density_all_groups[g] 是第 g 个 group 的 L×steps 矩阵
    density_all_groups = []

    num_layers = len(model.model.layers)

    for i, item in enumerate(filtered_test_data):
        begin_time = time.time()
        this_length = []
        this_result = []
        this_density = [[] for _ in range(num_layers)]

        question = item["question"] + " Answer: "
        first_context = item["context"] + "\n\n"
        gold_answer = item["answer"]

        print(f"\n[Group {i+1}]")

        # 当前累积 context（纯文本）
        context_text = first_context

        # 记录第一个 context 在整段 prompt 中的 token 长度（含 bos）
        # 与 latent 版本对齐：使用 add_special_tokens=True
        first_context_len = tokenizer(
            first_context, add_special_tokens=True
        )["input_ids"]
        first_context_len = len(first_context_len)

        # ---- Step 0：仅第一个 context ----
        candidate_answer_text = generate_with_context(
            model=model,
            tokenizer=tokenizer,
            context_text=context_text,
            question=question,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )

        print(f"Question: {question}")
        print(f"Gold answer: {gold_answer}")
        print(f"Candidate answer: {candidate_answer_text}")

        if gold_answer.replace("</s>", "").strip().lower() in candidate_answer_text.lower():
            this_result.append(1)
        else:
            this_result.append(0)

        # 当前 prompt（context+question）总 token 长度（含 bos）
        cur_total_len = len(tokenizer(
            context_text + question, add_special_tokens=True
        )["input_ids"])
        this_length.append(cur_total_len)

        # Step 0 attention density
        step0_densities = compute_attention_density_all_layers_text(
            model=model,
            tokenizer=tokenizer,
            context_text=context_text,
            question=question,
            first_context_len=first_context_len,
        )
        for l in range(num_layers):
            this_density[l].append(step0_densities[l])

        # ---- 逐步拼接 unrelated context ----
        for j, unrelated_text in enumerate(filtered_unrelated_data):
            # 截断 unrelated 到最多 512 个 token，并解码回文本
            unrelated_ids = tokenizer(
                unrelated_text, add_special_tokens=False
            )["input_ids"][:512]
            truncated_text = tokenizer.decode(unrelated_ids) + "\n\n"

            # 拼接到右侧（first context 始终位于最左，保持其 token 范围不变）
            context_text = context_text + truncated_text

            cur_total_len = len(tokenizer(
                context_text + question, add_special_tokens=True
            )["input_ids"])
            this_length.append(cur_total_len)

            # 生成
            candidate_answer_text = generate_with_context(
                model=model,
                tokenizer=tokenizer,
                context_text=context_text,
                question=question,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
            )

            if gold_answer.replace("</s>", "").strip().lower() in candidate_answer_text.lower():
                this_result.append(1)
            else:
                this_result.append(0)

            # Step j+1 attention density
            step_densities = compute_attention_density_all_layers_text(
                model=model,
                tokenizer=tokenizer,
                context_text=context_text,
                question=question,
                first_context_len=first_context_len,
            )
            for l in range(num_layers):
                this_density[l].append(step_densities[l])

            torch.cuda.empty_cache()

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


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text-level QA Benchmark with attention density investigation"
    )
    parser.add_argument("--model_name", type=str, required=True,
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
        filename="./investigation_text.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("Accuracy_Text")

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

    # eager attention 实现，确保 output_attentions=True 能拿到真实 attention 权重
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

    accuracy, avg_length, avg_density_matrix = run_qa_benchmark(
        filtered_test_data,
        filtered_unrelated_data,
        model=model,
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
    header = "Layer\\Step | " + " | ".join([f"step{s}" for s in range(num_steps)])
    logger.info(header)
    for l in range(num_layers):
        row_str = f"Layer {l:2d}  | " + " | ".join(
            [f"{avg_density_matrix[l, s]:.6f}" for s in range(num_steps)]
        )
        logger.info(row_str)
    logger.info("Density matrix (nested list):")
    logger.info(avg_density_matrix.tolist())
