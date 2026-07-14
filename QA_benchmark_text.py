"""
文本级别的 Knowledge Retention QA Benchmark
对应 main.py 的 latent memory 版本，但不使用 hidden states 前缀注入，
而是直接将 context 文本拼接后与 question 一起喂给模型做常规 generate。
"""

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
    get_max_memory,
    col_one_ratio_np,
)


# ============================================================
#  文本级别的 forget：按 token 从左侧截断旧 context
# ============================================================

def truncate_context_by_tokens(context_text, max_tokens, tokenizer):
    """
    当 context_text 的 token 数超过 max_tokens 时，
    从左侧截掉多余的 token，只保留最右边的 max_tokens 个 token 对应的文本。
    """
    ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return context_text
    kept_ids = ids[-max_tokens:]
    return tokenizer.decode(kept_ids, skip_special_tokens=False)


# ============================================================
#  数据加载（与 main.py 完全一致）
# ============================================================

def load_qa_data(dataset_name, group_num, unrelated_num, tokenizer, args):
    random.seed(args.seed)
    if args.using_llmlingua:
        if args.llmlingua2:
            llm_lingua = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=args.llmlingua2,
            )
        else:
            llm_lingua = PromptCompressor(
                model_name=args.model_name,
            )

    if dataset_name == "nqa":
        path = "./data_new/nqa/"
    elif dataset_name == "squad":
        path = "./data_new/squad/"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    if not args.using_summary:
        test_path = path + "val_data.json"
    else:
        test_path = path + "val_data_with_summary(gpt-5.2).json"

    if not args.using_summary:
        unrelated_path = path + "unrelated_contexts_from_train.json"
    else:
        unrelated_path = path + "train_data_with_summary(gpt-5.2).json"

    with open(test_path, 'r', encoding="utf-8-sig") as file:
        test_data = json.load(file)
    with open(unrelated_path, 'r', encoding="utf-8-sig") as file:
        unrelated_data = json.load(file)

    random.shuffle(test_data)
    random.shuffle(unrelated_data)

    filtered_test_data = []
    filtered_unrelated_data = []

    if args.using_summary or args.using_llmlingua:
        context_length = 0
        summary_length = 0

    for item in test_data:
        if len(filtered_test_data) >= group_num:
            break
        context = item["context"]
        if not args.using_summary and not args.using_llmlingua:
            if len(tokenizer(context)["input_ids"]) <= 512:
                filtered_test_data.append(item)
        elif args.using_summary:
            if len(tokenizer(context)["input_ids"]) <= 512 and item["summary"] != "Valid":
                filtered_test_data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "context": item["summary"]
                })
                context_length += len(tokenizer(context)["input_ids"])
                summary_length += len(tokenizer(item["summary"])["input_ids"])
        elif args.using_llmlingua:
            if len(tokenizer(context)["input_ids"]) <= 512:
                compressed = llm_lingua.compress_prompt(
                    context, instruction="", question="", rate=args.lingua_rate
                )
                compressed_context = compressed["compressed_prompt"]
                filtered_test_data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "context": compressed_context
                })
                context_length += len(tokenizer(context)["input_ids"])
                summary_length += len(tokenizer(compressed_context)["input_ids"])

    for item in unrelated_data:
        if len(filtered_unrelated_data) >= unrelated_num:
            break
        if not args.using_summary and not args.using_llmlingua:
            filtered_unrelated_data.append(item)
        elif args.using_summary:
            filtered_unrelated_data.append(item["summary"])
            context_length += min(len(tokenizer(item["context"])["input_ids"]), 512)
            summary_length += len(tokenizer(item["summary"])["input_ids"])
        elif args.using_llmlingua:
            truncated_item = tokenizer.decode(
                tokenizer(item, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :512][0]
            )
            compressed = llm_lingua.compress_prompt(
                truncated_item, instruction="", question="", rate=args.lingua_rate
            )
            compressed_context = compressed["compressed_prompt"]
            filtered_unrelated_data.append(compressed_context)
            context_length += min(len(tokenizer(item)["input_ids"]), 512)
            summary_length += len(tokenizer(compressed_context)["input_ids"])

    if args.using_summary or args.using_llmlingua:
        compression_ratio = summary_length / context_length
    else:
        compression_ratio = 1.0

    print(f"Compression ratio: {compression_ratio:.4f}")
    return filtered_test_data, filtered_unrelated_data, compression_ratio


# ============================================================
#  文本级别的生成
# ============================================================

def generate_with_context(model, tokenizer, context_text, question, max_new_tokens, do_sample):
    """
    把 context + question 拼成一个 prompt，直接用 model.generate 做生成。
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

    # 只取新生成的部分
    answer_ids = output_ids[:, input_ids.shape[1]:]
    answer_text = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    return answer_text


# ============================================================
#  QA Benchmark 主循环
# ============================================================

def run_qa_benchmark(filtered_test_data, filtered_unrelated_data, model, tokenizer, args):
    result_collection = []
    length_collection = []

    for i, item in enumerate(filtered_test_data):
        begin_time = time.time()
        this_length = []
        this_result = []
        question = item["question"] + " Answer: "
        context_text = item["context"] + "\n\n"
        gold_answer = item["answer"]

        print(f"\n[Group {i+1}]")

        # ---------- 第 0 步：只用 gold context 回答 ----------
        if not args.borderline_test:
            answer_text = generate_with_context(
                model=model,
                tokenizer=tokenizer,
                context_text=context_text,
                question=question,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample
            )
        else:
            # borderline test：不给 context，只用 question
            answer_text = generate_with_context(
                model=model,
                tokenizer=tokenizer,
                context_text="",
                question=question,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample
            )

        print(f"Question: {question}")
        print(f"Gold answer: {gold_answer}")
        print(f"Candidate answer: {answer_text}")

        if gold_answer.replace("</s>", "").strip().lower() in answer_text.lower():
            this_result.append(1)
        else:
            this_result.append(0)

        ctx_token_len = len(tokenizer(context_text, add_special_tokens=False)["input_ids"])
        this_length.append(ctx_token_len)

        # ---------- 逐步拼接 unrelated context ----------
        for j, unrelated_text in enumerate(filtered_unrelated_data):
            # 截断 unrelated 到最多 512 token
            unrelated_ids = tokenizer(
                unrelated_text, add_special_tokens=False
            )["input_ids"][:512]
            truncated_text = tokenizer.decode(unrelated_ids) + "\n\n"

            # 新内容拼到右边
            context_text = context_text + truncated_text

            # forget：如果超过 max_knowledge_tokens，从左侧截断
            if args.forget_strategy != "none":
                context_text = truncate_context_by_tokens(
                    context_text, args.max_knowledge_tokens, tokenizer
                )

            ctx_token_len = len(tokenizer(context_text, add_special_tokens=False)["input_ids"])
            this_length.append(ctx_token_len)

            # 生成答案
            answer_text = generate_with_context(
                model=model,
                tokenizer=tokenizer,
                context_text=context_text,
                question=question,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample
            )

            if gold_answer.replace("</s>", "").strip().lower() in answer_text.lower():
                this_result.append(1)
            else:
                this_result.append(0)

        result_collection.append(this_result)
        length_collection.append(this_length)
        accuracy = col_one_ratio_np(result_collection)
        avg_length = np.mean(length_collection, axis=0).tolist()
        print([f"{acc:.4f}" for acc in accuracy])
        print([f"{length:.1f}" for length in avg_length])
        end_time = time.time()
        print(f"Step {i+1}: Time taken = {end_time - begin_time:.2f} seconds")

    return accuracy, avg_length


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text-level Knowledge Retention QA Benchmark (no latent / no hidden prefix)"
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HF model name or local path")
    parser.add_argument("--device", type=str,
                        default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dataset", type=str, default="nqa",
                        choices=["nqa", "squad"])
    parser.add_argument("--group_num", type=int, default=500)
    parser.add_argument("--unrelated_num", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true")
    # --- summary / llmlingua ---
    parser.add_argument("--using_summary", action="store_true")
    parser.add_argument("--using_llmlingua", action="store_true")
    parser.add_argument("--llmlingua2", action="store_true")
    parser.add_argument("--lingua_rate", type=float, default=0.5)
    # --- borderline test ---
    parser.add_argument("--borderline_test", action="store_true",
                        help="不提供 context，纯靠模型自身知识回答")
    # --- forget ---
    parser.add_argument("--max_knowledge_tokens", type=int, default=12800,
                        help="context 的最大 token 数上限，超出后从左侧截断")
    parser.add_argument("--forget_strategy", type=str, default="none",
                        choices=["none", "truncate"],
                        help="forget 策略：none 不截断，truncate 从左侧截断超出部分")
    # --- misc ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map_mode", type=str, default="auto",
                        choices=["auto", "balanced", "balanced_low_0", "sequential"])

    args = parser.parse_args()
    print(args)

    logging.basicConfig(
        filename="./result_text.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("QA_Benchmark_Text")

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

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device_map_mode,
        "max_memory": max_memory,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.eval()

    filtered_test_data, filtered_unrelated_data, compression_ratio = load_qa_data(
        dataset_name=args.dataset,
        group_num=args.group_num,
        unrelated_num=args.unrelated_num,
        tokenizer=tokenizer,
        args=args,
    )

    accuracy, avg_length = run_qa_benchmark(
        filtered_test_data=filtered_test_data,
        filtered_unrelated_data=filtered_unrelated_data,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )

    logger.info("\n")
    logger.info(
        f"Model: {args.model_name}, Dataset: {args.dataset}, Group num: {args.group_num}, "
        f"Unrelated num: {args.unrelated_num}, Compression ratio: {compression_ratio:.4f}"
    )
    logger.info(args)
    print("\n[Final accuracy with increasing unrelated contexts]")
    logger.info(avg_length)
    logger.info(accuracy)
