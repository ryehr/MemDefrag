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
)
from sdpa_tracing import probe_densities_sdpa, defrag_from_densities


def load_qa_data(dataset_name: str, group_num: int, unrelated_num: int):
    global tokenizer, args
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


def run_qa_benchmark(filtered_test_data, filtered_unrelated_data, model_reorder, model_gen):
    global tokenizer, args
    result_collection = []
    length_collection = []
    k_by_step = {}  # adaptive-K 模式下记录每个 time step 选出的 K
    input_device = get_input_device(model_gen)

    # 只在指定 time step 评测（记忆演化不受影响）；为空则每步都评测
    eval_step_set = None
    if getattr(args, "eval_steps", ""):
        eval_step_set = set(int(s) for s in args.eval_steps.split(","))

    def _should_eval(step):
        return eval_step_set is None or step in eval_step_set

    for i, item in enumerate(filtered_test_data):
        begin_time = time.time()
        indices = [1]
        this_length = []
        this_result = []
        question = item["question"] + " Answer: "
        context = item["context"] + "\n\n"

        gold_answer = item["answer"]
        print(f"\n[Group {i+1}]")

        if not args.borderline_test:
            knowledge_latent = model_knowledge(
                model_obj=model_gen,
                tokenizer=tokenizer,
                knowledge=context,
                add_special_tokens=True
            )
            if _should_eval(1):
                candidate_answer = generate_with_hidden_prefix(
                    model_obj=model_gen,
                    tokenizer=tokenizer,
                    knowledge_latent=knowledge_latent,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample
                )
            else:
                candidate_answer = None
        else:
            q_inputs = tokenizer(question, return_tensors="pt", add_special_tokens=True)
            answer_ids = model_gen.generate(
                input_ids=q_inputs["input_ids"].to(input_device),
                attention_mask=q_inputs["attention_mask"].to(input_device),
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
            )
            answer_ids = answer_ids[:, q_inputs["input_ids"].shape[1]:]
            candidate_answer = {
                'answer_text': tokenizer.decode(answer_ids[0], skip_special_tokens=True)
            }
            knowledge_latent = None

        print(f"Question: {question}")
        print(f"Gold answer: {gold_answer}")

        if candidate_answer is not None:
            print(f"Candidate answer: {candidate_answer['answer_text']}")
            if gold_answer.replace("</s>", "").strip().lower() in candidate_answer['answer_text'].lower():
                this_result.append(1)
            else:
                this_result.append(0)

        if not args.borderline_test:
            this_length.append(knowledge_latent["input_ids"].shape[1])
            indices.append(knowledge_latent["input_ids"].shape[1])
        else:
            this_length.append(0)

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

            step = j + 2  # time step n（step 1 是仅存 target 后的评测）
            if not _should_eval(step):
                continue

            k_record = [] if args.adaptive_k else None

            if args.reorder_base_layer >= 0 and args.tracing == "sdpa":
                # SDPA tracing：单模型（model_gen）上探针取密度 + 纯切片 defrag
                if args.shuffle_knowledge:
                    latent_used, indices_used = shuffle_knowledge(
                        knowledge_latent, indices)
                else:
                    latent_used, indices_used = knowledge_latent, indices
                densities = probe_densities_sdpa(
                    model_obj=model_gen, tokenizer=tokenizer,
                    knowledge_latent=latent_used, question=question,
                    layer_idx=args.reorder_base_layer, indices=indices_used,
                    query_mode=args.query_mode)
                reordered_knowledge_latent = defrag_from_densities(
                    latent_used, indices_used, densities, args.keep_num)

            elif args.reorder_base_layer >= 0:
                if not args.shuffle_knowledge:
                    reordered_knowledge_latent = reorder(
                        model_obj=model_reorder,
                        tokenizer=tokenizer,
                        knowledge_latent=knowledge_latent,
                        question=question,
                        reorder_base_layer=args.reorder_base_layer,
                        indices=indices,
                        keep_num=args.keep_num,
                        query_mode=args.query_mode,
                        adaptive_k=args.adaptive_k,
                        adaptive_tau=args.adaptive_tau,
                        adaptive_k_max=args.adaptive_k_max,
                        k_record=k_record
                    )
                else:
                    shuffled_knowledge_latent, shuffled_indices = shuffle_knowledge(
                        knowledge_latent, indices
                    )
                    reordered_knowledge_latent = reorder(
                        model_obj=model_reorder,
                        tokenizer=tokenizer,
                        knowledge_latent=shuffled_knowledge_latent,
                        question=question,
                        reorder_base_layer=args.reorder_base_layer,
                        indices=shuffled_indices,
                        keep_num=args.keep_num,
                        query_mode=args.query_mode,
                        adaptive_k=args.adaptive_k,
                        adaptive_tau=args.adaptive_tau,
                        adaptive_k_max=args.adaptive_k_max,
                        k_record=k_record
                    )

                if args.adaptive_k and k_record:
                    k_by_step.setdefault(step, []).append(k_record[-1])

            elif args.rotate_base_layer >= 0:
                reordered_knowledge_latent = rotate(
                    model_obj=model_reorder,
                    tokenizer=tokenizer,
                    knowledge_latent=knowledge_latent,
                    question=question,
                    rotate_base_layer=args.rotate_base_layer,
                    indices=indices,
                    window_size=args.window_num,
                    rotate_cutoff=args.rotate_cutoff,
                    query_mode=args.query_mode
                )

            else:
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

        result_collection.append(this_result)
        length_collection.append(this_length)
        accuracy = col_one_ratio_np(result_collection)
        avg_length = np.mean(length_collection, axis=0).tolist()
        print([f"{acc:.4f}" for acc in accuracy])
        print([f"{length:.1f}" for length in avg_length])
        if args.adaptive_k and k_by_step:
            mean_k = {s: float(np.mean(v)) for s, v in sorted(k_by_step.items())}
            print("Adaptive-K mean per step:", {s: f"{m:.2f}" for s, m in mean_k.items()})
        end_time = time.time()
        print(f"Step {i+1}: Time taken = {end_time - begin_time:.2f} seconds")

    return accuracy, avg_length, k_by_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-model, batch=1, no-vLLM KV test")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HF model name or local path") 
    # e.g. Qwen/Qwen2.5-7B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, google/gemma-2-9b-it
    parser.add_argument("--device", type=str,
                        default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dataset", type=str, default="nqa",
                        help="Dataset to use for QA benchmark", choices=["nqa", "squad"])
    parser.add_argument("--group_num", type=int, default=500)
    parser.add_argument("--unrelated_num", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--using_summary", action="store_true",
                        help="Whether to run borderline test with very long context")
    parser.add_argument("--using_llmlingua", action="store_true",
                        help="Whether to use LLM-Lingua for prompt compression")
    parser.add_argument("--llmlingua2", action="store_true",
                        help="Whether to use LLM-Lingua 2 for prompt compression")
    parser.add_argument("--lingua_rate", type=float, default=0.5,
                        help="Compression rate for LLM-Lingua")
    parser.add_argument("--borderline_test", action="store_true",
                        help="Whether to run borderline test with very long context")
    parser.add_argument("--reorder_base_layer", type=int, default=-1)
    parser.add_argument("--keep_num", type=int, default=5,
                        help="保留的最高 density 知识块数量；负数表示保留全部")
    parser.add_argument("--adaptive_k", action="store_true",
                        help="自适应 K：按 tracer 层降序密度的相邻比值 rho_K/rho_{K+1} > tau 确定截断点")
    parser.add_argument("--adaptive_tau", type=float, default=2.0,
                        help="自适应 K 的 margin 阈值 tau")
    parser.add_argument("--adaptive_k_max", type=int, default=4,
                        help="自适应 K 的上限 K_max")
    parser.add_argument("--eval_steps", type=str, default="",
                        help="逗号分隔的评测 time step（如 '1,10,20,30,40,50'）；为空则每步评测")
    parser.add_argument("--tracing", type=str, default="sdpa", choices=["sdpa", "eager"],
                        help="sdpa（默认，推荐）=单模型 SDPA 探针 tracing：不物化 S×S 注意力、"
                             "不加载 eager 模型副本，显存开销显著更低；"
                             "eager=参考实现（--adaptive_k / --rotate_base_layer 需要）")
    parser.add_argument("--log_file", type=str, default="./result_April.log",
                        help="结果日志文件路径")
    parser.add_argument("--shuffle_knowledge", action="store_true",
                        help="Whether to shuffle knowledge blocks before reordering")
    parser.add_argument("--rotate_base_layer", type=int, default=-1)
    parser.add_argument("--window_num", type=int, default=5,
                        help="rotate 模式下滑动窗口包含的 chunk 数量")
    parser.add_argument("--rotate_cutoff", action="store_true")
    parser.add_argument("--max_knowledge_tokens", type=int, default=12800)
    parser.add_argument("--forget_strategy", type=str, default="none",
                        choices=["none", "random", "perplexity"])
    parser.add_argument("--query_mode", type=str, default="last",
                        choices=["last", "mean"])
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device_map_mode", type=str, default="auto",
                        choices=["auto", "balanced", "balanced_low_0", "sequential"],
                        help="HF accelerate device_map mode")

    args = parser.parse_args()
    print(args)

    if args.reorder_base_layer >= 0 and args.rotate_base_layer >= 0:
        raise ValueError("reorder_base_layer 和 rotate_base_layer 不能同时 >= 0，请二选一。")
    if args.tracing == "sdpa" and (args.adaptive_k or args.rotate_base_layer >= 0):
        raise ValueError("--tracing sdpa 暂不支持 --adaptive_k / --rotate_base_layer。")

    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("QA_Benchmark")

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

    if (args.reorder_base_layer >= 0 or args.rotate_base_layer >= 0) and args.tracing == "eager":
        load_kwargs_reorder = {
            "torch_dtype": dtype,
            "device_map": args.device_map_mode,
            "max_memory": max_memory,
            "attn_implementation": "eager",
        }
        model_reorder = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **load_kwargs_reorder
        )
        model_reorder.eval()
    else:
        model_reorder = None

    load_kwargs_gen = {
        "torch_dtype": dtype,
        "device_map": args.device_map_mode,
        "max_memory": max_memory,
    }
    model_gen = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **load_kwargs_gen
    )
    model_gen.eval()

    filtered_test_data, filtered_unrelated_data, compression_ratio = load_qa_data(
        dataset_name=args.dataset,
        group_num=args.group_num,
        unrelated_num=args.unrelated_num
    )

    accuracy, avg_length, k_by_step = run_qa_benchmark(
        filtered_test_data,
        filtered_unrelated_data,
        model_reorder=model_reorder,
        model_gen=model_gen
    )

    logger.info("\n")
    logger.info(
        f"Model: {args.model_name}, Dataset: {args.dataset}, Group num: {args.group_num}, "
        f"Unrelated num: {args.unrelated_num}, Compression ratio: {compression_ratio:.4f}"
    )
    logger.info(args)
    if args.eval_steps:
        logger.info(f"Eval steps: {sorted(int(s) for s in args.eval_steps.split(','))}")
    print("\n[Final accuracy with increasing unrelated contexts]")
    logger.info(avg_length)
    logger.info(accuracy)
    if args.adaptive_k and k_by_step:
        mean_k_per_step = {s: round(float(np.mean(v)), 3) for s, v in sorted(k_by_step.items())}
        all_ks = [k for v in k_by_step.values() for k in v]
        k_hist = {k: all_ks.count(k) for k in sorted(set(all_ks))}
        logger.info(f"Adaptive-K (tau={args.adaptive_tau}, K_max={args.adaptive_k_max}) "
                    f"mean K per step: {mean_k_per_step}")
        logger.info(f"Adaptive-K overall histogram: {k_hist}, mean K = {float(np.mean(all_ks)):.3f}")
