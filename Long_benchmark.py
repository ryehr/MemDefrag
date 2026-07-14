import argparse
import torch
import json
import random
import math
import logging
import numpy as np
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from datasets import load_dataset

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
from bert_score import score as bert_score_fn
import nltk

from utilities import (
    get_input_device,
    get_max_memory,
    model_knowledge,
    concatenate_knowledge_latent,
    rebuild_indices_from_segment_lengths,
    maybe_forget_before_concat,
    generate_with_hidden_prefix,
    reorder,
    shuffle_knowledge,
    rotate,
    qa_f1_score,
)


def load_data(dataset_name: str, max_new_tokens: int):
    global tokenizer, args

    def split_into_chunks(context: str, min_tokens: int):
        if not context:
            return []

        enc = tokenizer(
            context,
            add_special_tokens=False,
            return_offsets_mapping=True
        )

        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        if not input_ids:
            return []

        valid_breaks = []
        for i, (_, end) in enumerate(offsets):
            if end > 0 and context[end - 1] in {"\n", "。", "."}:
                valid_breaks.append(i + 1)

        if not valid_breaks:
            return [context]

        chunks = []
        start_tok = 0
        break_ptr = 0

        while start_tok < len(input_ids):
            target_tok = start_tok + min_tokens

            while break_ptr < len(valid_breaks) and valid_breaks[break_ptr] < target_tok:
                break_ptr += 1

            if break_ptr < len(valid_breaks):
                end_tok = valid_breaks[break_ptr]

                start_char = offsets[start_tok][0]
                end_char = offsets[end_tok - 1][1]
                chunks.append(context[start_char:end_char])

                start_tok = end_tok
                break_ptr += 1
            else:
                start_char = offsets[start_tok][0]
                rest = context[start_char:]

                if chunks:
                    chunks[-1] += rest
                else:
                    chunks.append(rest)
                break

        if len(chunks) >= 2:
            last_ids = tokenizer(chunks[-1], add_special_tokens=False)["input_ids"]
            if len(last_ids) <= min_tokens - 1:
                chunks[-2] += chunks[-1]
                chunks.pop()

        return chunks

    def split_into_passages(context: str):
        if not context:
            return []

        pattern = re.compile(r'Passage\s*:?\s*\d+\s*:')

        matches = list(pattern.finditer(context))
        if not matches:
            return [context]

        chunks = []

        first_start = matches[0].start()
        if first_start > 0:
            prefix = context[:first_start]
            if prefix.strip():
                chunks.append(prefix)

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
            chunk = context[start:end]
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    data = load_dataset(
        "THUDM/LongBench",
        dataset_name,
        split="test",
        trust_remote_code=True
    )

    data_filtered = []
    for item in data:
        context = item["context"]
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]

        if len(context_ids) > args.max_input_length:
            continue

        if args.one_chunk:
            item["chunks"] = [context]
        else:
            if dataset_name in ["multifieldqa_en", "narrativeqa", "qasper"]:
                item["chunks"] = split_into_chunks(context, min_tokens=args.chunk_size)
            elif dataset_name in ["2wikimqa", "hotpotqa", "musique"]:
                item["chunks"] = split_into_passages(context)

        data_filtered.append(item)

    return data_filtered


def _compute_exact_match(prediction: str, gold_answers: list) -> float:
    """Exact Match: 任一 gold answer 作为子串出现在 prediction 中即为 1。"""
    pred_lower = prediction.lower()
    for ga in gold_answers:
        if ga.replace("</s>", "").strip().lower() in pred_lower:
            return 1.0
    return 0.0


def _compute_f1(prediction: str, gold_answers: list) -> float:
    """Token-level F1（来自 utilities.qa_f1_score），取所有 gold answers 的最大值。"""
    return max(qa_f1_score(prediction, ga) for ga in gold_answers)


def _compute_rouge_l(prediction: str, gold_answers: list) -> float:
    """ROUGE-L F-measure，取所有 gold answers 的最大值。"""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return max(
        scorer.score(ga.replace("</s>", "").strip(), prediction)["rougeL"].fmeasure
        for ga in gold_answers
    )


def _compute_bleu(prediction: str, gold_answers: list) -> float:
    """
    Sentence-level BLEU-4（SmoothingFunction method1），
    取所有 gold answers 的最大值。
    """
    smooth = SmoothingFunction().method1
    pred_tokens = prediction.lower().split()
    if len(pred_tokens) == 0:
        return 0.0
    return max(
        sentence_bleu(
            [ga.replace("</s>", "").strip().lower().split()],
            pred_tokens,
            smoothing_function=smooth,
        )
        for ga in gold_answers
    )


def _compute_meteor(prediction: str, gold_answers: list) -> float:
    """METEOR，取所有 gold answers 的最大值。"""
    pred_tokens = prediction.lower().split()
    if len(pred_tokens) == 0:
        return 0.0
    return max(
        nltk_meteor_score(
            [ga.replace("</s>", "").strip().lower().split()],
            pred_tokens,
        )
        for ga in gold_answers
    )


def _compute_bertscore_batch(predictions: list, gold_answers_list: list) -> list:
    """
    BERTScore（批量计算以提高效率）。
    对每个样本，取所有 gold answers 的最大 F1。
    返回 list[float]，长度 = len(predictions)。
    """
    # 展开成 (pred, ref) 对，记录对应关系
    flat_preds = []
    flat_refs = []
    sample_indices = []   # 每个 flat entry 属于哪个样本
    for idx, (pred, golds) in enumerate(zip(predictions, gold_answers_list)):
        for ga in golds:
            flat_preds.append(pred)
            flat_refs.append(ga.replace("</s>", "").strip())
            sample_indices.append(idx)

    if len(flat_preds) == 0:
        return [0.0] * len(predictions)

    _, _, f1s = bert_score_fn(
        flat_preds, flat_refs,
        lang="en",
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    f1_list = f1s.tolist()

    # 按样本取 max
    result = [0.0] * len(predictions)
    for flat_idx, sample_idx in enumerate(sample_indices):
        result[sample_idx] = max(result[sample_idx], f1_list[flat_idx])

    return result


METRIC_NAMES = ["exact_match", "f1", "rouge_l", "bleu", "meteor", "bert_score"]


def run_long_benchmark(data_filtered, max_new_tokens, prompt_format, model_reorder, model_gen):
    global tokenizer, args

    # 每个指标独立收集
    scores = {m: [] for m in METRIC_NAMES}

    # BERTScore 最后批量算，先收集 prediction 和 gold
    all_predictions = []
    all_gold_answers = []

    for i, item in enumerate(data_filtered):
        begin_time = time.time()
        indices = [1]
        question_prompt = prompt_format.format(**item)
        chunks = item["chunks"]
        gold_answers = item["answers"]

        print(f"\n[Group {i+1}]")
        print(f"Original context token length: {len(tokenizer(item['context'], add_special_tokens=False)['input_ids'])}")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Question: {question_prompt}")
        print(f"Gold answer: {gold_answers}")

        knowledge_latent = None
        for j, chunk in enumerate(chunks):
            if j == 0:
                knowledge_latent = model_knowledge(
                    model_obj=model_gen,
                    tokenizer=tokenizer,
                    knowledge=chunk,
                    add_special_tokens=True
                )
                indices.append(knowledge_latent["input_ids"].shape[1])
            else:
                temp_knowledge_latent = model_knowledge(
                    model_obj=model_gen,
                    tokenizer=tokenizer,
                    knowledge=chunk,
                    add_special_tokens=False
                )

                # --- forget 机制：在拼接新 chunk 前，对旧 knowledge 做剪枝 ---
                if args.forget_strategy != "none":
                    knowledge_latent = maybe_forget_before_concat(
                        knowledge_latent=knowledge_latent,
                        new_knowledge_latent=temp_knowledge_latent,
                        forget_strategy=args.forget_strategy,
                        max_knowledge_tokens=args.max_knowledge_tokens
                    )
                    # forget 后重建 indices
                    indices = rebuild_indices_from_segment_lengths(
                        knowledge_latent["segment_lengths"]
                    )

                knowledge_latent = concatenate_knowledge_latent(
                    knowledge_latent, temp_knowledge_latent
                )
                # 拼接后重建 indices
                indices = rebuild_indices_from_segment_lengths(
                    knowledge_latent["segment_lengths"]
                )

        print(f"Final knowledge token length: {knowledge_latent['input_ids'].shape[1]}")

        if args.reorder_base_layer >= 0:
            if not args.shuffle_knowledge:
                reordered_knowledge_latent = reorder(
                    model_obj=model_reorder,
                    tokenizer=tokenizer,
                    knowledge_latent=knowledge_latent,
                    question=question_prompt,
                    reorder_base_layer=args.reorder_base_layer,
                    indices=indices,
                    keep_num=args.keep_num,
                    query_mode=args.query_mode
                )
            else:
                shuffled_knowledge_latent, shuffled_indices = shuffle_knowledge(
                    knowledge_latent, indices
                )
                reordered_knowledge_latent = reorder(
                    model_obj=model_reorder,
                    tokenizer=tokenizer,
                    knowledge_latent=shuffled_knowledge_latent,
                    question=question_prompt,
                    reorder_base_layer=args.reorder_base_layer,
                    indices=shuffled_indices,
                    keep_num=args.keep_num,
                    query_mode=args.query_mode
                )
                del shuffled_knowledge_latent
        elif args.rotate_base_layer >= 0:
            reordered_knowledge_latent = rotate(
                model_obj=model_reorder,
                tokenizer=tokenizer,
                knowledge_latent=knowledge_latent,
                question=question_prompt,
                rotate_base_layer=args.rotate_base_layer,
                indices=indices,
                window_size=math.ceil(args.window_ratio * len(chunks)),
                rotate_cutoff=args.rotate_cutoff,
                query_mode=args.query_mode
            )
        else:
            reordered_knowledge_latent = knowledge_latent

        del knowledge_latent
        torch.cuda.empty_cache()

        candidate_answer = generate_with_hidden_prefix(
            model_obj=model_gen,
            tokenizer=tokenizer,
            knowledge_latent=reordered_knowledge_latent,
            question=question_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        del reordered_knowledge_latent
        torch.cuda.empty_cache()

        pred_text = candidate_answer["answer_text"]

        # --- 逐样本计算 5 个指标（BERTScore 最后批量算） ---
        em = _compute_exact_match(pred_text, gold_answers)
        f1 = _compute_f1(pred_text, gold_answers)
        rl = _compute_rouge_l(pred_text, gold_answers)
        bl = _compute_bleu(pred_text, gold_answers)
        mt = _compute_meteor(pred_text, gold_answers)

        scores["exact_match"].append(em)
        scores["f1"].append(f1)
        scores["rouge_l"].append(rl)
        scores["bleu"].append(bl)
        scores["meteor"].append(mt)

        all_predictions.append(pred_text)
        all_gold_answers.append(gold_answers)

        n = i + 1
        print(f"Candidate answer: {pred_text}")
        print(f"  EM={em:.4f}  F1={f1:.4f}  ROUGE-L={rl:.4f}  BLEU={bl:.4f}  METEOR={mt:.4f}")
        print(f"  Running avg  EM={np.mean(scores['exact_match']):.4f}  "
              f"F1={np.mean(scores['f1']):.4f}  "
              f"ROUGE-L={np.mean(scores['rouge_l']):.4f}  "
              f"BLEU={np.mean(scores['bleu']):.4f}  "
              f"METEOR={np.mean(scores['meteor']):.4f}")

        end_time = time.time()
        print(f"Step {n}: Time taken = {end_time - begin_time:.2f} seconds")

    # --- BERTScore 批量计算 ---
    print("\nComputing BERTScore for all samples ...")
    bert_scores = _compute_bertscore_batch(all_predictions, all_gold_answers)
    scores["bert_score"] = bert_scores

    # --- 汇总 ---
    avg_scores = {m: float(np.mean(scores[m])) for m in METRIC_NAMES}
    return avg_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long-context benchmark with hidden prefix")
    parser.add_argument("--model_name", type=str,
                        required=True,
                        help="HF model name or local path")
    parser.add_argument("--device", type=str,
                        default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                        choices=["2wikimqa", "hotpotqa", "qasper",
                                 "musique", "multifieldqa_en", "narrativeqa"])
    parser.add_argument("--max_input_length", default=16384, type=int)
    parser.add_argument("--one_chunk", action="store_true",
                        help="Whether to put the entire context in one chunk")
    parser.add_argument("--chunk_size", type=int, default=512)
    # --- reorder ---
    parser.add_argument("--reorder_base_layer", type=int, default=-1)
    parser.add_argument("--keep_num", type=int, default=5)
    parser.add_argument("--shuffle_knowledge", action="store_true",
                        help="Whether to shuffle knowledge blocks before reordering")
    parser.add_argument("--query_mode", type=str, default="last",
                        choices=["last", "mean"])
    # --- rotate ---
    parser.add_argument("--rotate_base_layer", type=int, default=-1)
    parser.add_argument("--window_ratio", type=float, default=0.3,
                        help="滑动窗口所占的 chunk 数量比例")
    parser.add_argument("--rotate_cutoff", action="store_true")
    # --- forget ---
    parser.add_argument("--max_knowledge_tokens", type=int, default=12800,
                        help="knowledge latent 的最大 token 数上限")
    parser.add_argument("--forget_strategy", type=str, default="none",
                        choices=["none", "random", "perplexity"],
                        help="forget 策略：none 不剪枝，random 随机删，perplexity 按 PPL 删")
    # --- misc ---
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device_map_mode", type=str, default="auto",
                        choices=["auto", "balanced", "balanced_low_0", "sequential"],
                        help="HF accelerate device_map mode")

    args = parser.parse_args()
    print(args)

    if args.reorder_base_layer >= 0 and args.rotate_base_layer >= 0:
        raise ValueError("reorder_base_layer 和 rotate_base_layer 不能同时 >= 0，请二选一。")

    # --- 只给自己的 logger 加 FileHandler，不用 basicConfig 污染全局 ---
    logger = logging.getLogger("Long_Benchmark")
    logger.setLevel(logging.INFO)
    _fh = logging.FileHandler("./Long_Benchmark_result_April.log", mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_fh)

    # 静默第三方库（rouge_score/absl）的 INFO 日志
    logging.getLogger("absl").setLevel(logging.WARNING)

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

    if args.reorder_base_layer >= 0 or args.rotate_base_layer >= 0:
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

    max_gen_dict = json.load(
        open("./longbench_config/dataset2maxlen.json", "r", encoding="utf-8-sig")
    )
    max_new_tokens = max_gen_dict[args.dataset]
    data_filtered = load_data(args.dataset, max_new_tokens)

    # 确保 NLTK 数据可用
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

    prompt_format = "Question: {input}\nAnswer:"
    avg_scores = run_long_benchmark(
        data_filtered=data_filtered,
        max_new_tokens=max_new_tokens,
        prompt_format=prompt_format,
        model_reorder=model_reorder,
        model_gen=model_gen
    )

    logger.info(args)
    logger.info(
        f"Dataset: {args.dataset}, Size: {len(data_filtered)}, "
        f"Max new tokens: {max_new_tokens}"
    )
    for metric_name in METRIC_NAMES:
        logger.info(f"Average {metric_name}: {avg_scores[metric_name]:.4f}")

    print(f"\n{'='*60}")
    print(f"[Final Results]  Dataset: {args.dataset}  |  Samples: {len(data_filtered)}")
    print(f"{'='*60}")
    for metric_name in METRIC_NAMES:
        print(f"  {metric_name:>15s} : {avg_scores[metric_name]:.4f}")
    print(f"{'='*60}")
