"""
LoCoMo_benchmark.py
===================
LoCoMo 多轮对话长期记忆评测（rebuttal 任务：single-hop, MemDefrag only）。

协议：
  - 每个对话的每个 session 序列化为一个 knowledge fragment（含 session 日期头与
    图片轮的 BLIP caption），按时间顺序注入 latent memory；
    超过 N_max 时触发 perplexity 比例遗忘（与论文 5.1 节协议一致）。
  - 全部 session 注入完毕后，用 single-hop（category 4）问题逐题查询：
    MemDefrag 先做 per-prompt 去碎片化（eager tracing @ tracer 层，Top-K 过滤），
    再以截断后的记忆为前缀生成答案。
  - 指标：token-level F1（LoCoMo 官方主指标）+ substring 准确率（论文 2.2 节规则）；
    另记录 tracing 命中率（evidence 所在 session 是否进入 Top-K）。

用法示例：
  python LoCoMo_benchmark.py --keep_num 2                      # MemDefrag Top-2
  python LoCoMo_benchmark.py --reorder_base_layer -1           # vanilla latent memory
"""

import argparse
import json
import logging
import random
import re
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utilities import (
    get_input_device,
    get_max_memory,
    model_knowledge,
    concatenate_knowledge_latent,
    rebuild_indices_from_segment_lengths,
    maybe_forget_before_concat,
    generate_with_hidden_prefix,
    reorder,
    qa_f1_score,
)


def serialize_session(conv, k):
    """一个 session 序列化为一个 fragment：日期头 + 逐轮文本（图片轮附 BLIP caption）。"""
    date = conv.get(f"session_{k}_date_time", "")
    lines = [f"Session {k} ({date}):"]
    for t in conv[f"session_{k}"]:
        txt = t["text"]
        if t.get("blip_caption"):
            txt += f' [shared photo: {t["blip_caption"]}]'
        lines.append(f'{t["speaker"]}: {txt}')
    return "\n".join(lines) + "\n\n"


def first_line(text):
    """取预测的第一个非空行（截掉模型自问自答的续写）。"""
    for line in text.strip().split("\n"):
        if line.strip():
            return line.strip()
    return text.strip()


def parse_evidence_sessions(evidence):
    """evidence 形如 ['D3:12', ...]（或其字符串形式）→ 涉及的 session 序号集合。"""
    if isinstance(evidence, str):
        try:
            evidence = eval(evidence)
        except Exception:
            evidence = [evidence]
    sessions = set()
    for e in evidence:
        m = re.match(r"D(\d+):", str(e))
        if m:
            sessions.add(int(m.group(1)))
    return sessions


def load_locomo(path, tokenizer, category, max_conv_tokens, conv_ids=None):
    """返回 [(sample_id, fragments[(session_k, text)], qa_list)]，只保留指定类别问题、
    且序列化总 token 数 < max_conv_tokens 的对话。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for s in data:
        if conv_ids and s["sample_id"] not in conv_ids:
            continue
        conv = s["conversation"]
        ks = sorted(int(k.split("_")[1]) for k in conv
                    if k.startswith("session_") and not k.endswith("date_time")
                    and isinstance(conv[k], list))
        frags = [(k, serialize_session(conv, k)) for k in ks]
        total = sum(len(tokenizer(t, add_special_tokens=False)["input_ids"])
                    for _, t in frags)
        if max_conv_tokens > 0 and total >= max_conv_tokens:
            continue
        qa = [q for q in s["qa"] if str(q["category"]) == str(category)]
        if not qa:
            continue
        out.append((s["sample_id"], frags, qa, total))
    return out


def build_memory(model_gen, tokenizer, frags, forget_strategy, max_knowledge_tokens):
    """按时间顺序注入所有 session fragments（首个片段带 bos），返回 (latent, indices)。"""
    knowledge_latent = None
    for _, text in frags:
        temp = model_knowledge(
            model_obj=model_gen, tokenizer=tokenizer, knowledge=text,
            add_special_tokens=(knowledge_latent is None),
        )
        if knowledge_latent is None:
            knowledge_latent = temp
            continue
        knowledge_latent = maybe_forget_before_concat(
            knowledge_latent=knowledge_latent, new_knowledge_latent=temp,
            forget_strategy=forget_strategy, max_knowledge_tokens=max_knowledge_tokens,
        )
        knowledge_latent = concatenate_knowledge_latent(knowledge_latent, temp)
    indices = rebuild_indices_from_segment_lengths(knowledge_latent["segment_lengths"])
    return knowledge_latent, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data_path", type=str, default="./data_new/locomo/locomo10.json")
    parser.add_argument("--category", type=str, default="4",
                        help="LoCoMo QA 类别：4=single-hop（已核实：1=multi-hop, 2=temporal, "
                             "3=open-domain, 5=adversarial）")
    parser.add_argument("--max_conv_tokens", type=int, default=18000,
                        help="只评测序列化总 token 数小于该值的对话；<=0 表示不过滤")
    parser.add_argument("--conv_ids", type=str, default="",
                        help="逗号分隔的 sample_id 白名单（如 conv-30,conv-26）；为空则不限制")
    parser.add_argument("--reorder_base_layer", type=int, default=13,
                        help="tracer 层；-1 表示 vanilla（不做 defragmentation）")
    parser.add_argument("--keep_num", type=int, default=2)
    parser.add_argument("--query_mode", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--max_knowledge_tokens", type=int, default=12800)
    parser.add_argument("--forget_strategy", type=str, default="perplexity",
                        choices=["none", "random", "perplexity"])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--verbose", action="store_true",
                        help="打印每道题的问题/gold/预测")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default="./adaptive_logs/locomo_result.log")
    parser.add_argument("--device_map_mode", type=str, default="auto")
    args = parser.parse_args()
    print(args)

    logging.basicConfig(filename=args.log_file, level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", force=True)
    logger = logging.getLogger("LoCoMo")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16
    max_memory = get_max_memory()

    if args.reorder_base_layer >= 0:
        model_reorder = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=dtype, device_map=args.device_map_mode,
            max_memory=max_memory, attn_implementation="eager").eval()
    else:
        model_reorder = None
    model_gen = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map=args.device_map_mode,
        max_memory=max_memory).eval()

    conv_ids = set(args.conv_ids.split(",")) if args.conv_ids else None
    convs = load_locomo(args.data_path, tokenizer, args.category,
                        args.max_conv_tokens, conv_ids)
    print(f"qualified conversations: {[(c[0], c[3], len(c[2])) for c in convs]}")

    all_f1, all_acc, all_hit = [], [], []
    all_f1_raw, all_acc_raw = [], []
    per_conv = {}

    for sample_id, frags, qa, total_tokens in convs:
        t0 = time.time()
        knowledge_latent, indices = build_memory(
            model_gen, tokenizer, frags, args.forget_strategy, args.max_knowledge_tokens)
        mem_len = knowledge_latent["input_ids"].shape[1]
        session_ks = [k for k, _ in frags]  # fragment 顺序 → session 序号
        print(f"\n[{sample_id}] {len(frags)} sessions, raw {total_tokens} tokens, "
              f"memory {mem_len} tokens ({time.time()-t0:.1f}s)")

        f1s, accs, hits = [], [], []
        f1s_raw, accs_raw = [], []
        for qi, q in enumerate(qa):
            question = q["question"] + " Answer: "
            gold = str(q["answer"])
            ev_sessions = parse_evidence_sessions(q.get("evidence", []))

            if args.reorder_base_layer >= 0:
                t = {}
                reordered = reorder(
                    model_obj=model_reorder, tokenizer=tokenizer,
                    knowledge_latent=knowledge_latent, question=question,
                    reorder_base_layer=args.reorder_base_layer, indices=indices,
                    keep_num=args.keep_num, query_mode=args.query_mode, timings=t)
                dens = t["densities"]
                topk_frag_ids = list(np.argsort(dens)[-args.keep_num:])
                topk_sessions = {session_ks[i] for i in topk_frag_ids
                                 if i < len(session_ks)}
                if ev_sessions:
                    hits.append(1 if ev_sessions & topk_sessions else 0)
            else:
                reordered = knowledge_latent

            out = generate_with_hidden_prefix(
                model_obj=model_gen, tokenizer=tokenizer,
                knowledge_latent=reordered, question=question,
                max_new_tokens=args.max_new_tokens, do_sample=False)
            pred = out["answer_text"]
            pred_trim = first_line(pred)

            gold_norm = gold.replace("</s>", "").strip().lower()
            f1s.append(qa_f1_score(pred_trim, gold))
            accs.append(1 if gold_norm in pred_trim.lower() else 0)
            f1s_raw.append(qa_f1_score(pred, gold))
            accs_raw.append(1 if gold_norm in pred.lower() else 0)
            if args.verbose:
                print(f"  Q{qi+1}: {q['question']}")
                print(f"    gold: {gold} | pred: {pred_trim[:100]}")

            if args.reorder_base_layer >= 0:
                del reordered
        per_conv[sample_id] = (np.mean(f1s), np.mean(accs),
                               np.mean(hits) if hits else float("nan"), len(f1s))
        all_f1 += f1s
        all_acc += accs
        all_hit += hits
        all_f1_raw += f1s_raw
        all_acc_raw += accs_raw
        print(f"[{sample_id}] n={len(f1s)}, F1={np.mean(f1s)*100:.2f}, "
              f"acc={np.mean(accs)*100:.2f}, "
              f"tracing hit@{args.keep_num}={np.mean(hits)*100:.2f}" if hits else
              f"[{sample_id}] n={len(f1s)}, F1={np.mean(f1s)*100:.2f}, acc={np.mean(accs)*100:.2f}")

        del knowledge_latent
        torch.cuda.empty_cache()

    tag = (f"MemDefrag Top-{args.keep_num} (layer {args.reorder_base_layer}, "
           f"{args.query_mode})" if args.reorder_base_layer >= 0 else "Vanilla")
    print(f"\n===== LoCoMo single-hop (category {args.category}), {tag} =====")
    print(f"conversations={len(convs)}, questions={len(all_f1)}")
    print(f"overall F1 (first-line) = {np.mean(all_f1)*100:.2f}")
    print(f"overall substring acc (first-line) = {np.mean(all_acc)*100:.2f}")
    print(f"overall F1 (raw) = {np.mean(all_f1_raw)*100:.2f}")
    print(f"overall substring acc (raw) = {np.mean(all_acc_raw)*100:.2f}")
    if all_hit:
        print(f"overall tracing hit@{args.keep_num} = {np.mean(all_hit)*100:.2f}")

    logger.info("\n")
    logger.info(f"LoCoMo category={args.category}, {tag}, "
                f"max_conv_tokens={args.max_conv_tokens}, conv_ids={args.conv_ids or 'all'}")
    logger.info(args)
    logger.info(f"per-conv (F1, acc, hit, n): { {k: tuple(round(float(x), 4) for x in v) for k, v in per_conv.items()} }")
    hit_str = f", hit={np.mean(all_hit)*100:.2f}" if all_hit else ""
    logger.info(f"OVERALL: convs={len(convs)}, n={len(all_f1)}, "
                f"F1={np.mean(all_f1)*100:.2f}, acc={np.mean(all_acc)*100:.2f}, "
                f"F1_raw={np.mean(all_f1_raw)*100:.2f}, "
                f"acc_raw={np.mean(all_acc_raw)*100:.2f}{hit_str}")


if __name__ == "__main__":
    main()
