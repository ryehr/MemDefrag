#!/bin/bash
# Adaptive-K (tau=2, K_max=4) knowledge-retention experiments:
# 4 models x 2 datasets, 50 steps, 500 groups, eval at n in {1,10,20,30,40,50}.
# Tracer layer & attention mode per (model, dataset) follow Table 10 of the paper.

PY=${PY:-python}
LOGDIR=./adaptive_logs
mkdir -p ${LOGDIR}

COMMON="--group_num 500 --unrelated_num 49 --adaptive_k --adaptive_tau 2.0 --adaptive_k_max 4 \
  --shuffle_knowledge --forget_strategy perplexity --max_knowledge_tokens 12800 \
  --eval_steps 1,10,20,30,40,50 --seed 42"

run_cfg () {  # gpu model dataset layer query_mode tag
  CUDA_VISIBLE_DEVICES=$1 ${PY} QA_benchmark.py \
    --model_name "$2" --dataset "$3" --reorder_base_layer "$4" --query_mode "$5" \
    ${COMMON} --log_file ${LOGDIR}/adaptive_$6.log > ${LOGDIR}/adaptive_$6.out 2>&1
  echo "[$(date '+%F %T')] finished: $6 (exit $?)"
}

echo "[$(date '+%F %T')] Round 1: Llama + Qwen"
run_cfg 0 meta-llama/Llama-3.1-8B-Instruct  nqa   13 last llama_nqa   &
run_cfg 1 meta-llama/Llama-3.1-8B-Instruct  squad 13 mean llama_squad &
run_cfg 2 Qwen/Qwen2.5-7B-Instruct          nqa   14 mean qwen_nqa    &
run_cfg 3 Qwen/Qwen2.5-7B-Instruct          squad 14 mean qwen_squad  &
wait

echo "[$(date '+%F %T')] Round 2: Mistral + Gemma"
run_cfg 0 mistralai/Mistral-7B-Instruct-v0.3 nqa   15 mean mistral_nqa   &
run_cfg 1 mistralai/Mistral-7B-Instruct-v0.3 squad 15 mean mistral_squad &
run_cfg 2 google/gemma-2-9b-it               nqa   15 last gemma_nqa     &
run_cfg 3 google/gemma-2-9b-it               squad 15 mean gemma_squad   &
wait

echo "[$(date '+%F %T')] All adaptive-K experiments done."
