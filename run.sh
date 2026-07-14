#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=0 python QA_benchmark.py --dataset nqa --query_mode mean --unrelated_num 49  --reorder_base_layer 13 --keep_num 1 --shuffle_knowledge --forget_strategy perplexity &
wait


# DATASETS=(multifieldqa_en narrativeqa qasper)

# for DATASET in "${DATASETS[@]}"; do
#     echo "========== Dataset: ${DATASET} | query_mode: last =========="
#     # CUDA_VISIBLE_DEVICES=0 python Long_benchmark.py --dataset "${DATASET}" --keep_num 1 --query_mode last &
#     # CUDA_VISIBLE_DEVICES=1 python Long_benchmark.py --dataset "${DATASET}" --keep_num 2 --query_mode last &
#     # CUDA_VISIBLE_DEVICES=2 python Long_benchmark.py --dataset "${DATASET}" --keep_num 3 --query_mode last &
#     CUDA_VISIBLE_DEVICES=3 python Long_benchmark.py --dataset "${DATASET}" --keep_num 4 --query_mode last &
#     wait

#     # echo "========== Dataset: ${DATASET} | query_mode: mean =========="
#     # CUDA_VISIBLE_DEVICES=0 python Long_benchmark.py --dataset "${DATASET}" --keep_num 1 --query_mode mean &
#     # CUDA_VISIBLE_DEVICES=1 python Long_benchmark.py --dataset "${DATASET}" --keep_num 2 --query_mode mean &
#     # CUDA_VISIBLE_DEVICES=2 python Long_benchmark.py --dataset "${DATASET}" --keep_num 3 --query_mode mean &
#     # CUDA_VISIBLE_DEVICES=3 python Long_benchmark.py --dataset "${DATASET}" --keep_num 4 --query_mode mean &
#     # wait
# done

echo "All done."


echo "All jobs finished."